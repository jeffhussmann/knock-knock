import logging
import shutil
import subprocess
import sys

from urllib.parse import urlparse
from pathlib import Path
from collections import Counter, defaultdict

import pysam
import yaml

import Bio.SeqIO
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from hits import fastq, genomes, interval, mapping_tools, sam, sw, utilities
import knock_knock.editing_strategy
import knock_knock.effector
import knock_knock.experiment
import knock_knock.pegRNAs

import knock_knock.utilities

logger = logging.getLogger(__name__)

feature_colors = {
    'HA_1': '#c7b0e3',
    'HA_RT': '#c7b0e3',
    'RTT': '#c7b0e3',
    'HA_2': '#85dae9',
    'HA_PBS': '#85dae9',
    'PBS': '#85dae9',
    'forward_primer': '#75C6A9',
    'reverse_primer': '#9eafd2',
    'sgRNA': '#c6c9d1',
    'donor_specific': '#b1ff67',
    'PCR_adapter_1': '#F8D3A9',
    'PCR_adapter_2': '#D59687',
    'protospacer': '#ff9ccd',
    'scaffold': '#b7e6d7',
}

class EditingStrategyBuilder:
    ''' 
    Attempts to identify the genomic location where an sgRNA sequence
    is flanked by amplicon primers.
    
    '''

    def __init__(self,
                 base_dir,
                 name,
                 genome,
                 amplicon_primer_names,
                 sgRNA_names,
                 genomes=None,
                ):

        self.base_dir = Path(base_dir)
        self.strategies_dir = knock_knock.editing_strategy.get_strategies_dir(self.base_dir)

        self.name = name
        self.dir = self.strategies_dir / self.name

        self.genome = genome

        def ensure_list(possible_string):
            converted = possible_string

            if isinstance(converted, str):
                if converted == '':
                    converted = []
                else:
                    converted = converted.split(';')
            
            return converted

        self.amplicon_primer_names = ensure_list(amplicon_primer_names)
        self.sgRNA_names = ensure_list(sgRNA_names)

        self.index_locations = knock_knock.editing_strategy.locate_supplemental_indices(self.base_dir)

        if genomes is None:
            genomes = knock_knock.editing_strategy.Genomes(self.base_dir)

        self.genomes = genomes

        self.dir = knock_knock.editing_strategy.get_strategies_dir(self.base_dir) / self.name

    @utilities.memoized_property
    def extra_sequences(self):
        fasta_records, _ = knock_knock.editing_strategy.load_all_fasta_records(self.strategies_dir)
        genbank_records, _ = knock_knock.editing_strategy.load_all_genbank_records(self.strategies_dir)

        all_records = fasta_records + genbank_records

        name_counts = Counter(record.name for record in all_records)

        duplicates = {name for name, count in name_counts.items() if count > 1} 

        if len(duplicates) > 0:
            raise ValueError('multiple records for names {duplicates}')

        extra_sequences = {record.name: str(record.seq).upper() for record in all_records}

        return extra_sequences

    @utilities.memoized_property
    def sgRNAs(self):
        csv_fn = self.strategies_dir / 'sgRNAs.csv'

        if csv_fn.exists():
            all_sgRNAs = knock_knock.pegRNAs.read_csv(csv_fn)
        else:
            all_sgRNAs = {}

        missing_names = set(self.sgRNA_names) - set(all_sgRNAs)

        if len(missing_names) > 0:
            raise ValueError(f'no entry for sgRNAs {missing_names}')

        sgRNAs = {name: components for name, components in all_sgRNAs.items() if name in self.sgRNA_names}

        return sgRNAs

    @utilities.memoized_property
    def primers(self):
        amplicon_primers_fn = self.strategies_dir / 'amplicon_primers.csv'

        if amplicon_primers_fn.exists():
            all_primers = knock_knock.utilities.read_and_sanitize_csv(amplicon_primers_fn, index_col='name').to_dict()
        else:
            all_primers = {}

        missing_names = set(self.amplicon_primer_names) - set(all_primers)

        if len(missing_names) > 0:
            raise ValueError(f'no entry for primers {missing_names}')

        full_primers = {name: seq for name, seq in all_primers.items() if name in self.amplicon_primer_names}

        # Only align primer sequence downstream of any N's.
        primers = {name: seq.upper().split('N')[-1] for name, seq in full_primers.items()}

        return primers

    def identify_protospacer_features_in_amplicon(self,
                                                  amplicon_sequence,
                                                  amplicon_description=None,
                                                 ):

        protospacer_features_in_amplicon = {}

        for sgRNA_name, components in sorted(self.sgRNAs.items()):
            try:
                effector = knock_knock.effector.effectors[components['effector']]
                protospacer_feature = effector.identify_protospacer_in_target(amplicon_sequence, components['protospacer'])
                protospacer_features_in_amplicon[sgRNA_name] = protospacer_feature

            except ValueError:
                if amplicon_description is not None:
                    sgRNA_description = f'{sgRNA_name} {components['effector']} protospacer: {components['protospacer']}'
                    logger.warning(f'A protospacer sequence adjacent to an appropriate PAM could not be located for {sgRNA_description} in target {amplicon_description}')

                if components['extension'] != '':
                    # pegRNAs must have a protospacer in target.
                    raise ValueError

        return protospacer_features_in_amplicon

    @utilities.memoized_property
    def extracted_target_sequence(self):
        if len(self.primers) > 0:
            left_primer_al, right_primer_al = self.concordant_primer_alignments_to_genome

            ref_name = left_primer_al.reference_name
            region_start = left_primer_al.reference_start
            region_end = right_primer_al.reference_end

            buffer = 200

        else:
            ref_name = self.protospacer_alignment.reference_name
            region_start = self.protospacer_alignment.reference_start
            region_end = self.protospacer_alignment.reference_end

            buffer = 6000

        target_start = max(0, region_start - buffer)
        target_end = min(self.reference_length(ref_name), region_end + buffer)

        target_sequence = self.region_fetcher(ref_name, target_start, target_end).upper()

        return target_sequence

    def build(self):
        self.dir.mkdir(exist_ok=True)

        convert_strand = {
            '+': 1,
            '-': -1,
        }

        primer_als = self.concordant_primer_alignments_to_target

        target_features = [
            SeqFeature(location=FeatureLocation(al.reference_start,
                                                al.reference_end,
                                                strand=convert_strand[sam.get_strand(al)],
                                               ),
                       id=al.query_name,
                       type='misc_feature',
                       qualifiers={
                           'label': al.query_name,
                           'ApEinfo_fwdcolor': feature_colors['forward_primer'],
                       },
                      )
            for al in primer_als
        ]

        target_name = 'target'
        target_Seq = Seq(self.extracted_target_sequence)
        target_record = SeqRecord(target_Seq,
                                  name=target_name,
                                  features=target_features,
                                  annotations={
                                      'molecule_type': 'DNA',
                                  },
                                 )

        gb_fn = self.dir / f'{target_name}.gb'

        Bio.SeqIO.write(target_record, gb_fn, 'genbank')

        parameters_fn = self.dir / 'parameters.yaml'

        parameters = {
            'target': target_name,
        }

        #manifest['genome_source'] = self.info.get('genome_source', self.genome)

        parameters['primer_names'] = self.amplicon_primer_names

        parameters_fn.write_text(yaml.dump(parameters, default_flow_style=False))

        #gb_records = list(gb_records.values())

        #strat = knock_knock.editing_strategy.EditingStrategy(self.base_dir, self.name, gb_records=gb_records)

        #strat.make_protospacer_fastas()
        #if strat.genome_source in self.index_locations:
        #    strat.map_protospacers(strat.genome_source)

    @utilities.memoized_property
    def region_fetcher(self):
        if self.genome in self.index_locations:
            region_fetcher = self.genomes.region_fetcher(self.genome)

        else:
            if self.genome not in self.extra_sequences:
                raise ValueError(f'no sequence record found for "{self.genome}"')

            def region_fetcher(seq_name, start, end):
                return self.extra_sequences[seq_name][start:end]

        return region_fetcher

    def reference_length(self, reference_name):
        if self.genome in self.index_locations:
            genome_index = self.genomes.fasta_index(self.genome)
            length = genome_index[reference_name].length
        else:
            length = len(self.extra_sequences[reference_name])

        return length

    def identify_virtual_primer(self, primer_alignments):
        ''' Given alignments of one primer, identify any alignments such that
            a concordant virtual primer a reasonable distance away would produce
            an amplicon containing valid protospacer locations for all required
            protospacers.
        '''

        virtual_primer_distances = [
            2000,
            4000,
            8000,
        ]

        virtual_primer_length = 20

        valids = []

        primer_name = list(primer_alignments)[0]

        for primer_al in primer_alignments[primer_name]:
            strand = sam.get_strand(primer_al)

            ref_name = primer_al.reference_name
            if ref_name == 'target':
                ref_length = len(self.extracted_target_sequence)
            else:
                ref_length = self.reference_length(ref_name)

            header = pysam.AlignmentHeader.from_references([ref_name], [ref_length])

            virtual_al = pysam.AlignedSegment(header)
            virtual_al.query_name = 'virtual_primer'
            virtual_al.cigar = [(sam.BAM_CMATCH, virtual_primer_length)]
            virtual_al.reference_name = primer_al.reference_name

            found_all_protospacers = False
            
            for virtual_primer_distance in virtual_primer_distances:
                if strand == '+':
                    virtual_primer_end = min(ref_length, primer_al.reference_end + virtual_primer_distance + virtual_primer_length)
                    virtual_primer_start = virtual_primer_end - virtual_primer_length
                    
                    left_al = primer_al
                    right_al = virtual_al
                    
                    virtual_al.is_reverse = True

                else:
                    virtual_primer_start = max(0, primer_al.reference_start - virtual_primer_distance - virtual_primer_length)
                    virtual_primer_end = virtual_primer_start + virtual_primer_length
                    
                    right_al = primer_al
                    left_al = virtual_al

                virtual_al.reference_start = virtual_primer_start
                    
                if ref_name == 'target':
                    amplicon_sequence = self.extracted_target_sequence[left_al.reference_start:right_al.reference_end].upper()
                else:
                    amplicon_sequence = self.region_fetcher(ref_name, left_al.reference_start, right_al.reference_end).upper()
                
                try:
                    self.identify_protospacer_features_in_amplicon(amplicon_sequence)
                    found_all_protospacers = True
                    break

                except:
                    continue

            if found_all_protospacers:
                valids.append((left_al, right_al))
            else:
                continue

        if len(valids) == 0:
            raise ValueError
        
        if len(valids) > 1:
            logger.warning(f'Found {len(valids)} possible valid primer alignments.')

        return valids[0]

    @utilities.memoized_property
    def concordant_primer_alignments_to_genome(self):
        if len(self.primers) == 2:
            alignment_tester = sw.identify_concordant_primer_alignment_pair

        elif len(self.primers) == 1:
            alignment_tester = self.identify_virtual_primer

        else:
            raise ValueError

        if self.genome in self.index_locations:
            try:
                primer_alignments = self.align_primers_to_reference_genome_with_STAR()
                concordant_primer_alignments = alignment_tester(primer_alignments)

            except:
                logger.warning('Failed to find concordant primer alignments with STAR, falling back to manual search.')
                primer_alignments = self.align_primers_to_reference_genome_manually()
                concordant_primer_alignments = alignment_tester(primer_alignments)

        else:
            primer_alignments = self.align_primers_to_extra_sequence()
            concordant_primer_alignments = alignment_tester(primer_alignments)

        return concordant_primer_alignments

    @utilities.memoized_property
    def concordant_primer_alignments_to_target(self):
        if len(self.primers) == 0:
            concordant_primer_alignments = []

        else:
            if len(self.primers) == 2:
                alignment_tester = sw.identify_concordant_primer_alignment_pair

            elif len(self.primers) == 1:
                alignment_tester = self.identify_virtual_primer

            else:
                raise ValueError

            primer_alignments = sw.align_primers_to_sequence(self.primers, 'target', self.extracted_target_sequence)
            concordant_primer_alignments = alignment_tester(primer_alignments)

        return concordant_primer_alignments

    def align_primers_to_reference_genome_with_STAR(self):
        if self.genome not in self.index_locations:
            raise ValueError(f'Can\'t locate indices for {self.genome}')

        primers_dir = self.dir / 'primer_alignment'
        primers_dir.mkdir(exist_ok=True)

        fastq_fn = primers_dir / 'primers.fastq'

        with fastq_fn.open('w') as fh:
            for primer_name, primer_seq in self.primers.items():
                quals = fastq.encode_sanger([40]*len(primer_seq))
                read = fastq.Read(primer_name, primer_seq, quals)
                fh.write(str(read))

        STAR_prefix = primers_dir / 'primers_'
        bam_fn = primers_dir / 'primers.bam'

        mapping_tools.map_STAR(fastq_fn,
                               self.index_locations[self.genome]['STAR'],
                               STAR_prefix,
                               mode='permissive',
                               bam_fn=bam_fn,
                              )

        primer_alignments = defaultdict(list)

        # Retain only alignments that include the 3' end of the primer.

        with pysam.AlignmentFile(bam_fn) as bam_fh:
            for al in bam_fh:
                covered = interval.get_covered(al)
                if len(al.query_sequence) - 1 in covered:
                    primer_alignments[al.query_name].append(al)
                    
        shutil.rmtree(primers_dir)

        return primer_alignments

    def align_primers_to_reference_genome_manually(self):
        if self.genome not in self.index_locations:
            raise ValueError(f'Can\'t locate indices for {self.genome}')

        loaded_genome = self.genomes.loaded(self.genome)

        primer_alignments = sw.align_primers_to_genome(self.primers, loaded_genome, suffix_length=18)

        return primer_alignments

    def align_primers_to_extra_sequence(self):
        seq = self.region_fetcher(self.genome, None, None).upper()

        primer_alignments = sw.align_primers_to_sequence(self.primers, self.genome, seq)

        return primer_alignments

    @utilities.memoized_property
    def protospacer_alignment(self):
        if len(self.sgRNAs) != 1:
            raise ValueError('WGS mode requires exactly one sgRNA')

        protospacer_name = list(self.sgRNAs)[0]
        components = self.sgRNAs[protospacer_name]

        protospacer_alignemnts = self.genomes.align_protospacer_to_genome(protospacer_name,
                                                                          components['protospacer'],
                                                                          components['effector'],
                                                                          self.genome,
                                                                         )

        if len(protospacer_alignemnts) == 0:
            raise ValueError('failed to align {protospacer_name} to {self.genome}')

        elif len(protospacer_alignemnts) > 1:
            raise ValueError('multiple alignments of {protospacer_name} to {self.genome}')
        
        protospacer_alignemnt = protospacer_alignemnts[0]

        return protospacer_alignemnt

def build_strategies(base_dir, batch_name, ignore_existing=False):
    data_dir = Path(base_dir) / 'data' / batch_name

    sample_sheet_fn = data_dir / 'sample_sheet.csv'

    sample_sheet = knock_knock.utilities.read_and_sanitize_csv(sample_sheet_fn, index_col='sample_name')

    strategies_dir = knock_knock.editing_strategy.get_strategies_dir(base_dir)

    genomes = knock_knock.editing_strategy.Genomes(base_dir)

    for _, row in sample_sheet.iterrows():
        editing_strategy_name = knock_knock.experiment.sample_sheet_row_to_editing_strategy_name(row)
        strategy_dir = strategies_dir / editing_strategy_name

        parameters_fn = strategy_dir / 'parameters.yaml'

        # Checking for existence of strategy_dir can be wrong
        # if there was an earlier failed attempt. Instead, check for
        # parameters since it is the last file written. 
        if not parameters_fn.exists() or ignore_existing:
            logger.info(f'Building {editing_strategy_name}')

            builder = EditingStrategyBuilder(base_dir,
                                             editing_strategy_name,
                                             row['genome'],
                                             row['amplicon_primers'],
                                             row['sgRNAs'],
                                             genomes=genomes,
                                            )

            builder.build()

def build_indices(base_dir, name, num_threads=1, RAM_limit=int(60e9), **STAR_index_kwargs):
    base_dir = Path(base_dir)

    logger.info(f'Building indices for {name}')
    fasta_dir = base_dir / 'indices' / name / 'fasta'

    fasta_fns = genomes.get_all_fasta_file_names(fasta_dir)
    if len(fasta_fns) == 0:
        raise ValueError(f'No fasta files found in {fasta_dir}')
    elif len(fasta_fns) > 1:
        raise ValueError(f'Can only build minimap2 index from a single fasta file')

    logger.info('Indexing fastas...')
    genomes.make_fais(fasta_dir)

    fasta_fn = fasta_fns[0]

    logger.info('Building STAR index...')
    STAR_dir = base_dir / 'indices' / name / 'STAR'
    STAR_dir.mkdir(exist_ok=True)
    mapping_tools.build_STAR_index([fasta_fn], STAR_dir,
                                   num_threads=num_threads,
                                   RAM_limit=RAM_limit,
                                   **STAR_index_kwargs,
                                  )

    logger.info('Building minimap2 index...')
    minimap2_dir = base_dir / 'indices' / name / 'minimap2'
    minimap2_dir.mkdir(exist_ok=True)
    minimap2_index_fn = minimap2_dir / f'{name}.mmi'
    mapping_tools.build_minimap2_index(fasta_fn, minimap2_index_fn)

def download_genome_and_build_indices(base_dir, genome_name, num_threads=8):
    urls = {
        'hg38': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz',
        'hg19': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz',
        'bosTau7': 'http://hgdownload.cse.ucsc.edu/goldenPath/bosTau7/bigZips/bosTau7.fa.gz',
        'macFas5': 'http://hgdownload.cse.ucsc.edu/goldenPath/macFas5/bigZips/macFas5.fa.gz',
        'mm10': 'ftp://ftp.ensembl.org/pub/release-98/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna.toplevel.fa.gz',
        'e_coli': 'ftp://ftp.ensemblgenomes.org/pub/bacteria/release-44/fasta/bacteria_0_collection/escherichia_coli_str_k_12_substr_mg1655/dna/Escherichia_coli_str_k_12_substr_mg1655.ASM584v2.dna.chromosome.Chromosome.fa.gz',
        'phiX': 'https://webdata.illumina.com/downloads/productfiles/igenomes/phix/PhiX_Illumina_RTA.tar.gz',
    }

    if genome_name not in urls:
        print(f'No URL known for {genome_name}.')
        print('Valid options are:')
        for gn in sorted(urls):
            print(f'\t- {gn}')
        sys.exit(1)

    base_dir = Path(base_dir)
    genome_dir = base_dir / 'indices' / genome_name
    fasta_dir = genome_dir / 'fasta'

    logger.info(f'Downloading {genome_name}...')

    wget_command = [
        'wget',
        '--quiet',
        urls[genome_name],
        '-P', str(fasta_dir),
    ]
    subprocess.run(wget_command, check=True)

    logger.info('Uncompressing...')

    file_name = Path(urlparse(urls[genome_name]).path).name

    if genome_name == 'phiX':
        tar_command = [
            'tar', 'xz',  f'--file={fasta_dir / file_name}', f'--directory={fasta_dir}',
        ]
        subprocess.run(tar_command, check=True)

        extracted_path = fasta_dir / 'PhiX' / 'Illumina' / 'RTA' / 'Sequence' / 'Chromosomes' / 'phix.fa'
        new_path = fasta_dir / 'phix.fa'
        extracted_path.rename(new_path)

        shutil.rmtree(fasta_dir / 'PhiX')
        (fasta_dir / 'README.txt').unlink()

        for tar_gz_fn in fasta_dir.glob('*.tar.gz'):
            tar_gz_fn.unlink()

    else:
        gunzip_command = [
            'gunzip', '--force',  str(fasta_dir / file_name),
        ]
        subprocess.run(gunzip_command, check=True)

    STAR_index_kwargs = {}

    if genome_name in ['e_coli', 'phiX']:
        STAR_index_kwargs['wonky_param'] = 4

    build_indices(base_dir, genome_name, num_threads=num_threads, **STAR_index_kwargs)