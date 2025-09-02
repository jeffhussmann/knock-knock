import logging
import shutil
import subprocess
import sys
import warnings

from urllib.parse import urlparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import pysam
import yaml

import Bio.SeqIO
from Bio import BiopythonWarning
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from hits import fasta, fastq, genomes, interval, mapping_tools, sam, sw, utilities
import knock_knock.editing_strategy
import knock_knock.effector
import knock_knock.pegRNAs

import knock_knock.utilities

logger = logging.getLogger(__name__)

def identify_homology_arms(donor_seq, donor_type, target_seq, cut_after, required_match_length=15):
    header = pysam.AlignmentHeader.from_references(['donor', 'target'], [len(donor_seq), len(target_seq)])
    mapper = sw.SeedAndExtender(donor_seq.encode(), 8, header, 'donor')
    
    target_bytes = target_seq.encode()
    
    alignments = {
        'before_cut': [],
        'after_cut': [],
    }

    seed_starts = {
        'before_cut': range(cut_after - required_match_length, 0, -1),
        'after_cut': range(cut_after, len(target_seq) - required_match_length),
    }

    for side in ['before_cut', 'after_cut']:
        for seed_start in seed_starts[side]:  
            alignments[side] = mapper.seed_and_extend(target_bytes, seed_start, seed_start + required_match_length, 'target')
            if alignments[side]:
                break

        else:
            results = {'failed': f'cannot locate homology arm on {side}'}
            return results
        
    possible_HA_boundaries = []
    
    for before_al in alignments['before_cut']:
        before_strand = sam.get_strand(before_al)

        for after_al in alignments['after_cut']:
            after_strand = sam.get_strand(after_al)

            if before_strand == after_strand:
                strand = before_strand

                if strand == '+':
                    if before_al.reference_end < after_al.reference_start:
                        possible_HA_boundaries.append((donor_seq, before_al.reference_start, after_al.reference_end))

                elif strand == '-':
                    if before_al.reference_start > after_al.reference_end:
                        flipped_seq = utilities.reverse_complement(donor_seq)
                        start = len(donor_seq) - 1 - (before_al.reference_end - 1)
                        end = len(donor_seq) - 1 - after_al.reference_start + 1
                        possible_HA_boundaries.append((flipped_seq, start, end))

    possible_HAs = []
    for possibly_flipped_donor_seq, HA_start, HA_end in possible_HA_boundaries:
        donor_window = possibly_flipped_donor_seq[HA_start:HA_end]

        donor_prefix = donor_window[:required_match_length]

        donor_suffix = donor_window[-required_match_length:]

        # Try to be resilient against multiple occurrence of HA substrings in the target
        # by prioritizing matches closest to the cut site.
        target_HA_start = target_seq.rfind(donor_prefix, 0, cut_after + required_match_length)
        target_HA_end = target_seq.find(donor_suffix, cut_after - required_match_length) + len(donor_suffix)

        if target_HA_start == -1 or target_HA_end == -1 or target_HA_start >= target_HA_end:
            results = {'failed': f'cannot locate homology arms in target'}
            return results

        relevant_target_seq = target_seq[target_HA_start:target_HA_end]

        total_HA_length = target_HA_end - target_HA_start

        mismatches_before_deletion = np.cumsum([t != d for t, d in zip(relevant_target_seq, donor_window)])

        flipped_target = relevant_target_seq[::-1]
        flipped_donor = donor_window[::-1]
        mismatches_after_deletion = np.cumsum([0] + [t != d for t, d in zip(flipped_target, flipped_donor)][:-1])[::-1]

        total_mismatches = mismatches_before_deletion + mismatches_after_deletion

        last_index_in_HA_1 = int(np.argmin(total_mismatches))
        min_mismatches = total_mismatches[last_index_in_HA_1]

        lengths = {}
        lengths['HA_1'] = last_index_in_HA_1 + 1
        lengths['HA_2'] = total_HA_length - lengths['HA_1']
        lengths['donor_specific'] = len(donor_seq) - total_HA_length
        
        info = {
            'min_mismatches': min_mismatches,
            'possibly_flipped_donor_seq': possibly_flipped_donor_seq,
            'donor_HA_start': HA_start,
            'donor_HA_end': HA_end,
            'target_HA_start': target_HA_start,
            'target_HA_end': target_HA_end,
            'lengths': lengths,
        }
        possible_HAs.append((info))
        
    def priority(info):
        return info['min_mismatches'], -min(info['lengths']['HA_1'], info['lengths']['HA_2'])

    if not possible_HAs:
        results = {'failed': 'cannot locate homology arms'}
    else:
        results = min(possible_HAs, key=priority)

    lengths = results['lengths']

    donor_starts = {
        'HA_1': results['donor_HA_start'],
        'donor_specific': results['donor_HA_start'] + lengths['HA_1'],
        'HA_2': results['donor_HA_end'] - lengths['HA_2'],
    }
    donor_ends = {
        'HA_1': donor_starts['HA_1'] + lengths['HA_1'],
        'donor_specific': donor_starts['HA_2'],
        'HA_2': donor_starts['HA_2'] + lengths['HA_2'],
    }

    if donor_type == 'PCR':
        if donor_starts['HA_1'] != 0:
            donor_starts['PCR_adapter_1'] = 0
            donor_ends['PCR_adapter_1'] = donor_starts['HA_1']

        if donor_ends['HA_2'] != len(donor_seq):
            donor_starts['PCR_adapter_2'] = donor_ends['HA_2']
            donor_ends['PCR_adapter_2'] = len(donor_seq)

    target_starts = {
        'HA_1': results['target_HA_start'],
        'HA_2': results['target_HA_end'] - lengths['HA_2'],
    }
    target_ends = {key: target_starts[key] + lengths[key] for key in target_starts}

    donor_strand = 1
    target_strand = 1

    donor_features = [
        SeqFeature(location=FeatureLocation(donor_starts[feature_name], donor_ends[feature_name], strand=donor_strand),
                   id=feature_name,
                   type='misc_feature',
                   qualifiers={
                       'label': feature_name,
                       'ApEinfo_fwdcolor': feature_colors[feature_name],
                   },
                  )
        for feature_name in donor_starts
    ]

    target_features = ([
        SeqFeature(location=FeatureLocation(target_starts[feature_name], target_ends[feature_name], strand=target_strand),
                   id=feature_name,
                   type='misc_feature',
                   qualifiers={
                       'label': feature_name,
                       'ApEinfo_fwdcolor': feature_colors[feature_name],
                   },
                  )
        for feature_name in target_starts
    ])

    HA_info = {
        'possibly_flipped_donor_seq': results['possibly_flipped_donor_seq'],
        'donor_features': donor_features,
        'target_features': target_features,
    }

    return HA_info

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
    
    info should have keys:
        name
        genome
        amplicon_primers
        sgRNAs
    optional keys:
        donor
        nonhomologous_donor_sequence
        extra_sequences
    '''

    def __init__(self,
                 base_dir,
                 info,
                 defer_HA_identification=False,
                ):

        self.base_dir = Path(base_dir)
        self.info = info
        self.index_locations = knock_knock.editing_strategy.locate_supplemental_indices(self.base_dir)

        self.extra_sequences = load_extra_sequences(self.base_dir)

        self.extra_genbank_records = load_extra_genbank_records(self.base_dir)

        self.defer_HA_identification = defer_HA_identification

        self.name = self.info['name']

        self.target_dir = knock_knock.editing_strategy.get_strategies_dir(self.base_dir) / self.name
        self.target_dir.mkdir(parents=True, exist_ok=True)

        # Only align primer sequence downstream of any N's.
        self.primers_name, primers = self.info['amplicon_primers']

        self.primers = {}

        primers = [primer.upper().split('N')[-1] for primer in primers.split(';')]

        if len(primers) == 1:
            self.primers[self.primers_name] = primers[0]
        else:
            for primer_i, primer in enumerate(primers):
                self.primers[f'{self.primers_name}_{primer_i}'] = primer

        self.genome = self.info['genome']

        if self.primers_name is None:
            self.target_name = self.name
        else:
            self.target_name = self.primers_name

        # self.target_name will used as a path component, so can't have a forward slash.
        self.target_name = self.target_name.replace('/', '_SLASH_')

        self.sgRNAs = self.info['sgRNAs']
        if self.sgRNAs is None:
            self.sgRNAs = []

    def identify_protospacer_features_in_amplicon(self,
                                                  amplicon_sequence,
                                                  amplicon_description=None,
                                                 ):

        protospacer_features_in_amplicon = {}

        for sgRNA_name, components in sorted(self.sgRNAs):
            try:
                effector = knock_knock.effector.effectors[components['effector']]
                protospacer_feature = effector.identify_protospacer_in_target(amplicon_sequence, components['protospacer'])
                protospacer_features_in_amplicon[sgRNA_name] = protospacer_feature

            except ValueError:
                if amplicon_description is not None:
                    sgRNA_description = f'{sgRNA_name} {components["effector"]} protospacer: {components["protospacer"]}'
                    logger.warning(f'A protospacer sequence adjacent to an appropriate PAM could not be located for {sgRNA_description} in target {amplicon_description}')

                if components['extension'] != '':
                    # pegRNAs must have a protospacer in target.
                    raise ValueError

        return protospacer_features_in_amplicon

    def build(self, generate_pegRNA_genbanks=False):
        donor_info = self.info.get('donor')

        if donor_info is None:
            donor_name = None
            donor_seq = None
        else:
            donor_name, donor_seq = donor_info
            if donor_name is None:
                donor_name = f'{self.name}_donor'

        if donor_seq is None:
            has_donor = False
        else:
            has_donor = True

        if self.info.get('donor_type') is None:
            donor_type = None
        else:
            _, donor_type = self.info['donor_type']

        nh_donor_info = self.info.get('nonhomologous_donor_sequence')

        if nh_donor_info is None:
            nh_donor_name = None
            nh_donor_seq = None
        else:
            nh_donor_name, nh_donor_seq = nh_donor_info
            if nh_donor_name is None:
                nh_donor_name = f'{self.name}_NH_donor'

        if nh_donor_seq is None:
            has_nh_donor = False
        else:
            has_nh_donor = True

        left_primer_al, right_primer_al = self.align_primers()

        ref_name = left_primer_al.reference_name
        amplicon_start = left_primer_al.reference_start
        amplicon_end = right_primer_al.reference_end
        amplicon_sequence = self.region_fetcher(ref_name, amplicon_start, amplicon_end).upper()

        amplicon_description = f'{left_primer_al.reference_name}:{left_primer_al.reference_start:,}-{right_primer_al.reference_end:,}'
        protospacer_features_in_amplicon = self.identify_protospacer_features_in_amplicon(amplicon_sequence, amplicon_description=amplicon_description)

        final_window_around = min(500, amplicon_start)

        def amplicon_coords_to_target_coords(p):
            return p + final_window_around

        def genomic_coords_to_target_coords(p):
            return p - amplicon_start + final_window_around

        convert_strand = {
            '+': 1,
            '-': -1,
        }

        target_sequence = self.region_fetcher(ref_name, amplicon_start - final_window_around, amplicon_end + final_window_around).upper()
        
        left_primer_location = FeatureLocation(genomic_coords_to_target_coords(left_primer_al.reference_start),
                                               genomic_coords_to_target_coords(left_primer_al.reference_end),
                                               strand=convert_strand['+'],
                                              )

        right_primer_location = FeatureLocation(genomic_coords_to_target_coords(right_primer_al.reference_start),
                                                genomic_coords_to_target_coords(right_primer_al.reference_end),
                                                strand=convert_strand['-'],
                                               )

        target_features = [
            SeqFeature(location=left_primer_location,
                       id=left_primer_al.query_name,
                       type='misc_feature',
                       qualifiers={
                           'label': left_primer_al.query_name,
                           'ApEinfo_fwdcolor': feature_colors['forward_primer'],
                       },
                      ),
            SeqFeature(location=right_primer_location,
                       id=right_primer_al.query_name,
                       type='misc_feature',
                       qualifiers={
                           'label': right_primer_al.query_name,
                           'ApEinfo_fwdcolor': feature_colors['reverse_primer'],
                       },
                      ),
            SeqFeature(location=left_primer_location,
                       id='anchor',
                       type='misc_feature',
                       qualifiers={
                           'label': 'anchor',
                       },
                      ),
        ]

        protospacer_features = []
        for sgRNA_name, feature in protospacer_features_in_amplicon.items():
            location = FeatureLocation(amplicon_coords_to_target_coords(feature.start),
                                       amplicon_coords_to_target_coords(feature.end), 
                                       strand=convert_strand[feature.strand],
                                      )
            protospacer_feature = SeqFeature(location=location,
                                             id=sgRNA_name,
                                             type=f'protospacer',
                                             qualifiers={
                                                 'label': sgRNA_name,
                                                 'ApEinfo_fwdcolor': feature_colors['sgRNA'],
                                             },
                                            )
            protospacer_features.append(protospacer_feature)

        gb_records = {}

        if has_donor:
            if not self.defer_HA_identification:
                if len(self.sgRNAs) > 1:
                    raise ValueError

                sgRNA_name, components = self.sgRNAs[0]

                protospacer_feature = protospacer_features[0]
                effector = knock_knock.effector.effectors[components['effector']]

                # TODO: untested code branch here
                cut_after_offset = [offset for offset in effector.cut_after_offset if offset is not None][0]

                if protospacer_feature.location.strand == 1:
                    # protospacer_feature.end is the first nt of the PAM
                    cut_after = protospacer_feature.location.end + cut_after_offset
                else:
                    # protospacer_feature.start - 1 is the first nt of the PAM
                    cut_after = protospacer_feature.location.start - 1 - cut_after_offset - 1

                HA_info = identify_homology_arms(donor_seq, donor_type, target_sequence, cut_after)

                if 'failed' in HA_info:
                    raise ValueError

                donor_Seq = Seq(HA_info['possibly_flipped_donor_seq'])
                donor_features = HA_info['donor_features']
                target_features.extend(HA_info['target_features'])

            else:
                donor_Seq = Seq(donor_seq)
                donor_features = []

            donor_record = SeqRecord(donor_Seq,
                                     name=donor_name,
                                     features=donor_features,
                                     annotations={
                                         'molecule_type': 'DNA',
                                     },
                                    )

            gb_records[donor_name] = donor_record

        sgRNAs_with_extensions = [(name, components) for name, components in self.sgRNAs if components['extension'] != '']

        if len(sgRNAs_with_extensions) > 0:

            bad_pegRNAs = []

            for name, components in sgRNAs_with_extensions:
                try:
                    pegRNA = knock_knock.pegRNAs.pegRNA(name, components, self.target_name, target_sequence)
                except Exception as err:
                    bad_pegRNAs.append((name, str(err)))
                    continue

                pegRNA_SeqFeatures = [
                    SeqFeature(id=feature_name,
                               location=FeatureLocation(feature.start, feature.end + 1, strand=convert_strand[feature.strand]),
                               type='misc_feature',
                               qualifiers={
                                   'label': feature_name,
                                   'ApEinfo_fwdcolor': feature.attribute['color'],
                               },
                              )
                    for (seq_name, feature_name), feature in pegRNA.features.items() if seq_name == name
                ]

                pegRNA_Seq = Seq(components['full_sequence'])
                pegRNA_record = SeqRecord(pegRNA_Seq,
                                          name=name,
                                          features=pegRNA_SeqFeatures,
                                          annotations={
                                              'molecule_type': 'DNA',
                                          },
                                         )

                gb_records[name] = pegRNA_record

            if len(bad_pegRNAs) > 0:
                full_error_message = ['Error identifying valid protospacer/PBS for pegRNA(s):']
                for name, error_message in bad_pegRNAs:
                    full_error_message.append(f'{name}: {error_message}')

                full_error_message = '\n'.join(full_error_message)

                raise ValueError(full_error_message)
            
        if has_nh_donor:
            nh_donor_Seq = Seq(nh_donor_seq)
            nh_donor_record = SeqRecord(nh_donor_Seq, name=nh_donor_name, annotations={'molecule_type': 'DNA'})
            gb_records[nh_donor_name] = nh_donor_record

        target_Seq = Seq(target_sequence)
        target_record = SeqRecord(target_Seq,
                                  name=self.target_name,
                                  features=target_features,
                                  annotations={
                                      'molecule_type': 'DNA',
                                  },
                                 )
        gb_records[self.target_name] = target_record

        if self.info.get('extra_sequences') is not None:
            for extra_seq_name, extra_seq in self.info['extra_sequences']:
                if extra_seq_name in self.extra_genbank_records:
                    record = self.extra_genbank_records[extra_seq_name]
                else:
                    record = SeqRecord(Seq(extra_seq), name=extra_seq_name, annotations={'molecule_type': 'DNA'})

                gb_records[extra_seq_name] = record

        if self.info.get('donor') is not None:
            donor_name, donor_sequence = self.info['donor']
            if donor_name in self.extra_genbank_records:
                record = self.extra_genbank_records[donor_name]

                gb_records[donor_name] = record

        # Note: for debugging convenience, genbank files can be written for pegRNAs,
        # but these are NOT supplied as genbank records to make the final EditingStrategy,
        # since relevant features are either represented by the intial decomposition into
        # components or inferred on instantiation of the EditingStrategy.
        pegRNA_names = [name for name, components in sgRNAs_with_extensions]
        non_pegRNA_records = {name: record for name, record in gb_records.items() if name not in pegRNA_names}
        gb_records_for_manifest = non_pegRNA_records

        truncated_name_i = 0
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=BiopythonWarning)

            for which_seq, record in gb_records.items(): 

                if which_seq not in pegRNA_names or generate_pegRNA_genbanks:
                    gb_fn = self.target_dir / f'{which_seq}.gb'
                    try:
                        Bio.SeqIO.write(record, gb_fn, 'genbank')
                    except ValueError:
                        # locus line too long, can't write genbank file with BioPython
                        old_name = record.name

                        truncated_name = f'{record.name[:11]}_{truncated_name_i}'
                        record.name = truncated_name
                        Bio.SeqIO.write(record, gb_fn, 'genbank')

                        record.name = old_name

                        truncated_name_i += 1

        manifest_fn = self.target_dir / 'manifest.yaml'

        sources = sorted(gb_records_for_manifest)
            
        manifest = {
            'sources': sources,
            'target': self.target_name,
        }

        if self.info.get('donor') is not None:
            donor_name, donor_sequence = self.info['donor']
            manifest['donor'] = donor_name

        if has_donor:
            manifest['donor'] = donor_name
            manifest['donor_specific'] = 'donor_specific'
            if donor_type is not None:
                manifest['donor_type'] = donor_type

        if has_nh_donor:
            manifest['nonhomologous_donor'] = nh_donor_name

        manifest['features_to_show'] = [
            [self.target_name, left_primer_al.query_name],
            [self.target_name, right_primer_al.query_name],
        ]

        if has_donor:
            manifest['features_to_show'].extend([
                [donor_name, 'HA_1'],
                [donor_name, 'HA_2'],
                [donor_name, 'donor_specific'],
                [donor_name, 'PCR_adapter_1'],
                [donor_name, 'PCR_adapter_2'],
                [self.target_name, 'HA_1'],
                [self.target_name, 'HA_2'],
            ])

        manifest['genome_source'] = self.info.get('genome_source', self.genome)

        manifest['primer_names'] = [left_primer_al.query_name, right_primer_al.query_name]

        manifest_fn.write_text(yaml.dump(manifest, default_flow_style=False))

        gb_records = list(gb_records_for_manifest.values())

        strat = knock_knock.editing_strategy.EditingStrategy(self.base_dir, self.name, gb_records=gb_records)

        sgRNAs_df = load_sgRNAs(self.base_dir, process=False)
        sgRNA_names = sorted([name for name, _ in self.sgRNAs])
        sgRNAs_df.loc[sgRNA_names].to_csv(strat.fns['sgRNAs'])

        strat.make_protospacer_fastas()
        if strat.genome_source in self.index_locations:
            strat.map_protospacers(strat.genome_source)

    @utilities.memoized_property
    def region_fetcher(self):
        if self.genome in self.index_locations:
            region_fetcher = genomes.build_region_fetcher(self.index_locations[self.genome]['fasta'])
        else:
            if self.genome not in self.extra_sequences:
                raise ValueError(f'no sequence record found for "{self.genome}"')

            def region_fetcher(seq_name, start, end):
                return self.extra_sequences[seq_name][start:end]

        return region_fetcher

    def reference_length(self, reference_name):
        if self.genome in self.index_locations:
            genome_index = genomes.get_genome_index(self.index_locations[self.genome]['fasta'])
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
                    
                amplicon_sequence = self.region_fetcher(primer_al.reference_name, left_al.reference_start, right_al.reference_end).upper()
                
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

    def align_primers(self):
        if len(self.primers) == 2:
            alignment_tester = self.identify_concordant_primer_alignment_pair
        elif len(self.primers) == 1:
            alignment_tester = self.identify_virtual_primer

        if self.genome in self.index_locations:
            try:
                primer_alignments = self.align_primers_to_reference_genome_with_STAR()
                concordant_primer_alignments = alignment_tester(primer_alignments)

            except ValueError:
                logger.warning('Failed to find concordant primer alignments with STAR, falling back to manual search.')
                primer_alignments = self.align_primers_to_reference_genome_manually()
                concordant_primer_alignments = alignment_tester(primer_alignments)

        else:
            primer_alignments = self.align_primers_to_extra_sequence()
            concordant_primer_alignments = alignment_tester(primer_alignments)

        return concordant_primer_alignments

    def align_primers_to_reference_genome_with_STAR(self):
        if self.genome not in self.index_locations:
            raise ValueError(f'Can\'t locate indices for {self.genome}')

        primers_dir = self.target_dir / 'primer_alignment'
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

        genome_dictionary = genomes.load_entire_genome(self.index_locations[self.genome]['fasta'])

        primer_alignments = sw.align_primers_to_genome(self.primers, genome_dictionary, suffix_length=18)

        return primer_alignments

    def align_primers_to_extra_sequence(self):
        seq = self.region_fetcher(self.genome, None, None).upper()

        primer_alignments = sw.align_primers_to_sequence(self.primers, self.genome, seq)

        return primer_alignments

    def identify_concordant_primer_alignment_pair(self, primer_alignments, max_length=10000):
        ''' Find pairs of alignments that are on the same chromosome and point towards each other. '''

        def same_reference_name(first_al, second_al):
            return first_al.reference_name == second_al.reference_name

        def in_correct_orientation(first_al, second_al):
            ''' Check if first_al and second_al point towards each other on the same chromosome. '''
            
            if not same_reference_name(first_al, second_al):
                return None
            
            first_interval = interval.get_covered_on_ref(first_al)
            second_interval = interval.get_covered_on_ref(second_al)
            if interval.are_overlapping(first_interval, second_interval):
                return None
            
            left_al, right_al = sorted([first_al, second_al], key=lambda al: al.reference_start)

            if sam.get_strand(left_al) != '+' or sam.get_strand(right_al) != '-':
                return None
            else:
                return left_al, right_al

        def reference_extent(first_al, second_al):
            if not same_reference_name(first_al, second_al):
                return np.inf
            else:
                start = min(first_al.reference_start, second_al.reference_start)
                end = max(first_al.reference_end, second_al.reference_end)

                return end - start

        correct_orientation_pairs = []
        not_correct_orientation_pairs = []

        first_name, second_name = sorted(primer_alignments)

        for first_al in primer_alignments[first_name]:
            for second_al in primer_alignments[second_name]:
                if (oriented_pair := in_correct_orientation(first_al, second_al)) is not None:
                    correct_orientation_pairs.append(oriented_pair)

                elif reference_extent(first_al, second_al) < max_length:
                    not_correct_orientation_pairs.append((first_al, second_al))
                    
        if len(correct_orientation_pairs) == 0:
            if len(not_correct_orientation_pairs) > 0:
                for first_al, second_al in not_correct_orientation_pairs:
                    logger.warning(f'Found nearby primer alignments that don\'t point towards each other: {first_al.reference_name} {sam.get_strand(first_al)} {first_al.reference_start:,}-{first_al.reference_end:,}, {sam.get_strand(second_al)} {second_al.reference_start:,}-{second_al.reference_end:,}')

            raise ValueError(f'Could not identify primer binding sites in {self.genome} that point towards each other.')

        # Rank pairs in the correct orientation by (shortest) amplicon length.

        correct_orientation_pairs = sorted(correct_orientation_pairs, key=lambda pair: reference_extent(*pair))

        if len(correct_orientation_pairs) > 1:
            logger.warning(f'{len(correct_orientation_pairs)} concordant primer alignments found.')
            for left_al, right_al in correct_orientation_pairs[:10]:
                logger.warning(f'Found {reference_extent(left_al, right_al):,} nt amplicon on {left_al.reference_name} from {left_al.reference_start:,} to {right_al.reference_end - 1:,}')

            if len(correct_orientation_pairs) > 10:
                logger.warning(f'... and {len(correct_orientation_pairs) - 10} more')

        short_amplicons = [pair for pair in correct_orientation_pairs if reference_extent(*pair) <= 1000]
        if len(short_amplicons) > 1:
            logger.warning(f'{len(short_amplicons)} potential amplicons <= 1kb found. There is a risk the shortest amplicon may not be the intended target.')
            
        left_al, right_al = correct_orientation_pairs[0]

        if reference_extent(left_al, right_al) > max_length:
            logger.warning(f'Found {reference_extent(left_al, right_al):,} nt amplicon on {left_al.reference_name} from {left_al.reference_start:,} to {right_al.reference_end - 1:,}')
            raise ValueError(f'Could not identify an amplicon shorter than {max_length} in {self.genome}')

        return left_al, right_al

def load_sgRNAs(base_dir, process=True):
    '''
    If process == False, just pass along the DataFrame for subsetting.
    '''
    csv_fn = knock_knock.editing_strategy.get_strategies_dir(base_dir) / 'sgRNAs.csv'

    if not csv_fn.exists():
        return None
    else:
        return knock_knock.pegRNAs.read_csv(csv_fn, process=process)

def load_extra_sequences(base_dir):
    strategies_dir = knock_knock.editing_strategy.get_strategies_dir(base_dir)

    extra_sequences = {}

    fasta_extensions = {
        '.fasta',
        '.fa',
    }

    fasta_fns = sorted(fn for fn in strategies_dir.iterdir() if fn.suffix in fasta_extensions)

    for fasta_fn in fasta_fns:
        records = fasta.to_dict(fasta_fn)
        records = {name: seq.upper() for name, seq in records.items()}
        duplicates = set(extra_sequences) & set(records)
        if len(duplicates) > 0:
            raise ValueError(f'multiple records for {duplicates}')

        extra_sequences.update(records)

    extra_genbank_records = load_extra_genbank_records(base_dir)
    duplicates = set(extra_sequences) & set(extra_genbank_records)
    if len(duplicates) > 0:
        raise ValueError(f'multiple records for {duplicates}')

    extra_sequences.update({name: str(record.seq).upper() for name, record in extra_genbank_records.items()})

    return extra_sequences

def load_extra_genbank_records(base_dir):
    strategies_dir = knock_knock.editing_strategy.get_strategies_dir(base_dir)

    records = {}

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=BiopythonWarning)

        genbank_fns = sorted(strategies_dir.glob('*.gb'))
        for genbank_fn in genbank_fns:
            for record in Bio.SeqIO.parse(genbank_fn, 'gb'):
                if record.name in records:
                    raise ValueError(f'multiple records for {record.name}')

                records[record.name] = record

    return records

def build_component_registry(base_dir):
    base_dir = Path(base_dir)

    strategies_dir = knock_knock.editing_strategy.get_strategies_dir(base_dir)

    registry = {}

    registry['sgRNAs'] = load_sgRNAs(base_dir)

    amplicon_primers_fn = strategies_dir / 'amplicon_primers.csv'

    if amplicon_primers_fn.exists():
        registry['amplicon_primers'] = knock_knock.utilities.read_and_sanitize_csv(amplicon_primers_fn, index_col='name')
    else:
        registry['amplicon_primers'] = {}

    registry['extra_sequence'] = load_extra_sequences(base_dir)

    donors_fn = strategies_dir / 'donors.csv'

    if donors_fn.exists():
        donors = pd.read_csv(donors_fn, index_col='name')
        registry['donor_sequence'] = donors['donor_sequence']
        registry['donor_type'] = donors['donor_type']
    else:
        registry['donor_sequence'] = {}
        registry['donor_type'] = {}

    return registry

def build_editing_strategies_from_csv(base_dir, defer_HA_identification=False):
    base_dir = Path(base_dir)

    strategies_dir = knock_knock.editing_strategy.get_strategies_dir(base_dir)

    csv_fn = strategies_dir / 'strategies.csv'

    strategies_df = pd.read_csv(csv_fn, comment='#', index_col='name').replace({np.nan: None})

    registry = build_component_registry(base_dir)

    def lookup(row, column_to_lookup, registry_column, validate_sequence=True, multiple_lookups=False):
        value_to_lookup = row.get(column_to_lookup)
        if value_to_lookup is None:
            return None

        if multiple_lookups:
            values_to_lookup = value_to_lookup.split(';')
        else:
            values_to_lookup = [value_to_lookup]

        registered_values = registry[registry_column]
        valid_chars = set('TCAGN;')

        looked_up = []
        for value_to_lookup in values_to_lookup:
            value_to_lookup = value_to_lookup.strip()

            if value_to_lookup in registered_values:
                value_name = value_to_lookup
                seq = registered_values[value_to_lookup]
                possible_error_message = f'invalid char in {row.name} {column_to_lookup} registry entry {value_to_lookup}\n{seq}'
            else:
                raise ValueError(value_to_lookup)

            if seq is not None and validate_sequence:
                seq = seq.upper()
                invalid_chars = set(seq) - valid_chars
                if invalid_chars:
                    print(possible_error_message)
                    print(f'Valid sequence characters are {valid_chars}; {seq} contains {invalid_chars}')
                    sys.exit(1)

            looked_up.append((value_name, seq))

        if not multiple_lookups:
            looked_up = looked_up[0]
            
        return looked_up

    for target_name, row in strategies_df.iterrows():
        if '/' in target_name:
            raise ValueError(f'target names cannot contain a forward slash: {target_name}')

        info = {
            'name': target_name,
            'genome': row['genome'],
            'genome_source': row['genome_source'],
            'amplicon_primers': lookup(row, 'amplicon_primers', 'amplicon_primers'),
            'sgRNAs': lookup(row, 'sgRNAs', 'sgRNAs', multiple_lookups=True, validate_sequence=False),
            'donor': lookup(row, 'donor', 'extra_sequence'),
            'extra_sequences': lookup(row, 'extra_sequences', 'extra_sequence', multiple_lookups=True),
            'nonhomologous_donor_sequence': lookup(row, 'nonhomologous_donor_sequence', 'donor_sequence'),
        }

        if info['sgRNAs'] is not None:
            for sgRNA_name, sgRNA_components in info['sgRNAs']:
                if sgRNA_name is None:
                    # Because of how lookup works, sgRNA_components will hold value of 
                    # name that wasn't found.
                    raise ValueError(f'{sgRNA_components} not found')

        logger.info(f'Building {target_name}...')

        builder = EditingStrategyBuilder(base_dir,
                                         info,
                                         defer_HA_identification=defer_HA_identification,
                                        )
        builder.build(generate_pegRNA_genbanks=False)

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