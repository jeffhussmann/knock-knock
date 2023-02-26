import gzip
import logging
import sys
from itertools import chain, islice

from collections import defaultdict

import pysam

from knock_knock.experiment import Experiment, ensure_list
from knock_knock import visualize
from knock_knock import layout as layout_module

from hits import adapters, fastq, sam, sw, utilities
from hits.utilities import memoized_property

class IlluminaExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.fns.update({
            'no_overlap_outcome_counts': self.results_dir / 'no_overlap_outcome_counts.csv',
            'no_overlap_outcome_list': self.results_dir / 'no_overlap_outcome_list.txt',
            'too_short_outcome_list': self.results_dir / 'too_short_outcome_list.txt',
        })

        self.sequencing_primers = self.description.get('sequencing_primers', 'truseq')
        self.x_tick_multiple = 100

        self.layout_mode = 'illumina'

        self.outcome_fn_keys = [
            'outcome_list',
            'no_overlap_outcome_list',
            'too_short_outcome_list',
        ]

        for k in ['R1', 'R2', 'I1', 'I2']:
            if k in self.description:
                fastq_fns = ensure_list(self.description[k])
                self.fns[k] = [self.data_dir / name for name in fastq_fns]
        
                for fn in self.fns[k]:
                    if not fn.exists():
                        pass
                        #logging.warning(f'{self.group} {self.sample_name} specifies non-existent {fn}')

        self.paired_end = 'R2' in self.description

        if self.paired_end:
            self.read_types = [
                'stitched',
                'stitched_by_name',
                'nonredundant',
                'R1_no_overlap',
                'R2_no_overlap',
            ]
        else:
            self.read_types = [
                'trimmed',
                'trimmed_by_name',
                'nonredundant',
            ]

    @property
    def preprocessed_read_type(self):
        if self.paired_end:
            read_type = 'stitched_by_name'
        else:
            read_type = 'trimmed_by_name'
        return read_type

    @property
    def default_read_type(self):
        if self.paired_end:
            read_type = 'stitched_by_name'
        else:
            read_type = 'trimmed_by_name'
        return read_type

    @property
    def read_types_to_align(self):
        return [
            'stitched_by_name',
            'R1_no_overlap',
            'R2_no_overlap',
        ]

    @property
    def reads(self):
        # Standardizing names is important for sorting.
        return fastq.reads(self.fns['R1'], standardize_names=True)

    @property
    def read_pairs(self):
        # Standardizing names is important for sorting.
        return fastq.read_pairs(self.fns['R1'], self.fns['R2'], standardize_names=True, up_to_space=True)

    @memoized_property
    def no_overlap_qnames(self):
        return {r.name for r in self.reads_by_type('R1_no_overlap')}

    @memoized_property
    def R1_read_length(self):
        R1, R2 = next(self.read_pairs)
        return len(R1)
    
    @memoized_property
    def R2_read_length(self):
        R1, R2 = next(self.read_pairs)
        return len(R2)

    def check_combined_read_length(self):
        if self.paired_end:
            combined_read_length = self.R1_read_length + self.R2_read_length
            if combined_read_length < self.target_info.amplicon_length:
                logging.warning(f'Warning: {self.group} {self.name} combined read length ({combined_read_length}) less than expected amplicon length ({self.target_info.amplicon_length:,}).')

    @memoized_property
    def max_relevant_length(self):
        if self.paired_end:
            return self.R1_read_length + self.R2_read_length + 100
        else:
            return 600

    def get_read_layout(self, read_id, fn_key='bam_by_name', outcome=None, read_type=None):
        if self.paired_end and read_id in self.no_overlap_qnames:
            als = self.get_read_alignments(read_id, outcome=outcome)
            layout = layout_module.NonoverlappingPairLayout(als['R1'], als['R2'], self.target_info)
            return layout
        else:
            return super().get_read_layout(read_id, fn_key=fn_key, outcome=outcome, read_type=read_type)

    def get_read_diagram(self, qname, outcome=None, relevant=True, read_type=None, **kwargs):
        if self.paired_end and qname in self.no_overlap_qnames:
            als = self.get_read_alignments(qname, outcome=outcome)

            if relevant:
                layout = layout_module.NonoverlappingPairLayout(als['R1'], als['R2'], self.target_info)
                layout.categorize()
                to_plot = layout.relevant_alignments
            else:
                to_plot = als

            kwargs['inferred_amplicon_length'] = self.qname_to_inferred_length[qname]
            for k, v in self.diagram_kwargs.items():
                kwargs.setdefault(k, v)

            diagram = visualize.ReadDiagram(to_plot, self.target_info, **kwargs)

        else:
            diagram = super().get_read_diagram(qname, outcome=outcome, relevant=relevant, read_type=read_type, **kwargs)

        return diagram
    
    def no_overlap_alignment_groups(self, outcome=None):
        R1_read_type = 'R1_no_overlap'
        R2_read_type = 'R2_no_overlap'
        
        if outcome is not None:
            R1_fn_key = 'R1_no_overlap_bam_by_name'
            R2_fn_key = 'R2_no_overlap_bam_by_name'
            
        else:
            R1_fn_key = 'bam_by_name'
            R2_fn_key = 'bam_by_name'

        R1_groups = super().alignment_groups(outcome=outcome, fn_key=R1_fn_key, read_type=R1_read_type)
        R2_groups = super().alignment_groups(outcome=outcome, fn_key=R2_fn_key, read_type=R2_read_type)

        group_pairs = zip(R1_groups, R2_groups)

        for (R1_name, R1_als), (R2_name, R2_als) in group_pairs:
            if R1_name != R2_name:
                raise ValueError(R1_name, R2_name)
            else:
                yield R1_name, {'R1': R1_als, 'R2': R2_als}

    def alignment_groups(self, fn_key='bam_by_name', outcome=None, read_type=None):
        overlap_al_groups = super().alignment_groups(fn_key=fn_key, outcome=outcome, read_type=read_type)
        to_chain = [overlap_al_groups]

        if isinstance(outcome, str):
            # Note: this is confusing, but outcomes that are categories will get the 
            # no-overlap als included by the calls to self.alignment_groups for each
            # relevant (category, subcategory) pair.
            pass
        else:
            if 'R1_no_overlap' in self.read_types:
                no_overlap_al_groups = self.no_overlap_alignment_groups(outcome)
                to_chain.append(no_overlap_al_groups)

        return chain(*to_chain)
    
    def categorize_no_overlap_outcomes(self, max_reads=None):
        outcomes = defaultdict(list)

        with self.fns['no_overlap_outcome_list'].open('w') as fh:
            fh.write(f'## Generated at {utilities.current_time_string()}\n')

            alignment_groups = self.no_overlap_alignment_groups()

            if max_reads is not None:
                alignment_groups = islice(alignment_groups, max_reads)

            for name, als in self.progress(alignment_groups, desc='Categorizing non-overlapping read pairs'):
                try:
                    pair_layout = layout_module.NonoverlappingPairLayout(als['R1'], als['R2'], self.target_info)
                    pair_layout.categorize()
                except:
                    print(self.sample_name, name)
                    raise
                
                outcomes[pair_layout.category, pair_layout.subcategory].append(name)

                outcome = self.final_Outcome.from_layout(pair_layout)
                fh.write(f'{outcome}\n')

        # To make plotting easier, for each outcome, make a file listing all of
        # qnames for the outcome and a bam file (sorted by name) with all of the
        # alignments for these qnames.

        qname_to_outcome = {}
        full_bam_fns = {which: self.fns_by_read_type['bam_by_name'][f'{which}_no_overlap'] for which in ['R1', 'R2']}

        header = sam.get_header(full_bam_fns['R1'])
        alignment_sorters = sam.multiple_AlignmentSorters(header, by_name=True)

        for outcome, qnames in outcomes.items():
            outcome_fns = self.outcome_fns(outcome)
            outcome_fns['dir'].mkdir(exist_ok=True)

            for which in ['R1', 'R2']:
                alignment_sorters[outcome, which] = outcome_fns['bam_by_name'][f'{which}_no_overlap']
            
            with outcome_fns['no_overlap_query_names'].open('w') as fh:
                for qname in qnames:
                    qname_to_outcome[qname] = outcome
                    fh.write(qname + '\n')
        
        with alignment_sorters:
            saved_verbosity = pysam.set_verbosity(0)
            for which in ['R1', 'R2']:
                with pysam.AlignmentFile(full_bam_fns[which]) as full_bam_fh:
                    for al in full_bam_fh:
                        if al.query_name in qname_to_outcome:
                            outcome = qname_to_outcome[al.query_name]
                            alignment_sorters[outcome, which].write(al)
            pysam.set_verbosity(saved_verbosity)

    def trim_reads(self):
        ''' Trim a (potentially variable-length) barcode from the beginning of a read
        by searching for the expected sequence that the amplicon should begin with.
        '''

        ti = self.target_info

        primer_prefix_length = 6

        if ti.sequencing_direction == '+':
            start = ti.sequencing_start.start
            prefix = ti.target_sequence[start:start + primer_prefix_length]
        else:
            end = ti.sequencing_start.end
            prefix = utilities.reverse_complement(ti.target_sequence[end + 1 - primer_prefix_length:end + 1])

        prefix = prefix.upper()

        fns = self.fns_by_read_type['fastq']
        with gzip.open(fns['trimmed'], 'wt', compresslevel=1) as trimmed_fh, \
             open(self.fns['too_short_outcome_list'], 'w') as too_short_fh:

            too_short_fh.write(f'## Generated at {utilities.current_time_string()}\n')

            for read in self.progress(self.reads, desc='Trimming reads'):
                try:
                    start = read.seq.index(prefix, 0, 30)
                except ValueError:
                    start = 0

                end = adapters.trim_by_local_alignment(adapters.truseq_R2_rc, read.seq)

                trimmed = read[start:end]

                if len(trimmed) == 0:
                    outcome = self.final_Outcome(trimmed.name, len(trimmed), 'nonspecific amplification', 'primer dimer', 'n/a')
                    too_short_fh.write(f'{outcome}\n')
                else:
                    trimmed_fh.write(str(trimmed))

    def sort_trimmed_reads(self):
        reads = sorted(self.reads_by_type('trimmed'), key=lambda read: read.name)
        fn = self.fns_by_read_type['fastq']['trimmed_by_name']
        with gzip.open(fn, 'wt', compresslevel=1) as sorted_fh:
            for read in reads:
                sorted_fh.write(str(read))

    def stitch_read_pairs(self):
        before_R1 = adapters.primers[self.sequencing_primers]['R1']
        before_R2 = adapters.primers[self.sequencing_primers]['R2']

        # Setup for post-stitching trimming process. 
        match_length_required = 6
        window_size = 30

        ti = self.target_info

        start = ti.primers_by_side_of_target[5].start
        prefix = ti.target_sequence[start:start + match_length_required]

        end = ti.primers_by_side_of_target[3].end
        suffix = ti.target_sequence[end + 1 - match_length_required:end + 1]

        if ti.sequencing_direction == '-':
            prefix, suffix = utilities.reverse_complement(suffix), utilities.reverse_complement(prefix)

        fns = self.fns_by_read_type['fastq']

        with gzip.open(fns['stitched'], 'wt', compresslevel=1) as stitched_fh, \
             gzip.open(fns['R1_no_overlap'], 'wt', compresslevel=1) as R1_fh, \
             gzip.open(fns['R2_no_overlap'], 'wt', compresslevel=1) as R2_fh, \
             open(self.fns['too_short_outcome_list'], 'w') as too_short_fh:

            too_short_fh.write(f'## Generated at {utilities.current_time_string()}\n')

            description = 'Stitching read pairs'
            for R1, R2 in self.progress(self.read_pairs, desc=description):
                if R1.name != R2.name:
                    print(f'Error: read pairs are out of sync in {self.group} {self.name}.')
                    R1_fns = ','.join(str(fn) for fn in self.fns['R1'])
                    R2_fns = ','.join(str(fn) for fn in self.fns['R2'])
                    print(f'R1 file name: {R1_fns}')
                    print(f'R2 file name: {R2_fns}')
                    print(f'R1 read {R1.name} paired with R2 read {R2.name}.')
                    sys.exit(1)

                stitched = sw.stitch_read_pair(R1, R2, before_R1, before_R2, indel_penalty=-1000)

                if len(stitched) == len(R1) + len(R2):
                    # No overlap was detected.
                    R1_fh.write(str(R1))
                    R2_fh.write(str(R2))

                elif len(stitched) <= 10:
                    # 22.07.18: Should this check be done on the final trimmed read instead?
                    outcome = self.final_Outcome(stitched.name, len(stitched), 'nonspecific amplification', 'primer dimer', 'n/a')
                    too_short_fh.write(f'{outcome}\n')

                else:
                    # Trim after stitching to leave adapters in expected place during stitching.

                    try:
                        start = stitched.seq.index(prefix, 0, window_size)
                    except ValueError:
                        start = 0

                    try:
                        min_possible_end = len(stitched) - window_size
                        end = stitched.seq.index(suffix, min_possible_end, len(stitched)) + match_length_required
                    except ValueError:
                        end = len(stitched)

                    trimmed = stitched[start:end]

                    stitched_fh.write(str(trimmed))

    def sort_stitched_reads(self):
        reads = sorted(self.reads_by_type('stitched'), key=lambda read: read.name)
        fn = self.fns_by_read_type['fastq']['stitched_by_name']
        with gzip.open(fn, 'wt', compresslevel=1) as sorted_fh:
            for read in reads:
                sorted_fh.write(str(read))

    def preprocess(self):
        if self.paired_end:
            self.stitch_read_pairs()
            self.sort_stitched_reads()
        else:
            self.trim_reads()
            self.sort_trimmed_reads()

    def align(self):
        self.make_nonredundant_sequence_fastq()

        for read_type in self.read_types_to_align:
            self.generate_alignments(read_type)
            self.generate_supplemental_alignments_with_STAR(read_type)
            self.combine_alignments(read_type)

    def categorize(self):
        self.categorize_outcomes()

        self.generate_outcome_counts()
        self.generate_read_lengths()

        self.record_sanitized_category_names()