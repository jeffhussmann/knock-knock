import gzip
import shutil
import sys
from itertools import chain, islice

from collections import defaultdict
from contextlib import ExitStack

import pysam

from knock_knock.experiment import Experiment, ensure_list
from knock_knock import visualize
from knock_knock import layout as layout_module

import hits.visualize
from hits import adapters, fastq, sw, utilities
from hits.utilities import memoized_property, group_by

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
                        print(f'Warning: {self.group} {self.sample_name} specifies non-existent {fn}')

        self.read_types = [
            'stitched',
            'R1_no_overlap',
            'R2_no_overlap',
        ]

        self.trim_from_R1 = self.description.get('trim_from_R1', 0)
        self.trim_from_R2 = self.description.get('trim_from_R2', 0)

        self.diagram_kwargs.update(dict(draw_sequence=True,
                                        max_qual=41,
                                        center_on_primers=True,
                                        ),
                                  )

    def __repr__(self):
        return f'IlluminaExperiment: batch={self.batch}, sample_name={self.sample_name}, base_dir={self.base_dir}'

    @property
    def preprocessed_read_type(self):
        return 'stitched'

    @property
    def default_read_type(self):
        return 'stitched'

    @property
    def read_types_to_align(self):
        return [
            'stitched',
            'R1_no_overlap',
            'R2_no_overlap',
        ]

    @memoized_property
    def R1_read_length(self):
        R1, R2 = next(self.read_pairs)
        return len(R1) - self.trim_from_R1
    
    @memoized_property
    def R2_read_length(self):
        R1, R2 = next(self.read_pairs)
        return len(R2) - self.trim_from_R2

    def check_combined_read_length(self):
        combined_read_length = self.R1_read_length + self.R2_read_length
        if combined_read_length < self.target_info.amplicon_length:
            print(f'Warning: {self.group} {self.name} combined read length ({combined_read_length}) less than expected amplicon length ({self.target_info.amplicon_length:,}).')

    @memoized_property
    def max_relevant_length(self):
        return self.R1_read_length + self.R2_read_length + 100

    def get_read_layout(self, read_id, fn_key='bam_by_name', outcome=None, read_type=None):
        if read_id in self.no_overlap_qnames:
            als = self.get_no_overlap_read_alignments(read_id, outcome=outcome)
            layout = layout_module.NonoverlappingPairLayout(als['R1'], als['R2'], self.target_info)
            return layout
        else:
            return super().get_read_layout(read_id, fn_key=fn_key, outcome=outcome, read_type=read_type)

    def get_read_alignments(self, read_id, fn_key='bam_by_name', outcome=None, read_type=None):
        if read_type is None:
            if read_id in self.no_overlap_qnames:
                als = self.get_no_overlap_read_alignments(read_id, outcome=outcome)
            else:
                als = super().get_read_alignments(read_id, fn_key=fn_key, outcome=outcome, read_type=self.default_read_type)
        else:
            als = super().get_read_alignments(read_id, fn_key=fn_key, outcome=outcome, read_type=read_type)
            
        return als

    def get_no_overlap_read_alignments(self, read_id, outcome=None):
        als = {}

        for which in ['R1', 'R2']:
            als[which] = self.get_read_alignments(read_id, fn_key='bam_by_name', outcome=outcome, read_type=f'{which}_no_overlap')

        return als

    def get_read_diagram(self, qname, outcome=None, relevant=True, read_type=None, **kwargs):
        if qname in self.no_overlap_qnames:
            als = self.get_no_overlap_read_alignments(qname, outcome=outcome)

            if relevant:
                layout = layout_module.NonoverlappingPairLayout(als['R1'], als['R2'], self.target_info)
                layout.categorize()
                to_plot = layout.relevant_alignments
                length = layout.inferred_amplicon_length
                if length == -1:
                    length = self.length_to_store_unknown
                kwargs['inferred_amplicon_length'] = length
            else:
                to_plot = als

            for k, v in self.diagram_kwargs.items():
                kwargs.setdefault(k, v)

            diagram = visualize.ReadDiagram(to_plot, self.target_info, **kwargs)

        else:
            diagram = super().get_read_diagram(qname, outcome=outcome, relevant=relevant, read_type=read_type, **kwargs)

        return diagram
    
    @property
    def read_pairs(self):
        read_pairs = fastq.read_pairs(self.fns['R1'], self.fns['R2'], up_to_space=True)
        return read_pairs

    @memoized_property
    def no_overlap_qnames(self):
        return {r.name for r in self.reads_by_type('R1_no_overlap')}

    def no_overlap_alignment_groups(self, outcome=None):
        R1_read_type = 'R1_no_overlap'
        R2_read_type = 'R2_no_overlap'
        
        if outcome is not None:
            R1_fn_key = 'R1_no_overlap_bam_by_name'
            R2_fn_key = 'R2_no_overlap_bam_by_name'
            
        else:
            R1_fn_key = 'bam_by_name'
            R2_fn_key = 'bam_by_name'

        saved_verbosity = pysam.set_verbosity(0)
        R1_groups = self.alignment_groups(outcome=outcome, fn_key=R1_fn_key, read_type=R1_read_type)
        R2_groups = self.alignment_groups(outcome=outcome, fn_key=R2_fn_key, read_type=R2_read_type)
        pysam.set_verbosity(saved_verbosity)

        group_pairs = zip(R1_groups, R2_groups)

        for (R1_name, R1_als), (R2_name, R2_als) in group_pairs:
            if R1_name != R2_name:
                raise ValueError(R1_name, R2_name)
            else:
                yield R1_name, {'R1': R1_als, 'R2': R2_als}
    
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
        bam_fhs = {}

        with ExitStack() as stack:
            full_bam_fns = {which: self.fns_by_read_type['bam_by_name'][f'{which}_no_overlap'] for which in ['R1', 'R2']}
            full_bam_fhs = {which: stack.enter_context(pysam.AlignmentFile(full_bam_fns[which])) for which in ['R1', 'R2']}
        
            for outcome, qnames in outcomes.items():
                outcome_fns = self.outcome_fns(outcome)
                outcome_fns['dir'].mkdir(exist_ok=True)
                for which in ['R1', 'R2']:
                    bam_fn = outcome_fns['bam_by_name'][f'{which}_no_overlap']
                    bam_fhs[outcome, which] = stack.enter_context(pysam.AlignmentFile(bam_fn, 'wb', template=full_bam_fhs[which]))
                
                fh = stack.enter_context(outcome_fns['no_overlap_query_names'].open('w'))
                for qname in qnames:
                    qname_to_outcome[qname] = outcome
                    fh.write(qname + '\n')
            
            for which in ['R1', 'R2']:
                for al in full_bam_fhs[which]:
                    if al.query_name in qname_to_outcome:
                        outcome = qname_to_outcome[al.query_name]
                        bam_fhs[outcome, which].write(al)

    def stitch_read_pairs(self):
        before_R1 = adapters.primers[self.sequencing_primers]['R1']
        before_R2 = adapters.primers[self.sequencing_primers]['R2']

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

                if len(stitched) == self.R1_read_length + self.R2_read_length:
                    # No overlap was detected.
                    R1_fh.write(str(R1))
                    R2_fh.write(str(R2))

                elif len(stitched) <= 10:
                    outcome = self.final_Outcome(stitched.name, len(stitched), 'malformed layout', 'too short', 'n/a')
                    too_short_fh.write(f'{outcome}\n')

                else:
                    # Trim after stitching to leave adapters in expected place during stitching.
                    stitched = stitched[self.trim_from_R1:len(stitched) - self.trim_from_R2]

                    stitched_fh.write(str(stitched))

    def generate_length_range_figures(self, specific_outcome=None, num_examples=1):
        def extract_length(als):
            if isinstance(als, dict):
                pair_layout = layout_module.NonoverlappingPairLayout(als['R1'], als['R2'], self.target_info)
                pair_layout.categorize()
                length = pair_layout.inferred_amplicon_length
            else:
                length = als[0].query_length
                
            if length == -1:
                converted = self.length_to_store_unknown
            elif length > self.max_relevant_length:
                converted = self.max_relevant_length
            else:
                converted = length
            
            return converted

        by_length = defaultdict(lambda: utilities.ReservoirSampler(num_examples))

        al_groups = self.alignment_groups(outcome=specific_outcome)
        no_overlap_al_groups = self.no_overlap_alignment_groups(outcome=specific_outcome)

        for name, als in chain(al_groups, no_overlap_al_groups):
            length = extract_length(als)
            by_length[length].add((name, als))

        if specific_outcome is None:
            fns = self.fns
        else:
            fns = self.outcome_fns(specific_outcome)

        fig_dir = fns['length_ranges_dir']
            
        if fig_dir.is_dir():
            shutil.rmtree(str(fig_dir))
        fig_dir.mkdir()

        if specific_outcome is not None:
            description = ': '.join(specific_outcome)
        else:
            description = 'Generating length-specific diagrams'

        items = self.progress(by_length.items(), desc=description, total=len(by_length))

        for length, sampler in items:
            als = sampler.sample
            diagrams = self.alignment_groups_to_diagrams(als, num_examples=num_examples, **self.diagram_kwargs)
            im = hits.visualize.make_stacked_Image([d.fig for d in diagrams])
            fn = fns['length_range_figure'](length, length)
            im.save(fn)

    def preprocess(self):
        self.stitch_read_pairs()

    def process(self, stage):
        try:
            if stage == 'preprocess':
                self.preprocess()

            elif stage == 'align':
                
                for read_type in self.read_types_to_align:
                    self.generate_alignments(read_type)
                    self.generate_supplemental_alignments_with_STAR(read_type)
                    self.combine_alignments(read_type)

            elif stage == 'categorize':
                self.categorize_outcomes(read_type='stitched')
                self.categorize_no_overlap_outcomes()

                self.generate_outcome_counts()
                self.generate_read_lengths()

                self.extract_donor_microhomology_lengths()

                self.record_sanitized_category_names()
            
            elif stage == 'visualize':
                self.generate_figures()
        except:
            print(self.group, self.sample_name)
            raise
