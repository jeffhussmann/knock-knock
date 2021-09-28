import shutil
from collections import defaultdict

import numpy as np
import pandas as pd

import hits.visualize
from hits import utilities, interval

from knock_knock.experiment import Experiment, ensure_list

memoized_property = hits.utilities.memoized_property

class PacbioExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.paired_end_read_length = None
        self.x_tick_multiple = 500

        self.layout_mode = 'pacbio'

        ccs_fastq_fns = ensure_list(self.description['CCS_fastq_fn'])
        self.fns['CCS_fastqs'] = [self.data_dir / name for name in ccs_fastq_fns]

        for fn in self.fns['CCS_fastqs']:
            if not fn.exists():
                #raise ValueError(f'{self.group}: {self.name} specifies non-existent {fn}')
                pass

        self.read_types = ['CCS']
        self.default_read_type = 'CCS'

        self.outcome_fn_keys = ['outcome_list']

        self.length_plot_smooth_window = self.description.get('length_plot_smooth_window', 7)

        self.diagram_kwargs.update(draw_sequence=False,
                                  )

    def __repr__(self):
        return f'PacbioExperiment: batch={self.batch}, sample_name={self.sample_name}, base_dir={self.base_dir}'

    @memoized_property
    def max_relevant_length(self):
        auto_length = int((self.target_info.amplicon_length * 2.5 // 1000 + 1)) * 1000
        return self.description.get('max_relevant_length', auto_length)

    def length_ranges(self, outcome=None):
        interval_length = self.max_relevant_length // 50
        starts = np.arange(0, self.max_relevant_length + interval_length, interval_length)

        if outcome is None:
            lengths = self.read_lengths
        else:
            lengths = self.outcome_stratified_lengths[outcome]

        ranges = []
        for start in starts:
            if sum(lengths[start:start + interval_length]) > 0:
                ranges.append((start, start + interval_length - 1))

        return pd.DataFrame(ranges, columns=['start', 'end'])

    def generate_length_range_figures(self, specific_outcome=None, num_examples=1):
        by_length_range = defaultdict(lambda: utilities.ReservoirSampler(num_examples))
        length_ranges = [interval.Interval(row['start'], row['end']) for _, row in self.length_ranges(specific_outcome).iterrows()]

        fn_key = 'bam_by_name'

        al_groups = self.alignment_groups(outcome=specific_outcome, fn_key=fn_key)
        for name, group in al_groups:
            length = group[0].query_length

            # Need to make sure that the last interval catches anything longer than
            # self.max_relevant_length.

            if length >= self.max_relevant_length:
                last_range = length_ranges[-1]
                if last_range.start == self.max_relevant_length:
                    by_length_range[last_range.start, last_range.end].add((name, group))
            else:
                for length_range in length_ranges:
                    if length in length_range:
                        by_length_range[length_range.start, length_range.end].add((name, group))

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

        items = self.progress(by_length_range.items(), desc=description, total=len(by_length_range))

        for (start, end), sampler in items:
            diagrams = self.alignment_groups_to_diagrams(sampler.sample,
                                                         num_examples=num_examples,
                                                         **self.diagram_kwargs,
                                                        )
            im = hits.visualize.make_stacked_Image([d.fig for d in diagrams])
            fn = fns['length_range_figure'](start, end)
            im.save(fn)
    
    def process(self, stage):
        try:
            if stage == 'preprocess':
                pass
            elif stage == 'align':
                for read_type in self.read_types:
                    self.generate_alignments(read_type=read_type)
                    self.generate_supplemental_alignments_with_minimap2(read_type=read_type)
                    self.combine_alignments(read_type=read_type)

            elif stage == 'categorize':
                self.categorize_outcomes(read_type='CCS')

                self.generate_outcome_counts()
                self.generate_read_lengths()

                self.extract_donor_microhomology_lengths()

                self.record_sanitized_category_names()

            elif stage == 'visualize':
                self.generate_figures()
        except:
            print(self.group, self.sample_name)
            raise
