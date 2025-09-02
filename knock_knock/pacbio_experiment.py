import gzip

import numpy as np

import hits.fastq
import hits.utilities

from knock_knock.experiment import Experiment, ensure_list

memoized_property = hits.utilities.memoized_property
memoized_with_kwargs = hits.utilities.memoized_with_kwargs

class PacbioExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.x_tick_multiple = 500

        self.platform = 'pacbio'

        ccs_fastq_fns = ensure_list(self.description['CCS_fastq_fn'])
        self.fns['CCS_fastqs'] = [self.data_dir / name for name in ccs_fastq_fns]

        self.outcome_fn_keys = ['outcome_list']

        self.length_plot_smooth_window = self.description.get('length_plot_smooth_window', 7)

    @property
    def preprocessed_read_type(self):
        return 'CCS_by_name'

    @property
    def read_types(self):
        return {self.preprocessed_read_type}

    @property
    def reads(self):
        return hits.fastq.reads(self.fns['CCS_fastqs'], standardize_names=True, up_to_space=True)

    @memoized_property
    def max_relevant_length(self):
        auto_length = int((self.editing_strategy.amplicon_length * 2.5 // 1000 + 1)) * 1000
        return self.description.get('max_relevant_length', auto_length)

    @memoized_with_kwargs
    def length_ranges(self, *, outcome=None):
        interval_length = self.max_relevant_length // 50
        starts = np.arange(0, self.max_relevant_length + interval_length, interval_length)

        lengths = self.outcome_stratified_lengths.by_outcome(outcome)

        ranges = []
        for start in starts:
            if sum(lengths[start:start + interval_length]) > 0:
                ranges.append((start, start + interval_length - 1))

        return ranges

    def preprocess(self):
        fn = self.fns_by_read_type['fastq'][self.preprocessed_read_type]

        with gzip.open(fn, 'wt', compresslevel=1) as sorted_fh:
            for read in sorted(self.reads, key=lambda read: read.name):
                sorted_fh.write(str(read))

    def align(self):
        for read_type in self.read_types:
            self.generate_alignments_with_blast(read_type=read_type)
            self.generate_supplemental_alignments_with_minimap2(read_type=read_type)
            self.combine_alignments(read_type=read_type)

    def categorize(self):
        self.categorize_outcomes()

        self.generate_outcome_counts()
        self.generate_outcome_stratified_lengths()

        self.record_sanitized_category_names()