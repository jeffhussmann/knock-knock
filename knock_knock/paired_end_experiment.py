import gzip
import sys

from hits import adapters, fastq, utilities

memoized_property = utilities.memoized_property

class PairedEndExperiment:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.outcome_fn_keys = [
            'outcome_list',
            #'no_overlap_outcome_list',
        ]

        self.read_types = [
            'stitched',
            'stitched_by_name',
            'nonredundant',
            'R1_no_overlap',
            'R2_no_overlap',
        ]

        self.error_corrected = False
        self.layout_mode = 'amplicon'

    @property
    def preprocessed_read_type(self):
        return 'stitched_by_name'

    @property
    def default_read_type(self):
        return 'stitched_by_name'

    @property
    def read_types_to_align(self):
        return [
            'nonredundant',
            #'R1_no_overlap',
            #'R2_no_overlap',
            ]

    @property
    def read_pairs(self):
        read_pairs = fastq.read_pairs(self.fns['R1'], self.fns['R2'], standardize_names=True)
        return read_pairs

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

    def sort_stitched_read_pairs(self):
        reads = sorted(self.reads_by_type('stitched'), key=lambda read: read.name)
        fn = self.fns_by_read_type['fastq']['stitched_by_name']
        with gzip.open(fn, 'wt', compresslevel=1) as sorted_fh:
            for read in reads:
                sorted_fh.write(str(read))

    def preprocess(self):
        self.stitch_read_pairs()
        self.sort_stitched_read_pairs()

    def process(self, stage):
        try:
            if stage == 'preprocess':
                self.preprocess()

            elif stage == 'align':
                self.make_nonredundant_sequence_fastq()
                
                for read_type in self.read_types_to_align:
                    self.generate_alignments(read_type)
                    self.generate_supplemental_alignments_with_STAR(read_type, min_length=20)
                    self.combine_alignments(read_type)

            elif stage == 'categorize':
                self.categorize_outcomes()

                if 'R1_no_overlap' in self.read_types_to_align:
                    self.categorize_no_overlap_outcomes()

                self.count_read_lengths()
                self.count_outcomes()

                self.record_sanitized_category_names()
            
            elif stage == 'visualize':
                self.generate_figures()
        except:
            print(self.group, self.sample_name)
            raise
