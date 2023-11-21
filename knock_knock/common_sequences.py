import shutil
import gzip
from collections import Counter

import hits.fastq
import hits.utilities
import knock_knock.outcome_record

memoized_property = hits.utilities.memoized_property

class CommonSequenceSplitter:
    def __init__(self, experiment_group, max_sequences=None, reads_per_chunk=1000):
        self.experiment_group = experiment_group
        self.max_sequences = max_sequences
        self.reads_per_chunk = reads_per_chunk
        self.current_chunk_fh = None
        self.seq_counts = Counter()
        self.distinct_samples_per_seq = Counter()
        
        common_sequences_dir = self.experiment_group.fns['common_sequences_dir']

        if common_sequences_dir.is_dir():
            shutil.rmtree(str(common_sequences_dir))
            
        common_sequences_dir.mkdir()

    def update_counts(self, seqs):
        counts = Counter(seqs)
        self.seq_counts.update(counts)
        for seq in counts:
            self.distinct_samples_per_seq[seq] += 1
            
    def close(self):
        if self.current_chunk_fh is not None:
            self.current_chunk_fh.close()
            
    def possibly_make_new_chunk(self, i):
        if i % self.reads_per_chunk == 0:
            self.close()
            chunk_name = f'{i:010d}-{i + self.reads_per_chunk - 1:010d}'

            chunk_exp = self.experiment_group.common_sequence_chunk_exp_from_name(chunk_name)
            chunk_exp.results_dir.mkdir()
            fn = chunk_exp.fns_by_read_type['fastq'][chunk_exp.preprocessed_read_type]
            self.current_chunk_fh = gzip.open(fn, 'wt', compresslevel=1)
            
    def write_read(self, i, read):
        self.possibly_make_new_chunk(i)
        self.current_chunk_fh.write(str(read))
        
    def write_files(self):
        # Include one value outside of the solexa range to allow automatic detection.
        qual = hits.fastq.unambiguous_sanger_Q40(1000)
   
        tuples = []

        i = 0 
        for seq, count in self.seq_counts.most_common(self.max_sequences):
            distinct_samples = self.distinct_samples_per_seq[seq]

            if count > 1:
                name = f'{i:010}_{count:010}'
                read = hits.fastq.Read(name, seq, qual[:len(seq)])
                self.write_read(i, read)
                i += 1
            else:
                name = None

            tuples.append((name, seq, count))

        self.close()

class CommonSequencesExperiment:
    @property
    def final_Outcome(self):
        return knock_knock.outcome_record.CommonSequenceOutcomeRecord

    @property
    def uncommon_read_type(self):
        return self.preprocessed_read_type

    @memoized_property
    def common_sequence_to_outcome(self):
        return {}

    @memoized_property
    def common_sequence_to_alignments(self):
        return {}

    def extract_reads_with_uncommon_sequences(self):
        ''' Overload to prevent from overwriting its own sequences. '''
        pass