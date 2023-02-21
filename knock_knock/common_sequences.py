import shutil
import gzip
from collections import Counter

from hits import fastq

class CommonSequenceSplitter:
    def __init__(self, experiment_group, reads_per_chunk=1000):
        self.experiment_group = experiment_group
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
            fn = chunk_exp.fns_by_read_type['fastq']['nonredundant']
            self.current_chunk_fh = gzip.open(fn, 'wt', compresslevel=1)
            
    def write_read(self, i, read):
        self.possibly_make_new_chunk(i)
        self.current_chunk_fh.write(str(read))
        
    def write_files(self):
        # Include one value outside of the solexa range to allow automatic detection.
        quals = {
            0: '',
        }
        for length in range(1, 1000):
            quals[length] = fastq.encode_sanger([25] + [40] * (length - 1))
   
        tuples = []

        i = 0 
        for seq, count in self.seq_counts.most_common():
            distinct_samples = self.distinct_samples_per_seq[seq]

            if count > 1:
                name = f'{i:010}_{count:010}'
                read = fastq.Read(name, seq, quals[len(seq)])
                self.write_read(i, read)
                i += 1
            else:
                name = None

            tuples.append((name, seq, count))

        self.close()