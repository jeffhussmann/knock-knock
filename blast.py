import pysam
import subprocess
import tempfile
import os
import array

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

import Sequencing.sam as sam
import Sequencing.fasta as fasta
import Sequencing.fastq as fastq
import Sequencing.utilities as utilities

def h_to_s(kind):
    if kind == sam.BAM_CHARD_CLIP:
        kind = sam.BAM_CSOFT_CLIP
    return kind

def replace_hard_clip_with_soft(cigar):
    return [(h_to_s(k), l) for k, l in cigar]

def blast(ref_fn, reads, bam_fn, bam_by_name_fn):
    with tempfile.TemporaryDirectory(suffix='_blast') as temp_dir:
        temp_dir_path = Path(temp_dir)

        reads_fasta_fn = temp_dir_path / 'reads.fasta'
        reads_fasta_fai_fn = reads_fasta_fn.with_suffix('.fasta.fai')

        sam_fn = temp_dir_path / 'alignments.sam'

        fastq_dict = {
            '+': {},
            '-': {},
        }

        if isinstance(reads, (str, Path)):
            reads = fastq.reads(reads)

        with reads_fasta_fn.open('w') as fasta_fh:
            for read in reads:
                fastq_dict['+'][read.name] = read
                fastq_dict['-'][read.name] = read.reverse_complement()

                fasta_read = fasta.Read(read.name, read.seq)
                fasta_fh.write(str(fasta_read))
                
        pysam.faidx(str(reads_fasta_fn))
            
        blast_command = [
            'blastn',
            '-task', 'blastn', # default is megablast
            '-evalue', '0.1',
            '-max_target_seqs', '1000000',
            '-parse_deflines', # otherwise qnames/rnames are lost
            '-outfmt', '17', # SAM output
            '-subject', str(reads_fasta_fn), # for bowtie-like behavior, reads are subject ...
            '-query', str(ref_fn), # ... and refs are query
            '-out', str(sam_fn),
        ]
        subprocess.check_call(blast_command)

        def undo_hard_clipping(r):
            strand = sam.get_strand(r)
            read = fastq_dict[strand][r.query_name]

            r.query_sequence = read.seq
            r.query_qualities = fastq.decode_sanger(read.qual)

            r.cigar = replace_hard_clip_with_soft(r.cigar)
    
        def make_unaligned(read):
            unal = pysam.AlignedSegment()
            unal.query_name = read.name
            unal.is_unmapped = True
            unal.query_sequence = read.seq
            unal.query_qualities = fastq.decode_sanger(read.qual)
            return unal

        sam_fh = pysam.AlignmentFile(sam_fn)

        sorter = sam.AlignmentSorter(sam_fh.references, sam_fh.lengths, bam_fn)
        by_name_sorter = sam.AlignmentSorter(sam_fh.references, sam_fh.lengths, bam_by_name_fn, by_name=True)

        with sorter, by_name_sorter:
            aligned_names = set()
            for al in sam_fh:
                aligned_names.add(al.query_name)

                undo_hard_clipping(al)

                sorter.write(al)
                by_name_sorter.write(al)

            for name in fastq_dict['+']:
                if name not in aligned_names:
                    unal = make_unaligned(fastq_dict['+'][name])
                    by_name_sorter.write(unal)
