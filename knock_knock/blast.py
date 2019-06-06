import subprocess
import tempfile
from pathlib import Path

import pysam

from hits import sam, fasta, fastq

def h_to_s(kind):
    if kind == sam.BAM_CHARD_CLIP:
        kind = sam.BAM_CSOFT_CLIP
    return kind

def replace_hard_clip_with_soft(cigar):
    return [(h_to_s(k), l) for k, l in cigar]

def blast(ref_fn, reads, bam_fn, bam_by_name_fn, max_insertion_length=None):
    with tempfile.TemporaryDirectory(suffix='_blast') as temp_dir:
        temp_dir_path = Path(temp_dir)

        reads_fasta_fn = temp_dir_path / 'reads.fasta'

        sam_fn = temp_dir_path / 'alignments.sam'

        fastq_dict = {
            '+': {},
            '-': {},
        }

        if isinstance(reads, (str, Path, list)):
            reads = fastq.reads(reads, up_to_space=True)

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
            '-gapopen', '10',
            '-gapextend', '4',
            '-max_target_seqs', '1000000',
            '-parse_deflines', # otherwise qnames/rnames are lost
            '-outfmt', '17', # SAM output
            '-subject', str(reads_fasta_fn), # for bowtie-like behavior, reads are subject ...
            '-query', str(ref_fn), # ... and refs are query
            '-out', str(sam_fn),
        ]
        subprocess.run(blast_command, check=True)

        def undo_hard_clipping(al):
            strand = sam.get_strand(al)
            read = fastq_dict[strand][al.query_name]

            al.query_sequence = read.seq
            al.query_qualities = fastq.decode_sanger(read.qual)

            al.cigar = replace_hard_clip_with_soft(al.cigar)
    
        def make_unaligned(read):
            unal = pysam.AlignedSegment()
            unal.query_name = read.name
            unal.is_unmapped = True
            unal.query_sequence = read.seq
            unal.query_qualities = fastq.decode_sanger(read.qual)
            return unal

        try:
            sam_fh = pysam.AlignmentFile(str(sam_fn))
            header = sam_fh.header
        except ValueError:
            # blast had no output
            header = sam.header_from_fasta(ref_fn)
            pysam.AlignmentFile(str(sam_fn), 'wb', header=header).close()
            sam_fh = pysam.AlignmentFile(str(sam_fn))

        sorter = sam.AlignmentSorter(bam_fn, header)
        by_name_sorter = sam.AlignmentSorter(bam_by_name_fn, header, by_name=True)

        with sorter, by_name_sorter:
            aligned_names = set()
            for al in sam_fh:
                aligned_names.add(al.query_name)

                undo_hard_clipping(al)

                if max_insertion_length is not None:
                    split_als = sam.split_at_large_insertions(al, max_insertion_length + 1)

                    for split_al in split_als:
                        sorter.write(split_al)
                        by_name_sorter.write(split_al)
                else:
                    sorter.write(al)
                    by_name_sorter.write(al)

            for name in fastq_dict['+']:
                if name not in aligned_names:
                    unal = make_unaligned(fastq_dict['+'][name])
                    by_name_sorter.write(unal)
