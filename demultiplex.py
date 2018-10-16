#!/usr/bin/env python3

import argparse
import subprocess
import heapq
from pathlib import Path
from itertools import chain, islice
from collections import Counter, defaultdict

import pysam
import pandas as pd
import numpy as np
import yaml
import tqdm; progress = tqdm.tqdm

from sequencing import mapping_tools, fastq, sam

import collapse

class FastqQuartetSplitter(object):
    def __init__(self, base_path, reads_per_chunk=5000000):
        self.base_path = base_path
        self.reads_per_chunk = reads_per_chunk
        self.next_chunk_number = 0
        self.next_read_number = 0
        self.chunk_fhs = None
    
    def close(self):
        if self.chunk_fhs is not None:
            for fh in self.chunk_fhs.values():
                fh.close()
                
    def start_next_chunk(self):
        self.close()
  
        template = str(self.base_path)  + '/{}.{:05d}.fastq'
        fns = {which: template.format(which, self.next_chunk_number) for which in ['R1', 'R2']}
        self.chunk_fhs = {which: open(fn, 'w') for which, fn in fns.items()}
        
        self.next_chunk_number += 1
        
    def write(self, quartet):
        if self.next_read_number % self.reads_per_chunk == 0:
            self.start_next_chunk()
            
        new_name = collapse.UMI_Annotation(original_name=quartet.R1.name.split(' ')[0],
                                           UMI=quartet.I1.seq,
                                          )
        
        for which in ['R1', 'R2']:
            read = getattr(quartet, which)
            read.name = new_name
            self.chunk_fhs[which].write(str(read))

        self.next_read_number += 1

class UMISorter(object):
    def __init__(self, output_prefix, chunk_size=50000):
        self.sorted_fn = Path(str(output_prefix) + '_R2.fastq')
        self.chunk_size = chunk_size
        self.chunk = []
        self.chunk_number = 0
        self.chunk_fns = []

    def add(self, read):
        self.chunk.append(read)

        if len(self.chunk) == self.chunk_size:
            self.finish_chunk()

    def finish_chunk(self):
        sorted_chunk = sorted(self.chunk, key=lambda r: r.name)

        suffix = '.{:06d}.fastq'.format(self.chunk_number)
        chunk_fn = self.sorted_fn.with_suffix(suffix)

        with chunk_fn.open('w') as chunk_fh:
            for read in sorted_chunk:
                chunk_fh.write(str(read))

        self.chunk_fns.append(chunk_fn)
        self.chunk = []
        self.chunk_number += 1

    def close(self):
        if len(self.chunk_fns) == 1 and len(self.chunk) == 0:
            # Exactly one full chunk was written, so just rename it.
            self.chunk_fns[0].rename(self.sorted_fn)

        else:
            last_chunk = sorted(self.chunk, key=lambda r: r.name)

            previous_chunks = [fastq.reads(fn) for fn in self.chunk_fns]
            
            with self.sorted_fn.open('w') as sorted_fh:
                merged_reads = heapq.merge(last_chunk, *previous_chunks, key=lambda r: r.name)

                for read in merged_reads:
                    sorted_fh.write(str(read))

            for chunk_fn in self.chunk_fns:
                chunk_fn.unlink()

def hamming_distance(first, second):
    return sum(1 for f, s in zip(first, second) if f != s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--sample_dir', required=True)

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--demux_samples', action='store_true')
    mode_group.add_argument('--map_parallel', metavar='MAX_PROCS')
    mode_group.add_argument('--map_chunk', metavar='CHUNK')
    mode_group.add_argument('--demux_guides', action='store_true')

    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    sample_name = sample_dir.parts[-1]

    if args.demux_samples:
        sample_counts = Counter()

        sample_sheet = yaml.load((sample_dir / 'sample_sheet.yaml').read_text())
        sample_indices = list(sample_sheet['sample_indices'].items())
        group_name = sample_sheet['group_name']

        splitters = {}

        to_skip = set(sample_sheet.get('samples_to_skip', []))

        def get_sample_input_dir(sample):
            return sample_dir.parent / '{}_{}'.format(group_name, sample) / 'input'

        for sample, index in sample_indices + [('unknown', '')]:
            if sample not in to_skip:
                base_path = get_sample_input_dir(sample)
                base_path.mkdir(parents=True, exist_ok=True)

                splitters[sample] = FastqQuartetSplitter(base_path)

        fastq_fns = [[sample_dir / name for name in sample_sheet[which]] for which in fastq.quartet_order]
        fn_quartets = (fastq.read_quartets(fns) for fns in zip(*fastq_fns))
        quartets = chain.from_iterable(fn_quartets)

        for quartet in progress(quartets, total=sample_sheet['num_reads']): 
            sample = 'unknown'
            
            for name, index in sample_indices:
                if hamming_distance(quartet.I2.seq, index) <= 1:
                    sample = name
            
            if sample not in to_skip:
                splitters[sample].write(quartet)

            sample_counts[sample] += 1
        
        for splitter in splitters.values():
            splitter.close()

        for sample, index in sample_indices + [('unknown', '')]:
            if sample not in to_skip:
                stats = {
                    'num_reads': sample_counts[sample],
                }
                stats_fn = get_sample_input_dir(sample) / 'stats.yaml'
                stats_fn.write_text(yaml.dump(stats, default_flow_style=False))

        pd.Series(sample_counts).to_csv(sample_dir / 'sample_counts.txt', sep='\t')

    elif args.map_parallel is not None:
        max_procs = args.map_parallel
        R1_fns = sorted((sample_dir / 'input').glob('R1.*.fastq'))
        chunks = [fn.suffixes[0].strip('.') for fn in R1_fns]

        parallel_command = [
            'parallel',
            '-n', '1', 
            '--bar',
            '--max-procs', max_procs,
            './demultiplex.py',
            '--sample_dir', str(sample_dir),
            '--map_chunk', ':::',
        ] + chunks
        
        subprocess.run(parallel_command, check=True)

    elif args.map_chunk is not None:
        chunk = args.map_chunk

        R1_fn = sample_dir / 'input' / 'R1.{}.fastq'.format(chunk)
        STAR_index = '/home/jah/projects/britt/guides/STAR_index'
        output_dir = sample_dir / 'guide_mapping'
        output_dir.mkdir(exist_ok=True)
        output_prefix = output_dir / '{}.'.format(chunk)
        mapping_tools.map_STAR(R1_fn, STAR_index, output_prefix,
                               sort=False,
                               mode='guide_alignment',
                               include_unmapped=True,
                               )

    elif args.demux_guides:
        stats = yaml.load((sample_dir / 'input' / 'stats.yaml').read_text())

        bad_guides_fn = sample_dir / 'bad_guides.bam'

        guide_dir = sample_dir / 'by_guide'
        guide_dir.mkdir(exist_ok=True)

        R2_fns = sorted((sample_dir / 'input').glob('R2*.fastq'))
        bam_fns = sorted((sample_dir / 'guide_mapping').glob('*.bam'))

        with pysam.AlignmentFile(str(bam_fns[0])) as fh:
            header = fh.header

        fn_reads = (fastq.reads(fn) for fn in R2_fns)
        reads = chain.from_iterable(fn_reads)

        alignment_files = (pysam.AlignmentFile(str(fn)) for fn in bam_fns)
        mappings = chain.from_iterable(alignment_files)
        mapping_groups = sam.grouped_by_name(mappings)

        sorters = {}

        read_length = 45

        edit_distance = defaultdict(lambda: np.zeros(read_length + 1, int))
        indels = defaultdict(lambda: np.zeros(read_length + 1, int))

        MDs = defaultdict(Counter)

        def edit_info(al):
            if al.is_unmapped or al.is_reverse:
                return (read_length, read_length)
            else:
                return (sam.total_indel_lengths(al), al.get_tag('NM'))

        guide_counts = Counter()

        bad_sorter = sam.AlignmentSorter(bad_guides_fn, header)
        with bad_sorter:
            for (query_name, als), read in progress(zip(mapping_groups, reads), total=stats['num_reads']):
                # Record stats on all alignments for diagnostics.
                for al in als:
                    if not al.is_unmapped:
                        num_indels, NM = edit_info(al)
                        guide = al.reference_name
                        edit_distance[guide][NM] += 1
                        indels[guide][num_indels] += 1

                        if num_indels == 0 and NM == 1:
                            MDs[guide][al.get_tag('MD')] += 1


                edit_tuples = [(al, edit_info(al)) for al in als]
                min_edit = min(info for al, info in edit_tuples)
                
                min_edit_als = [al for al, info in edit_tuples if info == min_edit]
                    
                protospacer = al.query_sequence
                protospacer_qual = fastq.sanitize_qual(fastq.encode_sanger(al.query_qualities))

                old_annotation = collapse.UMI_Annotation.from_identifier(read.name)
                new_annotation = collapse.UMI_protospacer_Annotation(protospacer=protospacer,
                                                                    protospacer_qual=protospacer_qual,
                                                                    **old_annotation)
                read.name = str(new_annotation)
                
                min_indels, min_NM = min_edit
                if min_indels == 0 and min_NM <= 1 and len(min_edit_als) == 1:
                    al = min_edit_als[0]
                    guide = al.reference_name
                else:
                    guide = 'unknown'
                    for al in als:
                        bad_sorter.write(al)
            
                if guide not in sorters:
                    sorters[guide] = UMISorter(guide_dir / guide)
                
                sorters[guide].add(read)
                guide_counts[guide] += 1
        
        for sorter in progress(sorters.values()):
            sorter.close()
            
        pd.DataFrame(indels).T.to_csv(sample_dir / 'indel_distributions.txt')
        pd.DataFrame(edit_distance).T.to_csv(sample_dir / 'edit_distance_distributions.txt')
        pd.Series(guide_counts).to_csv(sample_dir / 'guide_counts.txt')
