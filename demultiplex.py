#!/usr/bin/env python3

import argparse
import subprocess
import heapq
import re
import gzip
from pathlib import Path
from itertools import chain, islice
from collections import Counter, defaultdict

import pysam
import pandas as pd
import numpy as np
import yaml
import tqdm; progress = tqdm.tqdm

from sequencing import mapping_tools, fastq, sam, utilities

from knockin import collapse
from knockin.collapse_cython import hamming_distance

class FastqQuartetSplitter(object):
    def __init__(self, base_path, quartet_name, reads_per_chunk=5000000):
        self.base_path = base_path
        self.quartet_name = quartet_name
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
  
        fns = {}
        chunk_string = chunk_to_string(self.next_chunk_number)
        for which in ['R1', 'R2']:
            fns[which] = self.base_path / '{}.{}.{}.fastq.gz'.format(which, self.quartet_name, chunk_string)

        self.chunk_fhs = {which: gzip.open(str(fn), 'wt', compresslevel=4) for which, fn in fns.items()}
        
        self.next_chunk_number += 1
        
    def write(self, quartet):
        if self.next_read_number % self.reads_per_chunk == 0:
            self.start_next_chunk()
            
        new_name = collapse.Annotations['UMI'](original_name=quartet.R1.name.split(' ')[0],
                                               UMI=quartet.I1.seq,
                                              )
        
        for which in ['R1', 'R2']:
            read = getattr(quartet, which)
            read.name = new_name
            self.chunk_fhs[which].write(str(read))

        self.next_read_number += 1

class UMISorters(object):
    def __init__(self, output_dir, progress=utilities.identity):
        self.output_dir = output_dir
        self.progress = progress

        self.sorters = defaultdict(list)

    def __enter__(self):
        return self

    def write(self, guide, read):
        self.sorters[guide].append(read)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        for guide in self.progress(sorted(self.sorters)):
            sorted_reads = sorted(self.sorters[guide], key=lambda r: r.name)

            fn = self.output_dir / '{}_R2.fastq.gz'.format(guide)
            with gzip.open(str(fn), 'wt') as zfh:
                for read in sorted_reads:
                    zfh.write(str(read))

            del self.sorters[guide]
            del sorted_reads

def chunk_to_string(chunk):
    return '{:05d}'.format(int(chunk))

def demux_samples(sample_dir, quartet_name):
    sample_dir = Path(sample_dir)
    sample_counts = Counter()
    index_counts = Counter()

    sample_sheet = yaml.load((sample_dir / 'sample_sheet.yaml').read_text())

    index_to_sample_name = utilities.get_one_mismatch_resolver(sample_sheet['sample_indices'])
    sample_names = sorted(sample_sheet['sample_indices']) + ['unknown']

    group_name = sample_sheet['group_name']

    splitters = {}

    to_skip = set(sample_sheet.get('samples_to_skip', []))

    def get_sample_input_dir(sample_name):
        return sample_dir.parent / '{}_{}'.format(group_name, sample_name) / 'input'

    for sample_name in sample_names:
        if sample_name not in to_skip:
            base_path = get_sample_input_dir(sample_name)
            base_path.mkdir(parents=True, exist_ok=True)

            splitters[sample_name] = FastqQuartetSplitter(base_path, quartet_name)

    quartet_dict = sample_sheet['quartets'][quartet_name] 
    fastq_fns = [sample_dir / quartet_dict[which] for which in fastq.quartet_order]
    quartets = fastq.read_quartets(fastq_fns)

    for quartet in progress(quartets, total=quartet_dict['num_reads']): 
        I2_seq = quartet.I2.seq
        index_counts[I2_seq] += 1

        sample = index_to_sample_name.get(I2_seq, 'unknown')
        
        if sample not in to_skip:
            splitters[sample].write(quartet)

        sample_counts[sample] += 1
    
    for splitter in splitters.values():
        splitter.close()

    #for sample_name in sample_names:
    #    if sample_name not in to_skip:
    #        stats = {
    #            'num_reads': sample_counts[sample_name],
    #        }
    #        stats_fn = get_sample_input_dir(sample_name) / 'stats.yaml'
    #        stats_fn.write_text(yaml.dump(stats, default_flow_style=False))

    counts = pd.Series(sample_counts).sort_values(ascending=False)
    counts.to_csv(sample_dir / 'sample_counts_{}.txt'.format(quartet_name), sep='\t')

    expected_indices = set()
    for seqs in sample_sheet['sample_indices'].values():
        if not isinstance(seqs, list):
            seqs = [seqs]
        expected_indices.update(seqs)

    with (sample_dir / 'index_stats_{}.txt'.format(quartet_name)).open('w') as fh:
        total = sum(index_counts.values())
        for index, count in index_counts.most_common(50):
            name = index_to_sample_name.get(index, '')

            if index in expected_indices:
                mismatches = ''
            elif index in index_to_sample_name:
                mismatches = ' (1 mismatch)'
            else:
                mismatches = ''

            fraction = float(count) / total

            fh.write('{0}\t{2: >10,}\t({3: >6.2%})\t{4}{1}\n'.format(index, mismatches, count, fraction, name))

def demux_all_sample_guides(sample_dir, max_procs):
    sample_dir = Path(sample_dir)

    sample_sheet = yaml.load((sample_dir / 'sample_sheet.yaml').read_text())

    samples = {str(s) for s in sample_sheet['sample_indices']}
    group_name = sample_sheet['group_name']

    for sample in sorted(samples):
        print(sample)
        sample_dir = sample_dir.parent / '{}_{}'.format(group_name, sample)
        demux_guides_parallel(sample_dir, max_procs)

def demux_guides_parallel(sample_dir, max_procs):
    sample_dir = Path(sample_dir)
    R1_fns = sorted((sample_dir / 'input').glob('R1.*.fastq.gz'))
    chunk_pattern = 'R1\.(.+)\.fastq.gz'
    chunk_names = [re.match(chunk_pattern, fn.name).group(1) for fn in R1_fns]

    parallel_command = [
        'parallel',
        '-n', '1', 
        '--bar',
        '--max-procs', max_procs,
        './demultiplex.py',
        '--sample_dir', str(sample_dir),
        '--demux_guides_chunk', ':::',
    ] + chunk_names
    
    subprocess.run(parallel_command, check=True)

    chunk_dirs = sorted([p for p in (sample_dir / 'by_guide').iterdir() if p.is_dir()])
    for name in ['indel_distributions.txt', 'edit_distance_distributions.txt']:
        indel_fns = [d / name for d in chunk_dirs]

        dfs = [pd.read_csv(fn, index_col=0) for fn in indel_fns]
        df = pd.concat(dfs, keys=np.arange(len(indel_fns)))
        summed = df.sum(level=[1])
        
        summed_fn = sample_dir / name
        summed.to_csv(summed_fn)

    guide_count_fns = [d / 'guide_counts.txt' for d in chunk_dirs]

    series_list = [pd.read_csv(fn, index_col=0, header=None, squeeze=True) for fn in guide_count_fns]

    summed = pd.concat(series_list, axis=1).sum(axis=1).astype(int)
    summed.to_csv(sample_dir / 'guide_counts.txt')

    # Start the most abundant first to help maximize parallelization.
    guide_counts = pd.read_csv(sample_dir / 'guide_counts.txt', header=None, index_col=0, squeeze=True)
    guide_order = list(guide_counts.sort_values(ascending=False).index)
    
    parallel_command = [
        'parallel',
        '-n', '1', 
        '--bar',
        '--max-procs', max_procs,
        './demultiplex.py',
        '--sample_dir', str(sample_dir),
        '--merge_chunks', ':::',
    ] + guide_order
    
    subprocess.run(parallel_command, check=True)
    
    parallel_command = [
        'parallel',
        '-n', '1', 
        '--bar',
        '--max-procs', max_procs,
        './demultiplex.py',
        '--sample_dir', str(sample_dir),
        '--clean_up_chunk', ':::',
    ] + chunk_names
    
    subprocess.run(parallel_command, check=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--sample_dir', required=True)

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--demux_samples', metavar='QUARTET_NAME')
    mode_group.add_argument('--demux_all_sample_guides', metavar='MAX_PROCS')
    mode_group.add_argument('--demux_guides_parallel', metavar='MAX_PROCS')
    mode_group.add_argument('--demux_guides_chunk', metavar='CHUNK')
    mode_group.add_argument('--merge_chunks', metavar='GUIDE')
    mode_group.add_argument('--clean_up_chunk', metavar='CHUNK')

    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    sample_name = sample_dir.parts[-1]

    if args.demux_samples:
        quartet_name = args.demux_samples
        demux_samples(sample_dir, quartet_name)

    elif args.demux_all_sample_guides is not None:
        max_procs = args.demux_all_sample_guides
        demux_all_sample_guides(sample_dir, max_procs)

    elif args.demux_guides_parallel is not None:
        max_procs = args.demux_guides_parallel
        demux_guides_parallel(sample_dir, max_procs)

    elif args.demux_guides_chunk is not None:
        chunk_name = args.demux_guides_chunk

        R1_fn = sample_dir / 'input' / 'R1.{}.fastq.gz'.format(chunk_name)
        R2_fn = sample_dir / 'input' / 'R2.{}.fastq.gz'.format(chunk_name)

        STAR_index = '/home/jah/projects/britt/guides/STAR_index'

        output_dir = sample_dir / 'guide_mapping'
        output_dir.mkdir(exist_ok=True)
        STAR_output_prefix = output_dir / '{}.'.format(chunk_name)

        bam_fn = mapping_tools.map_STAR(R1_fn, STAR_index, STAR_output_prefix,
                                        sort=False,
                                        mode='guide_alignment',
                                        include_unmapped=True,
                                       )

        for suffix in ['Log.out', 'SJ.out.tab', 'Log.progress.out']:
            fn = STAR_output_prefix.parent / (STAR_output_prefix.name + suffix)
            fn.unlink()

        bad_guides_fn = output_dir / '{}.bad_guides.bam'.format(chunk_name)

        by_guide_dir = sample_dir / 'by_guide'
        by_guide_dir.mkdir(exist_ok=True)

        with pysam.AlignmentFile(bam_fn) as fh:
            header = fh.header

        reads = fastq.reads(R2_fn)

        mappings = pysam.AlignmentFile(str(bam_fn))
        mapping_groups = sam.grouped_by_name(mappings)

        chunk_dir = by_guide_dir / chunk_name
        chunk_dir.mkdir(exist_ok=True)
        sorters = UMISorters(chunk_dir)

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

        bad_guides_sorter = sam.AlignmentSorter(bad_guides_fn, header)

        with bad_guides_sorter, sorters:
            for (query_name, als), read in zip(mapping_groups, reads):
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
                    
                guide = al.query_sequence
                guide_qual = fastq.sanitize_qual(fastq.encode_sanger(al.query_qualities))

                old_annotation = collapse.Annotations['UMI'].from_identifier(read.name)
                new_annotation = collapse.Annotations['UMI_guide'](guide=guide,
                                                                   guide_qual=guide_qual,
                                                                   **old_annotation,
                                                                  )
                read.name = str(new_annotation)
                
                min_indels, min_NM = min_edit
                if min_indels == 0 and min_NM <= 1 and len(min_edit_als) == 1:
                    al = min_edit_als[0]
                    guide = al.reference_name
                else:
                    guide = 'unknown'
                    for al in als:
                        bad_guides_sorter.write(al)
            
                sorters.write(guide, read)
                guide_counts[guide] += 1
        
        pd.DataFrame(indels).T.to_csv(chunk_dir / 'indel_distributions.txt')
        pd.DataFrame(edit_distance).T.to_csv(chunk_dir / 'edit_distance_distributions.txt')
        pd.Series(guide_counts).to_csv(chunk_dir / 'guide_counts.txt')

    elif args.merge_chunks:
        guide = args.merge_chunks
        by_guide_dir = sample_dir / 'by_guide'
        chunk_fns = sorted(by_guide_dir.glob('*/{}_R2.fastq.gz'.format(guide)))
        merged_fn = by_guide_dir / '{}_R2.fastq.gz'.format(guide)

        chunks = [fastq.reads(fn) for fn in chunk_fns]
        
        with gzip.open(str(merged_fn), 'wt', compresslevel=4) as zfh:
            merged_reads = heapq.merge(*chunks, key=lambda r: r.name)
            for read in merged_reads:
                zfh.write(str(read))

    elif args.clean_up_chunk:
        chunk_name = args.clean_up_chunk
        chunk_dir = sample_dir / 'by_guide' / chunk_name

        for p in sorted(chunk_dir.iterdir()):
            p.unlink()
        
        chunk_dir.rmdir()