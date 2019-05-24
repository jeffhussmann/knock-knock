#!/usr/bin/env python3

import argparse
import array
import bisect
import subprocess
from collections import namedtuple, Counter
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import bokeh.palettes
import yaml
import tqdm
import pysam

from hits import fastq, utilities, sw, sam
from hits import annotation as annotation_module

from .collapse_cython import hq_mismatches_from_seed, hq_hamming_distance, hamming_distance_matrix, register_corrections

progress = tqdm.tqdm_notebook

CELL_BC_TAG = 'CB'
UMI_TAG = 'UR'
NUM_READS_TAG = 'ZR'
CLUSTER_ID_TAG = 'ZC'

HIGH_Q = 31
LOW_Q = 10
N_Q = 2

annotation_fields = {
    'cluster': [
        ('cell_BC', 's'),
        ('UMI', 's'),
        ('num_reads', '06d'),
        ('cluster_id', 's'),
    ],

    'read': [
        ('cell_BC', 's'),
        ('UMI', 's'),
        ('original_name', 's'),
    ],

    'UMI': [
        ('UMI', 's'),
        ('original_name', 's'),
    ],

    'UMI_guide': [
        ('UMI', 's'),
        ('guide', 's'),
        ('guide_qual', 's'),
        ('original_name', 's'),
    ],

    'collapsed_UMI': [
        ('UMI', 's'),
        ('guide', 's'),
        ('guide_qual', 's'),
        ('cluster_id', '06d'),
        ('num_reads', '06d'),
    ],

    'collapsed_UMI_mismatch': [
        ('UMI', 's'),
        ('cluster_id', '06d'),
        ('num_reads', '010d'),
        ('mismatch', 'd'),
    ],
}

Annotations = {key: annotation_module.Annotation_factory(fields) for key, fields in annotation_fields.items()}

def consensus_seq_and_qs(reads, max_read_length, bam):
    if max_read_length is None:
        max_read_length = len(reads[0].query_sequence)

    statistics = fastq.quality_and_complexity(reads, max_read_length, alignments=bam, min_q=30)
    shape = statistics['c'].shape

    rl_range = np.arange(max_read_length)
    
    fields = [
        ('c_above_min_q', int),
        ('c', int),
        ('average_q', float),
    ]

    stat_tuples = np.zeros(shape, dtype=fields)
    for k in ['c_above_min_q', 'c', 'average_q']:
        stat_tuples[k] = statistics[k]

    argsorted = stat_tuples.argsort()
    second_best_idxs, best_idxs = argsorted[:, -2:].T
    
    best_stats = stat_tuples[rl_range, best_idxs]

    majority = (best_stats['c'] / len(reads)) > 0.5
    at_least_one_hq = best_stats['c_above_min_q'] > 0
    
    qs = np.full(max_read_length, LOW_Q, dtype=int)
    qs[majority & at_least_one_hq] = HIGH_Q
    
    ties = (best_stats == stat_tuples[rl_range, second_best_idxs])

    best_idxs[ties] = utilities.base_to_index['N']
    qs[ties] = N_Q

    seq = ''.join(utilities.base_order[i] for i in best_idxs)

    return seq, qs

def call_consensus(reads, max_read_length, bam):
    seq, qs = consensus_seq_and_qs(reads, max_read_length, bam)

    if bam:
        consensus = pysam.AlignedSegment()
        consensus.query_sequence = seq
        consensus.query_qualities = array.array('B', qs)
        consensus.set_tag(NUM_READS_TAG, len(reads), 'i')
    else:
        guide_reads = []
        for read in reads:
            annotation = Annotations['UMI_guide'].from_identifier(read.name)
            guide_read = fastq.Read('PH', annotation['guide'], annotation['guide_qual'])
            guide_reads.append(guide_read)

        guide_seq, guide_qs = consensus_seq_and_qs(guide_reads, None, False)
        guide_qual = fastq.encode_sanger(guide_qs)

        annotation = Annotations['collapsed_UMI'](UMI='PH',
                                                  num_reads=len(reads),
                                                  guide=guide_seq,
                                                  guide_qual=guide_qual,
                                                  cluster_id=0,
                                                 )
        name = str(annotation)
        qual = fastq.encode_sanger(qs)
        consensus = fastq.Read(name, seq, qual)

    return consensus

def within_radius_of_seed(seed, reads, max_hq_mismatches):
    seed_b = seed.encode()
    ds = [hq_mismatches_from_seed(seed_b, read.query_sequence.encode(), read.query_qualities, 20)
          for read in reads]
    
    near_seed = []
    remaining = []
    
    for i, (d, al) in enumerate(zip(ds, reads)):
        if d <= max_hq_mismatches:
            near_seed.append(al)
        else:
            remaining.append(al)
    
    return near_seed, remaining

def propose_seed(reads, max_read_length, bam):
    seqs = (read.query_sequence for read in reads)

    seq_counts = Counter(seqs).most_common()

    highest_count = seq_counts[0][1]
    most_frequents = [s for s, c in seq_counts if c == highest_count]

    # If there is a tie, take the alphabetically first for determinism.
    seq = sorted(most_frequents)[0]
    
    if highest_count > 1:
        seed = seq
    else:
        consensus = call_consensus(reads, max_read_length, bam)
        seed = consensus.query_sequence
        
    return seed

def make_singleton_cluster(read, bam):
    if bam:
        singleton = pysam.AlignedSegment()
        singleton.query_sequence = read.query_sequence
        singleton.query_qualities = read.query_qualities
        singleton.set_tag(NUM_READS_TAG, 1, 'i')
    else:
        annotation = Annotations['UMI_guide'].from_identifier(read.name)
        name = Annotations['collapsed_UMI'](UMI=annotation['UMI'],
                                            guide=annotation['guide'],
                                            guide_qual=annotation['guide_qual'],
                                            cluster_id=0,
                                            num_reads=1,
                                           )
        singleton = fastq.Read(str(name), read.seq, read.qual)

    return singleton

def form_clusters(reads, max_read_length=None, max_hq_mismatches=0, bam=False):
    if len(reads) == 0:
        clusters = []
    
    elif len(reads) == 1:
        clusters = [make_singleton_cluster(read, bam) for read in reads]
    
    else:
        seed = propose_seed(reads, max_read_length, bam)
        near_seed, remaining = within_radius_of_seed(seed, reads, max_hq_mismatches)
        
        if len(near_seed) == 0:
            # didn't make progress, so give up
            clusters = [make_singleton_cluster(read, bam) for read in reads]
        
        else:
            consensus_near_seed = call_consensus(near_seed, max_read_length, bam)
            all_others = form_clusters(remaining, max_read_length, max_hq_mismatches, bam)
            clusters = [consensus_near_seed] + all_others
            
    return clusters

def align_clusters(first, second):
    al = sw.global_alignment(first.query_sequence, second.query_sequence)
    
    num_hq_mismatches = 0
    for q_i, t_i in al['mismatches']:
        if (first.query_qualities[q_i] > 20) and (second.query_qualities[t_i] > 20):
            num_hq_mismatches += 1
            
    return al['XO'], num_hq_mismatches

cell_key = lambda al: al.get_tag(CELL_BC_TAG)
UMI_key = lambda al: al.get_tag(UMI_TAG)

def fastq_sort_key(read):
    annotation = read_Annotation.from_identifier(read.name)
    return annotation['cell_BC'], annotation['UMI']

sort_key = lambda al: (al.get_tag(CELL_BC_TAG), al.get_tag(UMI_TAG))

empty_header = pysam.AlignmentHeader()

def sort_cellranger_bam(bam_fn, sorted_fn, show_progress=False):
    Path(sorted_fn).parent.mkdir(exist_ok=True)

    bam_fh = pysam.AlignmentFile(str(bam_fn))
    total_reads_in = bam_fh.mapped + bam_fh.unmapped

    als = bam_fh
    if show_progress:
        als = progress(als, total=total_reads_in, desc='Sorting')

    relevant = (al for al in als if al.is_unmapped and al.has_tag(CELL_BC_TAG))

    max_read_length = 0
    total_reads_out = 0
    
    chunk_fns = []
        
    for i, chunk in enumerate(utilities.chunks(relevant, 10000000)):
        suffix = '.{:06d}.bam'.format(i)
        chunk_fn = Path(sorted_fn).with_suffix(suffix)
        sorted_chunk = sorted(chunk, key=sort_key)
    
        with pysam.AlignmentFile(str(chunk_fn), 'wb', header=empty_header) as fh:
            for al in sorted_chunk:
                max_read_length = max(max_read_length, al.query_length)
                total_reads_out += 1
                fh.write(al)

        chunk_fns.append(chunk_fn)

    chunk_fhs = [pysam.AlignmentFile(str(fn), check_sq=False) for fn in chunk_fns]
    
    with pysam.AlignmentFile(str(sorted_fn), 'wb', header=empty_header) as fh:
        merged_chunks = heapq.merge(*chunk_fhs, key=sort_key)

        if show_progress:
            merged_chunks = progress(merged_chunks, total=total_reads_out, desc='Merging sorted chunks')

        for al in merged_chunks:
            fh.write(al)

    for fh in chunk_fhs:
        fh.close()

    for fn in chunk_fns:
        fn.unlink()
    
    yaml_fn = sorted_fn.with_suffix('.yaml')
    stats = {
        'total_reads': total_reads_out,
        'max_read_length': max_read_length,
    }
    yaml_fn.write_text(yaml.dump(stats, default_flow_style=False))

def sort_cellranger_bam_to_fastq(bam_fn, sorted_fn, gemgroup, show_progress=False):
    Path(sorted_fn).parent.mkdir(exist_ok=True)

    bam_fh = pysam.AlignmentFile(str(bam_fn))
    total_reads_in = bam_fh.mapped + bam_fh.unmapped

    als = bam_fh
    if show_progress:
        als = progress(als, total=total_reads_in, desc='Sorting')

    relevant = (al for al in als if al.is_unmapped and al.has_tag(CELL_BC_TAG))

    max_read_length = 0
    total_reads_out = 0
    
    chunk_fns = []
        
    for i, chunk in enumerate(utilities.chunks(relevant, 5000000)):
        suffix = '.{:06d}.fastq'.format(i)
        chunk_fn = Path(sorted_fn).with_suffix(suffix)
        sorted_chunk = sorted(chunk, key=sort_key)
    
        with chunk_fn.open('w') as fh:
            for al in sorted_chunk:
                max_read_length = max(max_read_length, al.query_length)

                cell_BC = al.get_tag(CELL_BC_TAG)
                cell_BC_gemgroup = '{}-{}'.format(cell_BC.split('-')[0], gemgroup)
                name = Annotations['Read'](cell_BC=cell_BC_gemgroup,
                                           UMI=al.get_tag(UMI_TAG),
                                           original_name=al.query_name,
                                          )
                read = fastq.Read(name,
                                  al.query_sequence,
                                  fastq.encode_sanger(al.query_qualities),
                                 )
                fh.write(str(read))

                total_reads_out += 1

        chunk_fns.append(chunk_fn)

    chunk_reads = [fastq.reads(fn) for fn in chunk_fns]
    
    with sorted_fn.open('w') as fh:
        merged_chunks = heapq.merge(*chunk_reads, key=lambda r: r.name)

        if show_progress:
            merged_chunks = progress(merged_chunks, total=total_reads_out, desc='Merging sorted chunks')

        for read in merged_chunks:
            fh.write(str(read))

    for fn in chunk_fns:
        fn.unlink()
    
    yaml_fn = sorted_fn.with_suffix('.yaml')
    stats = {
        'total_reads': total_reads_out,
        'max_read_length': max_read_length,
    }
    yaml_fn.write_text(yaml.dump(stats, default_flow_style=False))

def index_sorted_fastq(sorted_fastq_fn, show_progress=False):
    reads_per_landmark = 100000
    next_landmark = 0
    
    fh = sorted_fastq_fn.open()
    
    def lines():
        line = fh.readline()
        while line:
            yield line
            line = fh.readline()
    
    landmarks = []
    
    before_pos = fh.tell()
    
    reads = fastq.reads(lines())
    
    if show_progress:
        yaml_fn = sorted_fastq_fn.with_suffix('.yaml')
        stats = yaml.load(yaml_fn.read_text())
        total_reads = stats['total_reads']
        reads = progress(reads, total=total_reads, desc='Indexing')
    
    grouped = utilities.group_by(enumerate(reads), key=lambda i_and_read: fastq_sort_key(i_and_read[1]))
    
    for (cell_BC, UMI), group in grouped:
        i, n = group[0]
        if i >= next_landmark:           
            landmark = (cell_BC, UMI, before_pos)
            landmarks.append(landmark)
            while next_landmark <= i:
                next_landmark += reads_per_landmark
            
        before_pos = fh.tell()
    
    index_fn = sorted_fastq_fn.with_suffix('.index')
    pd.DataFrame(landmarks).to_csv(index_fn, header=False, index=False, sep='\t')
    
def load_index(sorted_fastq_fn):
    index_fn = sorted_fastq_fn.with_suffix('.index')
    index_df = pd.read_table(index_fn, header=None)
    landmarks = index_df.to_records(index=False)
    return landmarks
    
def find_closest_landmark_before(landmarks, cell_BC, UMI):
    cell_BC_UMIs = [(cell_BC, UMI) for cell_BC, UMI, offset in landmarks]
    # bisect logic here taken directly from bisect docs to find rightmost value less than or equal to
    i = bisect.bisect_right(cell_BC_UMIs, (cell_BC, UMI))
    if i:
        offset = landmarks[i - 1][-1]
    else:
        raise ValueError
    return offset

def get_cell_BC_UMI_reads(sorted_fastq_fn, cell_BC, UMI):
    landmarks = load_index(sorted_fastq_fn)
    offset = find_closest_landmark_before(landmarks, cell_BC, UMI)
    fh = sorted_fastq_fn.open()
    fh.seek(offset)
    
    relevant_reads = []
    
    target = (cell_BC, UMI)
    for read in fastq.reads(fh):
        key = fastq_sort_key(read)
        if key == target:
            relevant_reads.append(read)
        elif key > target:
            break
   
    return relevant_reads

def error_correct_UMIs(cell_group, max_UMI_distance=1):
    # sort UMIs in descending order by number of occurrences.
    UMI_counts = Counter(al.get_tag(UMI_TAG) for al in cell_group)
    UMIs = [UMI for UMI, count in UMI_counts.most_common()]

    ds = hamming_distance_matrix(UMIs)

    corrections = register_corrections(ds, max_UMI_distance, UMIs)

    for al in cell_group:
        correct_to = corrections.get(al.get_tag(UMI_TAG))
        if correct_to:
            al.set_tag(UMI_TAG, correct_to)
    
    return cell_group

def merge_annotated_clusters(biggest, other):
    merged_id = biggest.get_tag(CLUSTER_ID_TAG)
    if not merged_id.endswith('+'):
        merged_id = merged_id + '+'
    biggest.set_tag(CLUSTER_ID_TAG, merged_id, 'Z')

    total_reads = biggest.get_tag(NUM_READS_TAG) + other.get_tag(NUM_READS_TAG)
    biggest.set_tag(NUM_READS_TAG, total_reads, 'i')

    return biggest

def form_collapsed_clusters(sorted_fn,
                            collapsed_fn,
                            max_hq_mismatches,
                            max_indels,
                            max_UMI_distance,
                            show_progress=True):

    yaml_fn = sorted_fn.with_suffix('.yaml')
    stats = yaml.load(yaml_fn.read_text())
    max_read_length = stats['max_read_length']
    total_reads = stats['total_reads']

    sorted_als = pysam.AlignmentFile(str(sorted_fn), check_sq=False)
    if progress:
        sorted_als = progress(sorted_als, total=total_reads, desc='Collapsing')
    
    cell_groups = utilities.group_by(sorted_als, cell_key)
    
    with pysam.AlignmentFile(str(collapsed_fn), 'wb', header=empty_header) as collapsed_fh:
        for cell_BC, cell_group in cell_groups:
            if max_UMI_distance > 0:
                error_correct_UMIs(cell_group, max_UMI_distance)

            for UMI, UMI_group in utilities.group_by(cell_group, UMI_key):
                clusters = form_clusters(UMI_group, max_read_length, max_hq_mismatches)
                clusters = sorted(clusters, key=lambda c: c.get_tag(NUM_READS_TAG), reverse=True)

                for i, cluster in enumerate(clusters):
                    cluster.set_tag(CELL_BC_TAG, cell_BC, 'Z')
                    cluster.set_tag(UMI_TAG, UMI, 'Z')
                    cluster.set_tag(CLUSTER_ID_TAG, str(i), 'Z')

                biggest = clusters[0]
                rest = clusters[1:]

                not_collapsed = []

                for other in rest:
                    if other.get_tag(NUM_READS_TAG) == biggest.get_tag(NUM_READS_TAG):
                        not_collapsed.append(other)
                    else:
                        indels, hq_mismatches = align_clusters(biggest, other)

                        if indels <= max_indels and hq_mismatches <= max_hq_mismatches:
                            biggest = merge_annotated_clusters(biggest, other)
                        else:
                            not_collapsed.append(other)
                
                for cluster in [biggest] + not_collapsed:
                    annotation = Annotations['cluster'](cell_BC=cluster.get_tag(CELL_BC_TAG),
                                                       UMI=cluster.get_tag(UMI_TAG),
                                                       num_reads=cluster.get_tag(NUM_READS_TAG),
                                                       cluster_id=cluster.get_tag(CLUSTER_ID_TAG),
                                                      )

                    cluster.query_name = str(annotation)
                    collapsed_fh.write(cluster)

def make_cluster_fastqs(collapsed_fn, target, gemgroup, notebook=True):
    group_dir = Path(collapsed_fn).parent
    cell_identities = pd.read_csv('/home/jah/projects/britt/data/cell_identities.csv', index_col='cell_barcode') 
    guides = split_into_guide_fastqs(collapsed_fn, cell_identities, gemgroup, group_dir)
    make_sample_sheet(group_dir, target, guides)

def split_into_guide_fastqs(collapsed_fn, cell_identities, gemgroup, group_dir):
    clusters = pysam.AlignmentFile(str(collapsed_fn), check_sq=False)

    guide_fhs = {}

    for cluster in clusters:
        cell_BC = cluster.get_tag(CELL_BC_TAG)
        cell_BC = '{0}-{1}'.format(cell_BC.split('-')[0], gemgroup)
        if cell_BC not in cell_identities.index:
            guide = 'ambiguous'
        else:
            row = cell_identities.loc[cell_BC]
            if (row['guide_identity'] == '*' or 
                row['number_of_cells'] != 1 or
                not row['good_coverage']
               ):
                guide = 'ambiguous'
            else:
                guide = row['guide_identity']

        if guide not in guide_fhs:
            guide_fn = Path(group_dir) / (guide + '.fastq')
            guide_fhs[guide] = guide_fn.open('w')

        read = sam.mapping_to_Read(cluster)

        # temporary hack
        read.name = '{0}_{1}'.format(cell_BC, read.name.split('_', 1)[1])

        guide_fhs[guide].write(str(read))

    for guide, fh in guide_fhs.items():
        fh.close()

    guides = sorted(guide_fhs)
    return guides

def make_sample_sheet(group_dir, target, guides):
    color_list = bokeh.palettes.Category20c_20[:16] #+ bokeh.palettes.Category20b_20
    color_groups = itertools.cycle(list(zip(*[iter(color_list)]*4)))

    sample_sheet = {}

    grouped_guides = utilities.group_by(sorted(guides), lambda n: n.split('-')[0])
    for (group_name, group), color_group in zip(grouped_guides, color_groups):
        for name, color in zip(group, color_group[1:]):
            sample_sheet[name] = {
                'fastq_fns': name + '.fastq',
                'target_info': target,
                'project': 'screen',
                'color': color,
                'experiment_type': 'britt',
            }

    sample_sheet_fn = group_dir / 'sample_sheet.yaml'
    sample_sheet_fn.write_text(yaml.dump(sample_sheet, default_flow_style=False))

def load_sample_sheet(base_dir):
    sample_sheet_fn = Path(base_dir) / 'data' / 'sample_sheet.yaml'
    sample_sheet = yaml.load(sample_sheet_fn.read_text())
    return sample_sheet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--force_sort', action='store_true')
    parser.add_argument('--force_collapse', action='store_true')
    parser.add_argument('--no_progress', action='store_true')

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--parallel', metavar='MAX_PROCS')
    mode_group.add_argument('--collapse', metavar='NAME')

    args = parser.parse_args()

    sample_sheet = load_sample_sheet(args.base_dir)

    if args.parallel is not None:
        max_procs = args.parallel

        if args.force_sort:
            possibly_force_sort = ['--force_sort']
        else:
            possibly_force_sort = []

        parallel_command = [
            'parallel',
            '-n', '1', 
            '--verbose',
            '--max-procs', max_procs,
            './collapse.py',
            '--no_progress',
        ] + possibly_force_sort + [
            '--base_dir', str(base_dir),
            '--collapse', ':::',
        ] + sorted(sample_sheet)

        subprocess.check_call(parallel_command)

    elif args.collapse is not None:
        name = args.collapse
        info = sample_sheet[name]
        gemgroup = info.get('gemgroup', 1)

        max_hq_mismatches = info.get('max_hq_mismatches', 10)
        max_indels = info.get('max_indels', 2)
        max_UMI_distance = info.get('max_UMI_distance', 1)
        
        show_progress = not args.no_progress

        input_fn = Path(info['cellranger_dir']) / 'outs' / 'possorted_genome_bam.bam'
        sorted_fn = (base_dir / 'data' / name / name).with_suffix('.fastq')
        collapsed_fn = sorted_fn.with_name(sorted_fn.stem + '_collapsed.bam')

        if not sorted_fn.exists() or args.force_sort:
            sort_cellranger_bam_to_fastq(input_fn,
                                         sorted_fn,
                                         gemgroup,
                                         show_progress=show_progress,
                                        )

        index_sorted_fastq(sorted_fn, show_progress=show_progress)

        if not collapsed_fn.exists() or args.force_collapse:
            form_collapsed_clusters(sorted_fn,
                                    collapsed_fn,
                                    max_hq_mismatches,
                                    max_indels,
                                    max_UMI_distance,
                                    show_progress=show_progress,
                                   )
           
        make_cluster_fastqs(collapsed_fn, info['target'], info['gemgroup'])
