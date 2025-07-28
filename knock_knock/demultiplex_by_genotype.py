import gzip
import shutil
from collections import defaultdict, Counter
from contextlib import ExitStack
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hits.fasta
import hits.fastq

import knock_knock.arrayed_experiment_group

MATCHES_NONE = 'matches_none'
MATCHES_MULTIPLE = 'matches_multiple'
full_extension = '.fastq.gz'

def add_suffix_to_fn(fn, suffix, full_extension):
    fn = Path(fn)
    return f'{fn.name[:-len(full_extension)]}_{suffix}{full_extension}'

def get_unique_kmers(genotypes, k=20, relevant_slice=slice(None)):
    def get_kmers(seq, k):
        return {seq[i:i+k] for i in range(0, len(seq) - k)}

    kmers = {name: get_kmers(seq[relevant_slice], k) for name, seq in genotypes.items()}

    unique_kmers = {}

    for name in kmers:
        kmers_in_any_other = set.union(*[kmers[other_name] for other_name in kmers if other_name != name])
        unique_kmers[name] = kmers[name] - kmers_in_any_other

    return unique_kmers

def add_genotypes_to_sample_sheet(base_dir, batch_name, genotypes_fasta_fn):
    base_dir = Path(base_dir)

    input_dir = base_dir / 'documents' / batch_name

    demultiplexed_batch_name = f'{batch_name}_by_genotype'
    output_dir = base_dir / 'documents' / demultiplexed_batch_name
    output_dir.mkdir(exist_ok=True, parents=True)

    genotypes_fasta_fn = Path(genotypes_fasta_fn)
    genotypes = hits.fasta.to_dict(genotypes_fasta_fn, upper_case=True)

    sample_sheet_df = knock_knock.arrayed_experiment_group.sanitize_and_validate_sample_sheet(input_dir / 'sample_sheet.csv')

    suffixed_rows = []

    i = 0

    for _, row in sample_sheet_df.iterrows():
        for genotype in genotypes:
            suffixed_row = row.copy()
            suffixed_row['sample_name'] = f'{row["sample_name"]}_{genotype}'
            suffixed_rows.append(suffixed_row)
            
            name = Path(row['R1']).name

            suffixed_R1 = str(add_suffix_to_fn(Path(name), genotype, full_extension))
            
            suffixed_row['R1'] = suffixed_R1
            
            suffixed_row['genome'] = genotype

            suffixed_row.name = i

            i += 1

    suffixed_df = pd.DataFrame(suffixed_rows)

    suffixed_df.to_csv(output_dir / 'sample_sheet.csv', index=False)

    for fn in [genotypes_fasta_fn, input_dir / 'sgRNAs.csv', input_dir / 'amplicon_primers.csv']:
        shutil.copy(fn, output_dir / fn.name)

    knock_knock.arrayed_experiment_group.make_strategies(base_dir, suffixed_df)
    knock_knock.arrayed_experiment_group.make_group_descriptions_and_sample_sheet(base_dir, suffixed_df, demultiplexed_batch_name)

def demultiplex(base_dir, batch_name, genotypes_fasta_fn, relevant_slice=slice(None), progress=None):
    base_dir = Path(base_dir)

    input_dir = base_dir / 'data' / batch_name

    demultiplexed_batch_name = f'{batch_name}_by_genotype'
    output_dir = base_dir / 'data' / demultiplexed_batch_name
    output_dir.mkdir(exist_ok=True, parents=True)

    genotypes = hits.fasta.to_dict(genotypes_fasta_fn, upper_case=True)

    if progress is None:
        def ignore_kwargs(x, **kwargs):
            return x
        progress = ignore_kwargs

    unique_kmers = get_unique_kmers(genotypes, relevant_slice=relevant_slice)

    suffixes = sorted(genotypes) + [MATCHES_NONE, MATCHES_MULTIPLE]

    fastq_fns = sorted(input_dir.glob(f'*{full_extension}'))

    genotype_counts = defaultdict(Counter)

    for input_fn in progress(fastq_fns):
        with ExitStack() as stack:
            suffix_to_fh = {}
            for suffix in suffixes:
                fn = output_dir / add_suffix_to_fn(input_fn, suffix, full_extension)
                suffix_to_fh[suffix] = stack.enter_context(gzip.open(fn, 'wt', compresslevel=1))

            for read in hits.fastq.reads(input_fn):
                fingerprint = set()

                for name, ms in unique_kmers.items():
                    if any(m in read.seq for m in ms):
                        fingerprint.add(name)

                fingerprint = tuple(sorted(fingerprint))

                if len(fingerprint) == 1:
                    suffix = fingerprint[0]
                    key = fingerprint[0]
                elif len(fingerprint) > 1:
                    suffix = MATCHES_MULTIPLE
                    key = ','.join(fingerprint)
                else:
                    suffix = MATCHES_NONE
                    key = MATCHES_NONE

                genotype_counts[input_fn.name][key] += 1
                
                suffix_to_fh[suffix].write(str(read))

    genotype_count_fn = output_dir / 'genotype_counts.csv'
    genotype_counts = pd.DataFrame(genotype_counts).T.fillna(0).astype(int).sort_index(axis=1)
    genotype_counts.to_csv(genotype_count_fn)

    fig = plot_genotype_stats(genotype_counts)
    fig.savefig(output_dir / 'genotype_counts.png', dpi=200, bbox_inches='tight')

    add_genotypes_to_sample_sheet(base_dir, batch_name, genotypes_fasta_fn)

def plot_genotype_stats(genotype_counts):
    at_least_one = genotype_counts.drop('matches_none', axis=1)

    normalized = at_least_one.div(at_least_one.sum(axis=1), axis=0) * 100

    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axs[0].plot(genotype_counts.sum(axis=1).values, '.-', color='black')
    axs[0].set_ylabel('Total reads', size=14)

    axs[1].plot(at_least_one.sum(axis=1).values, '.-', color='black')
    axs[1].set_ylabel('Reads matching at\nleast one genotype', size=14)

    ax = axs[2]
    for col in sorted(normalized, key=lambda s: ',' in s):
        ys = normalized[col]
        xs = np.arange(len(ys))
        ax.plot(xs, ys, '.-', label=col, clip_on=False)
    ax.set_ylabel('% of reads matching\n each genotype', size=14)

    for ax in axs:
        ax.set_ylim(0)

    axs[2].axhline(50)
    axs[2].set_ylim(0, 100)

    axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1))

    axs[2].set_xlim(-0.5, len(genotype_counts.index) - 0.5)
    axs[2].set_xlabel('Sample', size=14)

    return fig
