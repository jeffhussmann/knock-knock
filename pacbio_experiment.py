#!/usr/bin/env python3.6

import argparse
import subprocess
from pathlib import Path

import pandas as pd
import yaml

import Sequencing.fastq as fastq

import target
import blast
import layout

base_dir = Path('/home/jah/projects/manu/experiments')

class PacbioExperiment(object):
    def __init__(self, name):
        self.name = name
        self.dir = base_dir / name
        description_fn = self.dir / 'description.yaml'
        description = yaml.load(description_fn.open())

        self.target = target.Target(description['target'])
        self.fns = {
            'fastq': self.dir / description['fastq_fn'],
            'bam': self.dir / 'alignments.bam',
            'bam_by_name': self.dir / 'alignments.by_name.bam',
            'outcomes_dir': self.dir / 'outcomes',
            'outcome_counts': self.dir / 'outcomes' / 'counts.csv',
            'lengths_figure': self.dir / 'outcomes' / 'all_lengths.png',
        }

        self.cell_line = description.get('cell_line')
        self.donor_type = description.get('donor_type')
        self.replicate = description.get('replicate')
        self.sorted = description.get('sorted')

    def query_names(self):
        for read in fastq.reads(self.fns['fastq']):
            yield read.name

    def generate_alignments(self):
        blast.blast(self.target.fns['ref_fasta'],
                    self.fns['fastq'],
                    self.fns['bam'],
                    self.fns['bam_by_name'],
                   )

    def load_outcome_counts(self):
        if self.fns['outcome_counts'].exists():
            counts = pd.read_csv(self.fns['outcome_counts'],
                                 index_col=(0, 1),
                                 header=None,
                                 squeeze=True,
                                )
        else:
            counts = None

        return counts

    def outcomes(self):
        counts = self.load_outcome_counts()
        if counts is None:
            return []
        else:
            return list(counts.index)

    def outcome_fns(self, outcome):
        outcome_dir = self.fns['outcomes_dir'] / '_'.join(outcome)
        fns = {
            'query_names': outcome_dir / 'qnames.txt',
            'bam': outcome_dir / 'alignments.bam',
            'bam_by_name': outcome_dir / 'alignments.by_name.bam',
            'first_example': outcome_dir / 'first_examples.png',
            'combined_figure': outcome_dir / 'combined.png',
            'lengths_figure': outcome_dir / 'lengths.png',
            'text_alignments': outcome_dir / 'alignments.txt',
        }
        return fns

    def outcome_query_names(self, outcome):
        fns = self.outcome_fns(outcome)
        qnames = [l.strip() for l in open(fns['qnames'])]
        return qnames

    def count_outcomes(self):
        outcomes = layout.count_outcomes(self.fns['bam_by_name'], self.target)
        counts = {description: len(names) for description, names in outcomes.items()}
        series = pd.Series(counts)
        series.to_csv(self.fns['outcome_counts'])

    def process(self):
        self.generate_alignments()

def get_all_experiments():
    names = (p.name for p in base_dir.glob('*') if p.is_dir())
    exps = [PacbioExperiment(n) for n in names]
    return exps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--process')
    group.add_argument('--parallel')

    args = parser.parse_args()

    if args.parallel is not None:
        max_procs = args.parallel

        exps = get_all_experiments()
        names = sorted(exp.name for exp in exps)

        parallel_command = [
            'parallel',
            '--verbose',
            '--max-procs', max_procs,
            './pacbio_experiment.py', '--process',
            ':::'] + names

        subprocess.check_call(parallel_command)

    elif args.process is not None:
        name = args.process
        exp = PacbioExperiment(name)
        exp.process()
