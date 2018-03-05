#!/usr/bin/env python3.6

import matplotlib
matplotlib.use('Agg', warn=False)
import argparse
import subprocess
import shutil
import tempfile
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import matplotlib.pyplot as plt
import bokeh
import pysam
import yaml

import Sequencing.fastq as fastq
import Sequencing.utilities as utilities

import target_info
import blast
import layout
import visualize

base_dir = Path('/home/jah/projects/manu/experiments')
    
palette = bokeh.palettes.Category20c_20
source_to_color = {}
for i, source in enumerate(['PCR', 'plasmid', 'ssDNA', 'CT']):
    for replicate in [1, 2, 3]:
        source_to_color[source, replicate] = palette[4 * i  + (replicate - 1)]

def priority_tuple_to_string(t):
    return '{0}: {1}'.format(*t)

def priority_string_to_description(s):
    return s.split(': ')[1]

class PacbioExperiment(object):
    def __init__(self, name):
        self.name = name
        self.dir = base_dir / name
        description_fn = self.dir / 'description.yaml'
        description = yaml.load(description_fn.open())

        self.target_name = description['target_info']
        self.target_info = target_info.TargetInfo(self.target_name)
        self.fns = {
            'fastq': self.dir / description['fastq_fn'],
            'bam': self.dir / 'alignments.bam',
            'bam_by_name': self.dir / 'alignments.by_name.bam',
            'outcomes_dir': self.dir / 'outcomes',
            'outcome_counts': self.dir / 'outcome_counts.csv',
            'lengths_figure': self.dir / 'all_lengths.png',
        }

        self.cell_line = description.get('cell_line')
        self.donor_type = description.get('donor_type')
        self.replicate = description.get('replicate')
        self.sorted = description.get('sorted')

        self.color = source_to_color.get((self.donor_type, self.replicate), 'grey')

    def query_names(self):
        for read in fastq.reads(self.fns['fastq']):
            yield read.name

    def outcome_query_names(self, outcome):
        outcome_fns = self.outcome_fns(outcome)
        als = pysam.AlignmentFile(outcome_fns['bam_by_name'])
        names = [n for n, _ in utilities.group_by(als, lambda al: al.query_name)]
        return names

    @utilities.memoized_property
    def read_lengths(self):
        lengths = Counter(len(r.seq) for r in fastq.reads(self.fns['fastq']))
        lengths = utilities.counts_to_array(lengths)
        return lengths

    def outcome_read_lengths(self, outcome):
        outcome_fns = self.outcome_fns(outcome)
        als = pysam.AlignmentFile(outcome_fns['bam_by_name'])

        lengths = Counter()
        for _, group in utilities.group_by(als, lambda al: al.query_name):
            lengths[group[0].query_length] += 1

        lengths = utilities.counts_to_array(lengths)
        return lengths

    def generate_alignments(self):
        blast.blast(self.target_info.fns['ref_fasta'],
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
                                 sep='\t',
                                )
        else:
            counts = None

        return counts

    @property
    def outcomes(self):
        counts = self.load_outcome_counts()
        if counts is None:
            return []
        else:
            return list(counts.index)

    def outcome_fns(self, outcome):
        outcome_string = '_'.join(map(str, outcome))
        outcome_dir = self.fns['outcomes_dir'] / outcome_string
        fns = {
            'dir': outcome_dir,
            'query_names': outcome_dir / 'qnames.txt',
            'bam_by_name': outcome_dir / 'alignments.by_name.bam',
            'first_example': outcome_dir / 'first_examples.png',
            'combined_figure': outcome_dir / 'combined.png',
            'lengths_figure': outcome_dir / 'lengths.png',
            'text_alignments': outcome_dir / 'alignments.txt',
        }
        return fns

    def outcome_query_names(self, outcome):
        fns = self.outcome_fns(outcome)
        qnames = [l.strip() for l in open(fns['query_names'])]
        return qnames

    def count_outcomes(self):
        bam_fh = pysam.AlignmentFile(self.fns['bam_by_name'])
        alignment_groups = utilities.group_by(bam_fh, lambda al: al.query_name)
        outcomes = defaultdict(list)

        for name, als in alignment_groups:
            layout_info = layout.characterize_layout(als, self.target_info)
            outcomes[layout_info['outcome']].append(name)

        bam_fh.close()

        shutil.rmtree(self.fns['outcomes_dir'])
        self.fns['outcomes_dir'].mkdir()

        counts = {description: len(names) for description, names in outcomes.items()}
        series = pd.Series(counts)

        series.to_csv(self.fns['outcome_counts'], sep='\t')
        
        # To make plotting easier, for each outcome, make a file listing all of
        # qnames for the outcome and a bam file (sorted by name) with all of the
        # alignments for these qnames.

        qname_to_outcome = {}
        bam_fhs = {}

        full_bam_fh = pysam.AlignmentFile(self.fns['bam_by_name'])
        
        for outcome, qnames in outcomes.items():
            outcome_fns = self.outcome_fns(outcome)
            outcome_fns['dir'].mkdir()
            bam_fhs[outcome] = pysam.AlignmentFile(outcome_fns['bam_by_name'], 'w', template=full_bam_fh)
            
            with outcome_fns['query_names'].open('w') as fh:
                for qname in qnames:
                    qname_to_outcome[qname] = outcome
                    fh.write(qname + '\n')
        
        for al in full_bam_fh:
            outcome = qname_to_outcome[al.query_name]
            bam_fhs[outcome].write(al)

        bam_fh.close()
        for outcome, fh in bam_fhs.items():
            fh.close()

    def make_outcome_plots(self, num_examples=10):
        fig = visualize.make_length_plot(self.read_lengths, self.color)
        fig.savefig(str(self.fns['lengths_figure']), bbox_inches='tight')
        plt.close(fig)

        for outcome in self.outcomes:
            outcome_fns = self.outcome_fns(outcome)
            
            with tempfile.TemporaryDirectory(suffix='_outcome_plots') as temp_dir:
                temp_fns = []
                for i in range(num_examples):
                    fig = visualize.plot_read(self.name, i, outcome=outcome, parsimonious=True)
                    if fig is None:
                        continue
                        
                    fig.axes[0].set_title('_', y=1.2, color='white')
                    
                    fn = Path(temp_dir) / '{0:05d}.png'.format(i)
                    temp_fns.append(fn)
                    fig.savefig(str(fn), bbox_inches='tight')
                    
                    if i == 0:
                        fig.axes[0].set_title('')
                        fig.savefig(str(outcome_fns['first_example']), bbox_inches='tight')
                        
                    plt.close(fig)

                lengths = self.outcome_read_lengths(outcome)

                fig = visualize.make_length_plot(self.read_lengths, self.color, lengths)
                fig.savefig(str(outcome_fns['lengths_figure']), bbox_inches='tight')
                plt.close(fig)
                
                to_concat = [outcome_fns['lengths_figure']] + temp_fns                
                convert_command = ['convert'] + to_concat + ['-background', 'white', '-gravity', 'center', '-append', outcome_fns['combined_figure']]
                subprocess.check_call(convert_command)
    
    def process(self):
        #self.generate_alignments()
        #self.count_outcomes()
        self.make_outcome_plots()

def get_all_experiments(conditions=None):
    if conditions is None:
        conditions = {}

    def check_conditions(exp):
        for k, v in conditions.items():
            if isinstance(v, (list, tuple, set)):
                if getattr(exp, k) not in v:
                    return False
            else:
                if getattr(exp, k) != v:
                    return False
        return True

    names = (p.name for p in base_dir.glob('*') if p.is_dir())
    exps = [PacbioExperiment(n) for n in names]
    filtered = [exp for exp in exps if check_conditions(exp)]
    return filtered

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
