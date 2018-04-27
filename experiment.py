import matplotlib
matplotlib.use('Agg', warn=False)

import shutil
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import bokeh.palettes
import pysam
import yaml
import scipy.signal

import sequencing.sam as sam
import sequencing.fastq as fastq
import sequencing.utilities as utilities
import sequencing.visualize_structure as visualize_structure
import sequencing.sw as sw
import sequencing.adapters as adapters

from . import target_info
from . import blast
from . import layout
from . import visualize

palette = bokeh.palettes.Category20c_20
source_to_color = {}
for i, source in enumerate(['PCR', 'plasmid', 'ssDNA', 'CT']):
    for replicate in [1, 2, 3]:
        source_to_color[source, replicate] = palette[4 * i  + (replicate - 1)]

palette = bokeh.palettes.Set2[8]
cap_to_color = {
    'AmMC6': palette[0],
    'Biotin': palette[1],
    'IDDT': palette[2],
}

def extract_color(description):
    if 'color' in description:
        color = description['color']
    elif description.get('capped', False):
        color = cap_to_color[description['cap']]
    else:
        donor = description.get('donor_type')
        rep = description.get('replicate', 1)
        color = source_to_color.get((donor, rep), 'grey')

    return color

class Experiment(object):
    def __init__(self, base_dir, group, name, description=None):
        self.group = group
        self.name = name

        base_dir = Path(base_dir)
        self.dir = base_dir / 'results' / group / name
        if not self.dir.is_dir():
            self.dir.mkdir(parents=True)

        data_dir = base_dir / 'data' / group

        if description is None:
            sample_sheet_fn = data_dir / 'sample_sheet.yaml'
            sample_sheet = yaml.load(sample_sheet_fn.read_text())
            self.description = sample_sheet[name]
        else:
            self.description = description

        # When checking if an Experiment meets filtering conditions, want to be
        # able to just test description.
        self.description['group'] = group
        self.description['name'] = name

        self.target_name = self.description['target_info']
        self.target_info = target_info.TargetInfo(base_dir, self.target_name)
        self.fns = {
            'bam': self.dir / 'alignments.bam',
            'bam_by_name': self.dir / 'alignments.by_name.bam',
            'outcomes_dir': self.dir / 'outcomes',
            'outcome_counts': self.dir / 'outcome_counts.csv',
            'outcome_sort_order': self.dir / 'outcome_sort_order.txt',
            'lengths_figure': self.dir / 'all_lengths.png',
            'length_ranges': self.dir / 'length_ranges.csv',
            'manual_length_ranges': self.dir / 'manual_length_ranges.csv',
        }

        if 'fastq_fn' in self.description:
            self.fns['fastq'] = data_dir / self.description['fastq_fn']

            if not self.fns['fastq'].exists():
                raise ValueError('{0}: {1} specifies non-existent {2}'.format(group, name, self.fns['fastq']))

        else:
            self.fns['R1'] = data_dir / self.description['R1_fn']
            self.fns['R2'] = data_dir / self.description['R2_fn']
            
            for k in ['R1', 'R2']:
                if not self.fns[k].exists():
                    raise ValueError('{0}: {1} specifies non-existent {2}'.format(group, name, self.fns[k]))

            self.fns['fastq'] = self.dir / 'stitched.fastq'

        self.color = extract_color(self.description)
    
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

    def query_names(self):
        for read in fastq.reads(self.fns['fastq'], up_to_space=True):
            yield read.name

    @utilities.memoized_property
    def read_lengths(self):
        lengths = Counter(len(r.seq) for r in fastq.reads(self.fns['fastq']))
        lengths = utilities.counts_to_array(lengths)
        return lengths

    @property
    def length_ranges(self):
        path = self.fns['length_ranges']
        if path.exists():
            ranges = pd.read_csv(path, sep='\t', header=None, names=['start', 'end'])
        else:
            ranges = pd.DataFrame(columns=['start', 'end'])
        return ranges

    def call_peaks_in_length_distribution(self):
        smoothed = utilities.smooth(self.read_lengths, 25)

        all_peaks, props = scipy.signal.find_peaks(smoothed, prominence=.1, distance=100)
        above_background = (props['prominences'] / smoothed[all_peaks]) > 0.5
        peaks = all_peaks[above_background]
        widths, *_ = scipy.signal.peak_widths(smoothed, peaks, rel_height=0.6)

        length_ranges = [(int(p - w / 2), int(p + w / 2)) for p, w in zip(peaks, widths)]
            
        df = pd.DataFrame(length_ranges)                  
        df.to_csv(self.fns['length_ranges'], index=False, header=None, sep='\t')

    def outcome_read_lengths(self, outcome):
        outcome_fns = self.outcome_fns(outcome)

        lengths = Counter()
        for _, group in sam.grouped_by_name(outcome_fns['bam_by_name']):
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

    def outcome_query_names(self, outcome):
        fns = self.outcome_fns(outcome)
        qnames = [l.strip() for l in open(fns['query_names'])]
        return qnames
    
    def load_outcome_sort_order(self):
        sort_order = {}
        for line in self.fns['outcome_sort_order'].open():
            outcome_string, priority_string = line.strip().split('\t')
            outcome = tuple(outcome_string.split('_'))
            priority = tuple(int(p) for p in priority_string.split('_'))
            sort_order[outcome] = priority
        return sort_order

    def count_outcomes(self):
        if self.fns['outcomes_dir'].is_dir():
            shutil.rmtree(str(self.fns['outcomes_dir']))

        self.fns['outcomes_dir'].mkdir()

        bam_fh = pysam.AlignmentFile(str(self.fns['bam_by_name']))
        alignment_groups = sam.grouped_by_name(bam_fh)
        outcomes = defaultdict(list)

        sort_order = {}
        for name, als in alignment_groups:
            layout_info = layout.characterize_layout(als, self.target_info)
            
            outcome = layout_info['outcome']
            outcomes[outcome['description']].append(name)

            sort_order[outcome['description']] = outcome['sort_order']

        bam_fh.close()

        counts = {description: len(names) for description, names in outcomes.items()}
        series = pd.Series(counts)

        series.to_csv(self.fns['outcome_counts'], sep='\t')

        with self.fns['outcome_sort_order'].open('w') as fh:
            for outcome, priority in sort_order.items():
                outcome_string = '_'.join(outcome)
                priority_string = '_'.join(map(str, priority))

                fh.write('{0}\t{1}\n'.format(outcome_string, priority_string))
        
        # To make plotting easier, for each outcome, make a file listing all of
        # qnames for the outcome and a bam file (sorted by name) with all of the
        # alignments for these qnames.

        qname_to_outcome = {}
        bam_fhs = {}

        full_bam_fh = pysam.AlignmentFile(str(self.fns['bam_by_name']))
        
        for outcome, qnames in outcomes.items():
            outcome_fns = self.outcome_fns(outcome)
            outcome_fns['dir'].mkdir()
            bam_fhs[outcome] = pysam.AlignmentFile(str(outcome_fns['bam_by_name']), 'w', template=full_bam_fh)
            
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
        fig = self.length_distribution_figure()
        fig.savefig(str(self.fns['lengths_figure']), bbox_inches='tight')
        plt.close(fig)

        for outcome in self.outcomes:
            outcome_fns = self.outcome_fns(outcome)
            
            als = self.get_read_alignments(0, outcome)
            fig = visualize.plot_read(als, self.target_info, parsimonious=True)
            fig.axes[0].set_title('')
            fig.savefig(str(outcome_fns['first_example']), bbox_inches='tight')
            plt.close(fig)
            
            als_iter = (self.get_read_alignments(i, outcome) for i in range(num_examples))
            stacked_im = visualize.make_stacked_Image(als_iter, self.target_info, parsimonious=True)
            stacked_im.save(outcome_fns['combined_figure'])

            lengths = self.outcome_read_lengths(outcome)
            fig = visualize.make_length_plot(self.read_lengths, self.color, lengths)
            fig.savefig(str(outcome_fns['lengths_figure']), bbox_inches='tight')
            plt.close(fig)
                
    def get_read_alignments(self, read_id, outcome=None):
        if outcome is not None:
            bam_fn = self.outcome_fns(outcome)['bam_by_name']
        else:
            bam_fn = self.fns['bam_by_name']
        
        read_groups = sam.grouped_by_name(bam_fn)

        if isinstance(read_id, int):
            try:
                for _ in range(read_id + 1):
                    name, group = next(read_groups)
                return group
            except StopIteration:
                return None
        else:
            name = None
            group = None

            for name, group in read_groups:
                if name == read_id:
                    break

            if name == read_id:
                return group
            else:
                return None

    def make_text_visualizations(self, num_examples=10):
        for outcome in self.outcomes:
            outcome_fns = self.outcome_fns(outcome)
            visualize_structure.visualize_bam_alignments(outcome_fns['bam_by_name'],
                                                         self.target_info.fns['ref_fasta'],
                                                         outcome_fns['text_alignments'],
                                                         num_examples,
                                                        )

    def length_distribution_figure(self, show_ranges=False, x_lims=None):
        ys = self.read_lengths

        if x_lims is None:
            x_lims = (0, len(ys))

        fig, ax = plt.subplots(figsize=(16, 5))

        ax.plot(ys, color=self.color)
        ax.set_ylim(0, 1.01 * max(ys))
        ax.set_xlim(*x_lims)
                           
        if show_ranges:
            for _, (start, end) in self.length_ranges.iterrows():
                ax.axvspan(start, end,
                           gid='length_range_{0:05d}_{1:05d}'.format(start, end),
                           alpha=0.1,
                           facecolor='white',
                           edgecolor='black',
                           zorder=100,
                          )
            
        major = np.arange(0, len(ys), 500)
        minor = [x for x in np.arange(0, len(ys), 100) if x % 500 != 0]
                    
        ax.set_xticks(major)
        ax.set_xticks(minor, minor=True)

        ax.set_ylabel('Number of reads')
        ax.set_xlabel('Length of read')

        return fig

    def span_to_Image(self, start, end, num_examples=5):
        groups = sam.grouped_by_name(self.fns['bam_by_name'])
        filtered = (group for name, group in groups
                    if start <= group[0].query_length <= end)

        sample = utilities.reservoir_sample(filtered, num_examples)
        
        return visualize.make_stacked_Image(sample, self.target_info, parsimonious=True)

    def stitch_read_pairs(self):
        before_R1 = adapters.primers['truseq']['R1']
        before_R2 = adapters.primers['truseq']['R2']
        with self.fns['fastq'].open('w') as fh:
            for R1, R2 in fastq.read_pairs(self.fns['R1'], self.fns['R2']):
                stitched = sw.stitch_read_pair(R1, R2, before_R1, before_R2)
                fh.write(str(stitched))
        
    def process(self):
        if 'R1' in self.fns:
            self.stitch_read_pairs()

        self.call_peaks_in_length_distribution()
        self.generate_alignments()
        self.count_outcomes()
        self.make_outcome_plots(num_examples=5)
        self.make_text_visualizations()

def get_all_experiments(base_dir, conditions=None):
    data_dir = Path(base_dir) / 'data'

    if conditions is None:
        conditions = {}

    def check_conditions(exp):
        for k, v in conditions.items():
            if isinstance(v, (list, tuple, set)):
                if exp.description.get(k) not in v:
                    return False
            else:
                if exp.description.get(k) != v:
                    return False
        return True

    exps = []
    groups = (p.name for p in data_dir.glob('*') if p.is_dir())
    
    for group in groups:
        sample_sheet_fn = data_dir / group / 'sample_sheet.yaml'
        sample_sheet = yaml.load(sample_sheet_fn.read_text())
        for name, description in sample_sheet.items():
            exp = Experiment(base_dir, group, name, description=description)
            exps.append(exp)

    filtered = [exp for exp in exps if check_conditions(exp)]
    return filtered
