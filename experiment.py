import matplotlib
matplotlib.use('Agg', warn=False)

import shutil
from pathlib import Path
from itertools import islice
from collections import defaultdict, Counter

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import bokeh.palettes
import pysam
import yaml
import scipy.signal

from sequencing import sam, fastq, utilities, visualize_structure, sw, adapters, mapping_tools

from . import target_info, blast, layout, britt_layout, jin_layout, visualize, coherence, collapse

group_by = utilities.group_by

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

        self.data_dir = base_dir / 'data' / group

        if description is None:
            sample_sheet_fn = self.data_dir / 'sample_sheet.yaml'
            sample_sheet = yaml.load(sample_sheet_fn.read_text())
            self.description = sample_sheet[name]
        else:
            self.description = description

        self.project = self.description.get('project', 'knockin')
        self.layout_module = layout
        self.split_at_large_insertions = True

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
            'outcome_list': self.dir / 'outcome_list.txt',

            'lengths_figure': self.dir / 'all_lengths.png',
            'length_ranges': self.dir / 'length_ranges.csv',
            'manual_length_ranges': self.dir / 'manual_length_ranges.csv',

            'deletion_edges': self.dir / 'deletion_edges.npz',
        }

        self.sequencing_primers = 'truseq'

        def ensure_list(possibly_list):
            if isinstance(possibly_list, list):
                definitely_list = possibly_list
            else:
                definitely_list = [possibly_list]
            return definitely_list

        if 'fastq_fns' in self.description:
            fastq_fns = ensure_list(self.description['fastq_fns'])
            self.fns['fastqs'] = [self.data_dir / name for name in fastq_fns]

            for fn in self.fns['fastqs']:
                if not fn.exists():
                    raise ValueError('{0}: {1} specifies non-existent {2}'.format(group, name, fn))

        else:
            for k in ['R1', 'R2', 'I1', 'I2']:
                if k in self.description:
                    fastq_fns = ensure_list(self.description[k])
                    self.fns[k] = [self.data_dir / name for name in fastq_fns]
            
                    for fn in self.fns[k]:
                        if not fn.exists():
                            raise ValueError('{0}: {1} specifies non-existent {2}'.format(group, name, fn))

            self.fns['stitched'] = self.dir / 'stitched.fastq'
            self.fns['fastqs'] = [self.fns['stitched']]

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

    @property
    def reads(self):
        return fastq.reads(self.fns['fastqs'], up_to_space=True)

    @property
    def query_names(self):
        for read in self.reads:
            yield read.name

    @utilities.memoized_property
    def read_lengths(self):
        #lengths = Counter(len(r.seq) for r in self.reads)
        #lengths = utilities.counts_to_array(lengths)
        lengths = np.zeros(300)
        lengths[290] += 1
        return lengths

    @property
    def length_ranges(self):
        path = self.fns['length_ranges']
        if path.exists():
            ranges = pd.read_csv(path, sep='\t', header=None, names=['start', 'end'])
        else:
            ranges = pd.DataFrame(columns=['start', 'end'])
        return ranges

    @property
    def alignments_by_name(self, fn_key='bam_by_name'):
        fh = pysam.AlignmentFile(str(self.fns[fn_key]))
        return sam.grouped_by_name(fh)

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
        bam_fns = []
        bam_by_name_fns = []

        for i, chunk in enumerate(utilities.chunks(self.reads, 10000)):
            suffix = '.{:06d}.bam'.format(i)
            bam_fn = self.fns['bam'].with_suffix(suffix)
            bam_by_name_fn = self.fns['bam_by_name'].with_suffix(suffix)

            blast.blast(self.target_info.fns['ref_fasta'],
                        chunk,
                        bam_fn,
                        bam_by_name_fn,
                        split_at_large_insertions=self.split_at_large_insertions,
                       )

            bam_fns.append(bam_fn)
            bam_by_name_fns.append(bam_by_name_fn)

        sam.merge_sorted_bam_files(bam_fns, self.fns['bam'])
        sam.merge_sorted_bam_files(bam_by_name_fns, self.fns['bam_by_name'], by_name=True)

        for fn in bam_fns:
            fn.unlink()
            fn.with_suffix('.bam.bai').unlink()
        
        for fn in bam_by_name_fns:
            fn.unlink()

    @property
    def reads(self):
        return fastq.reads(self.fns['fastqs'], up_to_space=True)
        for i, chunk in enumerate(utilities.chunks(self.reads, 10000)):
            suffix = '.{:06d}.bam'.format(i)
            bam_fn = self.fns['bam'].with_suffix(suffix)
            bam_by_name_fn = self.fns['bam_by_name'].with_suffix(suffix)

            blast.blast(self.target_info.fns['ref_fasta'],
                        chunk,
                        bam_fn,
                        bam_by_name_fn,
                        split_at_large_insertions=self.split_at_large_insertions,
                       )

            bam_fns.append(bam_fn)
            bam_by_name_fns.append(bam_by_name_fn)

        sam.merge_sorted_bam_files(bam_fns, self.fns['bam'])
        sam.merge_sorted_bam_files(bam_by_name_fns, self.fns['bam_by_name'], by_name=True)

        for fn in bam_fns:
            fn.unlink()
            fn.with_suffix('.bam.bai').unlink()
        
        for fn in bam_by_name_fns:
            fn.unlink()

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
        qnames = [l.strip() for l in open(str(fns['query_names']))]
        return qnames
    
    def count_outcomes(self, fn_key='bam_by_name'):
        if self.fns['outcomes_dir'].is_dir():
            shutil.rmtree(str(self.fns['outcomes_dir']))

        self.fns['outcomes_dir'].mkdir()

        bam_fh = pysam.AlignmentFile(str(self.fns[fn_key]))
        alignment_groups = sam.grouped_by_name(bam_fh)
        outcomes = defaultdict(list)

        sort_order = {}
        with self.fns['outcome_list'].open('w') as fh:
            for name, als in alignment_groups:
                layout_info = self.layout_module.characterize_layout(als, self.target_info)
                
                outcome = layout_info['outcome']
                outcomes[outcome['description']].append(name)

                sort_order[outcome['description']] = outcome['sort_order']

                category, subcat = outcome['description']
                details = str(layout_info['details'])
                fh.write('{0}\t{1}\t{2}\t{3}\n'.format(name, category, subcat, details))

        bam_fh.close()

        counts = {description: len(names) for description, names in outcomes.items()}
        pd.Series(counts).to_csv(self.fns['outcome_counts'], sep='\t')

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

        full_bam_fh.close()
        for outcome, fh in bam_fhs.items():
            fh.close()

    def make_outcome_plots(self, num_examples=10):
        fig = self.length_distribution_figure()
        fig.savefig(str(self.fns['lengths_figure']), bbox_inches='tight')
        plt.close(fig)

        kwargs = dict(
            parsimonious=False,
            #process_mappings=self.layout_module.characterize_layout,
        )

        for outcome in self.outcomes:
            outcome_fns = self.outcome_fns(outcome)
            
            als = self.get_read_alignments(0, outcome=outcome)
            fig = visualize.plot_read(als, self.target_info, **kwargs)
            fig.axes[0].set_title('')
            fig.savefig(str(outcome_fns['first_example']), bbox_inches='tight')
            plt.close(fig)
            
            als_iter = (self.get_read_alignments(i, outcome=outcome) for i in range(num_examples))
            stacked_im = visualize.make_stacked_Image(als_iter, self.target_info, **kwargs)
            stacked_im.save(outcome_fns['combined_figure'])

            lengths = self.outcome_read_lengths(outcome)
            fig = visualize.make_length_plot(self.read_lengths, self.color, lengths)
            fig.savefig(str(outcome_fns['lengths_figure']), bbox_inches='tight')
            plt.close(fig)
                
    def get_read_alignments(self, read_id, fn_key='bam_by_name', outcome=None):
        if outcome is not None:
            bam_fn = self.outcome_fns(outcome)[fn_key]
        else:
            bam_fn = self.fns[fn_key]
        
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
        before_R1 = adapters.primers[self.sequencing_primers]['R1']
        before_R2 = adapters.primers[self.sequencing_primers]['R2']
        with self.fns['stitched'].open('w') as fh:
            read_pairs = fastq.read_pairs(self.fns['R1'], self.fns['R2'])
            for R1, R2 in read_pairs:
                stitched = sw.stitch_read_pair(R1, R2, before_R1, before_R2)
                fh.write(str(stitched))
        
    def process(self):
        #if 'R1' in self.fns:
        #    self.stitch_read_pairs()

        #self.call_peaks_in_length_distribution()
        #self.generate_alignments()
        self.count_outcomes()
        #self.make_outcome_plots(num_examples=3)
        #self.make_text_visualizations()

        print('finished with {0}: {1}'.format(self.group, self.name))
        
class JinExperiment(Experiment):
    def __init__(self, base_dir, group, name, description=None):
        super().__init__(base_dir, group, name, description)
        self.layout_module = jin_layout
        self.split_at_large_insertions = False
        self.sequencing_primers = 'nextera'
    
    def count_outcomes(self, fn_key='bam_by_name'):
        if self.fns['outcomes_dir'].is_dir():
            shutil.rmtree(str(self.fns['outcomes_dir']))

        self.fns['outcomes_dir'].mkdir()

        bam_fh = pysam.AlignmentFile(str(self.fns[fn_key]))
        alignment_groups = sam.grouped_by_name(bam_fh)
        outcomes = defaultdict(list)

        with self.fns['outcome_list'].open('w') as fh:
            for name, als in alignment_groups:
                layout = self.layout_module.Layout(als, self.target_info)
                
                category, subcategory, details = layout.categorize()
                
                outcomes[category, subcategory].append(name)

                fh.write('{0}\t{1}\t{2}\t{3}\n'.format(name, category, subcategory, details))

        bam_fh.close()

        counts = {outcome: len(names) for outcome, names in outcomes.items()}
        pd.Series(counts).to_csv(self.fns['outcome_counts'], sep='\t')

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

        full_bam_fh.close()
        for outcome, fh in bam_fhs.items():
            fh.close()
    
    def make_outcome_plots(self, num_examples=10):
        fig = self.length_distribution_figure()
        fig.savefig(str(self.fns['lengths_figure']), bbox_inches='tight')
        plt.close(fig)

        kwargs = dict(
            parsimonious=False,
            paired=300,
        )

        for outcome in self.outcomes:
            outcome_fns = self.outcome_fns(outcome)
            
            als = self.get_read_alignments(0, outcome=outcome)
            fig = visualize.plot_read(als, self.target_info, **kwargs)
            fig.axes[0].set_title('')
            fig.savefig(str(outcome_fns['first_example']), bbox_inches='tight')
            plt.close(fig)
            
            als_iter = (self.get_read_alignments(i, outcome=outcome) for i in range(num_examples))
            stacked_im = visualize.make_stacked_Image(als_iter, self.target_info, **kwargs)
            stacked_im.save(outcome_fns['combined_figure'])

            lengths = self.outcome_read_lengths(outcome)
            fig = visualize.make_length_plot(self.read_lengths, self.color, lengths)
            fig.savefig(str(outcome_fns['lengths_figure']), bbox_inches='tight')
            plt.close(fig)

class BrittExperiment(Experiment):
    def __init__(self, base_dir, group, name, description=None):
        super().__init__(base_dir, group, name, description)
        self.fns.update({
            'supplemental_bam': self.dir / 'supplemental_alignments.bam',
            'supplemental_bam_by_name': self.dir / 'supplemental_alignments.by_name.bam',
            'combined_bam': self.dir / 'combined.bam',
            'combined_bam_by_name': self.dir / 'combined.by_name.bam',

            'collapsed_UMI_outcomes': self.dir / 'collapsed_UMI_outcomes.txt',
            'cell_outcomes': self.dir / 'cell_outcomes.txt',
            'coherent_cell_outcomes': self.dir / 'coherent_cell_outcomes.txt',
        })
        
        self.layout_module = britt_layout
        self.split_at_large_insertions = True

    def generate_supplemental_alignments(self, num_threads=1):
        ''' Use bowtie2 to produce local alignments to CRCh38, filtering out
        spurious alignmnents of polyA or polyG stretches. '''

        bowtie2_index = '/nvme/indices/bowtie2/GRCh38/genome'
        template, mappings = mapping_tools.map_bowtie2(
            bowtie2_index,
            reads=self.reads,
            local=True,
            score_min='C,60,0',
            memory_mapped_IO=True,
            report_up_to=10,
            yield_mappings=True,
            threads=num_threads,
            custom_binary=True,
        )

        bam_fn = str(self.fns['supplemental_bam'])
        with sam.AlignmentSorter(bam_fn, header=template.header) as sorter:
            homopolymer_length = 10
            homopolymers = {b*homopolymer_length for b in ['A', 'G']}

            for mapping in mappings:
                al_seq = mapping.query_alignment_sequence
                if mapping.is_reverse:
                    al_seq = utilities.reverse_complement(al_seq)

                contains_hp = any(hp in al_seq for hp in homopolymers)
                if not contains_hp and not mapping.is_unmapped:
                    sorter.write(mapping)
                        
        sam.sort_bam(self.fns['supplemental_bam'],
                     self.fns['supplemental_bam_by_name'],
                     by_name=True,
                    )

    def combine_alignments(self):
        sam.merge_sorted_bam_files([self.fns['bam'], self.fns['supplemental_bam']],
                                   self.fns['combined_bam'],
                                  )

        sam.merge_sorted_bam_files([self.fns['bam_by_name'], self.fns['supplemental_bam_by_name']],
                                   self.fns['combined_bam_by_name'],
                                   by_name=True,
                                  )
        
    def count_outcomes(self, fn_key='combined_bam_by_name'):
        if self.fns['outcomes_dir'].is_dir():
            shutil.rmtree(str(self.fns['outcomes_dir']))

        self.fns['outcomes_dir'].mkdir()

        bam_fh = pysam.AlignmentFile(str(self.fns[fn_key]))
        alignment_groups = sam.grouped_by_name(bam_fh)
        outcomes = defaultdict(list)

        with self.fns['outcome_list'].open('w') as fh:
            for name, als in alignment_groups:
                layout = self.layout_module.Layout(als, self.target_info)
                
                category, subcategory, details = layout.categorize()
                
                outcomes[category, subcategory].append(name)

                annotation = collapse.cluster_Annotation.from_identifier(name)
                UMI_outcome = coherence.UMI_Outcome(annotation['cell_BC'],
                                                    annotation['UMI'],
                                                    annotation['num_reads'],
                                                    category,
                                                    subcategory,
                                                    details,
                                                    name,
                                                   )

                fh.write(str(UMI_outcome) + '\n')

        bam_fh.close()

        counts = {outcome: len(names) for outcome, names in outcomes.items()}
        pd.Series(counts).to_csv(self.fns['outcome_counts'], sep='\t')

        # To make plotting easier, for each outcome, make a file listing all of
        # qnames for the outcome and a bam file (sorted by name) with all of the
        # alignments for these qnames.

        qname_to_outcome = {}
        bam_fhs = {}

        full_bam_fh = pysam.AlignmentFile(str(self.fns[fn_key]))
        
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

        full_bam_fh.close()
        for outcome, fh in bam_fhs.items():
            fh.close()

    def collapse_UMI_outcomes(self):
        most_abundant_outcomes = coherence.collapse_UMI_outcomes(self.fns['outcome_list'])
        with self.fns['collapsed_UMI_outcomes'].open('w') as fh:
            for outcome in most_abundant_outcomes:
                fh.write(str(outcome) + '\n')

    def collapse_cell_outcomes(self):
        cell_outcomes = coherence.collapse_cell_outcomes(self.fns['collapsed_UMI_outcomes'])
        with self.fns['cell_outcomes'].open('w') as fh:
            for outcome in cell_outcomes:
                fh.write(str(outcome) + '\n')

    def filter_coherent_cells(self):
        good_cells = coherence.filter_coherent_cells(self.fns['cell_outcomes'])
        good_cells.to_csv(self.fns['coherent_cell_outcomes'], sep='\t')

    def process(self):
        #self.generate_alignments()
        #self.generate_supplemental_alignments()
        #self.combine_alignments()
        self.count_outcomes(fn_key='combined_bam_by_name')
        self.collapse_UMI_outcomes()
        self.collapse_cell_outcomes()
        self.filter_coherent_cells()
        #self.make_outcome_plots(num_examples=3)
        #self.make_text_visualizations()

        print('finished with {0}: {1}'.format(self.group, self.name))

class BrittPooledExperiment(BrittExperiment):
    @property
    def reads(self):
        rs = fastq.reads(self.fns['R2'], up_to_space=True)
        return rs
    
    def count_outcomes(self, fn_key='bam_by_name'):
        if self.fns['outcomes_dir'].is_dir():
            shutil.rmtree(str(self.fns['outcomes_dir']))

        self.fns['outcomes_dir'].mkdir()

        bam_fh = pysam.AlignmentFile(str(self.fns[fn_key]))
        alignment_groups = sam.grouped_by_name(bam_fh)
        outcomes = defaultdict(list)

        with self.fns['outcome_list'].open('w') as fh:
            for name, als in alignment_groups:
                layout = self.layout_module.Layout(als, self.target_info)
                
                category, subcategory, details = layout.categorize()
                
                outcomes[category, subcategory].append(name)

                annotation = collapse.collapsed_UMI_Annotation.from_identifier(name)
                UMI_outcome = coherence.Pooled_UMI_Outcome(annotation['UMI'],
                                                           annotation['cluster_id'],
                                                           annotation['num_reads'],
                                                           category,
                                                           subcategory,
                                                           details,
                                                          )
                fh.write(str(UMI_outcome) + '\n')

        bam_fh.close()

        counts = {outcome: len(names) for outcome, names in outcomes.items()}
        pd.Series(counts).to_csv(self.fns['outcome_counts'], sep='\t')

        # To make plotting easier, for each outcome, make a file listing all of
        # qnames for the outcome and a bam file (sorted by name) with all of the
        # alignments for these qnames.

        qname_to_outcome = {}
        bam_fhs = {}

        full_bam_fh = pysam.AlignmentFile(str(self.fns[fn_key]))
        
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

        full_bam_fh.close()
        for outcome, fh in bam_fhs.items():
            fh.close()
    
    def collapse_UMI_outcomes(self):
        most_abundant_outcomes = coherence.collapse_pooled_UMI_outcomes(self.fns['outcome_list'])
        with self.fns['collapsed_UMI_outcomes'].open('w') as fh:
            for outcome in most_abundant_outcomes:
                fh.write(str(outcome) + '\n')

    def process(self):
        #self.generate_alignments()
        #self.generate_supplemental_alignments()
        #self.combine_alignments()
        #self.count_outcomes(fn_key='combined_bam_by_name')
        self.collapse_UMI_outcomes()
        #self.make_outcome_plots(num_examples=3)

class BrittAmpliconExperiment(BrittExperiment):
    @property
    def reads(self):
        rs = fastq.reads(self.fns['R1'], up_to_space=True)
        return rs
    
    def generate_alignments(self):
        mapping_tools.map_bowtie2(
            self.target_info.fns['bowtie2_index'],
            reads=self.reads,
            output_file_name=self.fns['bam'],
            bam_output=True,
            local=True,
            report_all=True,
            error_file_name='/home/jah/projects/britt/bowtie2_error.txt',
            custom_binary=True,
            threads=18,
        )

        sam.sort_bam(self.fns['bam'], self.fns['bam_by_name'], by_name=True)

    def count_outcomes(self, fn_key='bam_by_name'):
        if self.fns['outcomes_dir'].is_dir():
            shutil.rmtree(str(self.fns['outcomes_dir']))

        self.fns['outcomes_dir'].mkdir()

        bam_fh = pysam.AlignmentFile(str(self.fns[fn_key]))
        alignment_groups = sam.grouped_by_name(bam_fh)
        outcomes = defaultdict(list)

        with self.fns['outcome_list'].open('w') as fh:
            for name, als in alignment_groups:
                layout = self.layout_module.Layout(als, self.target_info)
                
                category, subcategory, details = layout.categorize()
                
                outcomes[category, subcategory].append(name)

                fh.write('{0}\t{1}\t{2}\t{3}\n'.format(name, category, subcategory, details))

        bam_fh.close()

        counts = {outcome: len(names) for outcome, names in outcomes.items()}
        pd.Series(counts).to_csv(self.fns['outcome_counts'], sep='\t')

        # To make plotting easier, for each outcome, make a file listing all of
        # qnames for the outcome and a bam file (sorted by name) with all of the
        # alignments for these qnames.

        qname_to_outcome = {}
        bam_fhs = {}

        full_bam_fh = pysam.AlignmentFile(str(self.fns[fn_key]))
        
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

        full_bam_fh.close()
        for outcome, fh in bam_fhs.items():
            fh.close()
    
    def collapse_UMI_outcomes(self):
        most_abundant_outcomes = coherence.collapse_pooled_UMI_outcomes(self.fns['outcome_list'])
        with self.fns['collapsed_UMI_outcomes'].open('w') as fh:
            for outcome in most_abundant_outcomes:
                fh.write(str(outcome) + '\n')

    def process(self):
        #self.generate_alignments()
        #self.generate_supplemental_alignments()
        #self.combine_alignments()
        self.count_outcomes(fn_key='combined_bam_by_name')
        #self.collapse_UMI_outcomes()
        #self.make_outcome_plots(num_examples=3)

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
            if description.get('experiment_type') == 'britt':
                exp_class = BrittExperiment
            elif description.get('experiment_type') == 'britt_pooled':
                exp_class = BrittPooledExperiment
            elif description.get('experiment_type') == 'britt_amplicon':
                exp_class = BrittAmpliconExperiment
            elif description.get('experiment_type') == 'jin':
                exp_class = JinExperiment
            else:
                exp_class = Experiment
            
            exp = exp_class(base_dir, group, name, description=description)
            exps.append(exp)

    filtered = [exp for exp in exps if check_conditions(exp)]
    if len(filtered) == 0:
        raise ValueError('No experiments met conditions')

    return filtered
