import matplotlib
matplotlib.use('Agg', warn=False)

import shutil
import functools
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
from sequencing.utilities import memoized_property

from . import target_info, blast, layout, pooled_layout, jin_layout, visualize, coherence, collapse, svg

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
    def __init__(self, base_dir, group, name, description=None, progress=None):
        self.group = group
        self.name = name

        if progress is None:
            self.progress = utilities.identity
        else:
            self.progress = progress

        base_dir = Path(base_dir)
        self.dir = base_dir / 'results' / group / name
        if not self.dir.is_dir():
            self.dir.mkdir(parents=True)

        self.data_dir = base_dir / 'data' / group

        if description is None:
            sample_sheet_fn = self.data_dir / 'sample_sheet.yaml'
            self.sample_sheet = yaml.load(sample_sheet_fn.read_text())
            if name in self.sample_sheet:
                self.description = self.sample_sheet[name]
            else:
                self.description = self.sample_sheet
        else:
            self.description = description

        self.project = self.description.get('project', 'knockin')
        self.layout_module = layout
        self.max_insertion_length = 20

        # When checking if an Experiment meets filtering conditions, want to be
        # able to just test description.
        self.description['group'] = group
        self.description['name'] = name

        if 'target_info' in self.description:
            self.target_name = self.description['target_info']
        elif 'target_info_prefix' in self.description:
            if self.name == 'unknown':
                self.target_name = self.description['target_info_prefix']
            else:
                self.target_name = '{}_{}'.format(self.description['target_info_prefix'], self.name)

        self.target_info = target_info.TargetInfo(base_dir, self.target_name)

        self.fns = {
            'bam': self.dir / 'alignments.bam',
            'bam_by_name': self.dir / 'alignments.by_name.bam',

            'outcomes_dir': self.dir / 'outcomes',
            'outcome_counts': self.dir / 'outcome_counts.csv',
            'outcome_list': self.dir / 'outcome_list.txt',

            'lengths': self.dir / 'lengths.txt',
            'lengths_figure': self.dir / 'all_lengths.png',
            'length_ranges': self.dir / 'length_ranges.csv',
            'manual_length_ranges': self.dir / 'manual_length_ranges.csv',

            'length_range_figures': self.dir / 'length_ranges',
            'lengths_svg': self.dir / (self.name + '_by_length.html'),
        }

        self.sequencing_primers = 'truseq'

        self.paired_end_read_length = self.description.get('paired_end_read_length', None)

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
        rs = fastq.reads(self.fns['fastqs'], up_to_space=True)
        rs = self.progress(rs)

        return rs

    @property
    def read_pairs(self):
        read_pairs = fastq.read_pairs(self.fns['R1'], self.fns['R2'])
        read_pairs = self.progress(read_pairs)

        return read_pairs

    @property
    def query_names(self):
        for read in self.reads:
            yield read.name

    @memoized_property
    def read_lengths(self):
        return np.loadtxt(self.fns['lengths'], dtype=int)
    
    def count_read_lengths(self):
        lengths = Counter(len(r.seq) for r in self.reads)
        lengths = utilities.counts_to_array(lengths)
        np.savetxt(self.fns['lengths'], lengths, '%d')

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
        if self.paired_end_read_length is not None:
            peaks = []
            already_seen = set()

            for p in pd.Series(self.read_lengths).sort_values(ascending=False).index:
                if p in already_seen:
                    continue
                else:
                    for i in range(p - 10, p + 10):
                        already_seen.add(i)

                    peaks.append(p)

                    if len(peaks) == 10:
                        break

            length_ranges = [(p - 5, p + 5) for p in peaks]
            
        else:
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
                        max_insertion_length=self.max_insertion_length,
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

    def load_outcome_counts(self, key='outcome_counts'):
        if self.fns[key].exists():
            counts = pd.read_csv(self.fns[key],
                                 index_col=(0, 1),
                                 header=None,
                                 squeeze=True,
                                 sep='\t',
                                )
        else:
            counts = None

        return counts

    @memoized_property
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
        alignment_groups = self.progress(alignment_groups)

        outcomes = defaultdict(list)

        with self.fns['outcome_list'].open('w') as fh:
            for name, als in alignment_groups:
                layout = self.layout_module.Layout(als, self.target_info)
                try:
                    category, subcategory, details = layout.categorize()
                except OverflowError:
                    print(self.name, name)
                    raise
                
                outcomes[category, subcategory].append(name)

                fh.write('{0}\t{1}\t{2}\t{3}\n'.format(name, category, subcategory, details))

        bam_fh.close()

        counts = {description: len(names) for description, names in outcomes.items()}
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
            paired_end_read_length=self.paired_end_read_length,
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

    def outcome_alignment_groups(self, outcome):
        bam_fn = self.outcome_fns(outcome)['bam_by_name']
        read_groups = sam.grouped_by_name(bam_fn)
        return read_groups

    def make_text_visualizations(self, num_examples=10):
        for outcome in self.outcomes:
            outcome_fns = self.outcome_fns(outcome)
            visualize_structure.visualize_bam_alignments(outcome_fns['bam_by_name'],
                                                         self.target_info.fns['ref_fasta'],
                                                         outcome_fns['text_alignments'],
                                                         num_examples,
                                                        )

    def length_distribution_figure(self, show_ranges=False, x_lims=None):
        ys = self.read_lengths / sum(self.read_lengths)

        if x_lims is None:
            x_lims = (0, len(ys))

        fig, ax = plt.subplots(figsize=(18, 5))

        ax.plot(ys, color=self.color)
        
        nonzero_xs = ys.nonzero()[0]
        nonzero_ys = ys[nonzero_xs]
        ax.scatter(nonzero_xs, nonzero_ys, s=4, c=self.color)
                           
        if show_ranges:
            #for _, (start, end) in self.length_ranges.iterrows():
            for start in range(501):
                end = start
                ax.axvspan(start - 0.5, end + 0.5,
                           gid='length_range_{0:05d}_{1:05d}'.format(start, end),
                           alpha=0.0,
                           facecolor='white',
                           edgecolor='black',
                           zorder=100,
                          )
            
        major = np.arange(0, len(ys), 50)
        minor = [x for x in np.arange(0, len(ys), 25) if x % 50 != 0]
                    
        ax.set_xticks(major)
        ax.set_xticks(minor, minor=True)

        ax.set_ylabel('Fraction of reads')
        ax.set_xlabel('Length of read')
        
        ax.set_ylim(0, 0.4)
        ax.set_xlim(*x_lims)

        return fig

    def span_to_Image(self, start, end, num_examples=5):
        groups = sam.grouped_by_name(self.fns['bam_by_name'])
        filtered = (group for name, group in groups
                    if start <= group[0].query_length <= end)
        return self.groups_to_Image(filtered, num_examples)

    def groups_to_Image(self, groups, num_examples):
        sample = utilities.reservoir_sample(groups, num_examples)
        
        kwargs = dict(
            parsimonious=True,
            paired_end_read_length=self.paired_end_read_length,
            label_layout=True,
            #process_mappings=self.layout_module.characterize_layout,
        )

        return visualize.make_stacked_Image(sample, self.target_info, **kwargs)

    def generate_individual_length_figures(self):
        groups = sam.grouped_by_name(self.fns['bam_by_name'])
        by_length = defaultdict(list)
        for name, group in groups:
            length = group[0].query_length
            by_length[length].append(group)

        self.fns['length_range_figures'].mkdir(exist_ok=True)

        items = by_length.items()
        items = self.progress(items)

        for length, groups in items:
            im = self.groups_to_Image(groups, 3)
            fn = self.fns['length_range_figures'] / '{}_{}.png'.format(length, length)
            im.save(fn)

    def generate_svg(self):
        html = svg.length_plot_with_popovers(self, standalone=True, x_lims=(150, 505))

        with self.fns['lengths_svg'].open('w') as fh:
            fh.write(html)

    def stitch_read_pairs(self):
        before_R1 = adapters.primers[self.sequencing_primers]['R1']
        before_R2 = adapters.primers[self.sequencing_primers]['R2']

        with self.fns['stitched'].open('w') as fh:
            for R1, R2 in self.read_pairs:
                stitched = sw.stitch_read_pair(R1, R2, before_R1, before_R2)
                fh.write(str(stitched))

    def process(self):
        #if self.paired_end_read_length is not None:
        #    self.stitch_read_pairs()

        #self.count_read_lengths()

        #self.call_peaks_in_length_distribution()
        #self.generate_alignments()
        #self.count_outcomes()
        #self.make_outcome_plots(num_examples=3)
        #self.generate_individual_length_figures()
        self.generate_svg()
        #self.make_text_visualizations()

class JinExperiment(Experiment):
    def __init__(self, base_dir, group, name, description=None, progress=None):
        super().__init__(base_dir, group, name, description, progress)
        self.max_insertion_length = None
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
                
                category, subcategory, details = layout.categorize_no_donor()
                
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
    
class BrittExperiment(Experiment):
    def __init__(self, base_dir, group, name, description=None, progress=None):
        super().__init__(base_dir, group, name, description, progress)
        self.fns.update({
            'supplemental_STAR_prefix': lambda name: self.dir / '{}_alignments_STAR.'.format(name),
            'supplemental_bam': lambda name: self.dir / '{}_alignments.bam'.format(name),
            'supplemental_bam_by_name': lambda name: self.dir / '{}_alignments.by_name.bam'.format(name),

            'combined_bam': self.dir / 'combined.bam',
            'combined_bam_by_name': self.dir / 'combined.by_name.bam',

            'collapsed_UMI_outcomes': self.dir / 'collapsed_UMI_outcomes.txt',
            'cell_outcomes': self.dir / 'cell_outcomes.txt',
            'filtered_cell_outcomes': self.dir / 'filtered_cell_outcomes.txt',
            'filtered_cell_outcome_counts': self.dir / 'filtered_cell_outcome_counts.txt',
        })
        
        self.layout_module = pooled_layout
        self.max_insertion_length = 4

    def generate_supplemental_alignments(self, reads, num_threads=1):
        ''' Use bowtie2 to produce local alignments to CRCh38, filtering out
        spurious alignmnents of polyA or polyG stretches. '''

        bowtie2_index = '/nvme/indices/bowtie2/GRCh38/genome'
        template, mappings = mapping_tools.map_bowtie2(
            bowtie2_index,
            reads=reads,
            local=True,
            score_min='C,60,0',
            memory_mapped_IO=True,
            #report_up_to=1000,
            report_up_to=20,
            #report_all=True,
            yield_mappings=True,
            threads=num_threads,
            custom_binary=True,
        )

        bam_fn = str(self.fns['supplemental_bam'])
        with sam.AlignmentSorter(bam_fn, header=template.header) as sorter:
            homopolymer_length = 10
            homopolymers = {b*homopolymer_length for b in ['A', 'G']}

            for mapping in mappings:
                if mapping.is_unmapped:
                    continue

                score_ratio = mapping.get_tag('AS') / mapping.query_alignment_length
                if score_ratio < 1.75:
                    continue

                al_seq = mapping.query_alignment_sequence
                if mapping.is_reverse:
                    al_seq = utilities.reverse_complement(al_seq)

                contains_hp = any(hp in al_seq for hp in homopolymers)
                if contains_hp:
                    continue
                
                sorter.write(mapping)
                        
        sam.sort_bam(self.fns['supplemental_bam'],
                     self.fns['supplemental_bam_by_name'],
                     by_name=True,
                    )
    
    def combine_alignments(self):
        supplemental_fns = [self.fns['supplemental_bam'](index_name) for index_name in self.supplemental_indices]
        sam.merge_sorted_bam_files([self.fns['bam']] + supplemental_fns,
                                   self.fns['combined_bam'],
                                  )

        supplemental_fns = [self.fns['supplemental_bam_by_name'](index_name) for index_name in self.supplemental_indices]
        sam.merge_sorted_bam_files([self.fns['bam_by_name']] + supplemental_fns,
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

class PooledExperiment(object):
    def __init__(self, base_dir, group, progress=None):
        self.base_dir = base_dir
        self.group = group

        if progress is None:
            progress = utilities.identity

        self.progress = progress

        sample_sheet_fn = base_dir / 'data' / group / 'sample_sheet.yaml'
        sample_sheet = yaml.load(sample_sheet_fn.read_text())

        self.target_name = sample_sheet['target_info_prefix']
        self.target_info = target_info.TargetInfo(base_dir, self.target_name)

        self.fns = {
            'guides': base_dir / 'guides' / 'guides.txt',
            'outcome_counts': base_dir / 'results' / group / 'outcome_counts.npz',
            'total_outcome_counts': base_dir / 'results' / group / 'total_outcome_counts.txt',
            'quantiles': base_dir / 'results' / group / 'quantiles.hdf5',
        }

    @memoized_property
    def guides_df(self):
        all_guides = pd.read_table(self.base_dir / 'guides' / 'guides.txt', index_col='name')
        top_3 = all_guides.query('top_3')
        in_order = top_3.sort_values(['gene', 'short_name', 'rank'])
        return in_order

    @memoized_property
    def guides(self):
        guides = self.guides_df.index.values
        return guides

    @memoized_property
    def non_targeting_guides(self):
        return [g for g in self.guides if 'non-targeting' in g]

    @memoized_property
    def targeting_guides(self):
        return [g for g in self.guides if 'non-targeting' not in g]

    @memoized_property
    def guide_to_short_name(self):
        return self.guides_df['short_name']

    @memoized_property
    def short_name_to_guide(self):
        return utilities.reverse_dictionary(self.guide_to_short_name)

    @memoized_property
    def genes(self):
        return sorted(set(self.guides_df['gene']))

    def gene_guides(self, gene):
        return self.guides_df.query('gene == @gene').index

    def guide_to_gene(self, guide):
        return self.guides_df.loc[guide]['gene']

    def make_outcome_counts(self):
        all_counts = {}

        for guide in self.progress(self.guides):
            exp = SingleGuideExperiment(self.base_dir, self.group, guide)
            if len(exp.outcome_counts) > 0:
                all_counts[guide] = exp.outcome_counts

        all_outcomes = set()

        for guide in all_counts:
            all_outcomes.update(all_counts[guide].index.values)
            
        outcome_order = sorted(all_outcomes)
        outcome_to_index = {outcome: i for i, outcome in enumerate(outcome_order)}

        counts = scipy.sparse.dok_matrix((len(outcome_order), len(self.guides)), dtype=int)

        for g, guide in enumerate(self.progress(self.guides)):
            for outcome, count in all_counts[guide].items():
                o = outcome_to_index[outcome]
                counts[o, g] = count
                
        scipy.sparse.save_npz(self.fns['outcome_counts'], counts.tocoo())

        df = pd.DataFrame(counts.todense(), columns=self.guides, index=pd.MultiIndex.from_tuples(outcome_order))

        df.sum(axis=1).to_csv(self.fns['total_outcome_counts'])

    @memoized_property
    def total_outcome_counts(self):
        return pd.read_csv(self.fns['total_outcome_counts'], header=None, index_col=[0, 1, 2], na_filter=False)

    @memoized_property
    def outcome_counts(self):
        sparse_counts = scipy.sparse.load_npz(self.fns['outcome_counts'])
        df = pd.DataFrame(sparse_counts.todense(),
                          index=self.total_outcome_counts.index,
                          columns=self.guides,
                         )
        df.index.names = ('category', 'subcategory', 'details')

        genomic_insertions = df.xs('genomic insertion').sum()
        df = df.drop('genomic insertion', level=0)
        df.loc['genomic insertion', 'genomic insertion', 'n/a'] = genomic_insertions
        
        donor_insertions = df.xs('donor insertion').sum()
        df = df.drop('donor insertion', level=0)
        df.loc['donor insertion', 'donor insertion', 'n/a'] = donor_insertions

        return df

    @memoized_property
    def non_targeting_outcomes(self):
        guide_outcomes = {}
        for nt_guide in self.non_targeting_guides:
            exp = SingleGuideExperiment(self.base_dir, self.group, nt_guide)
            fn = exp.fns['filtered_cell_outcomes']

            outcomes = [coherence.Pooled_UMI_Outcome.from_line(line) for line in fn.open()]

            for outcome in outcomes:
                if outcome.category == 'genomic insertion':
                    outcome.details = 'n/a'
                
                if outcome.category == 'donor insertion':
                    outcome.details = 'n/a'

            guide_outcomes[nt_guide] = outcomes

        return guide_outcomes

    @memoized_property
    def UMI_counts(self):
        return self.outcome_counts.sum()
    
    @memoized_property
    def all_non_targeting_counts(self):
        return self.outcome_counts[self.non_targeting_guides].sum(axis=1).sort_values(ascending=False)
    
    @memoized_property
    def all_non_targeting_fractions(self):
        return self.all_non_targeting_counts / self.all_non_targeting_counts.sum()

    @memoized_property
    def most_frequent_outcomes(self):
        return self.all_non_targeting_counts.index.values[:200]

    @memoized_property
    def common_non_targeting_fractions(self):
        counts = self.common_counts[self.non_targeting_guides].sum(axis=1)
        return counts / counts.sum()
    
    @memoized_property
    def common_counts(self):
        frequent_counts = self.outcome_counts.loc[self.most_frequent_outcomes] 
        leftover = self.UMI_counts - frequent_counts.sum()
        leftover_row = pd.DataFrame.from_dict({('uncommon', 'uncommon', 'n/a'): leftover}, orient='index')
        everything = pd.concat([frequent_counts, leftover_row])
        return everything

    @memoized_property
    def common_fractions(self):
        return self.common_counts / self.UMI_counts

    @memoized_property
    def fold_changes(self):
        return self.common_fractions.div(self.common_non_targeting_fractions, axis=0)

    @memoized_property
    def log2_fold_changes(self):
        fc = self.fold_changes
        smallest_nonzero = fc[fc > 0].min().min()
        floored = np.maximum(fc, smallest_nonzero)
        return np.log2(floored)

    def rational_outcome_order(self):
        def get_deletion_info(details):
            _, starts, length = pooled_layout.string_to_indels(details)[0]
            return {'num_MH_nts': len(starts) - 1,
                    'start': min(starts),
                    'length': length,
                }

        def has_MH(details):
            info = get_deletion_info(details)
            return info['num_MH_nts'] >= 2 and info['length'] > 1

        conditions = {
            'insertions': lambda c, sc, d: sc == 'insertion',
            'no_MH_deletions': lambda c, sc, d: sc == 'deletion' and not has_MH(d),
            'MH_deletions': lambda c, sc, d: sc == 'deletion' and has_MH(d),
            'donor': lambda c, sc, d: sc == 'donor' or sc == 'other',
            'wt': lambda c, sc, d: sc == 'wild type',
            'uncat': lambda c, sc, d: c == 'uncategorized',
            'genomic': lambda c, sc, d: c == 'genomic insertion',
            'donor insertion': lambda c, sc, d: c == 'donor insertion',
            'uncommon': [('uncommon', 'uncommon', 'n/a')],
        }

        group_order = [
            'uncat',
            'genomic',
            'donor insertion',
            'wt',
            'donor',
            'insertions',
            'no_MH_deletions',
            'MH_deletions',
            'uncommon',
        ]

        donor_order = [
                ('no indel', 'donor', 'ACGAGTTT'),
                ('no indel', 'other', '___AGTTT'),
                ('no indel', 'other', '____GTTT'),
                ('no indel', 'other', '___AGTT_'),
                ('no indel', 'other', '____GTT_'),
                ('no indel', 'other', '____GT__'),
                ('no indel', 'other', '____G___'),
                ('no indel', 'other', 'ACGAGTT_'),
                ('no indel', 'other', 'ACGAG___'),
                ('no indel', 'other', 'ACG_GTTT'),
                ('no indel', 'other', 'ambiguou'),
        ]

        groups = {
            name: [o for o in self.most_frequent_outcomes if condition(*o)] if name != 'uncommon' else condition
            for name, condition in conditions.items()
        }

        groups['donor'] = sorted(groups['donor'], key=donor_order.index)

        ordered = []
        for name in group_order:
            ordered.extend(groups[name])

        sizes = [len(groups[name]) for name in group_order]
        return ordered, sizes
    
class SingleGuideExperiment(BrittExperiment):
    def __init__(self, base_dir, group, name, description=None, progress=None):
        super().__init__(base_dir, group, name, description, progress)

        for which in ['R1', 'R2', 'I1', 'I2']:
            self.fns[which] = self.data_dir / 'by_guide' / '{}_{}.fastq'.format(self.name, which)

        self.fns['collapsed_R2'] = self.dir / '{}_collapsed_R2.fastq'.format(self.name)

        self.guide_target_name = '{}_{}'.format(self.target_name, self.name)
        try:
            self.target_info = target_info.TargetInfo(base_dir, self.guide_target_name)
        except FileNotFoundError:
            self.target_info = target_info.TargetInfo(base_dir, self.target_name)

        self.supplemental_indices = {
            'hg19': '/nvme/indices/refdata-cellranger-hg19-1.2.0/star',
            'bosTau7': '/nvme/indices/bosTau7',
        }

    @property
    def reads(self):
        reads = fastq.reads(self.fns['R2'], up_to_space=True)
        reads = self.progress(reads)

        return reads
    
    @property
    def collapsed_reads(self):
        reads = fastq.reads(self.fns['collapsed_R2'])
        reads = self.progress(reads)

        return reads

    def collapse_UMI_reads(self):
        ''' Takes R2_fn sorted by UMI and collapses reads with the same UMI and
        sufficiently similar sequence.
        '''

        def UMI_key(read):
            return collapse.UMI_Annotation.from_identifier(read.name)['UMI']
        def num_reads_key(read):
            return collapse.collapsed_UMI_Annotation.from_identifier(read.name)['num_reads']

        with self.fns['collapsed_R2'].open('w') as collapsed_fh:
            groups = utilities.group_by(self.reads, UMI_key)
            for UMI, UMI_group in groups:
                clusters = collapse.form_clusters(UMI_group, max_read_length=None, max_hq_mismatches=0)
                clusters = sorted(clusters, key=num_reads_key, reverse=True)

                for i, cluster in enumerate(clusters):
                    annotation = collapse.collapsed_UMI_Annotation.from_identifier(cluster.name)
                    annotation['UMI'] = UMI
                    annotation['cluster_id'] = i

                    cluster.name = str(annotation)

                    collapsed_fh.write(str(cluster))

    def generate_alignments(self):
        bam_fns = []
        bam_by_name_fns = []

        for i, chunk in enumerate(utilities.chunks(self.collapsed_reads, 10000)):
            suffix = '.{:06d}.bam'.format(i)
            bam_fn = self.fns['bam'].with_suffix(suffix)
            bam_by_name_fn = self.fns['bam_by_name'].with_suffix(suffix)

            blast.blast(self.target_info.fns['ref_fasta'],
                        chunk,
                        bam_fn,
                        bam_by_name_fn,
                        max_insertion_length=self.max_insertion_length,
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
            
    def generate_supplemental_alignments(self):
        ''' Use STAR to produce local alignments, post-filtering spurious alignmnents.
        '''

        for index_name, index in self.supplemental_indices.items():
            bam_fn = mapping_tools.map_STAR(self.fns['collapsed_R2'],
                                            index,
                                            self.fns['supplemental_STAR_prefix'](index_name),
                                            sort=False,
                                            mode='permissive',
                                           )

            all_mappings = pysam.AlignmentFile(bam_fn)
            header = all_mappings.header
            new_references = ['{}_{}'.format(index_name, ref) for ref in header.references]
            new_header = pysam.AlignmentHeader.from_references(new_references, header.lengths)
            filtered_fn = str(self.fns['supplemental_bam_by_name'](index_name))

            with pysam.AlignmentFile(filtered_fn, 'wb', header=new_header) as fh:
                for al in all_mappings:
                    if al.query_alignment_length <= 20:
                        continue

                    if al.get_tag('AS') / al.query_alignment_length <= 0.8:
                        continue

                    fh.write(al)

            sam.sort_bam(self.fns['supplemental_bam_by_name'](index_name),
                         self.fns['supplemental_bam'](index_name),
                        )

    def categorize_outcomes(self):
        if self.fns['outcomes_dir'].is_dir():
            shutil.rmtree(str(self.fns['outcomes_dir']))

        self.fns['outcomes_dir'].mkdir()

        bam_fh = pysam.AlignmentFile(str(self.fns['combined_bam_by_name']))
        alignment_groups = sam.grouped_by_name(bam_fh)
        outcomes = defaultdict(list)

        with self.fns['outcome_list'].open('w') as fh:
            for name, als in self.progress(alignment_groups):
                layout = self.layout_module.Layout(als, self.target_info)
                
                try:
                    category, subcategory, details = layout.categorize()
                except:
                    print()
                    print(self.name, name)
                    raise
                
                outcomes[category, subcategory].append(name)

                annotation = collapse.collapsed_UMI_Annotation.from_identifier(name)
                UMI_outcome = coherence.Pooled_UMI_Outcome(annotation['UMI'],
                                                           annotation['cluster_id'],
                                                           annotation['num_reads'],
                                                           category,
                                                           subcategory,
                                                           details,
                                                           name,
                                                          )
                fh.write(str(UMI_outcome) + '\n')

        bam_fh.close()

        # To make plotting easier, for each outcome, make a file listing all of
        # qnames for the outcome and a bam file (sorted by name) with all of the
        # alignments for these qnames.

        qname_to_outcome = {}
        bam_fhs = {}

        full_bam_fh = pysam.AlignmentFile(str(self.fns['combined_bam_by_name']))
        
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

    @memoized_property
    def outcome_counts(self):
        counts = pd.read_table(self.fns['filtered_cell_outcome_counts'],
                               header=None,
                               index_col=[0, 1, 2],
                               squeeze=True,
                               na_filter=False,
                              )
        counts.index.names = ['category', 'subcategory', 'details']
        return counts

    def collapse_UMI_outcomes(self):
        all_collapsed_outcomes, most_abundant_outcomes = coherence.collapse_pooled_UMI_outcomes(self.fns['outcome_list'])
        with self.fns['collapsed_UMI_outcomes'].open('w') as fh:
            for outcome in all_collapsed_outcomes:
                fh.write(str(outcome) + '\n')
        
        with self.fns['cell_outcomes'].open('w') as fh:
            for outcome in most_abundant_outcomes:
                fh.write(str(outcome) + '\n')
        
        counts = Counter()
        with self.fns['filtered_cell_outcomes'].open('w') as fh:
            for outcome in most_abundant_outcomes:
                if outcome.num_reads >= 10:
                    fh.write(str(outcome) + '\n')
                    counts[outcome.category, outcome.subcategory, outcome.details] += 1

        counts = pd.Series(counts).sort_values(ascending=False)
        counts.to_csv(self.fns['filtered_cell_outcome_counts'], sep='\t')

    @memoized_property
    def filtered_cell_outcomes(self):
        df = pd.read_table(self.fns['filtered_cell_outcomes'], header=None, names=coherence.Pooled_UMI_Outcome.columns)
        return df

    def process(self):
        try:
            #self.collapse_UMI_reads()
            self.generate_alignments()
            #self.generate_supplemental_alignments()
            #self.combine_alignments()
            self.categorize_outcomes()
            self.collapse_UMI_outcomes()
            #self.make_outcome_plots(num_examples=3)
        except:
            print(self.name)
            raise

class BrittAmpliconExperiment(BrittExperiment):
    @property
    def reads(self):
        rs = fastq.reads(self.fns['R1'], up_to_space=True)
        if self.progress is not None:
            rs = self.progress(rs)
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

    def count_outcomes(self):
        if self.fns['outcomes_dir'].is_dir():
            shutil.rmtree(str(self.fns['outcomes_dir']))

        self.fns['outcomes_dir'].mkdir()

        bam_fh = pysam.AlignmentFile(str(self.fns['combined_bam_by_name']))
        alignment_groups = sam.grouped_by_name(bam_fh)

        if self.progress is not None:
            alignment_groups = self.progress(alignment_groups)

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

        full_bam_fh = pysam.AlignmentFile(str(self.fns['combined_bam_by_name']))
        
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
        collapsed_outcomes = coherence.collapse_pooled_UMI_outcomes(self.fns['outcome_list'])
        with self.fns['collapsed_UMI_outcomes'].open('w') as fh:
            for outcome in collapsed_outcomes:
                fh.write(str(outcome) + '\n')

    def process(self):
        #self.generate_alignments()
        #self.generate_supplemental_alignments(num_threads=18)
        #self.combine_alignments()
        self.count_outcomes()
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
            elif description.get('experiment_type') == 'single_guide':
                exp_class = SingleGuideExperiment
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

def get_pooled_group(base_dir, group):
    sample_sheet_fn = base_dir / 'data' / group / 'sample_sheet.yaml'
    sample_sheet = yaml.load(sample_sheet_fn.read_text())
    counts_fn = base_dir / 'data' / group / sample_sheet['guide_counts']
    guide_counts = pd.read_table(counts_fn, header=None, index_col=0, squeeze=True)
    return guide_counts

def get_all_pooled_groups(base_dir):
    group_dirs = [p for p in (base_dir / 'data').iterdir() if p.is_dir()]

    pooled_groups = {}

    for group_dir in group_dirs:
        name = group_dir.name

        sample_sheet_fn = group_dir / 'sample_sheet.yaml'
        if sample_sheet_fn.exists():
            sample_sheet = yaml.load(sample_sheet_fn.read_text())
            pooled = sample_sheet.get('pooled', False)
            if pooled:
                pooled_groups[name] = get_pooled_group(base_dir, name)

    return pooled_groups
