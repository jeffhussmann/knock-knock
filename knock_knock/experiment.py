import matplotlib
matplotlib.use('Agg', warn=False)

import shutil
import sys
from pathlib import Path
from itertools import islice, chain
from collections import defaultdict, Counter
from contextlib import ExitStack

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import bokeh.palettes
import pysam
import yaml
import scipy.signal
import ipywidgets

from hits import sam, fastq, utilities, visualize_structure, sw, adapters, mapping_tools, interval
from hits.utilities import memoized_property, group_by

from . import target_info, blast, layout, visualize, read_outcome, svg, table

palette = bokeh.palettes.Category20c_20[1::4]

def extract_color(description):
    color = description.get('color')
    if color is None:
        color = 'grey'

    try:
        num = int(color)
        num = (num - 1) % len(palette)
        color = palette[num]
    except ValueError:
        pass

    return color
        
def ensure_list(possibly_list):
    if isinstance(possibly_list, list):
        definitely_list = possibly_list
    else:
        definitely_list = [possibly_list]
    return definitely_list

class Experiment(object):
    def __init__(self, base_dir, group, name, description=None, progress=None):
        self.group = group
        self.name = name

        if progress is None:
            def ignore_kwargs(x, **kwargs):
                return x
            progress = ignore_kwargs

        def pass_along_kwargs(iterable, **kwargs):
            return progress(iterable, **kwargs)

        self.progress = pass_along_kwargs

        self.base_dir = Path(base_dir)
        self.dir.mkdir(exist_ok=True, parents=True)

        self.data_dir = self.base_dir / 'data' / group

        if description is None:
            self.sample_sheet = load_sample_sheet(self.base_dir, self.group)
            if name in self.sample_sheet:
                self.description = self.sample_sheet[name]
            else:
                self.description = self.sample_sheet
        else:
            self.description = description

        self.project = self.description.get('project', 'knockin')
        self.layout_module = layout
        self.max_insertion_length = 20

        self.sgRNA = self.description.get('sgRNA')
        self.donor = self.description.get('donor')
        self.nonhomologous_donor = self.description.get('nonhomologous_donor')
        self.primer_names = self.description.get('primer_names', ['forward_primer', 'reverse_primer'])
        self.infer_homology_arms = self.description.get('infer_homology_arms', False)

        # When checking if an Experiment meets filtering conditions, want to be
        # able to just test description.
        self.description['group'] = group
        self.description['name'] = name

        self.fns = {
            'dir': self.dir,
            'outcomes_dir': self.dir / 'outcomes',
            'outcome_counts': self.dir / 'outcome_counts.csv',
            'outcome_list': self.dir / 'outcome_list.txt',

            'lengths': self.dir / 'lengths.txt',
            'lengths_figure': self.dir / 'all_lengths.png',
            'outcome_stratified_lengths_figure': self.dir / 'outcome_stratified_lengths.svg',
            'length_ranges': self.dir / 'length_ranges.csv',
            'manual_length_ranges': self.dir / 'manual_length_ranges.csv',

            'length_ranges_dir': self.dir / 'length_ranges',
            'lengths_svg': self.dir / (self.name + '_by_length.html'),
            'outcome_browser': self.dir / 'outcome_browser.html',
        }

        def make_length_range_fig_fn(start, end):
            return self.fns['length_ranges_dir'] / f'{start}_{end}.png'

        self.fns['length_range_figure'] = make_length_range_fig_fn
        
        self.color = extract_color(self.description)
        self.max_qual = 93
        
        index_names = self.description.get('supplemental_indices')
        if index_names is None:
            self.supplemental_index_names = []
        else:
            self.supplemental_index_names = index_names.split(';')

    @memoized_property
    def dir(self):
        return self.base_dir / 'results' / self.group / self.name

    @memoized_property
    def supplemental_indices(self):
        locations = target_info.locate_supplemental_indices(self.base_dir)
        return {name: locations[name] for name in self.supplemental_index_names}

    @memoized_property
    def supplemental_headers(self):
       return {name: sam.header_from_STAR_index(d['STAR']) for name, d in self.supplemental_indices.items()}

    @memoized_property
    def target_info(self):
        return target_info.TargetInfo(self.base_dir,
                                      self.target_name,
                                      donor=self.donor,
                                      nonhomologous_donor=self.nonhomologous_donor,
                                      sgRNA=self.sgRNA,
                                      primer_names=self.primer_names,
                                      supplemental_headers=self.supplemental_headers,
                                      infer_homology_arms=self.infer_homology_arms,
                                     )

    @memoized_property
    def target_name(self):
        return self.description['target_info']

    @memoized_property
    def fns_by_read_type(self):
        fns = defaultdict(dict)

        for read_type in self.read_types:
            if read_type is 'CCS':
                fns['fastq'][read_type] = self.fns['CCS_fastqs']
            else:
                fns['fastq'][read_type] = self.dir / f'{read_type}.fastq'

            fns['primary_bam'][read_type] = self.dir / f'{read_type}_alignments.bam'
            fns['primary_bam_by_name'][read_type] = self.dir / f'{read_type}_alignments.by_name.bam'

            for index_name in self.supplemental_index_names:
                fns['supplemental_STAR_prefix'][read_type, index_name] = self.dir / f'{read_type}_{index_name}_alignments_STAR.'
                fns['supplemental_bam'][read_type, index_name] = self.dir / f'{read_type}_{index_name}_alignments.bam'
                fns['supplemental_bam_by_name'][read_type, index_name] = self.dir / f'{read_type}_{index_name}_alignments.by_name.bam'
                fns['supplemental_bam_temp'][read_type, index_name] = self.dir / f'{read_type}_{index_name}_alignments.temp.bam'

            fns['bam'][read_type] = self.dir / f'{read_type}_combined_alignments.bam'
            fns['bam_by_name'][read_type] = self.dir / f'{read_type}_combined_alignments.by_name.bam'
        
        return fns

    def outcome_fns(self, outcome):
        # To allow outcomes to have arbitrary names without causing file path problems,
        # sanitize the names.
        outcome_string = self.layout_module.outcome_to_sanitized_string(outcome)
        outcome_dir = self.fns['outcomes_dir'] / outcome_string
        fns = {
            'dir': outcome_dir,
            'query_names': outcome_dir / 'qnames.txt',
            'no_overlap_query_names': outcome_dir / 'no_overlap_qnames.txt',
            'special_alignments': outcome_dir / 'special_alignments.bam',
            'filtered_cell_special_alignments': outcome_dir / 'filtered_cell_special_alignments.bam',
            'filtered_cell_bam': outcome_dir / 'filtered_cell_alignments.bam',
            'filtered_cell_bam_by_name': outcome_dir / 'filtered_cell_alignments.by_name.bam',
            'first_example': outcome_dir / 'first_examples.png',
            'combined_figure': outcome_dir / 'combined.png',
            'diagrams_html': outcome_dir / 'diagrams.html',
            'lengths_figure': outcome_dir / 'lengths.png',
            'text_alignments': outcome_dir / 'alignments.txt',
            'length_ranges_dir': outcome_dir / 'length_ranges',
            'lengths_svg': outcome_dir / 'by_length.html',
            'bam_by_name' : {read_type: outcome_dir / f'{read_type}.by_name.bam' for read_type in self.read_types},
        }

        def make_length_range_fig_fn(start, end):
            return fns['length_ranges_dir'] / f'{start}_{end}.png'

        fns['length_range_figure'] = make_length_range_fig_fn

        return fns

    def reads_by_type(self, read_type):
        reads = fastq.reads(self.fns_by_read_type['fastq'][read_type], up_to_space=True)
        return reads
    
    @memoized_property
    def read_lengths(self):
        return np.loadtxt(self.fns['lengths'], dtype=int)
    
    def count_read_lengths(self):
        lengths = sum(self.outcome_stratified_lengths.values())
        np.savetxt(self.fns['lengths'], lengths, '%d')

    def length_ranges(self, outcome=None):
        #path = self.fns['length_ranges']
        #if path.exists():
        #    ranges = pd.read_csv(path, sep='\t', header=None, names=['start', 'end'])
        #else:
        #    ranges = pd.DataFrame(columns=['start', 'end'])
        #return ranges
        interval_length = self.max_relevant_length // 50
        starts = np.arange(0, self.max_relevant_length + interval_length, interval_length)
        if outcome is None:
            lengths = self.read_lengths
        else:
            lengths = self.outcome_stratified_lengths[outcome]

        ranges = []
        for start in starts:
            if sum(lengths[start:start + interval_length]) > 0:
                ranges.append((start, start + interval_length - 1))

        return pd.DataFrame(ranges, columns=['start', 'end'])

    def alignment_groups(self, fn_key='bam_by_name', outcome=None, read_type=None):
        if isinstance(outcome, str):
            # outcome is a single category, so need to chain together all relevant
            # (category, subcategory) pairs.
            pairs = [(c, s) for c, s in self.outcomes if c == outcome]
            pair_groups = [self.alignment_groups(fn_key=fn_key, outcome=pair, read_type=read_type) for pair in pairs]
            return chain.from_iterable(pair_groups)

        else:
            if isinstance(outcome, tuple):
                # outcome is a (category, subcategory) pair
                fn = self.outcome_fns(outcome)['bam_by_name'][read_type]
            else:
                fn = self.fns_by_read_type[fn_key][read_type]

            if fn.exists():
                grouped = sam.grouped_by_name(fn)
            else:
                grouped = []

            return grouped

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

    def generate_alignments(self, read_type=None):
        reads = self.reads_by_type(read_type)
        reads = self.progress(reads, desc='Generating alignments')

        bam_fns = []
        bam_by_name_fns = []

        base_bam_fn = self.fns_by_read_type['primary_bam'][read_type]
        base_bam_by_name_fn = self.fns_by_read_type['primary_bam_by_name'][read_type]

        for i, chunk in enumerate(utilities.chunks(reads, 10000)):
            suffix = f'.{i:06d}.bam'
            bam_fn = base_bam_fn.with_suffix(suffix)
            bam_by_name_fn = base_bam_by_name_fn.with_suffix(suffix)

            blast.blast(self.target_info.fns['ref_fasta'],
                        chunk,
                        bam_fn,
                        bam_by_name_fn,
                        max_insertion_length=self.max_insertion_length,
                    )

            bam_fns.append(bam_fn)
            bam_by_name_fns.append(bam_by_name_fn)

        if len(bam_fns) == 0:
            # There weren't any reads. Make empty bam files.
            header = sam.header_from_fasta(self.target_info.fns['ref_fasta'])
            for fn in [base_bam_fn, base_bam_by_name_fn]:
                with pysam.AlignmentFile(fn, 'wb', header=header) as fh:
                    pass

        else:
            sam.merge_sorted_bam_files(bam_fns, base_bam_fn)
            sam.merge_sorted_bam_files(bam_by_name_fns, base_bam_by_name_fn, by_name=True)

        for fn in bam_fns:
            fn.unlink()
            fn.with_suffix('.bam.bai').unlink()
        
        for fn in bam_by_name_fns:
            fn.unlink()
    
    def generate_supplemental_alignments(self, read_type=None):
        ''' Use STAR to produce local alignments, post-filtering spurious alignmnents.
        '''
        for index_name in self.supplemental_indices:
            fastq_fn = self.fns_by_read_type['fastq'][read_type]
            STAR_prefix = self.fns_by_read_type['supplemental_STAR_prefix'][read_type, index_name]
            index = self.supplemental_indices[index_name]['STAR']

            bam_fn = mapping_tools.map_STAR(fastq_fn,
                                            index,
                                            STAR_prefix,
                                            sort=False,
                                            mode='permissive',
                                           )

            with pysam.AlignmentFile(bam_fn) as all_mappings:
                header = all_mappings.header
                new_references = ['{}_{}'.format(index_name, ref) for ref in header.references]
                new_header = pysam.AlignmentHeader.from_references(new_references, header.lengths)

                by_name_fn = self.fns_by_read_type['supplemental_bam_by_name'][read_type, index_name]
                by_name_sorter = sam.AlignmentSorter(by_name_fn, new_header, by_name=True)

                with by_name_sorter:
                    for al in all_mappings:
                        # To reduce noise, filter out alignments that are too short
                        # or that have too many edits (per aligned nt). Keep this in
                        # mind when interpretting short unexplained gaps in reads.

                        #if al.query_alignment_length <= 20:
                        #    continue

                        #if al.get_tag('AS') / al.query_alignment_length <= 0.8:
                        #    continue

                        by_name_sorter.write(al)

            Path(bam_fn).unlink()

            #sam.sort_bam(by_name_fn,
            #             self.fns_by_read_type['supplemental_bam'][read_type, index_name],
            #            )
    
    def combine_alignments(self, read_type=None):
        for by_name in [True]:
            if by_name:
                suffix = '_by_name'
            else:
                suffix = ''

            bam_key = 'primary_bam' + suffix
            supp_key = 'supplemental_bam' + suffix
            combined_key = 'bam' + suffix

            fns_to_merge = [self.fns_by_read_type[bam_key][read_type]]
            for index_name in self.supplemental_indices:
                fns_to_merge.append(self.fns_by_read_type[supp_key][read_type, index_name])

            sam.merge_sorted_bam_files(fns_to_merge,
                                       self.fns_by_read_type[combined_key][read_type],
                                       by_name=by_name,
                                      )

            for fn in fns_to_merge:
                fn.unlink()

    def load_outcome_counts(self, key='outcome_counts'):
        if self.fns[key].exists() and self.fns[key].stat().st_size > 0:
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
    
    def categorize_outcomes(self, fn_key='bam_by_name', read_type=None):
        if self.fns['outcomes_dir'].is_dir():
            shutil.rmtree(str(self.fns['outcomes_dir']))

        self.fns['outcomes_dir'].mkdir()

        outcomes = defaultdict(list)

        with self.fns['outcome_list'].open('w') as fh:
            alignment_groups = self.alignment_groups(fn_key, read_type=read_type)
            if read_type is None:
                description = 'Categorizing reads'
            else:
                description = f'Categorizing {read_type} reads'

            for name, als in self.progress(alignment_groups, desc=description):
                try:
                    layout = self.layout_module.Layout(als, self.target_info, mode=self.layout_mode)
                    if self.target_info.donor is not None or self.target_info.nonhomologous_donor is not None:
                        category, subcategory, details = layout.categorize()
                    else:
                        category, subcategory, details = layout.categorize_no_donor()
                except:
                    print(self.name, name)
                    raise
                
                outcomes[category, subcategory].append(name)

                if layout.seq is None:
                    length = 0
                else:
                    length = len(layout.seq)

                outcome = read_outcome.Outcome(name, length, category, subcategory, details)
                fh.write(f'{outcome}\n')

        counts = {description: len(names) for description, names in outcomes.items()}
        pd.Series(counts).to_csv(self.fns['outcome_counts'], sep='\t', header=False)

        # To make plotting easier, for each outcome, make a file listing all of
        # qnames for the outcome and a bam file (sorted by name) with all of the
        # alignments for these qnames.

        qname_to_outcome = {}
        bam_fhs = {}

        full_bam_fn = self.fns_by_read_type[fn_key][read_type]

        with pysam.AlignmentFile(full_bam_fn) as full_bam_fh:
        
            for outcome, qnames in outcomes.items():
                outcome_fns = self.outcome_fns(outcome)
                outcome_fns['dir'].mkdir()
                bam_fn = outcome_fns['bam_by_name'][read_type]
                bam_fhs[outcome] = pysam.AlignmentFile(bam_fn, 'wb', template=full_bam_fh)
                
                with outcome_fns['query_names'].open('w') as fh:
                    for qname in qnames:
                        qname_to_outcome[qname] = outcome
                        fh.write(qname + '\n')
            
            for al in full_bam_fh:
                outcome = qname_to_outcome[al.query_name]
                bam_fhs[outcome].write(al)

        for outcome, fh in bam_fhs.items():
            fh.close()

    def make_length_plot(self, outcome=None):
        def plot_nonzero(ax, xs, ys, color, highlight):
            nonzero = ys.nonzero()
            if highlight:
                alpha = 0.95
                markersize = 2
            else:
                alpha = 0.7
                markersize = 0

            ax.plot(xs[nonzero], ys[nonzero], 'o', color=color, markersize=markersize, alpha=alpha)
            ax.plot(xs, ys, '-', color=color, alpha=0.3 * alpha)

        fig, ax = plt.subplots(figsize=(14, 5))

        ys = self.read_lengths
        xs = np.arange(len(ys))

        if outcome is None:
            all_color = self.color
            highlight = True
        else:
            all_color = 'black'
            highlight = False

        plot_nonzero(ax, xs, ys, all_color, highlight=highlight)
        ax.set_ylim(0, max(ys) * 1.05)

        if outcome is not None:
            ys = self.outcome_stratified_lengths[outcome]
            xs = np.arange(len(ys))
            outcome_color = color
            plot_nonzero(ax, xs, ys, outcome_color, highlight=True)

        ax.set_xlabel('Length of read')
        ax.set_ylabel('Number of reads')
        ax.set_xlim(0, len(self.read_lengths) * 1.05)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        return fig
                
    def get_read_alignments(self, read_id, fn_key='bam_by_name', outcome=None, read_type=None):
        # iter() necessary because tqdm objects aren't iterators
        read_groups = iter(self.alignment_groups(fn_key, outcome, read_type))

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

    def get_read_diagram(self, read_id, outcome=None, relevant=True, read_type=None, **kwargs):
        fn_key = 'bam_by_name'

        als = self.get_read_alignments(read_id, fn_key=fn_key, outcome=outcome, read_type=read_type)
        if als is None:
            return None

        if relevant:
            layout = self.layout_module.Layout(als, self.target_info, mode=self.layout_mode)
            layout.categorize()
            to_plot = layout.relevant_alignments
        else:
            to_plot = als

        diagram = visualize.ReadDiagram(to_plot, self.target_info,
                                        features_to_show=self.target_info.features_to_show,
                                        **kwargs)

        return diagram

    def make_text_visualizations(self, num_examples=10):
        for outcome in self.outcomes:
            outcome_fns = self.outcome_fns(outcome)
            visualize_structure.visualize_bam_alignments(outcome_fns['bam_by_name'],
                                                         self.target_info.fns['ref_fasta'],
                                                         outcome_fns['text_alignments'],
                                                         num_examples,
                                                        )

    def length_distribution_figure(self, outcome=None, show_ranges=False, show_title=False):
        all_ys = self.read_lengths / self.total_reads

        fig, ax = plt.subplots(figsize=(12, 6))

        if outcome is None:
            ys_list = [
                (all_ys, self.color, 0.9, 'all reads'),
            ]
            max_y = max(all_ys)

            ys_to_check = all_ys
        else:
            if isinstance(outcome, tuple):
                outcome_lengths = self.outcome_stratified_lengths[outcome]
                color = self.outcome_to_color[outcome]
                label = ': '.join(outcome)
            else:
                outcome_lengths = sum([v for (c, s), v in self.outcome_stratified_lengths.items() if c == outcome])
                color = 'C0' # placeholder
                label = outcome

            outcome_ys = outcome_lengths / self.total_reads

            other_lengths = self.read_lengths - outcome_lengths
            other_ys = other_lengths / self.total_reads

            max_y = max(outcome_ys)

            ys_list = [
                (other_ys, 'black', 0.2, 'all other reads'),
                (outcome_ys, color, 0.9, label),
            ]

            ys_to_check = outcome_ys

        for ys, color, alpha, label in ys_list:
            ax.plot(ys, color=color, alpha=alpha, label=label)
            
            nonzero_xs = ys.nonzero()[0]
            nonzero_ys = ys[nonzero_xs]
            
            if label != 'all other reads':
                ax.scatter(nonzero_xs, nonzero_ys, s=2, c=color, alpha=alpha)
                           
        if show_ranges:
            for _, (start, end) in self.length_ranges.iterrows():
                if sum(ys_to_check[start:end + 1]) > 0:
                    ax.axvspan(start - 0.5, end + 0.5,
                               gid=f'length_range_{start:05d}_{end:05d}',
                               alpha=0.1,
                               facecolor='white',
                               edgecolor='black',
                               zorder=100,
                              )
            
        ax.set_ylabel('Fraction of reads')
        ax.set_xlabel('Amplicon length')
        
        y_lims = (0, max_y * 1.05)
        ax.set_ylim(*y_lims)

        x_max = int(self.max_relevant_length * 1.005)
        ax.set_xlim(0, x_max)
        
        if show_title:
            if outcome is None:
                title = f'{self.group}: {self.name}'
            else:
                category, subcategory = outcome
                title = f'{self.group}: {self.name}\n{category}: {subcategory}'

            ax.set_title(title)
            
        ax.legend(framealpha=0.5)
        
        expected_lengths = {
            'expected\nWT': self.target_info.amplicon_length,
        }
        if self.target_info.clean_HDR_length is not None:
            expected_lengths['expected\nHDR'] = self.target_info.clean_HDR_length
        
        for name, length in expected_lengths.items():
            ax.axvline(length, ymin=0, ymax=1.02, color='black', alpha=0.4, clip_on=False)

            ax.annotate(name,
                        xy=(length, 1.02), xycoords=('data', 'axes fraction'),
                        xytext=(0, 1), textcoords='offset points',
                        ha='center', va='bottom',
                        size=10,
                       )
        
        main_ticks = list(range(0, self.max_relevant_length, self.x_tick_multiple))
        main_tick_labels = [str(x) for x in main_ticks]

        extra_ticks = [self.max_relevant_length]
        extra_tick_labels = [f'$\geq${self.max_relevant_length}']

        if self.length_to_store_unknown is not None:
            extra_ticks.append(self.length_to_store_unknown)
            extra_tick_labels.append('?')

        ax.set_xticks(main_ticks + extra_ticks)
        ax.set_xticklabels(main_tick_labels + extra_tick_labels)
        
        minor = [x for x in np.arange(0, x_max, self.x_tick_multiple // 2) if x % self.x_tick_multiple != 0]
        ax.set_xticks(minor, minor=True)

        ax.set_ylabel('Fraction of reads', size=12)
        ax.set_xlabel('amplicon length', size=12)

        return fig

    def outcome_iter(self):
        fhs = [self.fns[key].open() for key in self.outcome_fn_keys]
        chained = chain.from_iterable(fhs)
        for line in chained:
            outcome = read_outcome.Outcome.from_line(line)
            yield outcome

    @memoized_property
    def outcome_stratified_lengths(self):
        outcome_lengths = defaultdict(Counter)

        outcomes = self.outcome_iter()
        description = 'Counting outcome-specific lengths'
        for outcome in self.progress(outcomes, desc=description):
            outcome_lengths[outcome.category, outcome.subcategory][outcome.length] += 1

        max_length = self.max_relevant_length
        if self.length_to_store_unknown is not None:
            max_length = max(max_length, self.length_to_store_unknown)

        outcome_length_arrays = {}
        for outcome, counts in outcome_lengths.items():
            array = np.zeros(max_length + 1)
            for length, value in counts.items():
                if length == -1:
                    array[self.length_to_store_unknown] = value
                elif length >= self.max_relevant_length:
                    array[self.max_relevant_length] += value
                else:
                    array[length] = value

            outcome_length_arrays[outcome] = array

        return outcome_length_arrays

    @memoized_property
    def total_reads(self):
        return self.read_lengths.sum()

    @memoized_property
    def outcome_highest_points(self):
        ''' Dictionary of {outcome: maximum of that outcome's read length frequency distribution} '''
        highest_points = {}

        for outcome, lengths in self.outcome_stratified_lengths.items():
            highest_points[outcome] = max(lengths / self.total_reads * 100)

        return highest_points

    @memoized_property
    def outcome_to_color(self):
        # To minimize the chance that a color will be used more than once in the same panel in 
        # outcome_stratified_lengths plots, sort color order by highest point.
        # Factored out here so that same colors can be used in svgs.
        color_order = sorted(self.outcome_stratified_lengths, key=self.outcome_highest_points.get, reverse=True)
        return {outcome: f'C{i % 10}' for i, outcome in enumerate(color_order)}

    def plot_outcome_stratified_lengths(self, x_lims=None, min_total_to_label=0.1, zoom_factor=0.1):
        outcome_lengths = self.outcome_stratified_lengths

        if x_lims is None:
            ys = list(outcome_lengths.values())[0]
            x_max = int(len(ys) * 1.005)
            x_lims = (0, x_max)

        ti = self.target_info

        panel_groups = []

        current_max = max(self.outcome_highest_points.values())

        left_after_previous = sorted(self.outcome_highest_points)

        while True:
            group = []
            still_left = []

            for outcome in left_after_previous:
                highest_point = self.outcome_highest_points[outcome]
                if current_max * zoom_factor < highest_point <= current_max:
                    group.append(outcome)
                else:
                    still_left.append(outcome)
                    
            if len(group) > 0:
                panel_groups.append((current_max, group))
                
            if len(still_left) == 0:
                break
                
            current_max = current_max * zoom_factor
            left_after_previous = still_left

        num_panels = len(panel_groups)

        fig, axs = plt.subplots(num_panels, 1, figsize=(14, 6 * num_panels), gridspec_kw=dict(hspace=0.12))

        if num_panels == 1:
            # Want to be able to treat axs as a 1D array.
            axs = [axs]

        ax = axs[0]
        ax.annotate(f'{self.group}: {self.name}',
                    xy=(0.5, 1), xycoords='axes fraction',
                    xytext=(0, 40), textcoords='offset points',
                    ha='center',
                    va='bottom',
                    size=14,
                   )

        y_maxes = []

        listed_order = sorted(outcome_lengths, key=self.layout_module.order)
        high_enough_to_show = []

        non_highlight_color = 'grey'

        for panel_i, (ax, (y_max, group)) in enumerate(zip(axs, panel_groups)):
            for outcome in listed_order:
                lengths = outcome_lengths[outcome]
                ys = lengths / self.total_reads

                sanitized_string = layout.outcome_to_sanitized_string(outcome)

                if outcome in group:
                    gid = f'line_highlighted_{sanitized_string}_{panel_i}'
                    color = self.outcome_to_color[outcome]
                    alpha = 1
                else:
                    color = non_highlight_color
                    # At higher zoom levels, fade the grey lines more to avoid clutter.
                    if panel_i == 0:
                        alpha = 0.6
                        gid = f'line_nonhighlighted_6_{sanitized_string}_{panel_i}'
                    elif panel_i == 1:
                        alpha = 0.3
                        gid = f'line_nonhighlighted_3_{sanitized_string}_{panel_i}'
                    else:
                        alpha = 0.05
                        gid = f'line_nonhighlighted_05_{sanitized_string}_{panel_i}'

                category, subcategory = outcome

                total = ys.sum() * 100
                if total > min_total_to_label:
                    high_enough_to_show.append(outcome)
                    label = f'{ys.sum():6.2%} {category}: {subcategory}'
                else:
                    label = None

                ax.plot(ys * 100, label=label, color=color, alpha=alpha, gid=gid)

                if outcome in group:
                    length_ranges = self.length_ranges(outcome)
                    for _, row in length_ranges.iterrows():
                        ax.axvspan(row.start - 0.5, row.end + 0.5,
                                   gid=f'length_range_{sanitized_string}_{row.start}_{row.end}',
                                   alpha=0.0,
                                   facecolor='white',
                                   edgecolor='black',
                                   zorder=100,
                                  )

            legend = ax.legend(bbox_to_anchor=(1.05, 1),
                               loc='upper left',
                               prop=dict(family='monospace', size=9),
                               framealpha=0.3,
                              )

            for outcome, line in zip(high_enough_to_show, legend.get_lines()):
                if line.get_color() != non_highlight_color:
                    line.set_linewidth(5)
                    sanitized_string = layout.outcome_to_sanitized_string(outcome)
                    line.set_gid(f'outcome_{sanitized_string}')

            expected_lengths = {
                'expected\nWT': ti.amplicon_length,
            }
            if ti.clean_HDR_length is not None:
                expected_lengths['expected\nHDR'] = ti.clean_HDR_length

            ax.set_ylim(0, y_max * 1.05)
            y_maxes.append(y_max)

        for panel_i, ax in enumerate(axs):
            main_ticks = list(range(0, self.max_relevant_length, self.x_tick_multiple))
            main_tick_labels = [str(x) for x in main_ticks]

            extra_ticks = [self.max_relevant_length]
            extra_tick_labels = [f'$\geq${self.max_relevant_length}']

            if self.length_to_store_unknown is not None:
                extra_ticks.append(self.length_to_store_unknown)
                extra_tick_labels.append('?')

            ax.set_xticks(main_ticks + extra_ticks)
            ax.set_xticklabels(main_tick_labels + extra_tick_labels)

            ax.set_xlim(*x_lims)
            ax.set_ylabel('Percentage of reads', size=12)
            ax.set_xlabel('amplicon length', size=12)

            for name, length in expected_lengths.items():
                if panel_i == 0:
                    ax.axvline(length, color='black', alpha=0.2)

                    ax.annotate(name,
                                xy=(length, 1), xycoords=('data', 'axes fraction'),
                                xytext=(0, 5), textcoords='offset points',
                                ha='center', va='bottom',
                                size=10,
                               )

        def draw_inset_guide(fig, top_ax, bottom_ax, bottom_y_max, panel_i):
            params_dict = {
                'top': {
                    'offset': 0.04,
                    'width': 0.007,
                    'transform': top_ax.get_yaxis_transform(),
                    'ax': top_ax,
                    'y': bottom_y_max,
                },
                'bottom': {
                    'offset': 0.01,
                    'width': 0.01,
                    'transform': bottom_ax.transAxes,
                    'ax': bottom_ax,
                    'y': 1,
                },
            }

            for which, params in params_dict.items():
                start = 1 + params['offset']
                end = start + params['width']
                y = params['y']
                transform = params['transform']
                ax = params['ax']

                params['start'] = start
                params['end'] = end

                params['top_corner'] = [end, y]
                params['bottom_corner'] = [end, 0]

                ax.plot([start, end, end, start],
                        [y, y, 0, 0],
                        transform=transform,
                        clip_on=False,
                        color='black',
                        linewidth=3,
                        )

                ax.fill([start, end, end, start],
                        [y, y, 0, 0],
                        transform=transform,
                        clip_on=False,
                        color='white',
                        gid=f'zoom_toggle_{which}_{panel_i}',
                       )

            inverted_fig_tranform = fig.transFigure.inverted().transform    

            for which, top_coords, bottom_coords in (('top', params_dict['top']['top_corner'], params_dict['bottom']['top_corner']),
                                                     ('bottom', params_dict['top']['bottom_corner'], params_dict['bottom']['bottom_corner']),
                                                    ):
                top_in_fig = inverted_fig_tranform(params_dict['top']['transform'].transform((top_coords)))
                bottom_in_fig = inverted_fig_tranform(params_dict['bottom']['transform'].transform((bottom_coords)))

                xs = [top_in_fig[0], bottom_in_fig[0]]
                ys = [top_in_fig[1], bottom_in_fig[1]]
                line = matplotlib.lines.Line2D(xs, ys,
                                               transform=fig.transFigure,
                                               clip_on=False,
                                               linestyle='--',
                                               color='black',
                                               alpha=0.5,
                                               gid=f'zoom_dotted_line_{panel_i}_{which}',
                                              )
                fig.lines.append(line)

        for panel_i, (y_max, top_ax, bottom_ax) in enumerate(zip(y_maxes[1:], axs, axs[1:])):
            draw_inset_guide(fig, top_ax, bottom_ax, y_max, panel_i)

        for ax in axs:
            ax.tick_params(axis='y', which='both', left=True, right=True)

        return fig

    def alignment_groups_to_diagrams(self, alignment_groups, num_examples, relevant=True, label_layout=False):
        subsample = utilities.reservoir_sample(alignment_groups, num_examples)

        if relevant:
            only_relevant = []
            for qname, als in subsample:
                if isinstance(als, dict):
                    l = self.layout_module.NonoverlappingPairLayout(als['R1'], als['R2'], self.target_info)
                else:
                    l = self.layout_module.Layout(als, self.target_info, mode=self.layout_mode)
                l.categorize()
                
                only_relevant.append(l.relevant_alignments)

            subsample = only_relevant
        
        kwargs = dict(
            ref_centric=True,
            label_layout=label_layout,
            read_label='amplicon',
            force_left_aligned=True,
            title='',
            features_to_show=self.target_info.features_to_show,
        )
        
        for als in subsample:
            if isinstance(als, dict):
                d = visualize.ReadDiagram(als['R1'], self.target_info, R2_alignments=als['R2'], **kwargs)
            else:
                d = visualize.ReadDiagram(als, self.target_info, **kwargs)
                
            yield d
            
    def generate_svg(self, outcome=None):
        html = svg.length_plot_with_popovers(self, outcome=outcome, standalone=True, x_lims=(0, 505), inline_images=False)

        if outcome is None:
            fns = self.fns
        else:
            fns = self.outcome_fns(outcome)

        with fns['lengths_svg'].open('w') as fh:
            fh.write(html)

    def generate_all_outcome_length_range_figures(self):
        outcomes = sorted(self.outcome_stratified_lengths)
        for outcome in self.progress(outcomes, desc='Generating outcome-specific length range diagrams'):
            self.generate_length_range_figures(outcome=outcome)

    def generate_figures(self):
        self.generate_all_outcome_length_range_figures()
        self.generate_all_outcome_example_figures()
        svg.decorate_outcome_browser(self)
    
    def generate_outcome_example_figures(self, outcome, num_examples):
        if isinstance(outcome, tuple):
            description = ': '.join(outcome)
        else:
            description = outcome

        al_groups = self.alignment_groups(outcome=outcome)
        diagrams = self.alignment_groups_to_diagrams(al_groups, num_examples=num_examples)
        
        def fig_to_img_tag(fig):
            URI, width, height = table.fig_to_png_URI(fig)
            plt.close(fig)
            tag = f"<img src={URI} height='{height}' width='{width}' class='center'>"
            return tag
        
        outcome_fns = self.outcome_fns(outcome)
        outcome_dir = outcome_fns['dir']
        if not outcome_dir.is_dir():
            outcome_dir.mkdir()

        fn = outcome_fns['diagrams_html']
        with fn.open('w') as fh:
            fh.write(f'''\
<html>
<head>
<title>{description}</title>
<style>
h2 {{
  text-align: center;
}}

p {{
  text-align: center;
  font-family: monospace;
}}

.center {{
  display: block;
  margin-left: auto;
  margin-right: auto;
}}

</style>
</head>
<body>
''')
            fh.write(f'<h2>{self.group}: {self.name}</h1>\n')
            fh.write(f'<h2>{description}</h2>\n')
            
            fig = self.length_distribution_figure(outcome=outcome)
            tag = fig_to_img_tag(fig)
            fh.write(f'{tag}\n<hr>\n')
                
            for i, diagram in enumerate(self.progress(diagrams, desc=description)):
                if i == 0:
                    diagram.fig.savefig(outcome_fns['first_example'], bbox_inches='tight')

                fh.write(f'<p>{diagram.query_name}</p>\n')
                tag = fig_to_img_tag(diagram.fig)
                fh.write(f'{tag}\n')
    
    def generate_all_outcome_example_figures(self, num_examples=25):
        for outcome in self.progress(self.outcomes, desc='Making diagrams for detailed subcategories'):
            self.generate_outcome_example_figures(outcome=outcome, num_examples=num_examples)
        
        categories = sorted(set(c for c, s in self.outcomes))
        for outcome in self.progress(categories, desc='Making diagrams for grouped categories'):
            self.generate_outcome_example_figures(outcome=outcome, num_examples=num_examples)
            
    def explore(self, by_outcome=True, **kwargs):
        return explore(self.base_dir, by_outcome=by_outcome, target=self.target_name, experiment=self, **kwargs)

    def get_read_layout(self, read_id, qname_to_als=None, fn_key='bam_by_name', outcome=None, read_type=None):
        # qname_to_als is to allow caching of many sets of als (e.g. for all
        # of a particular outcome category) to prevent repeated lookup
        if qname_to_als is None:
            als = self.get_read_alignments(read_id, fn_key=fn_key, outcome=outcome, read_type=read_type)
        else:
            als = qname_to_als[read_id]

        layout = self.layout_module.Layout(als, self.target_info, mode=self.layout_mode)

        return layout

class PacbioExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.paired_end_read_length = None
        auto_length = int((self.target_info.amplicon_length * 2.5 // 1000 + 1)) * 1000
        self.max_relevant_length = self.description.get('max_relevant_length', auto_length)
        self.length_to_store_unknown = None

        self.x_tick_multiple = 500

        self.layout_mode = 'pacbio'

        ccs_fastq_fns = ensure_list(self.description['CCS_fastq_fns'])
        self.fns['CCS_fastqs'] = [self.data_dir / name for name in ccs_fastq_fns]

        for fn in self.fns['CCS_fastqs']:
            if not fn.exists():
                #raise ValueError(f'{self.group}: {self.name} specifies non-existent {fn}')
                pass

        self.read_types = ['CCS']

        self.outcome_fn_keys = ['outcome_list']

    def alignment_groups(self, fn_key='bam_by_name', outcome=None, read_type='CCS'):
        groups = super().alignment_groups(fn_key=fn_key, outcome=outcome, read_type=read_type)
        return groups
    
    def get_read_alignments(self, read_id, fn_key='bam_by_name', outcome=None, read_type='CCS'):
        return super().get_read_alignments(read_id, fn_key=fn_key, outcome=outcome, read_type=read_type)

    def generate_supplemental_alignments(self, read_type=None):
        ''' Use minimap2 to produce local alignments.
        '''
        for index_name in self.supplemental_indices:
            # Note: this doesn't support multiple intput fastqs.
            fastq_fn = self.fns_by_read_type['fastq'][read_type][0]
            index = self.supplemental_indices[index_name]['minimap2']
            temp_bam_fn = self.fns_by_read_type['supplemental_bam_temp'][read_type, index_name]

            mapping_tools.map_minimap2(fastq_fn, index, temp_bam_fn)

            all_mappings = pysam.AlignmentFile(temp_bam_fn)
            header = all_mappings.header
            new_references = ['{}_{}'.format(index_name, ref) for ref in header.references]
            new_header = pysam.AlignmentHeader.from_references(new_references, header.lengths)

            by_name_fn = self.fns_by_read_type['supplemental_bam_by_name'][read_type, index_name]
            by_name_sorter = sam.AlignmentSorter(by_name_fn, new_header, by_name=True)

            # Unless I am missing an option, minimap2 omits seq and qual for mapped reads.
            # Add them back.
            with by_name_sorter:
                al_groups = sam.grouped_by_name(all_mappings)
                reads = self.reads_by_type(read_type)
                for (qname, als), read in zip(al_groups, reads):
                    if qname != read.name:
                        raise ValueError('iters out of sync')

                    seq = read.seq
                    seq_rc = utilities.reverse_complement(seq)

                    qual = fastq.decode_sanger_to_array(read.qual)
                    qual_rc = qual[::-1]

                    for al in als:
                        if not al.is_reverse:
                            al.query_sequence = seq
                            al.query_qualities = qual
                        else:
                            al.query_sequence = seq_rc
                            al.query_qualities = qual_rc

                        by_name_sorter.write(al)

            temp_bam_fn.unlink()

    def generate_length_range_figures(self, outcome=None, num_examples=1):
        by_length_range = defaultdict(lambda: utilities.ReservoirSampler(num_examples))
        length_ranges = [interval.Interval(row['start'], row['end']) for _, row in self.length_ranges(outcome).iterrows()]

        fn_key = 'bam_by_name'

        al_groups = self.alignment_groups(outcome=outcome, fn_key=fn_key)
        for name, group in al_groups:
            length = group[0].query_length
            for length_range in length_ranges:
                if length in length_range:
                    by_length_range[length_range.start, length_range.end].add((name, group))

        if outcome is None:
            fns = self.fns
        else:
            fns = self.outcome_fns(outcome)

        fig_dir = fns['length_ranges_dir']
            
        if fig_dir.is_dir():
            shutil.rmtree(str(fig_dir))
        fig_dir.mkdir()

        if outcome is not None:
            description = ': '.join(outcome)
        else:
            description = 'Generating length-specific diagrams'

        items = self.progress(by_length_range.items(), desc=description)

        for (start, end), sampler in items:
            diagrams = self.alignment_groups_to_diagrams(sampler.sample, num_examples=num_examples)
            im = visualize.make_stacked_Image(diagrams, titles='')
            fn = fns['length_range_figure'](start, end)
            im.save(fn)
    
    def generate_svg(self):
        html = svg.length_plot_with_popovers(self, standalone=True)

        with self.fns['lengths_svg'].open('w') as fh:
            fh.write(html)

    def preprocess(self):
        pass
    
    def process(self, stage):
        if stage == 'align':
            self.preprocess()

            for read_type in self.read_types:
                self.generate_alignments(read_type=read_type)
                self.generate_supplemental_alignments(read_type=read_type)
                self.combine_alignments(read_type=read_type)

        elif stage == 'categorize':
            self.categorize_outcomes(read_type='CCS')
            self.count_read_lengths()

        elif stage == 'visualize':
            self.generate_figures()

class IlluminaExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.fns.update({
            'no_overlap_outcome_counts': self.dir / 'no_overlap_outcome_counts.csv',
            'no_overlap_outcome_list': self.dir / 'no_overlap_outcome_list.txt',
        })

        self.sequencing_primers = self.description.get('sequencing_primers', 'truseq')
        self.x_tick_multiple = 100

        self.layout_mode = 'illumina'

        self.max_qual = 41

        self.outcome_fn_keys = ['outcome_list', 'no_overlap_outcome_list']

        for k in ['R1', 'R2', 'I1', 'I2']:
            if k in self.description:
                fastq_fns = ensure_list(self.description[k])
                self.fns[k] = [self.data_dir / name for name in fastq_fns]
        
                for fn in self.fns[k]:
                    if not fn.exists():
                        #raise ValueError(f'{self.group}: {self.name} specifies non-existent {fn}')
                        pass

        self.read_types = [
            'stitched',
            'R1_no_overlap',
            'R2_no_overlap',
        ]

    @memoized_property
    def R1_read_length(self):
        R1, R2 = next(self.read_pairs)
        return len(R1)
    
    @memoized_property
    def R2_read_length(self):
        R1, R2 = next(self.read_pairs)
        return len(R2)
    
    @memoized_property
    def max_relevant_length(self):
        return self.R1_read_length + self.R2_read_length + 100

    @memoized_property
    def length_to_store_unknown(self):
        return int(self.max_relevant_length * 1.05)

    def get_read_alignments(self, read_id, fn_key='bam_by_name', outcome=None, read_type=None):
        if read_type is None:
            if read_id in self.no_overlap_qnames:
                als = self.get_no_overlap_read_alignments(read_id, outcome=outcome)
            else:
                als = super().get_read_alignments(read_id, fn_key=fn_key, outcome=outcome, read_type='stitched')
        else:
            als = super().get_read_alignments(read_id, fn_key=fn_key, outcome=outcome, read_type=read_type)
            
        return als

    def get_no_overlap_read_alignments(self, read_id, outcome=None):
        als = {}

        for which in ['R1', 'R2']:
            als[which] = self.get_read_alignments(read_id, fn_key='bam_by_name', outcome=outcome, read_type=f'{which}_no_overlap')

        return als

    def get_read_diagram(self, qname, outcome=None, relevant=True, **kwargs):
        if qname in self.no_overlap_qnames:
            als = self.get_no_overlap_read_alignments(qname, outcome=outcome)

            if relevant:
                layout = self.layout_module.NonoverlappingPairLayout(als['R1'], als['R2'], self.target_info)
                layout.categorize()
                to_plot = layout.relevant_alignments
            else:
                to_plot = als

            diagram = visualize.ReadDiagram(to_plot['R1'], self.target_info, R2_alignments=to_plot['R2'],
                                            features_to_show=self.target_info.features_to_show,
                                            **kwargs)

        else:
            diagram = super().get_read_diagram(qname, outcome=outcome, relevant=relevant, read_type='stitched', **kwargs)

        return diagram
    
    @property
    def read_pairs(self):
        read_pairs = fastq.read_pairs(self.fns['R1'], self.fns['R2'], up_to_space=True)
        return read_pairs

    @memoized_property
    def no_overlap_qnames(self):
        fn = self.fns_by_read_type['fastq']['R1_no_overlap']
        return {r.name for r in fastq.reads(fn, up_to_space=True)}

    def length_ranges(self, outcome=None):
        if outcome is None:
            lengths = self.read_lengths
        else:
            lengths = self.outcome_stratified_lengths[outcome]

        nonzero, = np.nonzero(lengths)
        ranges = [(i, i) for i in nonzero]
        return pd.DataFrame(ranges, columns=['start', 'end'])
    
    def load_outcome_counts(self):
        stitched = super().load_outcome_counts(key='outcome_counts')
        no_overlap = super().load_outcome_counts(key='no_overlap_outcome_counts')

        if stitched is None and no_overlap is None:
            return None
        elif stitched is not None and no_overlap is None:
            return stitched
        elif stitched is None and no_overlap is not None:
            return no_overlap
        else:
            combined = stitched.add(no_overlap, fill_value=0).astype(int)
            return combined

    def no_overlap_alignment_groups(self, outcome=None):
        R1_read_type = 'R1_no_overlap'
        R2_read_type = 'R2_no_overlap'
        
        if outcome is not None:
            R1_fn_key = 'R1_no_overlap_bam_by_name'
            R2_fn_key = 'R2_no_overlap_bam_by_name'
            
        else:
            R1_fn_key = 'bam_by_name'
            R2_fn_key = 'bam_by_name'

        R1_groups = self.alignment_groups(outcome=outcome, fn_key=R1_fn_key, read_type=R1_read_type)
        R2_groups = self.alignment_groups(outcome=outcome, fn_key=R2_fn_key, read_type=R2_read_type)

        group_pairs = zip(R1_groups, R2_groups)

        for (R1_name, R1_als), (R2_name, R2_als) in group_pairs:
            if R1_name != R2_name:
                raise ValueError(R1_name, R2_name)
            else:
                yield R1_name, {'R1': R1_als, 'R2': R2_als}
    
    def categorize_no_overlap_outcomes(self):
        outcomes = defaultdict(list)

        with self.fns['no_overlap_outcome_list'].open('w') as fh:
            alignment_groups = self.no_overlap_alignment_groups()
            for name, als in self.progress(alignment_groups, desc='Categorizing non-overlapping read pairs'):
                try:
                    pair_layout = layout.NonoverlappingPairLayout(als['R1'], als['R2'], self.target_info)
                    category, subcategory, details = pair_layout.categorize()
                except:
                    print(self.name, name)
                    raise
                
                outcomes[category, subcategory].append(name)

                outcome = read_outcome.Outcome(name, pair_layout.length, category, subcategory, details)
                fh.write(f'{outcome}\n')

        counts = {description: len(names) for description, names in outcomes.items()}
        pd.Series(counts).to_csv(self.fns['no_overlap_outcome_counts'], sep='\t', header=False)

        # To make plotting easier, for each outcome, make a file listing all of
        # qnames for the outcome and a bam file (sorted by name) with all of the
        # alignments for these qnames.

        qname_to_outcome = {}
        bam_fhs = {}

        with ExitStack() as stack:
            full_bam_fns = {which: self.fns_by_read_type['bam_by_name'][f'{which}_no_overlap'] for which in ['R1', 'R2']}
            full_bam_fhs = {which: stack.enter_context(pysam.AlignmentFile(full_bam_fns[which])) for which in ['R1', 'R2']}
        
            for outcome, qnames in outcomes.items():
                outcome_fns = self.outcome_fns(outcome)
                outcome_fns['dir'].mkdir(exist_ok=True)
                for which in ['R1', 'R2']:
                    bam_fn = outcome_fns['bam_by_name'][f'{which}_no_overlap']
                    bam_fhs[outcome, which] = stack.enter_context(pysam.AlignmentFile(bam_fn, 'wb', template=full_bam_fhs[which]))
                
                fh = stack.enter_context(outcome_fns['no_overlap_query_names'].open('w'))
                for qname in qnames:
                    qname_to_outcome[qname] = outcome
                    fh.write(qname + '\n')
            
            for which in ['R1', 'R2']:
                for al in full_bam_fhs[which]:
                    outcome = qname_to_outcome[al.query_name]
                    bam_fhs[outcome, which].write(al)

    def stitch_read_pairs(self):
        before_R1 = adapters.primers[self.sequencing_primers]['R1']
        before_R2 = adapters.primers[self.sequencing_primers]['R2']

        fns = self.fns_by_read_type['fastq']

        with fns['stitched'].open('w') as stitched_fh, \
             fns['R1_no_overlap'].open('w') as R1_fh, \
             fns['R2_no_overlap'].open('w') as R2_fh:

            description = 'Stitching read pairs'
            for R1, R2 in self.progress(self.read_pairs, desc=description):
                stitched = sw.stitch_read_pair(R1, R2, before_R1, before_R2, indel_penalty=-1000)
                if len(stitched) == self.R1_read_length + self.R2_read_length:
                    R1_fh.write(str(R1))
                    R2_fh.write(str(R2))
                else:
                    stitched_fh.write(str(stitched))

    def alignment_groups(self, fn_key=None, outcome=None, read_type=None):
        if fn_key is not None or read_type is not None:
            groups = super().alignment_groups(fn_key=fn_key, outcome=outcome, read_type=read_type)
            for name, als in groups:
                yield name, als
        else:
            stitched_groups = super().alignment_groups(outcome=outcome, read_type='stitched')
            for name, als in stitched_groups:
                yield name, als
                
            no_overlap_groups = self.no_overlap_alignment_groups(outcome=outcome)
            for name, als in no_overlap_groups:
                yield name, als
    
    def generate_length_range_figures(self, outcome=None, num_examples=1):
        def extract_length(als):
            if isinstance(als, dict):
                pair_layout = self.layout_module.NonoverlappingPairLayout(als['R1'], als['R2'], self.target_info)
                pair_layout.categorize()
                length = pair_layout.length
            else:
                length = als[0].query_length
                
            if length == -1:
                converted = self.length_to_store_unknown
            elif length > self.max_relevant_length:
                converted = self.max_relevant_length
            else:
                converted = length
            
            return converted

        by_length = defaultdict(lambda: utilities.ReservoirSampler(num_examples))

        al_groups = self.alignment_groups(outcome=outcome)
        for name, als in al_groups:
            length = extract_length(als)
            by_length[length].add((name, als))
        
        if outcome is None:
            fns = self.fns
        else:
            fns = self.outcome_fns(outcome)

        fig_dir = fns['length_ranges_dir']
            
        if fig_dir.is_dir():
            shutil.rmtree(str(fig_dir))
        fig_dir.mkdir()

        if outcome is not None:
            description = ': '.join(outcome)
        else:
            description = 'Generating length-specific diagrams'

        items = self.progress(by_length.items(), desc=description)

        for length, sampler in items:
            diagrams = self.alignment_groups_to_diagrams(sampler.sample, num_examples=num_examples)
            im = visualize.make_stacked_Image(diagrams, titles='')
            fn = fns['length_range_figure'](length, length)
            im.save(fn)

    def preprocess(self):
        self.stitch_read_pairs()

    def process(self, stage=0):
        if stage == 'align':
            self.preprocess()
            
            for read_type in self.read_types:
                self.generate_alignments(read_type)
                self.generate_supplemental_alignments(read_type)
                self.combine_alignments(read_type)

        elif stage == 'categorize':
            self.categorize_outcomes(read_type='stitched')
            self.categorize_no_overlap_outcomes()

            self.count_read_lengths()
        
        elif stage == 'visualize':
            self.generate_figures()

def explore(base_dir, by_outcome=False, target=None, experiment=None, **kwargs):
    if target is None:
        target_names = sorted([t.name for t in target_info.get_all_targets(base_dir)])
    else:
        target_names = [target]

    default_filename = Path.cwd() / 'figure.png'

    widgets = {
        'target': ipywidgets.Select(options=target_names, value=target_names[0], layout=ipywidgets.Layout(height='200px')),
        'experiment': ipywidgets.Select(options=[], layout=ipywidgets.Layout(height='200px', width='450px')),
        'read_id': ipywidgets.Select(options=[], layout=ipywidgets.Layout(height='200px', width='600px')),
        'outcome': ipywidgets.Select(options=[], continuous_update=False, layout=ipywidgets.Layout(height='200px', width='450px')),
    }

    non_widgets = {
        'file_name': ipywidgets.Text(value=str(default_filename)),
        'save': ipywidgets.Button(description='Save snapshot'),
    }

    toggles = [
        ('parsimonious', False),
        ('relevant', True),
        ('ref_centric', True),
        ('draw_sequence', False),
        ('draw_qualities', False),
        ('draw_mismatches', True),
        ('draw_read_pair', False),
        ('force_left_aligned', False),
        ('split_at_indels', False),
    ]
    for key, default_value in toggles:
        widgets[key] = ipywidgets.ToggleButton(value=kwargs.pop(key, default_value))

    # For some reason, the target widget doesn't get a label without this.
    for k, v in widgets.items():
        v.description = k

    if experiment is None:
        conditions = {}
        exps = get_all_experiments(base_dir)
    else:
        exps = [experiment]

    output = ipywidgets.Output()

    @output.capture()
    def populate_experiments(change):
        target = widgets['target'].value
        previous_value = widgets['experiment'].value
        datasets = sorted([(f'{exp.group}: {exp.name}', exp)
                           for exp in exps
                           if exp.target_info.name == target
                          ])
        widgets['experiment'].options = datasets

        if datasets:
            if previous_value in datasets:
                widgets['experiment'].value = previous_value
                populate_outcomes(None)
            else:
                widgets['experiment'].index = 0
        else:
            widgets['experiment'].value = None

    @output.capture()
    def populate_outcomes(change):
        previous_value = widgets['outcome'].value
        exp = widgets['experiment'].value
        if exp is None:
            return

        outcomes = exp.outcomes
        widgets['outcome'].options = [('_'.join(outcome), outcome) for outcome in outcomes]
        if len(outcomes) > 0:
            if previous_value in outcomes:
                widgets['outcome'].value = previous_value
                populate_read_ids(None)
            else:
                widgets['outcome'].value = widgets['outcome'].options[0][1]
        else:
            widgets['outcome'].value = None

    @output.capture()
    def populate_read_ids(change):
        exp = widgets['experiment'].value

        if exp is None:
            return

        if by_outcome:
            outcome = widgets['outcome'].value
            if outcome is None:
                qnames = []
            else:
                qnames = exp.outcome_query_names(outcome)[:200]
        else:
            qnames = list(islice(exp.query_names, 200))

        widgets['read_id'].options = qnames

        if qnames:
            widgets['read_id'].value = qnames[0]
            widgets['read_id'].index = 0
        else:
            widgets['read_id'].value = None
            
    populate_experiments({'name': 'initial'})
    if by_outcome:
        populate_outcomes({'name': 'initial'})
    populate_read_ids({'name': 'initial'})

    widgets['target'].observe(populate_experiments, names='value')

    if by_outcome:
        widgets['outcome'].observe(populate_read_ids, names='value')
        widgets['experiment'].observe(populate_outcomes, names='value')
    else:
        widgets['experiment'].observe(populate_read_ids, names='value')

    @output.capture(clear_output=True)
    def plot(experiment, read_id, **plot_kwargs):
        exp = experiment

        if exp is None:
            return

        if by_outcome:
            als = exp.get_read_alignments(read_id, outcome=plot_kwargs['outcome'])
        else:
            als = exp.get_read_alignments(read_id)

        if als is None:
            return None

        print(als[0].query_name)
        print(als[0].get_forward_sequence())

        l = exp.layout_module.Layout(als, exp.target_info, mode=exp.layout_mode)
        info = l.categorize()
        
        if widgets['relevant'].value:
            als = l.relevant_alignments

        diagram = visualize.ReadDiagram(als, exp.target_info,
                                        max_qual=exp.max_qual,
                                        **plot_kwargs)
        fig = diagram.fig

        fig.axes[0].set_title(' '.join((l.name,) + info))

        return diagram.fig

    all_kwargs = {**{k: ipywidgets.fixed(v) for k, v in kwargs.items()}, **widgets}

    interactive = ipywidgets.interactive(plot, **all_kwargs)
    interactive.update()

    def make_row(keys):
        return ipywidgets.HBox([widgets[k] if k in widgets else non_widgets[k] for k in keys])

    if by_outcome:
        top_row_keys = ['target', 'experiment', 'outcome', 'read_id']
    else:
        top_row_keys = ['target', 'experiment', 'read_id']

    @output.capture(clear_output=False)
    def save(_):
        fig = interactive.result
        fn = non_widgets['file_name'].value
        fig.savefig(fn, bbox_inches='tight')

    non_widgets['save'].on_click(save)

    layout = ipywidgets.VBox(
        [make_row(top_row_keys),
         make_row([k for k, d in toggles]),
         make_row(['file_name', 'save']),
         interactive.children[-1],
         output,
        ],
    )

    return layout

def load_sample_sheet_from_csv(csv_fn):
    csv_fn = Path(csv_fn)

    # Note: can't include comment='#' because of '#' in hex color specifications.
    df = pd.read_csv(csv_fn, index_col='sample').replace({np.nan: None})
    if not df.index.is_unique:
        print(f'Error parsing sample sheet {csv_fn}')
        print(f'Sample names are not unique:')
        print(df[df.index.duplicated()])
        sys.exit(1)

    sample_sheet = df.to_dict(orient='index')

    for name, d in sample_sheet.items():
        if 'forward_primer' in d:
            d['primer_names'] = [d['forward_primer'], d['reverse_primer']]
            d.pop('forward_primer')
            d.pop('reverse_primer')

    return sample_sheet

def load_sample_sheet(base_dir, group):
    data_dir = Path(base_dir) / 'data'

    sample_sheet_yaml_fn = data_dir / group / 'sample_sheet.yaml'

    if sample_sheet_yaml_fn.exists():
        sample_sheet = yaml.safe_load(sample_sheet_yaml_fn.read_text())
    else:
        sample_sheet_csv_fn = sample_sheet_yaml_fn.with_suffix('.csv')
        if not sample_sheet_csv_fn.exists():
            sample_sheet = None
        else:
            sample_sheet = load_sample_sheet_from_csv(sample_sheet_csv_fn)

    return sample_sheet

def get_all_groups(base_dir):
    data_dir = Path(base_dir) / 'data'
    groups = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    return groups

def get_all_experiments(base_dir, conditions=None, as_dictionary=False, progress=None):
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
    groups = get_all_groups(base_dir)

    if 'group' in conditions:
        groups = (n for n in groups if n in conditions['group'])
    
    for group in groups:
        sample_sheet = load_sample_sheet(base_dir, group)

        if sample_sheet is None:
            print(f'Error: {group} has no sample sheet')
            continue

        for name, description in sample_sheet.items():
            if description.get('platform') == 'illumina':
                exp_class = IlluminaExperiment
            elif description.get('platform') == 'pacbio':
                exp_class = PacbioExperiment
            else:
                exp_class = Experiment
            
            exp = exp_class(base_dir, group, name, description=description, progress=progress)
            exps.append(exp)

    filtered = [exp for exp in exps if check_conditions(exp)]
    if len(filtered) == 0:
        raise ValueError('No experiments met conditions')

    if as_dictionary:
        d = {}
        for exp in filtered:
            d[exp.group, exp.name] = exp
        
        filtered = d

    return filtered
