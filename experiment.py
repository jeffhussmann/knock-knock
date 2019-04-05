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
import ipywidgets

from sequencing import sam, fastq, utilities, visualize_structure, sw, adapters, mapping_tools
from sequencing.utilities import memoized_property

from . import target_info, blast, layout, visualize, coherence, collapse, svg

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
            self.progress = utilities.identity
        else:
            self.progress = progress

        self.base_dir = Path(base_dir)
        self.dir.mkdir(exist_ok=True, parents=True)

        self.data_dir = self.base_dir / 'data' / group

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

        self.sgRNA = self.description.get('sgRNA')
        self.donor = self.description.get('donor')
        self.primer_names = self.description.get('primer_names')

        # When checking if an Experiment meets filtering conditions, want to be
        # able to just test description.
        self.description['group'] = group
        self.description['name'] = name

        self.fns = {
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
        
        self.color = extract_color(self.description)
        self.max_qual = 93
        
        self.supplemental_indices = {
            'hg19': '/nvme/indices/refdata-cellranger-hg19-1.2.0/star',
            'bosTau7': '/nvme/indices/bosTau7',
        }
        
        self.target_info = target_info.TargetInfo(self.base_dir,
                                                  self.target_name,
                                                  donor=self.donor,
                                                  sgRNA=self.sgRNA,
                                                  primer_names=self.primer_names,
                                                 )

    @memoized_property
    def dir(self):
        return self.base_dir / 'results' / self.group / self.name

    @memoized_property
    def target_name(self):
        return self.description['target_info']

    @memoized_property
    def fns_by_read_type(self):
        fns = {
            'fastq': {},
            'bam' : {},
            'bam_by_name': {},

            'supplemental_STAR_prefix': {},
            'supplemental_bam': {},
            'supplemental_bam_by_name': {},

            'combined_bam': {},
            'combined_bam_by_name': {},
        }

        for read_type in self.read_types:
            fns['fastq'][read_type] = self.dir / f'{read_type}.fastq'
            fns['bam'][read_type] = self.dir / f'{read_type}_alignments.bam'
            fns['bam_by_name'][read_type] = self.dir / f'{read_type}_alignments.by_name.bam'

            for index_name in self.supplemental_indices:
                fns['supplemental_STAR_prefix'][read_type, index_name] = self.dir / f'{read_type}_{index_name}_alignments_STAR.'
                fns['supplemental_bam'][read_type, index_name] = self.dir / f'{read_type}_{index_name}_alignments.bam'
                fns['supplemental_bam_by_name'][read_type, index_name] = self.dir / f'{read_type}_{index_name}_alignments.by_name.bam'

            fns['combined_bam'][read_type] = self.dir / f'{read_type}_combined_alignments.bam'
            fns['combined_bam_by_name'][read_type] = self.dir / f'{read_type}_combined_alignments.by_name.bam'
        
        return fns

    def outcome_fns(self, outcome):
        outcome_string = '_'.join(map(str, outcome))
        outcome_dir = self.fns['outcomes_dir'] / outcome_string
        fns = {
            'dir': outcome_dir,
            'query_names': outcome_dir / 'qnames.txt',
            'bam_by_name': outcome_dir / 'alignments.by_name.bam',
            'special_alignments': outcome_dir / 'special_alignments.bam',
            'filtered_cell_special_alignments': outcome_dir / 'filtered_cell_special_alignments.bam',
            'filtered_cell_bam': outcome_dir / 'filtered_cell_alignments.bam',
            'filtered_cell_bam_by_name': outcome_dir / 'filtered_cell_alignments.by_name.bam',
            'first_example': outcome_dir / 'first_examples.png',
            'combined_figure': outcome_dir / 'combined.png',
            'lengths_figure': outcome_dir / 'lengths.png',
            'text_alignments': outcome_dir / 'alignments.txt',
        }
        return fns

    @property
    def reads(self):
        reads = fastq.reads(self.fns['fastqs'], up_to_space=True)
        return self.progress(reads)
    
    def reads_by_type(self, read_type):
        reads = fastq.reads(self.fns_by_read_type['fastq'][read_type], up_to_space=True)
        return self.progress(reads)
    
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

    def alignment_groups(self, fn_key='bam_by_name', outcome=None, read_type=None):
        if outcome is not None:
            fn = self.outcome_fns(outcome)[fn_key]
        elif read_type is not None:
            fn = self.fns_by_read_type[fn_key][read_type]
        else:
            raise ValueError

        grouped = sam.grouped_by_name(fn)

        return self.progress(grouped)

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
        lengths = Counter()
        for _, group in self.alignment_groups(outcome=outcome):
            lengths[group[0].query_length] += 1

        lengths = utilities.counts_to_array(lengths)
        return lengths

    def generate_alignments(self, read_type):
        reads = self.reads_by_type(read_type)

        bam_fns = []
        bam_by_name_fns = []

        base_bam_fn = self.fns_by_read_type['bam'][read_type]
        base_bam_by_name_fn = self.fns_by_read_type['bam_by_name'][read_type]

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
            for fn in [base_bam_fh, base_bam_by_name_fn]:
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
    
    def generate_supplemental_alignments(self, read_type):
        ''' Use STAR to produce local alignments, post-filtering spurious alignmnents.
        '''
        for index_name, index in self.supplemental_indices.items():
            fastq_fn = self.fns_by_read_type['fastq'][read_type]
            STAR_prefix = self.fns_by_read_type['supplemental_STAR_prefix'][read_type, index_name]

            bam_fn = mapping_tools.map_STAR(fastq_fn,
                                            index,
                                            STAR_prefix,
                                            sort=False,
                                            mode='permissive',
                                           )

            all_mappings = pysam.AlignmentFile(bam_fn)
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

                    if al.query_alignment_length <= 20:
                        continue

                    if al.get_tag('AS') / al.query_alignment_length <= 0.8:
                        continue

                    by_name_sorter.write(al)

            sam.sort_bam(by_name_fn,
                         self.fns_by_read_type['supplemental_bam'][read_type, index_name],
                        )
    
    def combine_alignments(self, read_type):
        for by_name in [False, True]:
            if by_name:
                suffix = '_by_name'
            else:
                suffix = ''

            bam_key = 'bam' + suffix
            supp_key = 'supplemental_bam' + suffix
            combined_key = 'combined_bam' + suffix

            fns_to_merge = [self.fns_by_read_type[bam_key][read_type]]
            for index_name in self.supplemental_indices:
                fns_to_merge.append(self.fns_by_read_type[supp_key][read_type, index_name])

            sam.merge_sorted_bam_files(fns_to_merge,
                                       self.fns_by_read_type[combined_key][read_type],
                                       by_name=by_name,
                                      )

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
    
    def categorize_outcomes(self, fn_key='combined_bam_by_name', read_type=None):
        if self.fns['outcomes_dir'].is_dir():
            shutil.rmtree(str(self.fns['outcomes_dir']))

        self.fns['outcomes_dir'].mkdir()

        outcomes = defaultdict(list)

        with self.fns['outcome_list'].open('w') as fh:
            for name, als in self.alignment_groups(fn_key, read_type=read_type):
                layout = self.layout_module.Layout(als, self.target_info)

                try:
                    if self.target_info.donor is not None:
                        category, subcategory, details = layout.categorize()
                    else:
                        category, subcategory, details = layout.categorize_no_donor()
                except:
                    print(self.name, name)
                    raise
                
                outcomes[category, subcategory].append(name)

                fh.write(f'{name}\t{category}\t{subcategory}\t{details}\n')

        counts = {description: len(names) for description, names in outcomes.items()}
        pd.Series(counts).to_csv(self.fns['outcome_counts'], sep='\t')

        # To make plotting easier, for each outcome, make a file listing all of
        # qnames for the outcome and a bam file (sorted by name) with all of the
        # alignments for these qnames.

        qname_to_outcome = {}
        bam_fhs = {}

        if read_type is not None:
            full_bam_fn = self.fns_by_read_type[fn_key][read_type]
        else:
            full_bam_fh = self.fns[fn_key]

        with pysam.AlignmentFile(full_bam_fn) as full_bam_fh:
        
            for outcome, qnames in outcomes.items():
                outcome_fns = self.outcome_fns(outcome)
                outcome_fns['dir'].mkdir()
                bam_fhs[outcome] = pysam.AlignmentFile(outcome_fns['bam_by_name'], 'wb', template=full_bam_fh)
                
                with outcome_fns['query_names'].open('w') as fh:
                    for qname in qnames:
                        qname_to_outcome[qname] = outcome
                        fh.write(qname + '\n')
            
            for al in full_bam_fh:
                outcome = qname_to_outcome[al.query_name]
                bam_fhs[outcome].write(al)

        for outcome, fh in bam_fhs.items():
            fh.close()

    def make_outcome_plots(self, num_examples=10):
        fig = self.length_distribution_figure()
        fig.savefig(str(self.fns['lengths_figure']), bbox_inches='tight')
        plt.close(fig)

        kwargs = dict(
            parsimonious=True,
            paired_end_read_length=None,
            #ref_centric=True,
            size_multiple=0.3,
            detect_orientation=True,
            features_to_hide=['forward_primer_illumina', 'reverse_primer_illumina'],
            #process_mappings=self.layout_module.characterize_layout,
        )

        def relevant_alignments(i, outcome):
            als = self.get_read_alignments(i, outcome=outcome)
            if als is None:
                raise StopIteration
            l = layout.Layout(als, self.target_info)
            return l.alignments

        for outcome in self.progress(self.outcomes):
            outcome_fns = self.outcome_fns(outcome)
            
            als = relevant_alignments(0, outcome)
            diagram = visualize.ReadDiagram(als, self.target_info, **kwargs)
            diagram.fig.axes[0].set_title('')
            diagram.fig.savefig(str(outcome_fns['first_example']), bbox_inches='tight')
            plt.close(diagram.fig)
            
            als_iter = (relevant_alignments(i, outcome) for i in range(num_examples))
            stacked_im = visualize.make_stacked_Image(als_iter, self.target_info, **kwargs)
            stacked_im.save(outcome_fns['combined_figure'])

            lengths = self.outcome_read_lengths(outcome)
            fig = visualize.make_length_plot(self.read_lengths, self.color, lengths)
            fig.savefig(str(outcome_fns['lengths_figure']), bbox_inches='tight')
            plt.close(fig)
                
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

    def make_text_visualizations(self, num_examples=10):
        for outcome in self.outcomes:
            outcome_fns = self.outcome_fns(outcome)
            visualize_structure.visualize_bam_alignments(outcome_fns['bam_by_name'],
                                                         self.target_info.fns['ref_fasta'],
                                                         outcome_fns['text_alignments'],
                                                         num_examples,
                                                        )

    def length_distribution_figure(self, show_ranges=False, x_lims=None, tick_multiple=500):
        ys = self.read_lengths / sum(self.read_lengths)

        if x_lims is None:
            x_lims = (0, len(ys))

        fig, ax = plt.subplots(figsize=(18, 8))

        ax.plot(ys, color=self.color)
        
        nonzero_xs = ys.nonzero()[0]
        nonzero_ys = ys[nonzero_xs]
        ax.scatter(nonzero_xs, nonzero_ys, s=2, c=self.color)
                           
        if show_ranges:
            for _, (start, end) in self.length_ranges.iterrows():
                print(start, end)
            #for start in range(501):
                #end = start
                ax.axvspan(start - 0.5, end + 0.5,
                           gid=f'length_range_{start:05d}_{end:05d}',
                           alpha=0.1,
                           facecolor='white',
                           edgecolor='black',
                           zorder=100,
                          )
            
        major = np.arange(0, len(ys), tick_multiple)
        minor = [x for x in np.arange(0, len(ys), tick_multiple // 2) if x % tick_multiple != 0]
                    
        ax.set_xticks(major)
        ax.set_xticks(minor, minor=True)

        ax.set_ylabel('Fraction of reads')
        ax.set_xlabel('Length of read')
        
        ax.set_ylim(0, max(ys) * 1.05)
        #ax.set_ylim(0, 0.5)
        ax.set_xlim(*x_lims)

        return fig

    def span_to_Image(self, start, end, num_examples=5):
        filtered = (group for name, group in self.alignment_groups()
                    if start <= group[0].query_length <= end)
        return self.groups_to_Image(filtered, num_examples)

    def groups_to_Image(self, groups, num_examples, pairs=False):
        sample = utilities.reservoir_sample(groups, num_examples)
        
        kwargs = dict(
            parsimonious=True,
            ref_centric=True,
            label_layout=True,
            show_all_guides=True,
            paired_end_read_length=None,
            read_label='amplicon',
            size_multiple=0.5,
            detect_orientation=True,
            #process_mappings=self.layout_module.characterize_layout,
        )

        return vi.make_stacked_Image(sample, self.target_info, pairs=pairs, **kwargs)
    
    def generate_svg(self):
        html = svg.length_plot_with_popovers(self, standalone=True, x_lims=(0, 505))

        with self.fns['lengths_svg'].open('w') as fh:
            fh.write(html)

    def process(self, stage):
        #self.count_read_lengths()
        #self.generate_alignments(self.reads)
        self.categorize_outcomes()
        self.make_outcome_plots(num_examples=5)

    def explore(self, by_outcome=True, **kwargs):
        return explore(self.base_dir, by_outcome=by_outcome, target=self.target_name, experiment=(self.group, self.name), **kwargs)

class PacbioExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        super(PacbioExperiment, self).__init__(*args, **kwargs)
        self.paired_end_read_length = None

        fastq_fns = ensure_list(self.description['fastq_fns'])
        self.fns['fastqs'] = [self.data_dir / name for name in fastq_fns]

        for fn in self.fns['fastqs']:
            if not fn.exists():
                raise ValueError(f'{group}: {name} specifies non-existent {fn}')
    
    def generate_length_range_figures(self):
        self.fns['length_range_figures'].mkdir(exist_ok=True)

        rows = self.progress(list(self.length_ranges.iterrows()))

        for _, row in rows:
            im = self.span_to_Image(row.start, row.end)
            fn = self.fns['length_range_figures'] / f'{row.start}_{row.end}.png'
            im.save(fn)
    
    def generate_svg(self):
        html = svg.length_plot_with_popovers(self, standalone=True)

        with self.fns['lengths_svg'].open('w') as fh:
            fh.write(html)

class IlluminaExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        super(IlluminaExperiment, self).__init__(*args, **kwargs)

        self.sequencing_primers = self.description.get('sequencing_primers', 'truseq')
        self.paired_end_read_length = self.description.get('paired_end_read_length', None)
        self.max_qual = 41

        for k in ['R1', 'R2', 'I1', 'I2']:
            if k in self.description:
                fastq_fns = ensure_list(self.description[k])
                self.fns[k] = [self.data_dir / name for name in fastq_fns]
        
                for fn in self.fns[k]:
                    if not fn.exists():
                        raise ValueError(f'{self.group}: {self.name} specifies non-existent {fn}')

        self.read_types = [
            'stitched',
            'R1_no_overlap',
            'R2_no_overlap',
        ]

    def get_read_alignments(self, read_id, fn_key='bam_by_name', outcome=None, read_type=None):
        return super().get_read_alignments(read_id, fn_key=fn_key, outcome=outcome, read_type=read_type)
    
    @property
    def read_pairs(self):
        read_pairs = fastq.read_pairs(self.fns['R1'], self.fns['R2'])

        return self.progress(read_pairs)

    def stitched_alignment_groups(self, outcome=None):
        return self.alignment_groups(fn_key='stitched_bam_by_name', outcome=outcome)

    def no_overlap_alignment_groups(self):
        R1_groups = self.alignment_groups(fn_key='R1_no_overlap_bam_by_name')
        R2_groups = self.alignment_groups(fn_key='R2_no_overlap_bam_by_name')

        for (R1_name, R1_als), (R2_name, R2_als) in zip(R1_groups, R2_groups):
            if R1_name != R2_name:
                raise ValueError(R1_name, R2_name)
            else:
                yield R1_name, R1_als, R2_als

    def stitch_read_pairs(self):
        before_R1 = adapters.primers[self.sequencing_primers]['R1']
        before_R2 = adapters.primers[self.sequencing_primers]['R2']

        fns = self.fns_by_read_type['fastq']

        with fns['stitched'].open('w') as stitched_fh, \
             fns['R1_no_overlap'].open('w') as R1_fh, \
             fns['R2_no_overlap'].open('w') as R2_fh:

            for R1, R2 in self.read_pairs:
                stitched = sw.stitch_read_pair(R1, R2, before_R1, before_R2)
                if len(stitched) == 2 * self.paired_end_read_length:
                    R1_fh.write(str(R1))
                    R2_fh.write(str(R2))
                else:
                    stitched_fh.write(str(stitched))

    @property
    def query_names(self):
        for read in self.stitched_reads:
            yield read.name

    def count_read_lengths(self):
        lengths = Counter(len(r.seq) for r in self.reads_by_type('stitched'))

        no_overlap_length = self.paired_end_read_length * 2

        lengths[no_overlap_length] += sum(1 for _ in self.reads_by_type('R1_no_overlap'))

        lengths = utilities.counts_to_array(lengths)
        np.savetxt(self.fns['lengths'], lengths, '%d')
    
    def generate_individual_length_figures(self):
        by_length = defaultdict(list)
        for name, group in self.stitched_alignment_groups():
            length = group[0].query_length
            by_length[length].append(group)

        no_overlap_length = 2 * self.paired_end_read_length
        for name, R1_als, R2_als in self.no_overlap_alignment_groups():
            by_length[no_overlap_length].append((R1_als, R2_als))

        self.fns['length_range_figures'].mkdir(exist_ok=True)

        items = self.progress(by_length.items())

        for length, groups in items:
            if length == no_overlap_length:
                continue
            im = self.groups_to_Image(groups, 3, pairs=(length == no_overlap_length))
            fn = self.fns['length_range_figures'] / f'{length}_{length}.png'
            im.save(fn)

    def process(self, stage=0):
        #self.stitch_read_pairs()
        
        #self.count_read_lengths()

        #for read_type in self.read_types:
        #    self.generate_alignments(read_type)
        #    self.generate_supplemental_alignments(read_type)
        #    self.combine_alignments(read_type)

        self.categorize_outcomes(read_type='stitched')
        self.make_outcome_plots(num_examples=6)
        #self.generate_individual_length_figures()
        #self.generate_svg()
        #self.make_text_visualizations()

def explore(base_dir, by_outcome=False, target=None, experiment=None, **kwargs):
    if target is None:
        target_names = sorted([t.name for t in target_info.get_all_targets(base_dir)])
    else:
        target_names = [target]

    widgets = {
        'target': ipywidgets.Select(options=target_names, value=target_names[0], layout=ipywidgets.Layout(height='200px')),
        'experiment': ipywidgets.Select(options=[], layout=ipywidgets.Layout(height='200px', width='450px')),
        'read_id': ipywidgets.Select(options=[], layout=ipywidgets.Layout(height='200px', width='600px')),
        'outcome': ipywidgets.Select(options=[], continuous_update=False, layout=ipywidgets.Layout(height='200px', width='450px')),
        'zoom_in': ipywidgets.FloatRangeSlider(value=[-0.02, 1.02], min=-0.02, max=1.02, step=0.001, continuous_update=False, layout=ipywidgets.Layout(width='1200px')),
    }
    toggles = [
        'parsimonious',
        'relevant',
        'ref_centric',
        'draw_sequence',
        'draw_qualities',
        'draw_mismatches',
        'draw_read_pair',
        'force_left_aligned',
    ]
    for toggle in toggles:
        widgets[toggle] = ipywidgets.ToggleButton(value=kwargs.pop(toggle, False))

    # For some reason, the target widget doesn't get a label without this.
    for k, v in widgets.items():
        v.description = k

    if experiment is None:
        conditions = {}
    else:
        group_name, exp_name = experiment
        conditions = {'group': group_name, 'name': exp_name}

    exps = get_all_experiments(base_dir, conditions)

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
        if outcomes:
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

        if widgets['draw_read_pair'].value:
            paired_end_read_length = exp.paired_end_read_length
        else:
            paired_end_read_length = None
        
        if widgets['draw_sequence'].value:
            print(als[0].query_name)
            print(als[0].get_forward_sequence())

        l = exp.layout_module.Layout(als, exp.target_info)
        info = l.categorize()
        
        if widgets['relevant'].value:
            als = l.relevant_alignments

        diagram = visualize.ReadDiagram(als, exp.target_info,
                                        max_qual=exp.max_qual,
                                        paired_end_read_length=paired_end_read_length,
                                        read_label='amplicon',
                                        **plot_kwargs)
        fig = diagram.fig

        fig.axes[0].set_title(' '.join((l.name,) + info))

        return diagram.fig

    all_kwargs = {**{k: ipywidgets.fixed(v) for k, v in kwargs.items()}, **widgets}

    interactive = ipywidgets.interactive(plot, **all_kwargs)
    interactive.update()

    def make_row(keys):
        return ipywidgets.HBox([widgets[k] for k in keys])

    if by_outcome:
        top_row_keys = ['target', 'experiment', 'outcome', 'read_id']
    else:
        top_row_keys = ['target', 'experiment', 'read_id']

    layout = ipywidgets.VBox(
        [make_row(top_row_keys),
         make_row(toggles),
         interactive.children[-1],
         output,
        ],
    )

    return layout

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
    groups = (p.name for p in data_dir.iterdir() if p.is_dir())

    if 'group' in conditions:
        groups = (n for n in groups if n in conditions['group'])
    
    for group in groups:
        sample_sheet_fn = data_dir / group / 'sample_sheet.yaml'
        sample_sheet = yaml.load(sample_sheet_fn.read_text())

        for name, description in sample_sheet.items():
            if description.get('experiment_type') == 'illumina':
                exp_class = IlluminaExperiment
            elif description.get('experiment_type') == 'pacbio':
                exp_class = PacbioExperiment
            else:
                exp_class = Experiment
            
            exp = exp_class(base_dir, group, name, description=description)
            exps.append(exp)

    filtered = [exp for exp in exps if check_conditions(exp)]
    if len(filtered) == 0:
        raise ValueError('No experiments met conditions')

    return filtered