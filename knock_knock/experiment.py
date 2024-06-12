import matplotlib
if 'inline' not in matplotlib.get_backend():
    matplotlib.use('Agg')

import datetime
import heapq
import logging
import shutil
import sys

from collections import defaultdict, Counter
from contextlib import ExitStack
from itertools import islice, chain
from pathlib import Path
from textwrap import dedent

import bokeh.palettes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
import yaml

import hits.visualize
from hits import annotation, fastq, genomes, mapping_tools, sam, utilities
from hits.utilities import memoized_property

import knock_knock.blast
import knock_knock.lengths
import knock_knock.visualize.lengths
import knock_knock.target_info
import knock_knock.layout as layout_module
from . import outcome_record, svg, table, explore

class ColorGroupCycler:
    def __init__(self):
        starts_20c = np.arange(4) * 4
        starts_20b = np.array([3, 1, 0, 2, 4]) * 4

        groups_20c = [bokeh.palettes.Category20c_20[start:start + 3] for start in starts_20c]
        groups_20b = [bokeh.palettes.Category20b_20[start:start + 3] for start in starts_20b]

        self.all_groups = (groups_20c + groups_20b)
        
    def __getitem__(self, key):
        group_num, replicate = key
        group = self.all_groups[group_num % len(self.all_groups)]
        color = group[replicate % len(group)]
        return color
    
color_groups = ColorGroupCycler()

def extract_color(description):
    color = description.get('color')
    if color is None or color == 0:
        color = 'grey'
    else:
        num = int(color) - 1
        replicate = int(description.get('replicate', 1)) - 1
        color = color_groups[num, replicate]

    return color
        
def ensure_list(possibly_list):
    if isinstance(possibly_list, list):
        definitely_list = possibly_list
    else:
        definitely_list = [possibly_list]

    return definitely_list

UMI_annotation_fields = [
    ('original_name', 's'),
    ('UMI_seq', 's'),
    ('UMI_qual', 's'),
]

UMIAnnotation = annotation.Annotation_factory(UMI_annotation_fields)

class Experiment:
    def __init__(self, base_dir, batch_name, sample_name, description=None, progress=None):
        self.batch_name = batch_name
        self.sample_name = sample_name

        if progress is None or getattr(progress, '_silent', False):
            self.silent = True
            def ignore_kwargs(x, **kwargs):
                return x
            progress = ignore_kwargs
        else:
            self.silent = False

        def pass_along_kwargs(iterable, **kwargs):
            return progress(iterable, **kwargs)

        self.progress = pass_along_kwargs

        self.base_dir = Path(base_dir)

        if description is None:
            description = self.load_description()

        self.description = description

        self.max_insertion_length = 20

        self.sgRNAs = self.description.get('sgRNAs')
        self.donor = self.description.get('donor')
        self.nonhomologous_donor = self.description.get('nonhomologous_donor')
        self.primer_names = self.description.get('primer_names')
        self.sequencing_start_feature_name = self.description.get('sequencing_start_feature_name')
        self.infer_homology_arms = self.description.get('infer_homology_arms', False)
        self.min_relevant_length = self.description.get('min_relevant_length', False)

        # When checking if an Experiment meets filtering conditions, want to be
        # able to just test description.
        self.description['batch'] = batch_name
        self.description['sample_name'] = sample_name

        self.fns = {
            'results_dir': self.results_dir,
            'outcomes_dir': self.results_dir / 'outcomes',
            'sanitized_category_names': self.results_dir / 'outcomes' / 'sanitized_category_names.txt',
            'outcome_counts': self.results_dir / 'outcome_counts.csv',
            'outcome_list': self.results_dir / 'outcome_list.txt',

            'outcome_stratified_lengths': self.results_dir / 'lengths.hdf5',
            'lengths_figure': self.results_dir / 'all_lengths.png',

            'donor_microhomology_lengths': self.results_dir / 'donor_microhomology_lengths.txt', 

            'length_ranges_dir': self.results_dir / 'length_ranges',
            'outcome_browser': self.results_dir / 'outcome_browser.html',

            'snapshots_dir': self.results_dir / 'snapshots',
        }

        def make_length_range_fig_fn(start, end):
            return self.fns['length_ranges_dir'] / f'{start}_{end}.png'

        self.fns['length_range_figure'] = make_length_range_fig_fn
        
        self.color = extract_color(self.description)
        self.max_qual = 93
        
        index_names = self.description.get('supplemental_indices')

        if index_names is None:
            index_names = []

        if isinstance(index_names, str):
            index_names = index_names.split(';')

        self.supplemental_index_names = index_names

        # count_index_levels are level names for index on outcome counts.
        self.count_index_levels = ['category', 'subcategory', 'details']

        self.length_plot_smooth_window = 0

        self.has_UMIs = False

        self.outcome_fn_keys = [
            'outcome_list',
        ]

    def __repr__(self):
        return f'{self.__class__.__name__}: batch_name={self.batch_name}, sample_name={self.sample_name}, base_dir={self.base_dir}'

    @memoized_property
    def length_to_store_unknown(self):
        return int(self.max_relevant_length * 1.05)

    @property
    def final_Outcome(self):
        return outcome_record.OutcomeRecord

    def load_description(self):
        return load_sample_sheet(self.base_dir, self.batch_name)[self.sample_name]

    @property
    def categorizer(self):
        # This needs to be a property because subclasses use it as one.
        return layout_module.Layout

    @property
    def uncommon_read_type(self):
        # Will be overloaded by any subclass that separates out common sequences.
        return self.preprocessed_read_type

    @memoized_property
    def results_dir(self):
        d = self.base_dir / 'results' 

        if isinstance(self.batch_name, tuple):
            for level in self.batch_name:
                d /= level
        else:
            d /= self.batch_name

        d /= self.sample_name

        return d 

    @memoized_property
    def data_dir(self):
        d = self.base_dir / 'data'

        if isinstance(self.batch_name, tuple):
            for level in self.batch_name:
                d /= level
        else:
            d /= self.batch_name

        return d

    @memoized_property
    def supplemental_indices(self):
        locations = knock_knock.target_info.locate_supplemental_indices(self.base_dir)
        return {name: locations[name] for name in self.supplemental_index_names}

    @memoized_property
    def target_info(self):
        ti = knock_knock.target_info.TargetInfo(self.base_dir,
                                                self.target_name,
                                                donor=self.donor,
                                                nonhomologous_donor=self.nonhomologous_donor,
                                                sgRNAs=self.sgRNAs,
                                                primer_names=self.primer_names,
                                                sequencing_start_feature_name=self.sequencing_start_feature_name,
                                                supplemental_indices=self.supplemental_indices,
                                                infer_homology_arms=self.infer_homology_arms,
                                                min_relevant_length=self.min_relevant_length,
                                               )

        return ti

    @memoized_property
    def target_name(self):
        # Cast to str because yaml parsing produce a non-string.
        return str(self.description['target_info'])

    def check_combined_read_length(self):
        pass

    def make_nonredundant_sequence_fastq(self):
        pass

    @memoized_property
    def fns_by_read_type(self):
        fns = defaultdict(dict)

        for read_type in self.read_types:
            if read_type == 'CCS':
                fns['fastq'][read_type] = self.fns['CCS_fastqs']
            else:
                fns['fastq'][read_type] = self.results_dir / f'{read_type}.fastq.gz'

            fns['primary_bam'][read_type] = self.results_dir / f'{read_type}_alignments.bam'
            fns['primary_bam_by_name'][read_type] = self.results_dir / f'{read_type}_alignments.by_name.bam'

            for index_name in self.supplemental_index_names:
                fns['supplemental_STAR_prefix'][read_type, index_name] = self.results_dir / f'{read_type}_{index_name}_alignments_STAR.'
                fns['supplemental_bam'][read_type, index_name] = self.results_dir / f'{read_type}_{index_name}_alignments.bam'
                fns['supplemental_bam_by_name'][read_type, index_name] = self.results_dir / f'{read_type}_{index_name}_alignments.by_name.bam'
                fns['supplemental_bam_temp'][read_type, index_name] = self.results_dir / f'{read_type}_{index_name}_alignments.temp.bam'

            fns['bam'][read_type] = self.results_dir / f'{read_type}_combined_alignments.bam'
            fns['bam_by_name'][read_type] = self.results_dir / f'{read_type}_combined_alignments.by_name.bam'
        
        return fns

    def outcome_fns(self, outcome):
        # To allow outcomes to have arbitrary names without causing file path problems,
        # sanitize the names.
        outcome_string = self.categorizer.outcome_to_sanitized_string(outcome)
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
            'bam_by_name' : {read_type: outcome_dir / f'{read_type}.by_name.bam' for read_type in self.read_types},
        }

        def make_length_range_fig_fn(start, end):
            return fns['length_ranges_dir'] / f'{start}_{end}.png'

        fns['length_range_figure'] = make_length_range_fig_fn

        return fns

    def reads_by_type(self, read_type):
        fn_source = self.fns_by_read_type['fastq'][read_type]

        missing_file = False
        if isinstance(fn_source, list):
            if not all(fn.exists() for fn in fn_source):
                missing_file = True
        else:
            if not fn_source.exists():
                missing_file = True

        if missing_file:
            logging.warning(f'{self.batch_name}, {self.sample_name} {read_type} not found')
            reads = []
        else:
            reads = fastq.reads(fn_source, up_to_space=True)

        return reads
    
    @memoized_property
    def read_lengths(self):
        return self.outcome_stratified_lengths.lengths_for_all_outcomes

    def extract_reads_with_uncommon_sequences(self):
        pass
    
    def generate_outcome_counts(self):
        ''' Note that metadata lines start with '#' so category names can't. '''
        counts_fn = self.fns['outcome_counts']

        with open(counts_fn, 'w') as fh:
            for fn_key, metadata_lines in self.outcome_metadata():
                fh.write(f'# Metadata from {fn_key}:\n') 
                for line in metadata_lines:
                    fh.write(line)

        counts = Counter()
        for outcome in self.outcome_iter():
            counts[outcome.category, outcome.subcategory, outcome.details] += 1

        counts = pd.Series(counts, dtype=int).sort_values(ascending=False)
        counts.to_csv(counts_fn, mode='a', sep='\t', header=False)

    def generate_outcome_stratified_lengths(self):
        lengths = knock_knock.lengths.OutcomeStratifiedLengths(self.outcome_iter(),
                                                               self.max_relevant_length,
                                                               self.length_to_store_unknown,
                                                              )
        lengths.to_file(self.fns['outcome_stratified_lengths'])

    def record_snapshot(self):
        ''' Make copies of per-read outcome categorizations and
        outcome counts to allow comparison when categorization code
        changes.
        '''

        snapshot_name = f'{datetime.datetime.now():%Y-%m-%d_%H%M%S}'
        snapshot_dir = self.fns['snapshots_dir'] / snapshot_name
        snapshot_dir.mkdir(parents=True)

        fn_keys_to_snapshot = self.outcome_fn_keys + ['outcome_counts']

        for key in fn_keys_to_snapshot:
            shutil.copy(self.fns[key], snapshot_dir)

    def length_ranges(self, outcome=None):
        if outcome is None:
            lengths = self.read_lengths
        else:
            lengths = self.outcome_stratified_lengths.outcome_length_arrays[outcome]

        nonzero, = np.nonzero(lengths)
        ranges = [(i, i) for i in nonzero]
        return pd.DataFrame(ranges, columns=['start', 'end'])
    
    def alignment_groups(self, fn_key='bam_by_name', outcome=None, read_type=None):
        '''
            If outcome is not None, yield alignment groups from outcome-stratified
            bams.

            If read_type is not None, yield alignment groups from that
            read_type.

            If both outcome and read_type are None, yields groups that may be looked
            up from common sequences. 
        '''

        if outcome is not None:
            if read_type is None:
                read_type = self.preprocessed_read_type

            if isinstance(outcome, tuple):
                # outcome is a (category, subcategory) pair
                fn = self.outcome_fns(outcome)['bam_by_name'][read_type]

                if fn.exists():
                    grouped = sam.grouped_by_name(fn)
                else:
                    grouped = []

                yield from grouped

            elif isinstance(outcome, str):
                # outcome is a single category, so need to chain together all relevant
                # (category, subcategory) pairs.
                pairs = [(c, s) for c, s in self.categories_by_frequency if c == outcome]
                pair_groups = [self.alignment_groups(fn_key=fn_key, outcome=pair, read_type=read_type) for pair in pairs]
                yield from heapq.merge(*pair_groups)

        elif read_type is not None:
            fn = self.fns_by_read_type['bam_by_name'][read_type]

            if fn.exists():
                grouped = sam.grouped_by_name(fn)
            else:
                grouped = []

            yield from grouped

        else:
            uncommon_alignment_groups = self.alignment_groups(read_type=self.uncommon_read_type)
            all_reads = self.reads_by_type(self.preprocessed_read_type)

            for read in all_reads:
                if read.seq in self.common_sequence_to_alignments:
                    name = read.name
                    als = self.common_sequence_to_alignments[read.seq]

                else:
                    name, als = next(uncommon_alignment_groups)

                    if name != read.name:
                        raise ValueError('iters out of sync', name, read.name)

                yield name, als

    def query_names(self, read_type=None):
        for qname, als in self.alignment_groups(read_type=read_type):
            yield qname
    
    def generate_alignments_with_blast(self,
                                       read_type=None,
                                       supplemental_index_name=None,
                                       filter_to_discard=None,
                                      ):
        reads = self.reads_by_type(read_type)

        if read_type is None:
            description = 'Generating alignments'
        else:
            description = f'Generating {read_type} alignments'

        reads = self.progress(reads, desc=description)

        bam_by_name_fns = []

        if supplemental_index_name is None:
            bam_key = 'primary_bam_by_name'
            base_bam_by_name_fn = self.fns_by_read_type['primary_bam_by_name'][read_type]
            ref_seqs = self.target_info.reference_sequences
        else:
            base_bam_by_name_fn = self.fns_by_read_type['supplemental_bam_by_name'][read_type, supplemental_index_name]
            fasta_dir = self.supplemental_indices[supplemental_index_name]['fasta']
            ref_seqs = genomes.load_entire_genome(fasta_dir)

        for i, chunk in enumerate(utilities.chunks(reads, 10000)):
            suffix = f'.{i:06d}.bam'
            bam_by_name_fn = base_bam_by_name_fn.with_suffix(suffix)

            knock_knock.blast.blast(ref_seqs,
                                    chunk,
                                    bam_by_name_fn=bam_by_name_fn,
                                    max_insertion_length=self.max_insertion_length,
                                    ref_name_prefix_to_append=supplemental_index_name,
                                    filter_to_discard=filter_to_discard,
                                   )

            bam_by_name_fns.append(bam_by_name_fn)

        if len(bam_by_name_fns) == 0:
            # There weren't any reads. Make an empty bam file.
            with pysam.AlignmentFile(base_bam_by_name_fn, 'wb', header=self.target_info.header) as fh:
                pass

        else:
            sam.merge_sorted_bam_files(bam_by_name_fns, base_bam_by_name_fn, by_name=True)

        for fn in bam_by_name_fns:
            fn.unlink()

    def generate_supplemental_alignments_with_STAR(self, read_type=None, min_length=None):
        for index_name in self.supplemental_indices:
            if index_name == 'phiX':
                continue

            fastq_fn = self.fns_by_read_type['fastq'][read_type]
            STAR_prefix = self.fns_by_read_type['supplemental_STAR_prefix'][read_type, index_name]
            index = self.supplemental_indices[index_name]['STAR']

            bam_fn = mapping_tools.map_STAR(fastq_fn,
                                            index,
                                            STAR_prefix,
                                            sort=False,
                                            mode='permissive',
                                           )

            saved_verbosity = pysam.set_verbosity(0)
            with pysam.AlignmentFile(bam_fn) as all_mappings:
                header = all_mappings.header
                new_references = [f'{index_name}_{ref}' for ref in header.references]
                new_header = pysam.AlignmentHeader.from_references(new_references, header.lengths)

                by_name_fn = self.fns_by_read_type['supplemental_bam_by_name'][read_type, index_name]
                by_name_sorter = sam.AlignmentSorter(by_name_fn, new_header, by_name=True)

                with by_name_sorter:
                    for al in all_mappings:
                        # To reduce noise, filter out alignments that are too short
                        # or that have too many edits (per aligned nt). Keep this in
                        # mind when interpretting short unexplained gaps in reads.

                        if min_length is not None and (not al.is_unmapped) and al.query_alignment_length < min_length:
                            continue

                        #if al.get_tag('AS') / al.query_alignment_length <= 0.8:
                        #    continue

                        by_name_sorter.write(al)
            pysam.set_verbosity(saved_verbosity)

            mapping_tools.clean_up_STAR_output(STAR_prefix)

            Path(bam_fn).unlink()

    def generate_supplemental_alignments_with_minimap2(self,
                                                       report_all=True,
                                                       num_threads=1,
                                                       read_type=None,
                                                       use_ont_index=False,
                                                      ):

        for index_name in self.supplemental_indices:
            # Note: this doesn't support multiple intput fastqs.
            fastq_fn = ensure_list(self.fns_by_read_type['fastq'][read_type])[0]

            if use_ont_index:
                index_type = 'minimap2_ont'
            else:
                index_type = 'minimap2'

            try:
                index = self.supplemental_indices[index_name][index_type]
            except:
                raise ValueError(index_name, index_type)

            temp_bam_fn = self.fns_by_read_type['supplemental_bam_temp'][read_type, index_name]

            mapping_tools.map_minimap2(fastq_fn,
                                       index,
                                       temp_bam_fn,
                                       report_all=report_all,
                                       num_threads=num_threads,
                                       use_ont_index=use_ont_index,
                                      )

            header = sam.get_header(temp_bam_fn)
            new_references = [f'{index_name}_{ref}' for ref in header.references]
            new_header = pysam.AlignmentHeader.from_references(new_references, header.lengths)

            by_name_fn = self.fns_by_read_type['supplemental_bam_by_name'][read_type, index_name]
            by_name_sorter = sam.AlignmentSorter(by_name_fn, new_header, by_name=True)

            # Sorting key prioritizes first longer alignments, then ones with fewer edits.
            sorting_key = lambda al: (-al.query_alignment_length, al.get_tag('NM'))

            # Unless I am missing an option, minimap2 omits seq and qual for mapped reads.
            # Add them back.
            saved_verbosity = pysam.set_verbosity(0)
            with pysam.AlignmentFile(temp_bam_fn) as all_mappings, by_name_sorter:
                al_groups = sam.grouped_by_name(all_mappings)
                reads = self.reads_by_type(read_type)
                for (qname, als), read in zip(al_groups, reads):
                    if qname != read.name:
                        raise ValueError('iters out of sync')

                    seq = read.seq
                    seq_rc = utilities.reverse_complement(seq)

                    qual = fastq.decode_sanger_to_array(read.qual)
                    qual_rc = qual[::-1]

                    # Only retain the top 50 alignments.
                    sorted_als = sorted((al for al in als if not al.is_unmapped), key=sorting_key)[:50]

                    for al in sorted_als:
                        if not al.is_reverse:
                            al.query_sequence = seq
                            al.query_qualities = qual
                        else:
                            al.query_sequence = seq_rc
                            al.query_qualities = qual_rc

                        by_name_sorter.write(al)
            pysam.set_verbosity(saved_verbosity)

            temp_bam_fn.unlink()

    def combine_alignments(self, read_type=None):
        fns_to_merge = [self.fns_by_read_type['primary_bam_by_name'][read_type]]
        for index_name in self.supplemental_indices:
            fns_to_merge.append(self.fns_by_read_type['supplemental_bam_by_name'][read_type, index_name])

        sam.merge_sorted_bam_files(fns_to_merge,
                                   self.fns_by_read_type['bam_by_name'][read_type],
                                   by_name=True,
                                  )

        for fn in fns_to_merge:
            fn.unlink()

    @memoized_property
    def outcome_counts(self):
        fn = self.fns['outcome_counts']

        try:
            counts = pd.read_csv(fn,
                                 index_col=tuple(range(len(self.count_index_levels))),
                                 header=None,
                                 na_filter=False,
                                 sep='\t',
                                 comment='#',
                                ).squeeze('columns')
            counts.index.names = self.count_index_levels
            counts.name = None

        except (FileNotFoundError, pd.errors.EmptyDataError):
            counts = None

        return counts

    @memoized_property
    def outcome_fractions(self):
        if self.outcome_counts is None:
            return None
        else:
            return self.outcome_counts / self.outcome_counts.sum()

    @memoized_property
    def common_sequence_to_outcome(self):
        return {}

    @memoized_property
    def common_sequence_to_alignments(self):
        return {}

    @memoized_property
    def category_counts(self):
        if self.outcome_counts is None:
            return None
        else:
            return self.outcome_counts.groupby(level=['category', 'subcategory']).sum()

    @memoized_property
    def categories_by_frequency(self):
        if self.category_counts is None:
            return []
        else:
            return list(self.category_counts.sort_values(ascending=False).index)

    def outcome_query_names(self, outcome):
        fns = self.outcome_fns(outcome)
        all_qnames = []
        for fn_key in ['query_names', 'no_overlap_query_names']:
            fn = fns[fn_key]
            if fn.exists():
                qnames = fn.read_text().splitlines()
                all_qnames.extend(qnames)
        return all_qnames
    
    def record_sanitized_category_names(self):
        sanitized_to_original = {}
        for cat, subcats in self.categorizer.category_order:
            sanitized_string = self.categorizer.outcome_to_sanitized_string(cat)
            sanitized_to_original[sanitized_string] = cat
            for subcat in subcats:
                outcome = (cat, subcat)
                sanitized_string = self.categorizer.outcome_to_sanitized_string(outcome)
                sanitized_to_original[sanitized_string] = ', '.join(outcome)
        
        with open(self.fns['sanitized_category_names'], 'w') as fh:
            for k, v in sorted(sanitized_to_original.items()):
                fh.write(f'{k}\t{v}\n')

    @memoized_property
    def combined_header(self):
        return hits.sam.get_header(self.fns_by_read_type['bam_by_name'][self.uncommon_read_type])

    def categorize_outcomes(self, fn_key='bam_by_name', read_type=None, max_reads=None):
        if self.fns['outcomes_dir'].is_dir():
            shutil.rmtree(str(self.fns['outcomes_dir']))
           
        self.fns['outcomes_dir'].mkdir()

        outcome_to_qnames = defaultdict(list)
        outcome_to_no_overlap_qnames = defaultdict(list)

        with self.fns['outcome_list'].open('w') as outcome_fh:
            outcome_fh.write(f'## Generated at {utilities.current_time_string()}\n')

            alignment_groups = self.alignment_groups(fn_key=fn_key, read_type=read_type)

            if max_reads is not None:
                alignment_groups = islice(alignment_groups, max_reads)

            if read_type is None:
                description = 'Categorizing reads'
            else:
                description = f'Categorizing {read_type} reads'

            for name, als in self.progress(alignment_groups, desc=description):
                if 'UMI_key' in self.description:
                    UMI_annotation = UMIAnnotation.from_identifier(name)
                    UMI_seq = UMI_annotation['UMI_seq']
                    UMI_qual = UMI_annotation['UMI_qual']
                else:
                    UMI_seq = ''
                    UMI_qual = ''

                if isinstance(als, list):
                    seq = als[0].get_forward_sequence()
                    # TODO: this shouldn't use common sequence al for qualities.
                    Q30_fraction = np.mean(np.array(als[0].query_qualities) >=30)

                    # Special handling of empty sequence.
                    if seq is None:
                        seq = ''

                    if seq in self.common_sequence_to_outcome:
                        layout = self.common_sequence_to_outcome[seq]

                    else:
                        layout = self.categorizer(als,
                                                  self.target_info,
                                                  mode=self.layout_mode,
                                                  error_corrected=self.has_UMIs,
                                                 )

                        try:
                            layout.categorize()
                        except:
                            print()
                            print(self.sample_name, name)
                            raise

                    outcome_to_qnames[layout.category, layout.subcategory].append(name)

                elif isinstance(als, dict):
                    layout = self.nonoverlapping_categorizer(als,
                                                             self.target_info,
                                                            )
                    try:
                        layout.categorize()
                    except:
                        print()
                        print(self.sample_name, name)
                        raise

                    outcome_to_no_overlap_qnames[layout.category, layout.subcategory].append(name)
                
                else:
                    raise ValueError

                outcome = self.final_Outcome.from_layout(layout,
                                                         query_name=name,
                                                         Q30_fraction=Q30_fraction,
                                                         UMI_seq=UMI_seq,
                                                         UMI_qual=UMI_qual,
                                                        )
                outcome_fh.write(f'{outcome}\n')

        # To make plotting easier, for each outcome, make a file listing all of
        # qnames for the outcome and a bam file (sorted by name) with all of the
        # alignments for these qnames.

        if read_type is None:
            bam_read_type = self.preprocessed_read_type
        else:
            bam_read_type = read_type

        qname_to_outcome = {}

        alignment_sorters = sam.multiple_AlignmentSorters(self.combined_header, by_name=True)

        outcomes = set(outcome_to_qnames) | set(outcome_to_no_overlap_qnames)

        for outcome in outcomes:
            outcome_fns = self.outcome_fns(outcome)
            # This shouldn't be necessary due to rmtree of parent directory above
            # but empirically sometimes is.
            if outcome_fns['dir'].is_dir():
                shutil.rmtree(str(outcome_fns['dir']))

            outcome_fns['dir'].mkdir()

        for outcome, qnames in outcome_to_qnames.items():
            outcome_fns = self.outcome_fns(outcome)

            alignment_sorters[outcome] = outcome_fns['bam_by_name'][bam_read_type]

            with outcome_fns['query_names'].open('w') as fh:
                for qname in qnames:
                    qname_to_outcome[qname] = outcome
                    fh.write(qname + '\n')

        for outcome, qnames in outcome_to_no_overlap_qnames.items():
            outcome_fns = self.outcome_fns(outcome)

            for which in ['R1', 'R2']:
                alignment_sorters[outcome, which] = outcome_fns['bam_by_name'][f'{which}_no_overlap']
            
            with outcome_fns['no_overlap_query_names'].open('w') as fh:
                for qname in qnames:
                    qname_to_outcome[qname] = outcome
                    fh.write(qname + '\n')

        with alignment_sorters:
            for name, als in self.alignment_groups(fn_key=fn_key, read_type=read_type):
                if name in qname_to_outcome:
                    outcome = qname_to_outcome[name]

                    if isinstance(als, list):
                        for al in als:
                            al.query_name = name
                            alignment_sorters[outcome].write(al)

                    elif isinstance(als, dict):
                        for which in ['R1', 'R2']:
                            for al in als[which]:
                                al.query_name = name
                                alignment_sorters[outcome, which].write(al)

                    else:
                        raise ValueError

    def process(self, stage):
        self.results_dir.mkdir(exist_ok=True, parents=True)

        try:
            if stage == 'preprocess':
                self.preprocess()
            elif stage == 'align':
                self.align()
            elif stage == 'categorize':
                self.categorize()
                self.generate_summary_figures()
            elif stage == 'generate_example_diagrams':
                self.generate_example_diagrams()
            else:
                raise ValueError(f'invalid stage: {stage}')

        except:
            print(self.batch_name, self.sample_name)
            raise

    def get_read_alignments(self, read_id, fn_key='bam_by_name', outcome=None, read_type=None):
        # iter() necessary because tqdm objects aren't iterators
        read_groups = iter(self.alignment_groups(fn_key=fn_key, outcome=outcome, read_type=read_type))

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
        layout = self.get_read_layout(read_id, outcome=outcome, read_type=read_type)
        layout.categorize()

        diagram = layout.plot(relevant=relevant, **kwargs)

        return diagram

    def length_distribution_figure(self,
                                   outcome=None,
                                   show_ranges=False,
                                   show_title=False,
                                   fig_size=(12, 6),
                                   font_size=12,
                                   x_tick_multiple=None,
                                   max_relevant_length=None,
                                  ):
        if x_tick_multiple is None:
            x_tick_multiple = self.x_tick_multiple

        if max_relevant_length is None:
            max_relevant_length = self.max_relevant_length

        all_ys = self.read_lengths / self.total_reads

        def convert_to_smoothed_percentages(ys):
            window = self.length_plot_smooth_window * 2 + 1
            smoothed = pd.Series(ys).rolling(window=window, center=True, min_periods=1).sum()
            return smoothed * 100

        fig, ax = plt.subplots(figsize=fig_size)

        if outcome is None:
            ys_list = [
                (all_ys, self.color, 0.9, 'all reads', True),
            ]

            ys_to_check = all_ys
        else:
            if isinstance(outcome, tuple):
                outcome_lengths = self.outcome_stratified_lengths.outcome_length_arrays[outcome]
                color = self.outcome_stratified_lengths.outcome_to_color(smooth_window=self.length_plot_smooth_window)[outcome]
                label = ': '.join(outcome)
            else:
                outcome_lengths = sum([v for (c, s), v in self.outcome_stratified_lengths.outcome_length_arrays.items() if c == outcome])
                color = 'black'
                label = outcome

            outcome_ys = outcome_lengths / self.total_reads

            other_lengths = self.read_lengths - outcome_lengths
            other_ys = other_lengths / self.total_reads

            ys_list = [
                (other_ys, 'black', 0.2, 'all other reads', False),
                (outcome_ys, color, 0.9, label, True),
            ]

            ys_to_check = outcome_ys

        max_y = 0

        for ys, color, alpha, label, check_for_max in ys_list:
            ys = convert_to_smoothed_percentages(ys)

            if check_for_max:
                max_y = max(max_y, max(ys, default=0.1))

            if self.length_plot_smooth_window == 0:
                line_width = 1
            else:
                line_width = 2

            ax.plot(ys, color=color, alpha=alpha, linewidth=line_width, label=label)
            
            nonzero_xs = ys.to_numpy().nonzero()[0]
            nonzero_ys = ys[nonzero_xs]
            
            # Don't mark nonzero points if any smoothing was done.
            if self.length_plot_smooth_window == 0 and label != 'all other reads':
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
            
        y_lims = (0, max_y * 1.05)
        ax.set_ylim(*y_lims)

        x_max = int(max_relevant_length * 1.005)
        ax.set_xlim(0, x_max)
        
        if show_title:
            if outcome is None:
                title = f'{self.batch_name}: {self.sample_name}'
            else:
                category, subcategory = outcome
                title = f'{self.batch_name}: {self.sample_name}\n{category}: {subcategory}'

            ax.set_title(title)
            
        if outcome is not None:
            # No need to draw legend if only showing all reads
            ax.legend(framealpha=0.5)
        
        for i, (name, length) in enumerate(self.expected_lengths.items()):
            y = 1 + 0.02  + 0.04 * i
            ax.axvline(length, ymin=0, ymax=y, color='black', alpha=0.4, clip_on=False)

            ax.annotate(name,
                        xy=(length, y), xycoords=('data', 'axes fraction'),
                        xytext=(0, 1), textcoords='offset points',
                        ha='center', va='bottom',
                        size=10,
                       )
        
        main_ticks = list(range(0, max_relevant_length, x_tick_multiple))
        main_tick_labels = [f'{x:,}' for x in main_ticks]

        extra_ticks = [max_relevant_length]
        extra_tick_labels = [r'$\geq$' + f'{max_relevant_length}']

        if self.length_to_store_unknown is not None:
            extra_ticks.append(self.length_to_store_unknown)
            extra_tick_labels.append('?')

        ax.set_xticks(main_ticks + extra_ticks)
        ax.set_xticklabels(main_tick_labels + extra_tick_labels)
        
        minor = [x for x in np.arange(0, x_max, x_tick_multiple // 2) if x % x_tick_multiple != 0]
        ax.set_xticks(minor, minor=True)

        ax.set_ylabel('percentage of reads', size=font_size)
        ax.set_xlabel('amplicon length', size=font_size)

        return fig

    def outcome_iter(self, outcome_fn_keys=None):
        if outcome_fn_keys is None:
            outcome_fn_keys = self.outcome_fn_keys

        with ExitStack() as stack:
            fhs = []
            for key in outcome_fn_keys:
                fn = self.fns[key]
                if fn.exists():
                    fhs.append(stack.enter_context(fn.open()))

            chained = chain.from_iterable(fhs)
            for line in chained:
                # Metadata lines start with '##'.
                if line.startswith('##'):
                    continue

                outcome = self.final_Outcome.from_line(line)
                yield outcome

    def outcome_metadata(self, outcome_fn_keys=None):
        ''' Extract metadata lines from all outcome files. '''
        if outcome_fn_keys is None:
            outcome_fn_keys = self.outcome_fn_keys

        all_metadata_lines = []

        for key in outcome_fn_keys:
            fn = self.fns[key]
            if fn.exists():
                metadata_lines = []
                with open(fn) as fh:
                    for line in fh:
                        # Metadata lines start with '##'.
                        if line.startswith('##'):
                            metadata_lines.append(line)

                all_metadata_lines.append((key, metadata_lines))

        return all_metadata_lines

    @memoized_property
    def outcome_stratified_lengths(self):
        return knock_knock.lengths.OutcomeStratifiedLengths.from_file(self.fns['outcome_stratified_lengths'])

    @memoized_property
    def qname_to_inferred_length(self):
        qname_to_inferred_length = {}
        for outcome in self.outcome_iter():
            qname_to_inferred_length[outcome.query_name] = outcome.inferred_amplicon_length

        return qname_to_inferred_length

    @memoized_property
    def total_reads(self):
        return self.read_lengths.sum()

    @memoized_property
    def outcome_highest_points(self):
        return self.outcome_stratified_lengths.outcome_highest_points(smooth_window=self.length_plot_smooth_window)

    @memoized_property
    def expected_lengths(self):
        ti = self.target_info

        expected_lengths = {
            'WT': ti.amplicon_length,
        }

        if ti.clean_HDR_length is not None:
            expected_lengths['intended\nedit'] = ti.clean_HDR_length

        return expected_lengths

    def plot_outcome_stratified_lengths(self, **kwargs):
        kwargs = kwargs.copy()
        kwargs.setdefault('smooth_window', self.length_plot_smooth_window)

        return knock_knock.visualize.lengths.plot_outcome_stratified_lengths(self.outcome_stratified_lengths,
                                                                             self.categorizer,
                                                                             length_ranges=self.length_ranges,
                                                                             expected_lengths=self.expected_lengths,
                                                                             x_tick_multiple=self.x_tick_multiple,
                                                                             **kwargs,
                                                                            )

    def alignment_groups_to_diagrams(self,
                                     alignment_groups,
                                     num_examples,
                                     **diagram_kwargs,
                                    ):
        subsample = utilities.reservoir_sample(alignment_groups, num_examples)

        kwargs = {**diagram_kwargs}
        
        for qname, als in subsample:

            if isinstance(als, dict):
                kwargs['read_label'] = 'sequencing read pair'
                layout = layout_module.NonoverlappingPairLayout(als['R1'], als['R2'], self.target_info)
            else:
                layout = self.categorizer(als, self.target_info, mode=self.layout_mode)

            try:
                diagram = layout.plot(title='', **kwargs)
            except:
                print(self.sample_name, qname)
                raise
                
            yield diagram

    def generate_length_range_figures(self, specific_outcome=None, num_examples=1):
        by_length = defaultdict(lambda: utilities.ReservoirSampler(num_examples))

        al_groups = self.alignment_groups(outcome=specific_outcome)

        for name, als in al_groups:
            length = self.qname_to_inferred_length[name]
            if length == -1:
                length = self.length_to_store_unknown
            by_length[length].add((name, als))

        if specific_outcome is None:
            fns = self.fns
        else:
            fns = self.outcome_fns(specific_outcome)

        fig_dir = fns['length_ranges_dir']
            
        if fig_dir.is_dir():
            shutil.rmtree(str(fig_dir))

        # parents=True is necessary to handle 'too short' reads.
        fig_dir.mkdir(parents=True)

        if specific_outcome is not None:
            description = ': '.join(specific_outcome)
        else:
            description = 'Generating length-specific diagrams'

        items = self.progress(by_length.items(), desc=description, total=len(by_length))

        for length, sampler in items:
            als = sampler.sample
            diagrams = self.alignment_groups_to_diagrams(als, num_examples=num_examples)
            im = hits.visualize.make_stacked_Image([d.fig for d in diagrams])
            fn = fns['length_range_figure'](length, length)
            im.save(fn)

    def generate_all_outcome_length_range_figures(self):
        categories = sorted(self.categories_by_frequency)
        description = 'Generating outcome-specific length range diagrams'
        for category in self.progress(categories, desc=description):
            self.generate_length_range_figures(specific_outcome=category)

    def generate_outcome_browser(self, min_total_to_label=0.01):
        svg.decorate_outcome_browser(self, min_total_to_label=min_total_to_label)

    def generate_example_diagrams(self):
        self.generate_all_outcome_length_range_figures()
        self.generate_all_outcome_example_figures()

    def generate_summary_figures(self):
        lengths_fig = self.length_distribution_figure()
        lengths_fig.savefig(self.fns['lengths_figure'], bbox_inches='tight')

        self.generate_outcome_browser()

    def example_diagrams(self, outcome, num_examples):
        al_groups = self.alignment_groups(outcome=outcome)
        diagrams = self.alignment_groups_to_diagrams(al_groups, num_examples=num_examples)
        return diagrams
        
    def generate_outcome_example_figures(self, outcome, num_examples, **kwargs):
        if isinstance(outcome, tuple):
            description = ': '.join(outcome)
        else:
            description = outcome

        def fig_to_img_tag(fig):
            URI, width, height = table.fig_to_png_URI(fig)
            plt.close(fig)
            tag = f"<img src={URI} class='center'>"
            return tag

        diagrams = self.example_diagrams(outcome, num_examples)
        
        outcome_fns = self.outcome_fns(outcome)
        outcome_dir = outcome_fns['dir']
        if not outcome_dir.is_dir():
            outcome_dir.mkdir()

        fn = outcome_fns['diagrams_html']
        # TODO: this should be a jinja2 template.
        with fn.open('w') as fh:
            fh.write(dedent(f'''\
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
                max-height: 100%;
                max-width: 100%;
                height: auto;
                width: auto;
                }}

                </style>
                </head>
                <body>
                '''))
            fh.write(f'<h2>{self.batch_name}: {self.sample_name}</h1>\n')
            fh.write(f'<h2>{description}</h2>\n')
            
            fig = self.length_distribution_figure(outcome=outcome)
            if fig is not None:
                tag = fig_to_img_tag(fig)
                fh.write(f'{tag}\n<hr>\n')
                
            for i, diagram in enumerate(self.progress(diagrams, desc=description, total=num_examples, leave=False)):
                if i == 0:
                    diagram.fig.savefig(outcome_fns['first_example'], bbox_inches='tight')

                fh.write(f'<p>{diagram.query_name}</p>\n')
                tag = fig_to_img_tag(diagram.fig)
                fh.write(f'{tag}\n')

    def generate_all_outcome_example_figures(self, num_examples=10, **kwargs):
        categories = sorted(set(c for c, s in self.categories_by_frequency))
        for outcome in self.progress(categories, desc='Making diagrams for grouped categories'):
            self.generate_outcome_example_figures(outcome=outcome, num_examples=num_examples, **kwargs)

        subcategories = sorted(self.categories_by_frequency)
        for outcome in self.progress(subcategories, desc='Making diagrams for detailed subcategories'):
            self.generate_outcome_example_figures(outcome=outcome, num_examples=num_examples, **kwargs)
        
    def explore(self, by_outcome=True, **kwargs):
        explorer = explore.SingleExperimentExplorer(self, by_outcome, **kwargs)
        return explorer.layout

    def get_read_layout(self, read_id, qname_to_als=None, fn_key='bam_by_name', outcome=None, read_type=None):
        # qname_to_als is to allow caching of many sets of als (e.g. for all
        # of a particular outcome category) to prevent repeated lookup
        if qname_to_als is None:
            als = self.get_read_alignments(read_id, fn_key=fn_key, outcome=outcome, read_type=read_type)
        else:
            als = qname_to_als[read_id]

        layout = self.categorizer(als, self.target_info, mode=self.layout_mode)

        return layout

    def extract_donor_microhomology_lengths(self):
        MH_lengths = defaultdict(lambda: np.zeros(10000, int))

        for outcome in self.outcome_iter():
            category_and_sides = []

            if outcome.details == 'n/a':
                # no_overlap categorization doesn't record integration details
                continue

            if outcome.category == 'incomplete HDR':
                if outcome.subcategory == "5' HDR, 3' imperfect":
                    category_and_sides.append(('incomplete HDR, 3\' junction', 3))
                elif outcome.subcategory == "5' imperfect, 3' HDR":
                    category_and_sides.append(('incomplete HDR, 5\' junction', 5))

            elif outcome.category == 'donor fragment':
                integration = outcome_record.Integration.from_string(outcome.details)
                strand = integration.donor_strand
                category_and_sides.append((f'donor fragment, {strand}, 5\' junction', 5))
                category_and_sides.append((f'donor fragment, {strand}, 3\' junction', 3))
                
            if len(category_and_sides) > 0:
                integration = outcome_record.Integration.from_string(outcome.details)
                junctions = {
                    5: integration.mh_length_5,
                    3: integration.mh_length_3,
                }

                for category, side in category_and_sides:
                    length = junctions[side]
                    if length >= 0:
                        MH_lengths[category][length] += 1

            #elif outcome.category == 'non-homologous donor' and outcome.subcategory == 'simple':
            #    strand, NH_5, NH_3 = outcome.details.split(',')

            #    NH_junctions = {
            #        5: int(NH_5),
            #        3: int(NH_3),
            #    }

            #    for junction in [5, 3]:
            #        MH_category = f'non-homologous donor, {strand}, {junction}\' junction'
            #        if NH_junctions[junction] >= 0:
            #            MH_lengths[MH_category][NH_junctions[junction]] += 1

        with self.fns['donor_microhomology_lengths'].open('w') as fh:
            for MH_category, lengths in MH_lengths.items():
                fh.write(f'{MH_category}\n')
                counts_string = '\t'.join(map(str, lengths))
                fh.write(f'{counts_string}\n')

    @memoized_property
    def donor_microhomology_lengths(self):
        mh_lengths = {}
        lines = open(self.fns['donor_microhomology_lengths'])
        line_pairs = zip(*[lines]*2)
        for MH_category_line, lengths_line in line_pairs:
            MH_category = MH_category_line.strip()
            lengths = np.array([int(v) for v in lengths_line.strip().split()])
            mh_lengths[MH_category] = lengths

        return mh_lengths

def load_sample_sheet_from_csv(csv_fn):
    csv_fn = Path(csv_fn)

    # Note: can't include comment='#' because of '#' in hex color specifications.
    df = pd.read_csv(csv_fn, index_col='sample', dtype={'figures': str}).replace({np.nan: None})
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

def load_sample_sheet(base_dir, batch):
    data_dir = Path(base_dir) / 'data'
    data_yaml_fn = data_dir / batch / 'sample_sheet.yaml'

    results_dir = Path(base_dir) / 'results'
    results_yaml_fn = results_dir / batch / 'sample_sheet.yaml'

    if data_yaml_fn.exists():
        sample_sheet = yaml.safe_load(data_yaml_fn.read_text())
    elif results_yaml_fn.exists():
        sample_sheet = yaml.safe_load(results_yaml_fn.read_text())
    else:
        csv_fn = data_yaml_fn.with_suffix('.csv')
        if not csv_fn.exists():
            sample_sheet = None
        else:
            sample_sheet = load_sample_sheet_from_csv(csv_fn)

    return sample_sheet

def get_all_batches(base_dir):
    data_dir = Path(base_dir) / 'data'
    batches = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    return batches

def get_combined_sample_sheet(base_dir):
    batches = get_all_batches(base_dir)

    sample_sheets = {}

    for batch in batches:
        sample_sheet = load_sample_sheet(base_dir, batch)

        if sample_sheet is None:
            print(f'Error: {batch} has no sample sheet')
            continue

        sample_sheets[batch] = sample_sheet

    dfs = {k: pd.DataFrame.from_dict(v, orient='index') for k, v in sample_sheets.items()}

    combined = pd.concat(dfs, sort=True)

    return combined

def process_experiment_stage(base_dir,
                             batch_name,
                             sample_name,
                             stage,
                             progress=None,
                             print_timestamps=False,
                            ):
    sample_sheet = load_sample_sheet(base_dir, batch_name)

    if sample_sheet is None:
        print(f'Error: {batch_name} not found in {base_dir}')
        sys.exit(1)
    elif sample_name not in sample_sheet:
        print(f'Error: {sample_name} not found in {batch_name} sample sheet')
        sys.exit(1)
    else:
        description = sample_sheet[sample_name]

    exp_class = get_exp_class(description.get('platform'))
    exp = exp_class(base_dir, batch_name, sample_name, description=description, progress=progress)

    if print_timestamps:
        print(f'{utilities.current_time_string()} Started {batch_name}: {sample_name} {stage}')

    exp.process(stage)

    if print_timestamps:
        print(f'{utilities.current_time_string()} Finished {batch_name}: {sample_name} {stage}')

def get_exp_class(platform):
    if platform == 'illumina':
        from knock_knock.illumina_experiment import IlluminaExperiment
        exp_class = IlluminaExperiment
    elif platform == 'pacbio':
        from knock_knock.pacbio_experiment import PacbioExperiment
        exp_class = PacbioExperiment
    elif platform == 'length_bias':
        from knock_knock.length_bias_experiment import LengthBiasExperiment
        exp_class = LengthBiasExperiment
    else:
        exp_class = Experiment

    return exp_class

def get_all_experiments(base_dir, conditions=None, as_dictionary=True, progress=None, groups_to_exclude=None):
    if conditions is None:
        conditions = {}

    if groups_to_exclude is None:
        groups_to_exclude = set()

    def check_conditions(exp):
        for k, v in conditions.items():
            if not isinstance(v, (list, tuple, set)):
                vs = [v]
            else:
                vs = v

            exp_value = exp.description.get(k)
            if exp_value is None:
                exp_values = []
            else:
                if isinstance(exp_value, str):
                    exp_values = exp_value.split(';')
                else:
                    exp_values = [exp_value]

            if not any(exp_value in vs for exp_value in exp_values):
                return False

        return True

    exps = []
    batches = get_all_batches(base_dir)

    if 'batch' in conditions:
        v = conditions['batch']
        if not isinstance(v, (list, tuple, set)):
            vs = [v]
        else:
            vs = v
        batches = (n for n in batches if n in vs)
    
    for batch in batches:
        sample_sheet = load_sample_sheet(base_dir, batch)

        if sample_sheet is None:
            print(f'Error: {batch} has no sample sheet')
            continue

        for name, description in sample_sheet.items():
            if isinstance(description, str):
                continue

            exp_class = get_exp_class(description.get('platform'))
            
            exp = exp_class(base_dir, batch, name, description=description, progress=progress)
            exps.append(exp)

    filtered = [exp for exp in exps if check_conditions(exp) and exp.batch not in groups_to_exclude]
    if len(filtered) == 0:
        raise ValueError('No experiments met conditions')

    if as_dictionary:
        d = {}
        for exp in filtered:
            d[exp.batch, exp.sample_name] = exp
        
        filtered = d

    return filtered
