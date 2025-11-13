import knock_knock.architecture
import matplotlib
if 'inline' not in matplotlib.get_backend():
    matplotlib.use('Agg')

import dataclasses
import heapq
import logging
import shutil
import sys

from collections import defaultdict, Counter
from contextlib import ExitStack
from itertools import chain
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
from hits.utilities import memoized_property, memoized_with_kwargs

import knock_knock.architecture
import knock_knock.blast
import knock_knock.lengths
import knock_knock.outcome
import knock_knock.editing_strategy
import knock_knock.utilities
import knock_knock.visualize
import knock_knock.visualize.lengths

from . import svg, table, explore

logger = logging.getLogger(__name__)

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

@dataclasses.dataclass(frozen=True)
class Identifier:
    def __post_init__(self):
        for field in dataclasses.fields(self):
            if field.type == Path:
                object.__setattr__(self, field.name, Path(self.__getattribute__(field.name)))

    @classmethod
    def __field_names__(cls):
        return tuple(field.name for field in dataclasses.fields(cls))

    def __iter__(self):
        return iter(dataclasses.astuple(self))

    @property
    def specific_fields(self):
        return tuple(self)[1:]

    @property
    def summary(self):
        return ', '.join(map(str, self.specific_fields))

@dataclasses.dataclass(frozen=True)
class ExperimentIdentifier(Identifier):
    base_dir: Path
    batch_name: str
    sample_name: str

    def __str__(self):
        return f'{self.batch_name}, {self.sample_name}'

class Experiment:
    def __init__(self, identifier, description=None, experiment_group=None, progress=None):
        self.identifier = identifier

        self.experiment_group = experiment_group

        self.progress = knock_knock.utilities.possibly_default_progress(progress)

        if description is None:
            description = self.load_description()

        self.description = description

        self.max_insertion_length = 20

        self.sgRNAs = self.description.get('sgRNAs')
        self.donor = self.description.get('donor')
        self.nonhomologous_donor = self.description.get('nonhomologous_donor')
        self.primer_names = self.description.get('primer_names')
        self.sequencing_start_feature_name = self.description.get('sequencing_start_feature_name')
        self.infer_homology_arms = self.description.get('infer_homology_arms', True)
        self.max_reads = self.description.get('max_reads', None)
        if self.max_reads is not None:
            self.max_reads = int(self.max_reads)

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

        }

        self.chunks_dir = self.results_dir / 'chunks'

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
        return f'{self.__class__.__name__}: {self.identifier}'

    @memoized_property
    def length_to_store_unknown(self):
        return int(self.max_relevant_length * 1.05)

    def load_description(self):
        sample_sheet = load_sample_sheet(self.identifier.base_dir, self.identifier.batch_name)
        return sample_sheet[self.identifier.sample_name]

    @property
    def experiment_type(self):
        return self.description.get('experiment_type', 'HDR')

    @memoized_property
    def categorizer(self):
        return knock_knock.architecture.experiment_type_to_categorizer(self.experiment_type)

    @memoized_property
    def no_overlap_pair_categorizer(self):
        return knock_knock.architecture.experiment_type_to_no_overlap_categorizer(self.experiment_type)

    @property
    def uncommon_read_type(self):
        # Will be overloaded by any subclass that separates out common sequences.
        return self.preprocessed_read_type

    @memoized_property
    def results_dir(self):
        return self.identifier.base_dir / 'results' / self.identifier.batch_name / self.identifier.sample_name

    @memoized_property
    def data_dir(self):
        return self.identifier.base_dir / 'data' / self.identifier.batch_name

    @memoized_property
    def supplemental_indices(self):
        locations = knock_knock.editing_strategy.locate_supplemental_indices(self.identifier.base_dir)
        return {name: locations[name] for name in self.supplemental_index_names}

    @property
    def min_relevant_length(self):
        if knock_knock.utilities.is_one_sided(self.description.get('experiment_type')):
            min_relevant_length = 0
        else:
            min_relevant_length = self.description.get('min_relevant_length')

        return min_relevant_length

    @memoized_property
    def editing_strategy(self):
        strat = knock_knock.editing_strategy.EditingStrategy(self.identifier.base_dir,
                                                             self.editing_strategy_name,
                                                             donor=self.donor,
                                                             nonhomologous_donor=self.nonhomologous_donor,
                                                             sgRNAs=self.sgRNAs,
                                                             primer_names=self.primer_names,
                                                             sequencing_start_feature_name=self.sequencing_start_feature_name,
                                                             supplemental_indices=self.supplemental_indices,
                                                             infer_homology_arms=self.infer_homology_arms,
                                                             min_relevant_length=self.min_relevant_length,
                                                            )

        return strat

    @memoized_property
    def editing_strategy_name(self):
        # Cast to str because yaml parsing produce a non-string.
        return str(self.description['editing_strategy'])

    def check_combined_read_length(self):
        pass

    def make_nonredundant_sequence_fastq(self):
        pass

    @memoized_property
    def fns_by_read_type(self):
        fns = defaultdict(dict)

        for read_type in self.read_types:
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
            logger.warning(f'{self.identifier} {read_type} {fn_source} not found')
            reads = []
        else:
            reads = fastq.reads(fn_source, up_to_space=True)

        return reads
    
    @memoized_property
    def read_lengths(self):
        return self.outcome_stratified_lengths.lengths_for_all_reads

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
            counts[outcome.category, outcome.subcategory, str(outcome.details)] += 1

        counts = pd.Series(counts, dtype=int).sort_values(ascending=False)
        counts.to_csv(counts_fn, mode='a', sep='\t', header=False)

    def generate_outcome_stratified_lengths(self):
        lengths = knock_knock.lengths.OutcomeStratifiedLengths(self.outcome_iter(),
                                                               -self.max_relevant_length,
                                                               self.max_relevant_length,
                                                               self.length_to_store_unknown,
                                                               self.categorizer.non_relevant_categories,
                                                              )
        lengths.to_file(self.fns['outcome_stratified_lengths'])

    @memoized_with_kwargs
    def length_ranges(self, *, outcome=None):
        lengths = self.outcome_stratified_lengths.by_outcome(outcome)

        nonzero, = np.nonzero(lengths)
        ranges = [(i, i) for i in nonzero]

        return ranges

    @memoized_with_kwargs
    def length_to_length_range(self, *, outcome=None):
        length_to_length_range = {}
        
        for start, end in self.length_ranges(outcome=outcome):
            length_range = (start, end)
            for length in range(start, end + 1):
                length_to_length_range[length] = length_range

        length_to_length_range[None] = (self.length_to_store_unknown, self.length_to_store_unknown)
                
        return length_to_length_range
    
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
                        raise ValueError(f'iters out of sync: next read is {read.name}, next uncommmon alignment group is {name}')

                yield name, als

    def query_names(self, read_type=None):
        for qname, als in self.alignment_groups(read_type=read_type):
            yield qname
    
    def generate_alignments_with_blast(self,
                                       read_type=None,
                                       supplemental_index_name=None,
                                       filter_to_discard=None,
                                       reads_per_chunk=10000,
                                      ):
        reads = self.reads_by_type(read_type)

        if read_type is None:
            description = 'Generating alignments'
        else:
            description = f'Generating {read_type} alignments'

        reads = self.progress(reads, desc=description)

        bam_by_name_fns = []

        if supplemental_index_name is None:
            base_bam_by_name_fn = self.fns_by_read_type['primary_bam_by_name'][read_type]
            ref_seqs = self.editing_strategy.reference_sequences
        else:
            base_bam_by_name_fn = self.fns_by_read_type['supplemental_bam_by_name'][read_type, supplemental_index_name]
            fasta_dir = self.supplemental_indices[supplemental_index_name]['fasta']
            ref_seqs = genomes.load_entire_genome(fasta_dir)

        for i, chunk in enumerate(utilities.chunks(reads, reads_per_chunk)):
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
            with pysam.AlignmentFile(base_bam_by_name_fn, 'wb', header=self.editing_strategy.header) as fh:
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

    @memoized_with_kwargs
    def outcome_counts(self, *, level='details', only_relevant=True):
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

            if only_relevant:
                # Exclude reads that are not from the targeted locus (e.g. phiX, 
                # nonspecific amplification products, or cross-contamination
                # from other samples) and therefore are not relevant to the 
                # performance of the editing strategy.
                counts = counts.drop(self.categorizer.non_relevant_categories, errors='ignore')

            if level == 'details':
                pass
            else:
                if level == 'subcategory':
                    keys = ['category', 'subcategory']
                elif level == 'category':
                    keys = ['category']
                else:
                    raise ValueError

                counts = counts.groupby(keys).sum()

        except (FileNotFoundError, pd.errors.EmptyDataError):
            counts = None

        return counts

    @memoized_with_kwargs
    def outcome_fractions(self, *, level='details', only_relevant=True):
        counts = self.outcome_counts(level=level, only_relevant=only_relevant)
        if counts is None:
            fractions = None
        else:
            fractions = counts / counts.sum()

        return fractions

    @memoized_with_kwargs
    def deletion_boundaries(self, *, include_simple_deletions=True, include_edit_plus_deletions=False):
        return knock_knock.outcome.extract_deletion_boundaries(self.editing_strategy,
                                                               self.outcome_fractions,
                                                               include_simple_deletions=include_simple_deletions,
                                                               include_edit_plus_deletions=include_edit_plus_deletions,
                                                              )

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

    def categorize_outcomes(self, fn_key='bam_by_name', read_type=None):
        if self.fns['outcomes_dir'].is_dir():
            shutil.rmtree(self.fns['outcomes_dir'])
           
        self.fns['outcomes_dir'].mkdir()

        outcome_to_qnames = defaultdict(list)

        with self.fns['outcome_list'].open('w') as outcome_fh:
            outcome_fh.write(f'## Generated at {utilities.current_time_string()}\n')

            alignment_groups = self.alignment_groups(fn_key=fn_key, read_type=read_type)

            if read_type is None:
                description = 'Categorizing reads'
            else:
                description = f'Categorizing {read_type} reads'

            to_iter = zip(alignment_groups, self.reads_by_type(self.preprocessed_read_type))

            for (name, als), read in self.progress(to_iter, desc=description):
                if read.qname != name:
                    raise ValueError('iters out of sync')

                if 'UMI_key' in self.description:
                    UMI_annotation = UMIAnnotation.from_identifier(name)
                    UMI_seq = UMI_annotation['UMI_seq']
                    UMI_qual = UMI_annotation['UMI_qual']
                else:
                    UMI_seq = ''
                    UMI_qual = ''

                seq = read.seq

                # Special handling of empty sequence.
                if seq is None:
                    seq = ''

                if seq in self.common_sequence_to_outcome:
                    architecture = self.common_sequence_to_outcome[seq]

                else:
                    architecture = self.categorizer(als,
                                                    self.editing_strategy,
                                                    platform=self.platform,
                                                    error_corrected=self.has_UMIs,
                                                   )

                    try:
                        architecture.categorize()
                    except:
                        print()
                        print(self.identifier, name)
                        raise

                outcome_to_qnames[architecture.category, architecture.subcategory].append(name)

                outcome = knock_knock.outcome.CategorizationRecord.from_architecture(architecture,
                    query_name=name,
                    Q30_fraction=read.Q30_fraction,
                    mean_Q=read.mean_Q,
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

        for outcome in outcome_to_qnames:
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

            elif stage == 'generate_example_diagrams':
                self.generate_example_diagrams()

            elif stage == 'generate_summary_figures':
                self.generate_summary_figures()

            else:
                raise ValueError(f'invalid stage: {stage}')

        except:
            print(self)
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
        architecture = self.get_read_architecture(read_id, outcome=outcome, read_type=read_type)
        architecture.categorize()

        diagram = architecture.plot(relevant=relevant, **kwargs)

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
                label = ': '.join(outcome)
            else:
                label = outcome

            outcome_lengths = self.outcome_stratified_lengths.by_outcome(outcome)
            color = self.outcome_stratified_lengths.outcome_to_color(smooth_window=self.length_plot_smooth_window)[outcome]

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
                title = f"{self.identifier}"
            else:
                category, subcategory = outcome
                title = f"{self.identifier}\n{category}: {subcategory}"

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

                outcome = knock_knock.outcome.CategorizationRecord.from_string(line)

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

    def plot_outcome_stratified_lengths(self, **kwargs):
        kwargs = kwargs.copy()
        kwargs.setdefault('smooth_window', self.length_plot_smooth_window)

        outcome_stratified_lengths = self.outcome_stratified_lengths

        if kwargs.get('truncate_to_max_observed_length', False):
            outcome_stratified_lengths = outcome_stratified_lengths.truncate_to_max_observed_length()

        return knock_knock.visualize.lengths.plot_outcome_stratified_lengths(outcome_stratified_lengths,
                                                                             self.categorizer,
                                                                             self.editing_strategy,
                                                                             length_ranges=self.length_ranges,
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
                architecture = knock_knock.architecture.architecture.NonoverlappingPairArchitecture(als['R1'], als['R2'], self.editing_strategy)
            else:
                architecture = self.categorizer(als, self.editing_strategy, mode=self.platform)

            try:
                diagram = architecture.plot(title='', **kwargs)
            except:
                print(self.sample_name, qname)
                raise
                
            yield diagram

    def generate_length_range_figures(self, specific_outcome=None, num_examples=1):
        length_to_length_range = self.length_to_length_range(outcome=specific_outcome)

        # Downsampling these here is redundant (since alignment_groups_to_diagrams
        # will downsample anyways) but avoids excessive memory usage.
        by_length_range = defaultdict(lambda: utilities.ReservoirSampler(num_examples))

        al_groups = self.alignment_groups(outcome=specific_outcome)

        for name, als in al_groups:
            length = self.qname_to_inferred_length[name]

            length = min(length, self.max_relevant_length)

            length_range = length_to_length_range[length]
            by_length_range[length_range].add((name, als))

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

        items = self.progress(by_length_range.items(), desc=description, total=len(by_length_range))

        for (start, end), sampler in items:
            als = sampler.sample
            diagrams = self.alignment_groups_to_diagrams(als, num_examples=num_examples)
            im = hits.visualize.make_stacked_Image([d.fig for d in diagrams])
            fn = fns['length_range_figure'](start, end)
            im.save(fn)

    def generate_all_outcome_length_range_figures(self):
        description = 'Generating outcome-specific length range diagrams'

        outcomes = self.categories_by_frequency + self.subcategories_by_frequency
        for outcome in self.progress(outcomes, desc=description):
            self.generate_length_range_figures(specific_outcome=outcome)

    def generate_outcome_browser(self, **kwargs):
        svg.decorate_outcome_browser(self, **kwargs)

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
            fh.write(f"<h2>{self.identifier}</h1>\n")
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

    def get_read_architecture(self, read_id, qname_to_als=None, fn_key='bam_by_name', outcome=None, read_type=None):
        # qname_to_als is to allow caching of many sets of als (e.g. for all
        # of a particular outcome category) to prevent repeated lookup
        if qname_to_als is None:
            als = self.get_read_alignments(read_id, fn_key=fn_key, outcome=outcome, read_type=read_type)
        else:
            als = qname_to_als[read_id]

        architecture = self.categorizer(als, self.editing_strategy, platform=self.platform)

        return architecture

def load_sample_sheet_from_csv(csv_fn):
    # Note: can't include comment='#' because of '#' in hex color specifications.
    df = knock_knock.utilities.read_and_sanitize_csv(csv_fn, index_col='sample_name')

    if not df.index.is_unique:
        print(f'Error parsing sample sheet {csv_fn}')
        print(f'Sample names are not unique:')
        print(df[df.index.duplicated()])
        sys.exit(1)

    sample_sheet = pd.DataFrame(df).to_dict(orient='index')

    for name, d in sample_sheet.items():
        if 'forward_primer' in d:
            d['primer_names'] = [d['forward_primer'], d['reverse_primer']]
            d.pop('forward_primer')
            d.pop('reverse_primer')

    return sample_sheet

def load_sample_sheet(base_dir, batch_name):
    base_dir = Path(base_dir)

    data_dir = base_dir / 'data'
    data_yaml_fn = data_dir / batch_name / 'sample_sheet.yaml'

    results_dir = base_dir / 'results'
    results_yaml_fn = results_dir / batch_name / 'sample_sheet.yaml'

    csv_fn = data_yaml_fn.with_suffix('.csv')

    if data_yaml_fn.exists():
        sample_sheet = yaml.safe_load(data_yaml_fn.read_text())

    elif results_yaml_fn.exists():
        sample_sheet = yaml.safe_load(results_yaml_fn.read_text())

    elif csv_fn.exists():
        sample_sheet = load_sample_sheet_from_csv(csv_fn)
    else:
        sample_sheet = None

    return sample_sheet

def get_all_batch_names(base_dir):
    data_dir = Path(base_dir) / 'data'
    batches = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    return batches

def get_combined_sample_sheet(base_dir):
    batches = get_all_batch_names(base_dir)

    sample_sheets = {}

    for batch_name in batches:
        sample_sheet = load_sample_sheet(base_dir, batch_name)

        if sample_sheet is None:
            print(f'Error: {batch_name} has no sample sheet')
            continue

        sample_sheets[batch_name] = sample_sheet

    dfs = {k: pd.DataFrame.from_dict(v, orient='index') for k, v in sample_sheets.items()}

    combined = pd.concat(dfs, sort=True)

    return combined

def process_experiment_stage(identifier,
                             stage,
                             progress=None,
                            ):

    sample_sheet = load_sample_sheet(identifier.base_dir, identifier.batch_name)

    if sample_sheet is None:
        print(f'Error: {identifier.batch_name} not found in {identifier.base_dir}')
        sys.exit(1)
    elif identifier.sample_name not in sample_sheet:
        print(f'Error: {identifier.sample_name} not found in {identifier.batch_name} sample sheet')
        sys.exit(1)
    else:
        description = sample_sheet[identifier.sample_name]

    exp_class = get_exp_class(description.get('platform'))
    exp = exp_class(identifier, description=description, progress=progress)

    exp.process(stage)

def get_exp_class(platform):
    if platform == 'illumina':
        from knock_knock.illumina_experiment import IlluminaExperiment
        exp_class = IlluminaExperiment
    elif platform == 'pacbio':
        from knock_knock.pacbio_experiment import PacbioExperiment
        exp_class = PacbioExperiment
    else:
        exp_class = Experiment

    return exp_class

def get_all_experiments(base_dir,
                        conditions=None,
                        as_dictionary=True,
                        progress=None,
                        batch_names_to_exclude=None,
                       ):

    if conditions is None:
        conditions = {}

    if batch_names_to_exclude is None:
        batch_names_to_exclude = set()

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
    batch_names = get_all_batch_names(base_dir)

    if 'batch' in conditions:
        v = conditions['batch']
        if not isinstance(v, (list, tuple, set)):
            vs = [v]
        else:
            vs = v
        batch_names = (n for n in batch_names if n in vs)
    
    for batch_name in batch_names:
        sample_sheet = load_sample_sheet(base_dir, batch_name)

        if sample_sheet is None:
            print(f'Error: {batch_name} has no sample sheet')
            continue

        for sample_name, description in sample_sheet.items():
            if isinstance(description, str):
                continue

            exp_class = get_exp_class(description.get('platform'))
            
            identifier = ExperimentIdentifier(base_dir, batch_name, sample_name)
            exp = exp_class(identifier, description=description, progress=progress)
            exps.append(exp)

    filtered = [exp for exp in exps if check_conditions(exp) and exp.identifier.batch_name not in batch_names_to_exclude]
    if len(filtered) == 0:
        raise ValueError('No experiments met conditions')

    if as_dictionary:
        d = {}
        for exp in filtered:
            d[exp.identifier.batch_name, exp.identifier.sample_name] = exp
        
        filtered = d

    return filtered
