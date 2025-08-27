import itertools
import logging
import multiprocessing
import os
import shutil

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import hits.fastq
import hits.interval
import hits.utilities as utilities

import knock_knock.experiment
import knock_knock.arrayed_experiment_group

import knock_knock.architecture.integrase

memoized_property = utilities.memoized_property

logger = logging.getLogger(__name__)

class Experiment(knock_knock.experiment.Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.architecture_mode = 'nanopore'

        self.read_types = [
            'nanopore_by_name',
        ]

        self.fastq_dir = self.data_dir / self.sample_name

        self.preprocessed_read_type = 'nanopore_by_name'

        self.max_relevant_length = int(self.description.get('max_relevant_length', 10000))
        self.x_tick_multiple = 500
        self.length_plot_smooth_window = 1

        self.max_insertion_length = 20

        self.fns.update({
            'parsimonious_oriented_donor_als': self.results_dir / 'parsimonious_oriented_donor_als.bam',
        })

        #self.supplemental_index_names = []

    @memoized_property
    def categorizer(self):
        return knock_knock.architecture.integrase.Architecture

    @memoized_property
    def expected_lengths(self):
        return {}

    @property
    def fastq_fns(self):
        return sorted(self.fastq_dir.glob('*.fastq.gz'))

    @property
    def reads(self):
        reads = itertools.chain.from_iterable(hits.fastq.reads(fn, up_to_space=True) for fn in self.fastq_fns)

        for i, read in enumerate(reads):
            # samtools sorting appears to be funky with hex names.
            read.name = f'{i:010d}_{read.name}'
            yield read

    def preprocess(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)

        by_name_fn = self.fns_by_read_type['fastq']['nanopore_by_name']

        hits.fastq.ExternalSorter(self.reads, by_name_fn).sort()

    def align(self):
        self.generate_alignments_with_blast(self.preprocessed_read_type)

        self.generate_supplemental_alignments_with_minimap2(read_type=self.preprocessed_read_type,
                                                            report_all=False,
                                                            use_ont_index=True,
                                                           )

        self.combine_alignments(self.preprocessed_read_type)

    def categorize(self):
        self.categorize_outcomes()

        self.generate_outcome_counts()
        self.generate_outcome_stratified_lengths()

        self.record_sanitized_category_names()

    def length_ranges(self, outcome=None):
        interval_length = 25
        starts = np.arange(0, self.max_relevant_length + interval_length, interval_length)

        lengths = self.outcome_stratified_lengths.by_outcome(outcome)

        ranges = []

        for start in starts:
            if sum(lengths[start:start + interval_length]) > 0:
                ranges.append((start, start + interval_length - 1))

        return pd.DataFrame(ranges, columns=['start', 'end'])

    def generate_length_range_figures(self, specific_outcome=None, num_examples=1):
        by_length_range = defaultdict(lambda: utilities.ReservoirSampler(num_examples))
        length_ranges = [hits.interval.Interval(row['start'], row['end']) for _, row in self.length_ranges(specific_outcome).iterrows()]

        if isinstance(specific_outcome, str):
            def is_relevant(outcome):
                return outcome.category == specific_outcome

        elif isinstance(specific_outcome, tuple):
            category, subcategory = specific_outcome

            def is_relevant(outcome):
                return outcome.category == category and outcome.subcategory == subcategory
        
        elif specific_outcome is None:
            def is_relevant(outcome):
                return True

        else:
            raise ValueError(specific_outcome)

        outcomes = filter(is_relevant, self.outcome_iter())

        al_groups = self.alignment_groups(outcome=specific_outcome)

        for outcome, (name, group) in zip(outcomes, al_groups):
            if outcome.query_name != name:
                raise ValueError('iters out of sync')

            if outcome.inferred_amplicon_length >= self.max_relevant_length:
                last_range = length_ranges[-1]
                if last_range.start == self.max_relevant_length:
                    by_length_range[last_range.start, last_range.end].add((name, group))

            else:
                for length_range in length_ranges:
                    if outcome.inferred_amplicon_length in length_range:
                        by_length_range[length_range.start, length_range.end].add((name, group))

        if specific_outcome is None:
            fns = self.fns
        else:
            fns = self.outcome_fns(specific_outcome)

        fig_dir = fns['length_ranges_dir']
            
        if fig_dir.is_dir():
            shutil.rmtree(str(fig_dir))

        fig_dir.mkdir(exist_ok=True, parents=True)

        if specific_outcome is not None:
            description = ': '.join(specific_outcome)
        else:
            description = 'Generating length-specific diagrams'

        items = self.progress(by_length_range.items(), desc=description, total=len(by_length_range))

        for (start, end), sampler in items:
            diagrams = self.alignment_groups_to_diagrams(sampler.sample,
                                                         num_examples=num_examples,
                                                        )
            im = hits.visualize.make_stacked_Image([d.fig for d in diagrams])
            fn = fns['length_range_figure'](start, end)
            im.save(fn)

    def extract_parsimonious_oriented_donor_als(self):
        all_donor_als = []
    
        for qname, als in self.progress(self.alignment_groups()):
            architecture = self.categorizer(als, self.editing_strategy, mode='nanopore')

            if not architecture.is_malformed:
                donor_als = architecture.nonredundant_donor_alignments

                if architecture.sequencing_direction == '-':
                    for al in donor_als:
                        al.is_reverse = not al.is_reverse
                        
                all_donor_als.extend(donor_als)

        with hits.sam.AlignmentSorter(self.fns['parsimonious_oriented_donor_als'], header=self.editing_strategy.header) as fh:
            for al in all_donor_als:
                fh.write(al)
                
class ChunkedExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.chunk_number_to_fastq_fn = {}
        for i, fastq_fn in enumerate(self.fastq_fns):
            # Extract chunk number from systematic nanopore file name format
            # if present, otherwise use enumerate.

            try:
                chunk_number = int(fastq_fn.name.split('.')[0].split('_')[-1]) 
            except ValueError:
                chunk_number = i

            self.chunk_number_to_fastq_fn[chunk_number] = fastq_fn

        for chunk_number in self.chunk_number_to_fastq_fn:
            chunk_exp = self.chunk_experiment(chunk_number)
            chunk_exp.results_dir.mkdir(exist_ok=True, parents=True)

    @memoized_property
    def chunk_numbers(self):
        # To allow detection of chunks when results are present but not
        # fastqs, look at results, not fastqs. This means results need to be
        # made during initial processing.
        return sorted([int(d.name[len('chunk_'):]) for d in self.results_dir.glob('chunk_*') if d.is_dir()])

    def chunk_experiment(self, chunk_number):
        return ChunkExperiment(self, chunk_number)

    @memoized_property
    def chunk_experiments(self):
        return {chunk_number: ChunkExperiment(self, chunk_number) for chunk_number in self.chunk_numbers}

    def outcome_metadata(self):
        return {}

    @property
    def reads(self):
        for chunk_number in self.progress(self.chunk_numbers):
            chunk_exp = self.chunk_experiments[chunk_number]
            yield from chunk_exp.reads

    def outcome_iter(self):
        for chunk_number in self.progress(self.chunk_numbers):
            chunk_exp = self.chunk_experiments[chunk_number]
            yield from chunk_exp.outcome_iter()

    def alignment_groups(self, **kwargs):
        for chunk_number in self.chunk_numbers:
            chunk_exp = self.chunk_experiments[chunk_number]
            yield from chunk_exp.alignment_groups(**kwargs)

    def reads_by_type(self, read_type):
        for chunk_number in self.chunk_numbers:
            chunk_exp = self.chunk_experiments[chunk_number]
            yield from chunk_exp.reads_by_type(read_type)

    def get_read_alignments(self, read_id, **kwargs):
        annotation = ChunkReadAnnotation.from_identifier(read_id)
        chunk_exp = self.chunk_experiments[annotation['chunk_number']]
        return chunk_exp.get_read_alignments(read_id)

    @memoized_property
    def combined_header(self):
        # Is this guaranteed to have all supplementary references even if the first
        # chunk didn't have any reads align to them?
        return self.chunk_experiments[0].combined_header

    def process(self,
                max_chunks=None,
                num_processes=18,
                use_logger_thread=True,
                only_if_new=False,
               ):

        logger, file_handler = knock_knock.utilities.configure_standard_logger(self.results_dir)

        if use_logger_thread:
            process_pool = knock_knock.parallel.PoolWithLoggerThread(num_processes, logger)
        else:
            NICENESS = 3
            process_pool = multiprocessing.Pool(num_processes, maxtasksperchild=1, initializer=os.nice, initargs=(NICENESS,))

        with process_pool:
            arg_tuples = []

            chunks_to_process = self.chunk_numbers[:max_chunks]

            for chunk_number in chunks_to_process:
                arg_tuple = (
                    self.base_dir,
                    self.batch_name,
                    self.sample_name,
                    only_if_new,
                    chunk_number,
                    len(chunks_to_process),
                )

                arg_tuples.append(arg_tuple)

            process_pool.starmap(process_chunk_experiment, arg_tuples)

        self.generate_outcome_counts()
        self.generate_outcome_stratified_lengths()

        logger.removeHandler(file_handler)
        file_handler.close()

chunk_read_annotation_fields = [
    ('chunk_number', '06d'),
    ('read_number', '06d'),
    ('original_name', 's'),
]

ChunkReadAnnotation = hits.annotation.Annotation_factory(chunk_read_annotation_fields)

class ChunkExperiment(Experiment):
    def __init__(self, chunked_experiment, chunk_number):
        self.chunk_number = chunk_number
        self.chunked_experiment = chunked_experiment
        self.fastq_fn = self.chunked_experiment.chunk_number_to_fastq_fn[self.chunk_number]

        self.chunk_name = f'{self.chunk_number:06d}'

        Experiment.__init__(self,
                            chunked_experiment.base_dir,
                            chunked_experiment.batch_name,
                            chunked_experiment.sample_name,
                            progress=self.chunked_experiment.progress,
                           )

    @property
    def reads(self):
        original_reads = hits.fastq.reads(self.fastq_fn, up_to_space=True)

        for i, read in enumerate(original_reads):
            annotation = ChunkReadAnnotation(chunk_number=self.chunk_number, read_number=i, original_name=read.name)
            read.name = str(annotation)
            yield read

    @memoized_property
    def results_dir(self):
        return self.chunked_experiment.results_dir / f'chunk_{self.chunk_name}'

    def load_description(self):
        return self.chunked_experiment.description

def process_chunk_experiment(base_dir,
                             batch_name,
                             sample_name,
                             only_if_new,
                             chunk_number,
                             total_chunks,
                            ):
    chunked_exp = ChunkedExperiment(base_dir, batch_name, sample_name)
    chunk_exp = chunked_exp.chunk_experiment(chunk_number)

    progress_string = f'({chunk_number + 1: >7,} / {total_chunks: >7,})'

    for stage in [
        #'preprocess',
        #'align',
        'categorize',
    ]:
        stage_string = f'{chunk_number} {stage}'
        logger.info(f'{progress_string} Started {stage_string}')

        previously_processed = chunk_exp.fns['outcome_counts'].exists()

        if not(only_if_new and previously_processed):
            chunk_exp.process(stage)

        logger.info(f'{progress_string} Finished {stage_string}')

def convert_sample_sheet(base_dir, sample_sheet_df, batch_name):
    base_dir = Path(base_dir)
    sample_sheet_df = sample_sheet_df.copy()

    batch_dir = base_dir / 'data' / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    valid_supplemental_indices = set(knock_knock.editing_strategy.locate_supplemental_indices(base_dir))

    samples = {}

    strategy_keys = [
        'amplicon_primers',
        'genome',
        'genome_source',
        'extra_sequences',
        'donor',
    ]

    grouped = sample_sheet_df.groupby(strategy_keys)

    samples = {}

    for (amplicon_primers, genome, genome_source, extra_sequences, donor), rows in grouped:
        strategy_name = knock_knock.arrayed_experiment_group.make_default_editing_strategy_name(amplicon_primers, genome, genome_source, extra_sequences, donor)

        supplemental_indices = set()

        for name in [genome, genome_source] + extra_sequences.split(';'):
            if name in valid_supplemental_indices:
                supplemental_indices.add(name)

        if len(supplemental_indices) == 0:
            supplemental_indices.add('hg38')

        supplemental_indices = supplemental_indices & valid_supplemental_indices

        for _, row in rows.iterrows():
            sample_name = row['sample_name']

            samples[sample_name] = {
                'supplemental_indices': ';'.join(supplemental_indices),
                'editing_strategy': strategy_name,
                'experiment_type': 'nanopore',
                'sgRNAs': row['sgRNAs'],
            }

    samples_df = pd.DataFrame.from_dict(samples, orient='index')
    samples_df.index.name = 'sample_name'

    condition_columns = [c for c in sample_sheet_df.columns if c.startswith('condition:')]

    sample_sheet_df = sample_sheet_df.set_index('sample_name')

    for column in condition_columns:
        samples_df[column] = sample_sheet_df[column]

    samples_csv_fn = batch_dir / 'sample_sheet.csv'
    samples_df.to_csv(samples_csv_fn)

    return samples_df

def experiments(base_dir, batch_name, **kwargs):
    base_dir = Path(base_dir)

    sample_sheet = pd.read_csv(base_dir / 'data' / batch_name / 'sample_sheet.csv', index_col='sample_name')

    experiments = {}
    for sample_name in sample_sheet.index:
        experiments[sample_name] = ChunkedExperiment(base_dir, batch_name, sample_name, **kwargs)

    return experiments