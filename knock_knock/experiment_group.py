import bisect
import datetime
import logging

import pandas as pd
import pysam
import scipy.sparse

from hits import utilities
memoized_property = utilities.memoized_property
memoized_with_args = utilities.memoized_with_args

import knock_knock.outcome_record
import knock_knock.parallel

import knock_knock.common_sequences

def run_stage(GroupClass, group_args, sample_name, stage):
    group = GroupClass(*group_args)

    if sample_name in group.sample_names:
        # Important to do this branch first, since preprocessing happens before common sequence collection.
        exp = group.sample_name_to_experiment(sample_name, no_progress=True)
    elif sample_name in group.common_sequence_chunk_exp_names:
        exp = group.common_sequence_chunk_exp_from_name(sample_name)
    else:
        raise ValueError(sample_name)

    logging.info(f'Starting {sample_name} {stage}')
    exp.process(stage=stage)
    logging.info(f'Finished {sample_name} {stage}')

class ExperimentGroup:
    def __init__(self):
        self.fns = {
            'common_sequences_dir': self.results_dir / 'common_sequences',
            'common_sequence_outcomes': self.results_dir / 'common_sequences' / 'common_sequence_outcomes.txt',
            'common_sequence_special_alignments': self.results_dir / 'common_sequences' / 'all_special_alignments.bam',

            'total_outcome_counts': self.results_dir / 'total_outcome_counts.txt',
            'outcome_counts': self.results_dir  / 'outcome_counts.npz',

            'genomic_insertion_length_distributions': self.results_dir / 'genomic_insertion_length_distribution.txt',
        }

    def process(self, generate_figures=False, num_processes=18, verbose=True):
        self.results_dir.mkdir(exist_ok=True, parents=True)
        log_fn = self.results_dir / f'log_{datetime.datetime.now():%y%m%d-%H%M%S}.out'

        logger = logging.getLogger(__name__)
        logger.propagate = False
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_fn)
        formatter = logging.Formatter(fmt='%(asctime)s: %(message)s',
                                      datefmt='%y-%m-%d %H:%M:%S',
                                     )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        if verbose:
            print(f'Logging in {log_fn}')

        with knock_knock.parallel.PoolWithLoggerThread(num_processes, logger) as pool:
            logger.info('Preprocessing')

            args = [(type(self), self.group_args, sample_name, 'preprocess') for sample_name in self.sample_names]
            pool.starmap(run_stage, args)

            self.make_common_sequences()

            for stage in [
                'align',
                'categorize',
            ]:

                logger.info(f'Processing common sequences, stage {stage}')
                args = [(type(self), self.group_args, chunk_exp_name, stage) for chunk_exp_name in self.common_sequence_chunk_exp_names]
                pool.starmap(run_stage, args)

            self.merge_common_sequence_outcomes()

            stages = [
                'align',
                'categorize',
            ]

            if generate_figures:
                stages.append('generate_figures')

            for stage in stages:
                logger.info(f'Processing unique sequences, stage {stage}')
                args = [(type(self), self.group_args, sample_name, stage) for sample_name in self.sample_names]
                pool.starmap(run_stage, args)

        logger.info('Collecting outcome counts')
        self.make_outcome_counts()

        logger.info('Done!')

        logger.removeHandler(file_handler)
        file_handler.close()

    def make_common_sequences(self):
        ''' Identify all sequences that occur more than once across preprocessed
        reads for all experiments in the group and write them into common sequences
        experiments to be categorized.
        '''
        splitter = knock_knock.common_sequences.CommonSequenceSplitter(self)

        description = 'Collecting common sequences'
        exps = self.experiments(no_progress=True)
        for exp in self.progress(exps, desc=description, total=self.num_experiments):
            reads = exp.reads_by_type(self.preprocessed_read_type)
            splitter.update_counts((read.seq for read in reads))

        splitter.write_files()

    @memoized_property
    def common_sequence_chunk_exp_names(self):
        ''' Names of all common sequence chunk experiments. '''
        return sorted([d.name for d in self.fns['common_sequences_dir'].iterdir() if d.is_dir()])

    def common_sequence_chunk_exps(self):
        ''' Iterator over common sequence chunk experiments. ''' 
        for chunk_name in self.common_sequence_chunk_exp_names:
            yield self.common_sequence_chunk_exp_from_name(chunk_name)

    @memoized_property
    def common_names(self):
        ''' List of all names assigned to common sequence artificial reads. '''
        return [outcome.query_name for outcome in self.common_sequence_outcomes]

    @memoized_property
    def common_sequence_outcomes(self):
        outcomes = []
        for exp in self.common_sequence_chunk_exps():
            for outcome in exp.outcome_iter():
                outcomes.append(outcome)

        return outcomes

    @memoized_property
    def common_name_to_common_sequence(self):
        name_to_seq = {}
        for outcome in self.common_sequence_outcomes:
            name_to_seq[outcome.query_name] = outcome.seq

        return name_to_seq

    @memoized_property
    def common_sequence_to_common_name(self):
        return utilities.reverse_dictionary(self.common_name_to_common_sequence)

    @memoized_property
    def common_name_to_special_alignment(self):
        name_to_al = {}

        if self.fns['common_sequence_special_alignments'].exists():
            for al in pysam.AlignmentFile(self.fns['common_sequence_special_alignments']):
                name_to_al[al.query_name] = al

        return name_to_al

    @memoized_property
    def common_sequence_to_outcome(self):
        common_sequence_to_outcome = {}
        for outcome in self.common_sequence_outcomes:
            common_sequence_to_outcome[outcome.seq] = outcome

        return common_sequence_to_outcome

    def merge_common_sequence_outcomes(self):
        with self.fns['common_sequence_outcomes'].open('w') as fh:
            for outcome in self.common_sequence_outcomes:
                fh.write(f'{outcome}\n')

    @memoized_property
    def name_to_chunk(self):
        chunks = list(self.common_sequence_chunk_exps())
        starts = [int(chunk.sample_name.split('-')[0]) for chunk in chunks]

        def name_to_chunk(name):
            number = int(name.split('_')[0])
            start_index = bisect.bisect(starts, number) - 1 
            chunk = chunks[start_index]
            return chunk

        return name_to_chunk

    def get_common_seq_alignments(self, seq):
        name = self.common_sequence_to_common_name[seq]
        als = self.get_read_alignments(name)
        return als

    @memoized_property
    def common_sequence_to_alignments(self):
        common_sequence_to_alignments = {}
        for chunk_exp in self.common_sequence_chunk_exps():
            for common_name, als in chunk_exp.alignment_groups(read_type='nonredundant'):
                seq = self.common_name_to_common_sequence[common_name]
                common_sequence_to_alignments[seq] = als
        return common_sequence_to_alignments

    def get_read_alignments(self, name):
        if isinstance(name, int):
            name = self.common_names[name]

        chunk = self.name_to_chunk(name)

        als = chunk.get_read_alignments(name, read_type='nonredundant')

        return als

    def get_read_layout(self, name, **kwargs):
        als = self.get_read_alignments(name)
        layout = self.categorizer(als, self.target_info, mode=self.layout_mode, error_corrected=False, **kwargs)
        return layout

    def get_read_diagram(self, read_id, relevant=True, **diagram_kwargs):
        layout = self.get_read_layout(read_id)

        if relevant:
            layout.categorize()
            
        diagram = layout.plot(relevant=relevant, **diagram_kwargs)

        return diagram

    def make_outcome_counts(self):
        all_counts = {}

        description = 'Loading outcome counts'
        items = self.progress(self.full_condition_to_experiment.items(), desc=description)
        for condition, exp in items:
            try:
                all_counts[condition] = exp.outcome_counts
            except (FileNotFoundError, pd.errors.EmptyDataError):
                pass

        all_outcomes = set()

        for counts in all_counts.values():
            all_outcomes.update(counts.index.values)
            
        outcome_order = sorted(all_outcomes)
        outcome_to_index = {outcome: i for i, outcome in enumerate(outcome_order)}

        counts = scipy.sparse.dok_matrix((len(outcome_order), len(self.full_conditions)), dtype=int)

        description = 'Combining outcome counts'
        full_conditions = self.progress(self.full_conditions, desc=description)

        for c_i, condition in enumerate(full_conditions):
            if condition in all_counts:
                for outcome, count in all_counts[condition].items():
                    o_i = outcome_to_index[outcome]
                    counts[o_i, c_i] = count
                
        scipy.sparse.save_npz(self.fns['outcome_counts'], counts.tocoo())

        df = pd.DataFrame(counts.toarray(),
                          columns=self.full_conditions,
                          index=pd.MultiIndex.from_tuples(outcome_order),
                         )

        df.sum(axis=1).to_csv(self.fns['total_outcome_counts'], header=False)

        ## Collapse potentially equivalent outcomes together.
        #collapsed = pd.concat({pg: collapse_categories(df.loc[pg]) for pg in [True, False] if pg in df.index.levels[0]})

        #coo = scipy.sparse.coo_matrix(np.array(collapsed))
        #scipy.sparse.save_npz(self.fns['collapsed_outcome_counts'], coo)

        #collapsed.sum(axis=1).to_csv(self.fns['collapsed_total_outcome_counts'], header=False)

    @memoized_with_args
    def outcome_counts_df(self, collapsed):
        if collapsed:
            prefix = 'collapsed_'
        else:
            prefix = ''

        key = prefix + 'outcome_counts'

        sparse_counts = scipy.sparse.load_npz(self.fns[key])
        df = pd.DataFrame(sparse_counts.toarray(),
                          index=self.total_outcome_counts(collapsed).index,
                          columns=pd.MultiIndex.from_tuples(self.full_conditions),
                         )

        df.index.names = self.outcome_index_levels
        df.columns.names = self.outcome_column_levels

        return df

    @memoized_with_args
    def total_outcome_counts(self, collapsed):
        if collapsed:
            prefix = 'collapsed_'
        else:
            prefix = ''

        key = prefix + 'total_outcome_counts'

        return pd.read_csv(self.fns[key], header=None, index_col=list(range(len(self.outcome_index_levels))), na_filter=False)

    @memoized_property
    def genomic_insertion_length_distributions(self):
        df = pd.read_csv(self.fns['genomic_insertion_length_distributions'], index_col=[0, 1, 2])
        df.columns = [int(c) for c in df.columns]
        return df

class CommonSequencesExperiment:
    @property
    def final_Outcome(self):
        return knock_knock.outcome_record.CommonSequenceOutcomeRecord

    @memoized_property
    def results_dir(self):
        return self.experiment_group.fns['common_sequences_dir'] / self.sample_name

    @memoized_property
    def seq_to_outcome(self):
        return {}

    @memoized_property
    def seq_to_alignments(self):
        return {}

    @memoized_property
    def names_with_common_seq(self):
        return {}

    @property
    def preprocessed_read_type(self):
        return 'nonredundant'

    @property
    def read_types_to_align(self):
        return ['nonredundant']

    def make_nonredundant_sequence_fastq(self):
        pass