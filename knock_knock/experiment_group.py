import logging

import pandas as pd
import scipy.sparse

from hits import utilities
memoized_property = utilities.memoized_property
memoized_with_args = utilities.memoized_with_args

import knock_knock.parallel

logger = logging.getLogger(__name__)

def process_experiment(Group, identifier, stages, exp_i=None, num_exps=None):
    group = Group(identifier.group_id)

    exp = group.experiment(identifier)

    if exp_i is not None:
        progress_string = f'({exp_i + 1: >7,} / {num_exps: >7,}) '
    else:
        progress_string = ''

    for stage in stages:
        logger.info(f'{progress_string}Starting {identifier} {stage}')
        exp.process(stage=stage)
        logger.info(f'{progress_string}Finished {identifier} {stage}')

def postprocess_group(Group, identifier):
    group = Group(identifier)
    group.postprocess()

class ExperimentGroup:

    def experiment(self, identifier):
        return type(self).Experiment(identifier, experiment_group=self)

    def process(self,
                generate_example_diagrams=False,
                generate_summary_figures=False,
                num_processes=18,
                use_logger_thread=False,
               ):

        self.results_dir.mkdir(exist_ok=True, parents=True)

        pool = knock_knock.parallel.get_pool(num_processes=num_processes,
                                             use_logger_thread=use_logger_thread,
                                             log_dir=self.results_dir,
                                            )

        with pool:

            stages = [
                'preprocess',
                'align',
                'categorize',
            ]

            if generate_example_diagrams:
                stages.append('generate_example_diagrams')

            if generate_summary_figures:
                stages.append('generate_summary_figures')

            args = [
                (
                    type(self),
                    identifier,
                    stages,
                    exp_i,
                    len(self.all_experiment_ids),
                )
                for exp_i, identifier in enumerate(self.all_experiment_ids)
            ]

            pool.starmap(process_experiment, args)

        self.postprocess(generate_summary_figures=generate_summary_figures)

    def make_outcome_counts(self):
        all_counts = {}

        description = 'Loading outcome counts'
        items = self.progress(self.full_condition_to_experiment.items(), desc=description)
        for condition, exp in items:
            if exp.outcome_counts is not None:
                all_counts[condition] = exp.outcome_counts

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

    @memoized_with_args
    def outcome_counts_df(self, collapsed):
        if collapsed:
            prefix = 'collapsed_'
        else:
            prefix = ''

        key = prefix + 'outcome_counts'

        total_counts = self.total_outcome_counts(collapsed)

        if self.fns[key].exists() and total_counts is not None:
            sparse_counts = scipy.sparse.load_npz(self.fns[key])
            df = pd.DataFrame(sparse_counts.toarray(),
                              index=total_counts.index,
                              columns=pd.MultiIndex.from_tuples(self.full_conditions),
                             )

            df.index.names = self.outcome_index_levels
            df.columns.names = self.outcome_column_levels

        else:
            df = None

        return df

    @memoized_with_args
    def total_outcome_counts(self, collapsed):
        if collapsed:
            prefix = 'collapsed_'
        else:
            prefix = ''

        key = prefix + 'total_outcome_counts'

        if self.fns[key].exists():
            counts = pd.read_csv(self.fns[key], header=None, index_col=list(range(len(self.outcome_index_levels))), na_filter=False)
        else:
            counts = None

        return counts