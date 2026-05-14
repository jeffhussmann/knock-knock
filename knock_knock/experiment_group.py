import logging

import anndata
import pandas as pd
import scipy.sparse

from hits import utilities
memoized_property = utilities.memoized_property
memoized_with_args = utilities.memoized_with_args
memoized_with_kwargs = utilities.memoized_with_kwargs

import knock_knock.parallel

logger = logging.getLogger(__name__)

def process_experiment(Group, identifier, stages, exp_i=None, num_exps=None):
    group = Group(identifier.group_id)

    exp = group.experiment(identifier)

    if exp_i is not None:
        width = len(str(num_exps))
        progress_string = f'({exp_i + 1: >{width},} / {num_exps: >{width},}) '
    else:
        progress_string = ''

    for stage in stages:
        logger.info(f'{progress_string}Starting ({identifier.summary}), stage {stage}')
        exp.process(stage=stage)
        logger.info(f'{progress_string}Finished ({identifier.summary}), stage {stage}')

def postprocess_group(Group, identifier):
    group = Group(identifier)
    group.postprocess()

class ExperimentGroup:
    @property
    def column_names(self):
        return self.Experiment.Identifier.specific_field_names()

    @classmethod
    def from_identifier_fields(cls, fields, **kwargs):
        return cls(cls.Identifier(**fields), **kwargs)

    def experiment(self, identifier):
        return self.Experiment(identifier, experiment_group=self)

    def experiment_identifier_from_fields(self, **fields):
        return self.Experiment.Identifier(self.identifier, **fields)

    def experiment_from_identifier_fields(self, **fields):
        identifier = self.experiment_identifier_from_fields(**fields)
        return self.experiment(identifier)

    def filtered_experiment_ids(self, filters=None):
        if filters is None:
            filters = {}

        for identifier in self.all_experiment_ids:
            if identifier.passes_filters(filters):
                yield identifier

    def filtered_experiments(self, filters=None):
        for identifier in self.filtered_experiment_ids(filters):
            yield self.experiment(identifier)

    @memoized_property
    def outcome_counts_store_filters(self):
        filters = {
            'all': None
        }

        return filters

    @property
    def experiments(self):
        for identifier in self.all_experiment_ids:
            yield self.experiment(identifier)

    @memoized_property
    def first_experiment(self):
        return next(self.experiments)

    @property
    def categorizer(self):
        return self.first_experiment.categorizer

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
                                             task_name='process',
                                            )

        stages = [
            'preprocess',
            'align',
            'categorize',
        ]

        if generate_example_diagrams:
            stages.append('generate_example_diagrams')

        if generate_summary_figures:
            stages.append('generate_summary_figures')

        all_experiment_ids = list(self.all_experiment_ids)
        args = [
            (
                type(self),
                identifier,
                stages,
                exp_i,
                len(all_experiment_ids),
            )
            for exp_i, identifier in enumerate(all_experiment_ids)
        ]

        with pool:
            logger.info(f'Starting {self.identifier}')

            pool.starmap(process_experiment, args)

            logger.info(f'Finished!')

    def postprocess(self, **kwargs):
        self.make_outcome_counts_store()

    def make_outcome_counts_sparse(self, level='details', filters=None):
        # There can be long tails of outcome details observed only one time
        # (e.g. a specific pattern of sequencing errors) that can make it
        # slow to rely on pandas to merge outcome indexes.
        # Instead, build a DOK sparse matrix.
        # This is a scary potential point of failure that could scramble
        # count to (outcome, experiment) mappings - be careful!

        all_counts = {}

        description = 'Loading outcome counts'
        experiments = self.progress(self.filtered_experiments(filters), desc=description)
        for exp in experiments:
            counts = exp.outcome_counts(level=level, only_relevant=False)
            if counts is not None:
                all_counts[exp.identifier.specific_fields] = counts

        all_outcomes = set()

        for counts in all_counts.values():
            all_outcomes.update(counts.index.values)
            
        outcome_order = sorted(all_outcomes)
        outcome_to_index = {outcome: i for i, outcome in enumerate(outcome_order)}

        all_identifiers = list(identifier.specific_fields for identifier in self.all_experiment_ids)

        sparse_counts = scipy.sparse.dok_matrix((len(outcome_order), len(all_identifiers)), dtype=int)

        description = 'Combining outcome counts'

        for id_i, identifier in enumerate(self.progress(all_identifiers, desc=description)):
            if identifier in all_counts:
                for outcome, count in all_counts[identifier].items():
                    o_i = outcome_to_index[outcome]
                    sparse_counts[o_i, id_i] = count

        var = pd.DataFrame(all_identifiers,
                           columns=self.column_names,
                          )
        var.index = var.index.astype(str) # To avoid anndata warning
                          
        columns = ['category', 'subcategory', 'details']
        columns = columns[:columns.index(level) + 1]

        obs = pd.DataFrame(outcome_order,
                           columns=columns,
                          )
        obs.index = obs.index.astype(str) # To avoid anndata warning

        # Is CSC or CSR better here?
        adata = anndata.AnnData(X=sparse_counts.tocsc(),
                                obs=obs,
                                var=var,
                               )

        adata.write_h5ad(self.fns['outcome_counts_sparse'])

    def make_outcome_counts_store(self):
        logger.info('Aggregating outcome counts')

        empty_index = pd.MultiIndex.from_tuples([], names=['category', 'subcategory', 'details'])

        with pd.HDFStore(self.fns['outcome_counts_store'], mode='w') as store:
            for filter_i, (filter_name, filters) in enumerate(self.outcome_counts_store_filters.items()):
                logger.info(f'Aggregating filter {filter_i}')

                counts = {}
                
                for exp in self.filtered_experiments(filters):
                    exp_counts = exp.outcome_counts()
                    if exp_counts is None:
                        # Should this just be what outcome_counts returns when empty?
                        exp_counts = pd.Series(index=empty_index, dtype=int)

                    counts[exp.identifier.specific_fields] = exp_counts
                        
                counts = pd.concat(counts, axis=1, names=self.Experiment.Identifier.specific_field_names()).fillna(0).astype(int)

                store[filter_name] = counts

    @memoized_with_args
    def outcome_counts_store(self, filter_name):
        # None if fn doesn't exist or filter_name isn't in store.
        counts = None

        fn = self.fns['outcome_counts_store']

        if fn.exists():
            with pd.HDFStore(self.fns['outcome_counts_store'], 'r') as store:
                if filter_name in store:
                    counts = store[filter_name]

        return counts

    @memoized_with_kwargs
    def outcome_counts(self, *, level='details', only_relevant=True):
        outcome_counts = []

        for filter_name in self.outcome_counts_store_filters:
            if (filter_counts := self.outcome_counts_store(filter_name)) is not None:
                if level == 'details':
                    pass
                else:
                    if level == 'category':
                        keys = ['category']
                    elif level == 'subcategory':
                        keys = ['category', 'subcategory']
                    else:
                        raise ValueError

                    filter_counts = filter_counts.groupby(keys, observed=True).sum()

                outcome_counts.append(filter_counts)

        if len(outcome_counts) == 0:
            outcome_counts = None
        else:
            outcome_counts = pd.concat(outcome_counts, axis=1).fillna(0).astype(int)

            if only_relevant:
                # See comment in Experiment.outcome_counts
                outcome_counts = outcome_counts.drop(self.categorizer.non_relevant_categories, errors='ignore')

            # Sort columns to avoid annoying pandas PerformanceWarnings.
            outcome_counts.sort_index(axis='columns', inplace=True)

        return outcome_counts

    @memoized_with_kwargs
    def total_reads(self, *, only_relevant=True):
        total_reads = self.outcome_counts(only_relevant=only_relevant).sum()
        total_reads.name = 'reads'
        return total_reads

    @memoized_with_kwargs
    def outcome_fractions(self, *, level='details', only_relevant=True):
        counts = self.outcome_counts(level=level, only_relevant=only_relevant)

        if counts is not None:
            denominator = self.total_reads(only_relevant=only_relevant)
            fractions = counts / denominator
        
        else:
            fractions = None

        return fractions