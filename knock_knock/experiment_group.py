import logging

from collections import defaultdict

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

    @property
    def experiments(self):
        for identifier in self.all_experiment_ids:
            yield self.experiment(identifier)

    @memoized_property
    def first_experiment(self):
        return next(self.experiments)

    @memoized_property
    def experiment_partitions(self):
        experiment_partitions = {}

        for name, fields_list in self.experiment_id_fields_partitions.items():
            exps = [self.experiment_from_identifier_fields(**fields) for fields in fields_list]
            experiment_partitions[name] = exps

        return experiment_partitions

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

        concat_names = self.Experiment.Identifier.specific_field_names()

        with pd.HDFStore(self.fns['outcome_counts_store'], mode='w') as store:
            all_category_counts = []
            all_subcategory_counts = []

            for partition_i, (partition_name, exps) in enumerate(self.experiment_partitions.items()):
                logger.info(f'Aggregating partition {partition_i}')

                counts = {}
                
                for exp in exps:
                    exp_counts = exp.outcome_counts()
                    if exp_counts is not None:
                        counts[exp.identifier.specific_fields] = exp_counts
                        
                if counts:
                    counts = pd.concat(counts, axis=1, names=concat_names).fillna(0).astype(int)
                    store[partition_name] = counts

                    category_counts = counts.groupby('category', observed=True).sum()
                    subcategory_counts = counts.groupby(['category', 'subcategory'], observed=True).sum()

                    all_category_counts.append(category_counts)
                    all_subcategory_counts.append(subcategory_counts)

            store['category'] = pd.concat(all_category_counts, axis=1).fillna(0).astype(int)
            store['subcategory'] = pd.concat(all_subcategory_counts, axis=1).fillna(0).astype(int)

    @memoized_with_args
    def outcome_counts_store(self, key):
        counts = None

        if (fn := self.fns['outcome_counts_store']).exists():
            with pd.HDFStore(fn, 'r') as store:
                if key in store:
                    counts = store[key]

        if key in ['category', 'subcategory']:
            if counts is not None:
                # Add any missing columns
                all_columns = [experiment_id.specific_fields for experiment_id in self.all_experiment_ids]
                counts = counts.reindex(all_columns, axis=1).fillna(0).astype(int)

        elif key in self.experiment_id_fields_partitions:
            all_fields = self.experiment_id_fields_partitions[key]
            all_columns = pd.MultiIndex.from_tuples([self.experiment_identifier_from_fields(**fields).specific_fields for fields in all_fields], names=self.Experiment.Identifier.specific_field_names())

            if counts is not None:
                # Add any missing columns
                counts = counts.reindex(all_columns, axis=1).fillna(0).astype(int)
            else:
                empty_index = pd.MultiIndex.from_tuples([], names=['category', 'subcategory', 'details'])

                counts = pd.DataFrame(index=empty_index, columns=all_columns, dtype=int)

        else:
            raise ValueError(key)

        return counts

    @memoized_with_kwargs
    def outcome_counts(self, *, level='details', only_relevant=True):
        if level in ['category', 'subcategory']:
            outcome_counts = self.outcome_counts_store(level)
        
        else:
            outcome_counts = []

            for partition_name in self.progress(self.experiment_id_fields_partitions):
                outcome_counts.append(self.outcome_counts_store(partition_name))

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

        try:
            total_reads.name = 'reads'
        except AttributeError:
            # For some subclasses, total_reads could be a scalar
            pass

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