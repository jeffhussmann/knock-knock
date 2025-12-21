import dataclasses
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
        return [field.name for field in dataclasses.fields(type(self).Experiment.Identifier)][1:]

    @classmethod
    def from_identifier_fields(cls, fields, **kwargs):
        return cls(cls.Identifier(**fields), **kwargs)

    def experiment(self, identifier):
        return type(self).Experiment(identifier, experiment_group=self)

    def experiment_identifier_from_fields(self, **fields):
        return type(self).Experiment.Identifier(self.identifier, **fields)

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

            pool.starmap(process_experiment, args)

        self.postprocess(generate_summary_figures=generate_summary_figures)

    def make_outcome_counts(self, level='details'):
        # There can be long tails of outcome details observed only one time
        # (e.g. a specific pattern of sequencing errors) that can make it
        # slow to rely on pandas to merge outcome indexes.
        # Instead, build a DOK sparse matrix.
        # This is a scary potential point of failure that could scramble
        # count to (outcome, experiment) mappings - be careful!

        all_counts = {}

        description = 'Loading outcome counts'
        experiments = self.progress(self.experiments, desc=description)
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

        adata.write_h5ad(self.fns['outcome_counts'])
                                
    @memoized_property
    def outcome_counts_df(self):
        fn = self.fns['outcome_counts']

        if fn.exists():
            adata = anndata.read_h5ad(self.fns['outcome_counts'])
            df = knock_knock.utilities.adata_to_df(adata)
        else:
            df = None

        return df

    @memoized_with_kwargs
    def outcome_counts(self, *, level='details', only_relevant=True):
        outcome_counts = self.outcome_counts_df

        if outcome_counts is not None:
            if only_relevant:
                # See comment in Experiment.outcome_counts
                outcome_counts = outcome_counts.drop(self.categorizer.non_relevant_categories, errors='ignore')

            # Sort columns to avoid annoying pandas PerformanceWarnings.
            outcome_counts.sort_index(axis='columns', inplace=True)

            if level == 'details':
                pass
            else:
                keys = ['category', 'subcategory', 'details']
                keys = keys[:keys.index(level) + 1]

                outcome_counts = outcome_counts.groupby(keys, observed=True).sum()

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
