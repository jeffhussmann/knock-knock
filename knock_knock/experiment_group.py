import logging

import pandas as pd
import scipy.sparse

from hits import utilities
memoized_property = utilities.memoized_property
memoized_with_args = utilities.memoized_with_args

import knock_knock.visualize.stacked
import knock_knock.visualize.rejoining_boundaries
import knock_knock.parallel

logger = logging.getLogger(__name__)

def process_sample(GroupClass, base_dir, batch_name, group_name, sample_name, stages):
    group = GroupClass(base_dir, batch_name, group_name)

    exp = group.sample_name_to_experiment(sample_name, no_progress=True)

    for stage in stages:
        logger.info(f'Starting {sample_name} {stage}')
        exp.process(stage=stage)
        logger.info(f'Finished {sample_name} {stage}')

def postprocess_group(GroupClass, base_dir, batch_name, group_name):
    group = GroupClass(base_dir, batch_name, group_name)
    group.postprocess()

class ExperimentGroup:
    def __init__(self):
        self.fns = {
            'total_outcome_counts': self.results_dir / 'total_outcome_counts.txt',
            'outcome_counts': self.results_dir  / 'outcome_counts.npz',
            'pegRNA_conversion_fractions': self.results_dir / 'pegRNA_conversion_fractions.csv',

            'partial_incorporation_figure_high_threshold': self.results_dir / 'partial_incorporation.png',
            'partial_incorporation_figure_low_threshold': self.results_dir / 'partial_incorporation_low_threshold.png',
            'deletion_boundaries_figure': self.results_dir / 'deletion_boundaries.png',

            'single_flap_rejoining_boundaries_figure': self.results_dir / 'single_flap_rejoining_boundaries.png',
            'single_flap_rejoining_boundaries_figure_normalized': self.results_dir / 'single_flap_rejoining_boundaries_normalized.png',
            'single_flap_rejoining_boundaries_figure_individual_samples': self.results_dir / 'single_flap_rejoining_boundaries_individual_samples.png',
            'single_flap_rejoining_boundaries_figure_individual_samples_normalized': self.results_dir / 'single_flap_rejoining_boundaries_individual_samples_normalized.png',
        }

    def process(self, generate_example_diagrams=False, num_processes=18, use_logger_thread=False):
        self.results_dir.mkdir(exist_ok=True, parents=True)

        pool = knock_knock.parallel.get_pool(num_processes=num_processes, use_logger_thread=use_logger_thread, log_dir=self.results_dir)

        with pool:

            stages = [
                'preprocess',
                'align',
                'categorize',
            ]

            if generate_example_diagrams:
                stages.append('generate_example_diagrams')

            stages.append('generate_summary_figures')

            args = [(type(self), self.base_dir, self.batch_name, self.group_name, sample_name, stages) for sample_name in self.sample_names]
            pool.starmap(process_sample, args)

        self.postprocess()

    def postprocess(self):
        self.make_outcome_counts()
        self.write_pegRNA_conversion_fractions()
        self.make_group_figures()

    def make_group_figures(self):
        try:
            for fn_key, kwargs in [
                (
                    'partial_incorporation_figure_high_threshold',
                    dict(
                        frequency_cutoff=1e-2,
                    ),
                ),
                (
                    'partial_incorporation_figure_low_threshold',
                    dict(
                        frequency_cutoff=2e-3,
                    ),
                ),
            ]:

                grid = self.make_partial_incorporation_figure(condition_labels='with keys', **kwargs)

                if grid is not None:
                    grid.fig.savefig(self.fns[fn_key], dpi=200, bbox_inches='tight')

        except Exception as e:
            logger.warning(f'Failed to make partial incorporation figure for {self}')
            logger.warning(e)

        try:
            grid = self.make_deletion_boundaries_figure()

            if grid is not None:
                grid.fig.savefig(self.fns['deletion_boundaries_figure'], dpi=200, bbox_inches='tight')

        except Exception as e:
            logger.warning(f'Failed to make deletion boundaries figure for {self}')
            logger.warning(e)

        if len(self.editing_strategy.pegRNAs) == 1:
            try:
                for fn_key, kwargs in [
                    (
                        'single_flap_rejoining_boundaries_figure',
                        dict(
                            include_genome=False,
                        ),
                    ),
                    (
                        'single_flap_rejoining_boundaries_figure_normalized',
                        dict(
                            include_genome=False,
                            normalize=True,
                        ),
                    ),
                    (
                        'single_flap_rejoining_boundaries_figure_individual_samples',
                        dict(
                            include_genome=False,
                            aggregate_replicates=False,
                        ),
                    ),
                    (
                        'single_flap_rejoining_boundaries_figure_individual_samples_normalized',
                        dict(
                            include_genome=False,
                            aggregate_replicates=False,
                            normalize=True,
                        ),
                    ),
                ]:

                    fig, axs = self.make_single_flap_extension_chain_edge_figure(**kwargs)
                    fig.savefig(self.fns[fn_key], dpi=200, bbox_inches='tight')

            except Exception as e:
                logger.warning(f'Failed to make flap rejoining boundaries figures for {self}')
                logger.warning(e)

    def make_partial_incorporation_figure(self,
                                          unique_colors=False,
                                          min_reads=None,
                                          conditions=None,
                                          condition_colors=None,
                                          condition_labels=None,
                                          **kwargs,
                                         ):
        if conditions is None:
            conditions = self.full_conditions

        if condition_colors is None:
            condition_colors = self.condition_colors(unique=unique_colors)

        if min_reads is not None:
            conditions = [c for c in conditions if self.total_valid_reads.loc[c] >= min_reads]

        if condition_labels == 'with keys':
            condition_labels = self.condition_labels_with_keys
        elif condition_labels is None:
            condition_labels = self.condition_labels

        grid = knock_knock.visualize.stacked.make_partial_incorporation_figure(self.editing_strategy,
                                                                               self.outcome_fractions(),
                                                                               self.pegRNA_conversion_fractions,
                                                                               conditions=conditions,
                                                                               condition_colors=condition_colors,
                                                                               condition_labels=condition_labels,
                                                                               **kwargs,
                                                                              )

        return grid

    def make_deletion_boundaries_figure(self,
                                        unique_colors=False,
                                        min_reads=None,
                                        conditions=None,
                                        condition_colors=None,
                                        condition_labels=None,
                                        **kwargs,
                                       ):
        if conditions is None:
            conditions = self.full_conditions

        if min_reads is not None:
            conditions = [c for c in conditions if self.total_reads().loc[c] >= min_reads]

        if condition_colors is None:
            condition_colors = self.condition_colors(unique=unique_colors)

        if condition_labels == 'with keys':
            condition_labels = self.condition_labels_with_keys
        elif condition_labels is None:
            condition_labels = self.condition_labels

        grid = knock_knock.visualize.stacked.make_deletion_boundaries_figure(self.editing_strategy,
                                                                             self.outcome_fractions(),
                                                                             self.deletion_boundaries,
                                                                             conditions=conditions,
                                                                             condition_colors=condition_colors,
                                                                             condition_labels=self.condition_labels,
                                                                             **kwargs,
                                                                            )
        return grid

    def load_single_flap_boundary_properties(self,
                                             aggregate_replicates=True,
                                             palette='tab10',
                                             conditions=None,
                                             min_reads=None,
                                             samples_to_exclude=None,
                                             include_intended_edit=False,
                                             condition_colors=None,
                                             condition_labels=None,
                                            ):
        if samples_to_exclude is None:
            samples_to_exclude = set()

        if condition_colors is None:
            condition_colors = self.condition_colors(palette=palette)

        if condition_labels == 'with keys':
            condition_labels = self.condition_labels_with_keys
        elif condition_labels == 'sample_name':
            condition_labels = self.full_condition_to_sample_name
        elif condition_labels is None:
            condition_labels = self.condition_labels

        if aggregate_replicates and len(self.condition_keys) > 0:
            aggregate_conditions = self.condition_keys
        else:
            aggregate_conditions = None

        bps = knock_knock.visualize.rejoining_boundaries.EfficientBoundaryProperties(self.editing_strategy,
                                                                                     self.outcome_counts(),
                                                                                     aggregate_conditions=aggregate_conditions,
                                                                                     include_intended_edit=include_intended_edit,
                                                                                    )

        if conditions is None:
            conditions = bps.rejoining_counts.columns

        if min_reads is not None:
            # .sum() here handles if these are full conditions
            conditions = [c for c in conditions if self.total_reads().loc[c].sum() >= min_reads]

        columns_to_extract = [
            (condition_labels[condition], [condition], condition_colors[condition])
            for condition in conditions
        ]

        exp_sets = bps.to_exp_sets(columns_to_extract)

        return exp_sets

    def load_dual_flap_boundary_properties(self,
                                           aggregate_replicates=True,
                                           palette='tab10',
                                           conditions=None,
                                           min_reads=None,
                                           samples_to_exclude=None,
                                           condition_colors=None,
                                           condition_labels=None,
                                          ):
        if samples_to_exclude is None:
            samples_to_exclude = set()

        if condition_colors is None:
            condition_colors = self.condition_colors(palette=palette)

        if condition_labels == 'with keys':
            condition_labels = self.condition_labels_with_keys
        elif condition_labels is None:
            condition_labels = self.condition_labels

        if aggregate_replicates and len(self.condition_keys) > 0:
            aggregate_conditions = self.condition_keys
        else:
            aggregate_conditions = None

        bps = knock_knock.visualize.rejoining_boundaries.EfficientDualFlapBoundaryProperties(self.editing_strategy,
                                                                                             self.outcome_counts(),
                                                                                             aggregate_conditions=aggregate_conditions,
                                                                                            )

        if conditions is None:
            conditions = bps.left_rejoining_counts.columns

        if min_reads is not None:
            # .sum() here handles if these are full conditions
            conditions = [c for c in conditions if self.total_valid_reads.loc[c].sum() >= min_reads]

        columns_to_extract = [
            (condition_labels[condition], [condition], condition_colors[condition])
            for condition in conditions
        ]

        exp_sets = bps.to_exp_sets(columns_to_extract)

        return exp_sets

    def make_single_flap_extension_chain_edge_figure(self,
                                                     palette='tab10',
                                                     conditions=None,
                                                     condition_colors=None,
                                                     condition_labels=None,
                                                     aggregate_replicates=True,
                                                     samples_to_exclude=None,
                                                     include_intended_edit=False,
                                                     min_reads=None,
                                                     **plot_kwargs,
                                                    ):
        
        exp_sets = self.load_single_flap_boundary_properties(aggregate_replicates=aggregate_replicates,
                                                             palette=palette,
                                                             conditions=conditions,
                                                             samples_to_exclude=samples_to_exclude,
                                                             include_intended_edit=include_intended_edit,
                                                             condition_colors=condition_colors,
                                                             condition_labels=condition_labels,
                                                             min_reads=min_reads,
                                                            )

        fig, axs = knock_knock.visualize.rejoining_boundaries.plot_single_flap_extension_chain_edges(self.editing_strategy,
                                                                                                     exp_sets,
                                                                                                     **plot_kwargs,
                                                                                                    ) 

        if plot_kwargs.get('include_genome', True):
            if self.editing_strategy.sgRNAs is not None:
                sgRNAs = ','.join(self.editing_strategy.sgRNAs)
            else:
                sgRNAs = 'no sgRNAs'

            fig.suptitle(f'{self.editing_strategy.target} - {sgRNAs}')

        return fig, axs

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