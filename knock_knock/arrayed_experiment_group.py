import itertools
import logging
import re
import shutil
import warnings

from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

import knock_knock.build_strategies
import knock_knock.experiment
import knock_knock.experiment_group
import knock_knock.outcome
import knock_knock.illumina_experiment
import knock_knock.parallel
import knock_knock.pacbio_experiment
import knock_knock.editing_strategy
import knock_knock.utilities

import knock_knock.visualize.stacked
import knock_knock.visualize.rejoining_boundaries

from hits import adapters, fastq, sw, utilities

import knock_knock.utilities

memoized_property = utilities.memoized_property
memoized_with_kwargs = utilities.memoized_with_kwargs

logger = logging.getLogger(__name__)

def get_metadata_dir(base_dir):
    return Path(base_dir) / 'metadata'

@dataclass(frozen=True)
class BatchIdentifier(knock_knock.experiment.Identifier):
    base_dir: Path
    batch_name: str

    def __str__(self):
        return f'{self.batch_name}'

class Batch:
    def __init__(self,
                 identifier,
                 baseline_condition=None,
                 progress=None,
                ):

        self.identifier = identifier

        self.data_dir = self.identifier.base_dir / 'data' / self.identifier.batch_name
        self.results_dir = self.identifier.base_dir / 'results' / self.identifier.batch_name

        if progress is None or getattr(progress, '_silent', False):
            def ignore_kwargs(x, **kwargs):
                return x
            progress = ignore_kwargs

        self.progress = progress

        self.baseline_condition = baseline_condition

        self.sample_sheet_fn = self.data_dir / 'sample_sheet.csv'
        # Note: set_index after construction is necessary to force dtype=str for the index.
        self.sample_sheet = pd.read_csv(self.sample_sheet_fn, dtype=str).set_index('sample_name').fillna('').sort_index()

        self.group_descriptions_fn = self.data_dir / 'group_descriptions.csv'
        self.group_descriptions = pd.read_csv(self.group_descriptions_fn, index_col='group').replace({np.nan: None})

        self.fns = {
            'performance_metrics': self.results_dir / 'performance_metrics.csv',
            'group_name_to_sanitized_group_name': self.results_dir / 'group_name_to_sanitized_group_name.csv',
        }

    def __repr__(self):
        return f'Batch: {self.identifier}'

    @property
    def group_names(self):
        return self.sample_sheet['group'].unique()

    @memoized_property
    def sanitized_group_name_to_group_name(self):
        return utilities.reverse_dictionary(self.group_descriptions['sanitized_group_name'])

    def sanitized_group_name_to_group(self, sanitized_group_name):
        if isinstance(sanitized_group_name, int):
            sanitized_group_name = get_sanitized_group_name(sanitized_group_name)

        group_name = self.sanitized_group_name_to_group_name[sanitized_group_name]
        group = self.groups[group_name]

        return group

    def write_sanitized_group_name_lookup_table(self):
        if 'sanitized_group_name' in self.group_descriptions:
            table = self.group_descriptions['sanitized_group_name']
            self.results_dir.mkdir(exist_ok=True, parents=True)
            table.to_csv(self.fns['group_name_to_sanitized_group_name'])

    def write_performance_metrics(self):
        self.performance_metrics.to_csv(self.fns['performance_metrics'])

    def pegRNA_conversion_fractions_fns(self):
        ''' For adding to zips '''
        return self.results_dir.glob('pegRNA_conversion_fractions*')

    def write_pegRNA_conversion_fractions(self):
        for (genome, protospacer), df in self.pegRNA_conversion_fractions.items():
            fn = self.results_dir / f'pegRNA_conversion_fractions_{genome}_{protospacer}.csv'
            df.to_csv(fn)

    def group(self, group_name):
        identifier = GroupIdentifier(self.identifier, group_name)

        platform = self.group_descriptions.loc[group_name].get('platform', 'illumina')

        if platform == 'illumina':
            Group = ArrayedIlluminaExperimentGroup
        elif platform == 'pacbio':
            Group = ArrayedPacbioExperimentGroup
        else:
            raise ValueError

        return Group(identifier,
                     baseline_condition=self.baseline_condition,
                     progress=self.progress,
                    )

    @memoized_property
    def groups(self):
        groups = {group_name: self.group(group_name) for group_name in self.group_names}
        return groups

    @memoized_property
    def groups_by_sanitized_name(self):
        groups = {group.sanitized_group_name: group for group in self.groups.values()}
        return groups

    def group_query(self, query_string):
        groups = []

        for group_name, row in self.group_descriptions.query(query_string).iterrows():
            groups.append(self.groups[group_name])

        return groups

    def experiment_query(self, query_string):
        exps = []

        for sample_name, row in self.sample_sheet.query(query_string).iterrows():
            group = self.groups[row['group']]
            exp = group.sample_name_to_experiment(sample_name)
            exps.append(exp)

        return exps

    def sample_name_to_experiment(self, sample_name):
        exps = self.experiment_query(f'sample_name == "{sample_name}"')

        if len(exps) == 0:
            return None
        elif len(exps) > 1:
            raise ValueError
        else:
            return exps[0]

    @memoized_with_kwargs
    def outcome_counts(self, *, level='details', only_relevant=True):
        counts = {}

        for gn, group in self.groups.items():
            try:
                counts[gn] = group.outcome_counts(level=level, only_relevant=only_relevant)
            except:
                logger.warning(f'No read counts for {gn}')

        counts = pd.concat(counts, axis='columns').fillna(0).astype(int).sort_index()
        counts.columns.names = ['group'] + counts.columns.names[1:]
        
        return counts

    @memoized_with_kwargs
    def outcome_fractions(self, *, level='details', only_relevant=True):
        fs = {}

        for gn, group in self.groups.items():
            try:
                fs[gn] = group.outcome_fractions(level=level, only_relevant=only_relevant)
            except:
                logger.warning(f'No read counts for {gn}')

        fs = pd.concat(fs, axis='columns').fillna(0).sort_index()
        fs.columns.names = ['group'] + fs.columns.names[1:]
        
        return fs

    @memoized_with_kwargs
    def total_reads(self, *, only_relevant=True):
        counts = {}
        for gn, group in self.groups.items():
            try:
                counts[gn] = group.total_reads(only_relevant=only_relevant)
            except:
                logger.warning(f'No read counts for {gn}')

        counts = pd.concat(counts).fillna(0).sort_index()
        counts.index.names = ['group'] + counts.index.names[1:]
        
        return counts

    @memoized_property
    def performance_metrics(self):
        cat_ps = self.outcome_fractions(level='category').T * 100

        individual_columns = [
            'wild type',
            'intended edit',
            'partial edit',
        ]

        individual_column_ps = cat_ps.reindex(columns=individual_columns, fill_value=0)

        individual_column_ps.columns = [f'{n} %' for n in individual_column_ps.columns]

        unintended = 100 - individual_column_ps.sum(axis=1)
        unintended.name = 'all unintended edits %'

        total_reads = self.total_reads(only_relevant=False).copy()
        total_reads.name = 'total reads'

        total_relevant_reads = self.total_reads(only_relevant=True).copy()
        total_relevant_reads.name = 'total relevant reads'

        combined = pd.concat([total_reads, total_relevant_reads, individual_column_ps, unintended], axis=1)

        return combined

    @memoized_property
    def pegRNA_conversion_fractions(self):
        grouped = defaultdict(dict)

        for gn, group in self.groups.items():
            if group.editing_strategy.pegRNA is not None:
                sgRNAs = ' + '.join(group.editing_strategy.sgRNAs)
                genome = group.description['genome']
                protospacer = group.editing_strategy.pegRNA.components['protospacer']

                fn = group.fns['pegRNA_conversion_fractions']

                if fn.exists():
                    df = pd.read_csv(fn, index_col=group.full_condition_keys)
                elif group.pegRNA_conversion_fractions_by_edit_description is not None:
                    df = group.pegRNA_conversion_fractions_by_edit_description.T
                else:
                    df = None

                if df is not None:
                    grouped[genome, protospacer][sgRNAs] = df

        def by_position(description):
            match = re.match(r'\+(\d+)(.+)', description)
            return int(match.group(1)), match.group(2)

        def vectorized_by_position(columns):
            return pd.Index([by_position(description) for description in columns])

        pegRNA_conversion_fractions = {
            (genome, protospacer): pd.concat(data, names=['sgRNAs']).sort_index(axis=1, key=vectorized_by_position)
            for (genome, protospacer), data in grouped.items()
        }

        return pegRNA_conversion_fractions

    def preprocess(self):
        self.write_sanitized_group_name_lookup_table()

    def process(self, generate_example_diagrams=False, num_processes=18, use_logger_thread=False):
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

            stages.append('generate_summary_figures')

            args = []

            for group_name, group in self.groups.items():
                args.extend([(type(group), self.base_dir, self.batch_name, group_name, sample_name, stages) for sample_name in group.sample_names])

            pool.starmap(knock_knock.experiment_group.process_sample, args)

            args = [(type(group), self.base_dir, self.batch_name, group_name) for group_name, group in self.groups.items()]

            pool.starmap(knock_knock.experiment_group.postprocess_group, args)

    def postprocess(self):
        self.write_performance_metrics()
        self.write_pegRNA_conversion_fractions()

def get_batch(base_dir, batch_name, progress=None, **kwargs):
    group_dir = Path(base_dir) / 'data' / batch_name
    group_descriptions_fn = group_dir / 'group_descriptions.csv'

    if group_descriptions_fn.exists():
        identifier = BatchIdentifier(base_dir, batch_name)
        batch = Batch(identifier, progress=progress, **kwargs)
    else:
        batch = None

    return batch

def get_all_batches(base_dir, progress=None, **kwargs):
    possible_batch_dirs = [p for p in (Path(base_dir) / 'data').iterdir() if p.is_dir()]

    batches = {}

    for possible_batch_dir in sorted(possible_batch_dirs):
        batch_name = possible_batch_dir.name
        batch = get_batch(base_dir, batch_name, progress=progress, **kwargs)
        if batch is not None:
            batches[batch_name] = batch

    return batches

def get_all_experiments(base_dir=Path.home() / 'projects' / 'knock_knock', progress=None, conditions=None, **kwargs):
    if conditions is None:
        conditions = {}

    batches = get_all_batches(base_dir, progress, **kwargs)

    exps = {}

    for batch_name, batch in batches.items():
        if 'batch' in conditions and batch_name not in conditions['batch']:
            continue

        for sample_name, row in batch.sample_sheet.iterrows():
            group_name = row['group']

            if 'group' in conditions and group_name not in conditions['group']:
                continue

            group = batch.groups[group_name]

            if 'experiment_type' in conditions and group.experiment_type not in conditions['experiment_type']:
                continue

            exps[batch_name, group_name, sample_name] = group.experiment(sample_name)

    return exps

@dataclass(frozen=True)
class GroupIdentifier(knock_knock.experiment.Identifier):
    batch_id: BatchIdentifier
    group_name: str

    @property
    def base_dir(self):
        return self.batch_id.base_dir

    def __str__(self):
        return f'{self.batch_id}, {self.group_name}'

@dataclass(frozen=True)
class ExperimentIdentifier(knock_knock.experiment.Identifier):
    group_id: GroupIdentifier
    sample_name: str

    @property
    def base_dir(self):
        return self.group_id.base_dir

    def __str__(self):
        return f'{self.group_id}, {self.sample_name}'

class ArrayedExperiment:
    def load_description(self):
        description = {
            **self.experiment_group.description,
            **self.experiment_group.sample_sheet.loc[self.identifier.sample_name],
        }
        return description

    @memoized_property
    def data_dir(self):
        return self.experiment_group.data_dir

    @memoized_property
    def results_dir(self):
        sanitized_sample_name = self.description.get('sanitized_sample_name', self.identifier.sample_name)
        return self.experiment_group.results_dir / sanitized_sample_name

class ArrayedIlluminaExperiment(ArrayedExperiment, knock_knock.illumina_experiment.IlluminaExperiment):
    pass

class ArrayedPacbioExperiment(ArrayedExperiment, knock_knock.pacbio_experiment.PacbioExperiment):
    pass

class ArrayedExperimentGroup(knock_knock.experiment_group.ExperimentGroup):
    def __init__(self,
                 identifier,
                 progress=None,
                 baseline_condition=None,
                 batch=None,
                ):

        self.identifier = identifier

        if progress is None or getattr(progress, '_silent', False):
            def ignore_kwargs(x, **kwargs):
                return x

            progress = ignore_kwargs

        self.silent = True

        self.progress = progress

        if batch is None:
            batch = Batch(self.identifier.batch_id)

        self.batch = batch

        self.sample_sheet = self.batch.sample_sheet.query('group == @self.identifier.group_name').copy()

        self.description = self.batch.group_descriptions.loc[self.identifier.group_name].copy()

        self.sanitized_group_name = self.description.get('sanitized_group_name', self.identifier.group_name)

        if self.description.get('condition_keys') is None:
            self.condition_keys = []
        else:
            self.condition_keys = self.description['condition_keys'].split(';')

        self.full_condition_keys = tuple(self.condition_keys + ['replicate', 'sample'])

        if baseline_condition is not None:
            self.baseline_condition = baseline_condition
        elif self.description.get('baseline_condition') is not None:
            self.baseline_condition = tuple(str(self.description['baseline_condition']).split(';'))
        elif len(self.condition_keys) == 0:
            # If there are no (non-replicate) conditions, let self.baseline_condition
            # be a slice that will include everything.
            self.baseline_condition = pd.IndexSlice[:]
        else:
            raise ValueError('If there are any non-replicate conditions, a baseline condition needs to be specified')

        self.experiment_type = self.description['experiment_type']

        self.outcome_index_levels = ('category', 'subcategory', 'details')
        self.outcome_column_levels = self.full_condition_keys

        def condition_from_row(row):
            condition = tuple(str(row[key]) for key in self.condition_keys)
            if len(condition) == 1:
                condition = condition[0]
            return condition

        def full_condition_from_row(row):
            full_condition = []

            for key in self.full_condition_keys:
                if key == 'replicate':
                    value = int(row[key])
                elif key == 'sample':
                    value = row.name
                else:
                    value = row[key]

                full_condition.append(value)

            return tuple(full_condition)

        self.full_conditions = [full_condition_from_row(row) for _, row in self.sample_sheet.iterrows()]

        conditions_are_unique = len(set(self.full_conditions)) == len(self.full_conditions)
        if not conditions_are_unique:
            print(f'{self}\nconditions are not unique:')
            for k, v in Counter(self.full_conditions).most_common():
                if v > 1:
                    for _, row in self.sample_sheet.iterrows():
                        if full_condition_from_row(row) == k:
                            print(row)
            raise ValueError

        self.full_condition_to_sample_name = {full_condition_from_row(row): sample_name for sample_name, row in self.sample_sheet.iterrows()}

        if len(self.condition_keys) == 0:
            self.conditions = []
        else:
            self.conditions = sorted(set(tuple(condition) for *condition, replicate, sample_name in self.full_conditions))

        # Indexing breaks if it is a length 1 tuple.
        if len(self.condition_keys) == 1:
            self.baseline_condition = self.baseline_condition[0]
            self.conditions = [c[0] for c in self.conditions]

        self.condition_to_sample_names = defaultdict(list)
        for sample_name, row in self.sample_sheet.iterrows():
            condition = condition_from_row(row)
            self.condition_to_sample_names[condition].append(sample_name)

        self.fns = {
            'outcome_counts': self.results_dir  / 'outcome_counts.h5ad',
            'pegRNA_conversion_fractions': self.results_dir / 'pegRNA_conversion_fractions.csv',

            'partial_incorporation_figure_high_threshold': self.results_dir / 'partial_incorporation.png',
            'partial_incorporation_figure_low_threshold': self.results_dir / 'partial_incorporation_low_threshold.png',
            'deletion_boundaries_figure': self.results_dir / 'deletion_boundaries.png',

            'single_flap_rejoining_boundaries_figure': self.results_dir / 'single_flap_rejoining_boundaries.png',
            'single_flap_rejoining_boundaries_figure_normalized': self.results_dir / 'single_flap_rejoining_boundaries_normalized.png',
            'single_flap_rejoining_boundaries_figure_individual_samples': self.results_dir / 'single_flap_rejoining_boundaries_individual_samples.png',
            'single_flap_rejoining_boundaries_figure_individual_samples_normalized': self.results_dir / 'single_flap_rejoining_boundaries_individual_samples_normalized.png',
        }

    def __repr__(self):
        return f'ArrayedExperimentGroup: {self.identifier}'

    @memoized_property
    def sample_names(self):
        return sorted(self.sample_sheet.index)

    @memoized_property
    def all_experiment_ids(self):
        return [ExperimentIdentifier(self.identifier, sample_name) for sample_name in self.sample_names]

    @memoized_property
    def data_dir(self):
        return self.identifier.base_dir / 'data' / self.identifier.batch_id.batch_name

    @memoized_property
    def results_dir(self):
        return self.identifier.base_dir / 'results' / self.identifier.batch_id.batch_name / self.sanitized_group_name

    @property
    def preprocessed_read_type(self):
        return self.first_experiment.preprocessed_read_type

    @property
    def categorizer(self):
        return self.first_experiment.categorizer

    @property
    def platform(self):
        return self.first_experiment.platform

    @property
    def editing_strategy(self):
        return self.first_experiment.editing_strategy

    @memoized_property
    def num_experiments(self):
        return len(self.sample_sheet)

    def condition_replicates(self, condition):
        sample_names = self.condition_to_sample_names[condition]
        return [self.experiment(ExperimentIdentifier(self.identifier, sample_name)) for sample_name in sample_names]

    def condition_query(self, query_string):
        conditions_df = pd.DataFrame(self.full_conditions, index=self.full_conditions, columns=self.full_condition_keys)
        return conditions_df.query(query_string).index

    @memoized_property
    def sanitized_sample_name_to_sample_name(self):
        return utilities.reverse_dictionary(self.sample_sheet['sanitized_sample_name'])

    def sanitized_sample_name_to_experiment(self, sanitized_sample_name):
        if isinstance(sanitized_sample_name, int):
            sanitized_sample_name = get_sanitized_sample_name(sanitized_sample_name)

        sample_name = self.sanitized_sample_name_to_sample_name[sanitized_sample_name]
        identifier = ExperimentIdentifier(self.identifier, sample_name)
        return self.experiment(identifier)

    @memoized_property
    def full_condition_to_experiment(self):
        full_condition_to_experiment = {}

        for full_condition, sample_name in self.full_condition_to_sample_name.items():
            identifier = ExperimentIdentifier(self.identifier, sample_name)
            full_condition_to_experiment[full_condition] = self.experiment(identifier)

        return full_condition_to_experiment

    @memoized_with_kwargs
    def condition_colors(self, *, palette='husl', unique=False):
        if unique:
            condition_colors = {condition: f'C{i}' for i, condition in enumerate(self.full_conditions)}
        else:
            colors = sns.color_palette(palette, n_colors=len(self.conditions))
            condition_colors = dict(zip(self.conditions, colors))

            for full_condition in self.full_conditions:
                *condition, replicate, sample_name = full_condition
                condition = tuple(condition)

                if len(condition) == 0:
                    condition_colors[full_condition] = 'black'
                else:
                    if len(condition) == 1:
                        condition = condition[0]
                    
                    condition_colors[full_condition] = condition_colors[condition]

        return condition_colors

    @memoized_property
    def condition_labels(self):
        condition_labels = {}

        for full_condition in self.full_conditions:
            *condition, replicate, sample_name = full_condition

            num_reads = self.total_reads(only_relevant=True)[full_condition]

            if len(condition) > 0:
                partial_label = ', '.join(condition) + ', '
            else:
                partial_label = ''

            label = f'{partial_label}rep. {replicate} ({sample_name}, {num_reads:,} total relevant reads)'

            condition_labels[full_condition] = label

        for condition in self.conditions:
            if isinstance(condition, str):
                label = condition
            else:
                label = ', '.join(condition)

            condition_labels[condition] = label

        return condition_labels

    @memoized_property
    def condition_labels_with_keys(self):
        condition_labels = {}

        informative_condition_idxs = []

        conditions_array = np.array(self.conditions)
        if conditions_array.ndim == 1 and conditions_array.size > 0:
            conditions_array = conditions_array.reshape((conditions_array.size, -1))

        for c_i, row in enumerate(conditions_array.T):
            if len(set(row)) > 1:
                informative_condition_idxs.append(c_i)

        for full_condition in self.full_conditions:
            *condition, replicate, sample_name = full_condition

            if len(condition) > 0:
                partial_label = ', '.join(f'{self.condition_keys[c_i]}: {full_condition[c_i]}' for c_i in informative_condition_idxs) + ', '
            else:
                partial_label = ''

            label = f'{partial_label}rep.: {replicate}'

            condition_labels[full_condition] = label

        for condition in self.conditions:
            if isinstance(condition, str):
                label = condition
            else:
                label = ', '.join(f'{self.condition_keys[c_i]}: {condition[c_i]}' for c_i in informative_condition_idxs)

            condition_labels[condition] = label

        return condition_labels

    @memoized_with_kwargs
    def outcome_counts(self, *, level='details', only_relevant=True):
        outcome_counts = self.outcome_counts_df(False)

        if outcome_counts is not None:
            if only_relevant:
                # Exclude reads that are not from the targeted locus (e.g. phiX, 
                # nonspecific amplification products, or cross-contamination
                # from other samples) and therefore are not relevant to the 
                # performance of the editing strategy.
                outcome_counts = outcome_counts.drop(self.categorizer.non_relevant_categories, errors='ignore')

            # Sort columns to avoid annoying pandas PerformanceWarnings.
            outcome_counts = outcome_counts.sort_index(axis='columns')

            if level == 'details':
                pass
            else:
                if level == 'subcategory':
                    keys = ['category', 'subcategory']
                elif level == 'category':
                    keys = ['category']
                else:
                    raise ValueError

                outcome_counts = outcome_counts.groupby(keys).sum()

        return outcome_counts

    def group_by_condition(self, df):
        if len(self.condition_keys) == 0:
            # Supplying a constant function to 'by' means
            # all columns will be grouped together. Making
            # this constant value 'all' means that will be
            # the name of eventual aggregated column. 
            kwargs = dict(by=lambda x: 'all')
        else:
            kwargs = dict(level=self.condition_keys)

        return df.T.groupby(**kwargs)

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
            order = fractions[self.baseline_condition].mean(axis='columns').sort_values(ascending=False).index
            fractions = fractions.loc[order]
        
        else:
            fractions = None

        return fractions

    @memoized_with_kwargs
    def outcome_fraction_condition_means(self, *, level='details', only_relevant=True):
        fs = self.outcome_fractions(level=level, only_relevant=only_relevant)
        return self.group_by_condition(fs).mean().T

    @memoized_with_kwargs
    def outcome_fraction_baseline_means(self, *, level='details', only_relevant=True):
        return self.outcome_fraction_condition_means(level=level, only_relevant=only_relevant)[self.baseline_condition]

    @memoized_with_kwargs
    def outcome_fraction_condition_stds(self, *, level='details', only_relevant=True):
        fs = self.outcome_fractions(level=level, only_relevant=only_relevant)
        return self.group_by_condition(fs).std().T

    @memoized_with_kwargs
    def outcomes_by_baseline_frequency(self, *, level='details', only_relevant=True):
        return self.outcome_fraction_baseline_means(level=level, only_relevant=only_relevant).sort_values(ascending=False).index.values

    @memoized_with_kwargs
    def outcome_fraction_differences(self, *, level='details', only_relevant=True):
        fs = self.outcome_fractions(level=level, only_relevant=only_relevant)
        means = self.outcome_fraction_baseline_means(level=level, only_relevant=only_relevant)
        return fs.sub(means, axis=0)

    @memoized_with_kwargs
    def outcome_fraction_difference_condition_means(self, *, level='details', only_relevant=True):
        diffs = self.outcome_fraction_differences(level=level, only_relevant=only_relevant)
        return self.group_by_condition(diffs).mean().T

    @memoized_with_kwargs
    def outcome_fraction_difference_condition_stds(self, *, level='details', only_relevant=True):
        diffs = self.outcome_fraction_differences(level=level, only_relevant=only_relevant)
        return self.group_by_condition(diffs).std().T

    @memoized_with_kwargs
    def log2_fold_changes(self, *, level='details', only_relevant=True):
        # Using the warnings context manager doesn't work here, maybe because of pandas multithreading?
        warnings.filterwarnings('ignore')

        fs = self.outcome_fractions(level=level, only_relevant=only_relevant)
        means = self.outcome_fraction_baseline_means(level=level, only_relevant=only_relevant)

        fold_changes = fs.div(means, axis=0)
        log2_fold_changes = np.log2(fold_changes)

        warnings.resetwarnings()

        return log2_fold_changes

    @memoized_with_kwargs
    def log2_fold_change_condition_means(self, *, level='details', only_relevant=True):
        # Calculate mean in linear space, not log space
        means = self.outcome_fraction_condition_means(level=level, only_relevant=only_relevant)
        baseline_means = self.outcome_fraction_baseline_means(level=level, only_relevant=only_relevant)
        fold_changes = means.div(baseline_means, axis=0)
        return np.log2(fold_changes)

    @memoized_with_kwargs
    def log2_fold_change_condition_stds(self, *, level='details', only_relevant=True):
        # Calculate effective log2 fold change of mean +/- std in linear space
        means = self.outcome_fraction_condition_means(level=level, only_relevant=only_relevant)
        stds = self.outcome_fraction_condition_stds(level=level, only_relevant=only_relevant)
        baseline_means = self.outcome_fraction_baseline_means(level=level, only_relevant=only_relevant)

        return {
            'lower': np.log2((means - stds).div(baseline_means, axis=0)),
            'upper': np.log2((means + stds).div(baseline_means, axis=0)),
        }

    def reassign_outcomes(self, outcomes, reassign_to):
        ''' Returns a copy of category_fractions in which outcome fractions
            for outcomes have been reassigned to to category reassign_to. 
        '''
        cat_fs = self.category_fractions.copy()

        for c, s, d in outcomes:
            if c not in cat_fs.index:
                continue

            cat_fs.loc[c] -= self.outcome_fractions.loc[c, s, d]

            if reassign_to not in cat_fs.index:
                cat_fs[reassign_to] = 0

            cat_fs.loc[reassign_to] += self.outcome_fractions.loc[c, s, d]

        return cat_fs

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

    def postprocess(self, generate_summary_figures=True):
        self.make_outcome_counts()
        self.write_pegRNA_conversion_fractions()

        if generate_summary_figures:
            self.make_group_figures()

    # Duplication of code in pooled_screen
    def donor_outcomes_containing_SNV(self, SNV_name):
        strat = self.editing_strategy
        SNV_index = sorted(strat.donor_SNVs['target']).index(SNV_name)
        donor_base = strat.donor_SNVs['donor'][SNV_name]['base']
        nt_fracs = self.outcome_fraction_baseline_means
        outcomes = [(c, s, d) for c, s, d in nt_fracs.index.values if c == 'donor' and d[SNV_index] == donor_base]
        return outcomes

    @memoized_property
    def conversion_fractions(self):
        conversion_fractions = {}

        SNVs = self.editing_strategy.donor_SNVs['target']

        for SNV_name in SNVs:
            outcomes = self.donor_outcomes_containing_SNV(SNV_name)
            fractions = self.outcome_fractions().loc[outcomes].sum()
            conversion_fractions[SNV_name] = fractions

        conversion_fractions = pd.DataFrame.from_dict(conversion_fractions, orient='index').sort_index()
        
        return conversion_fractions

    @memoized_property
    def outcomes_containing_pegRNA_programmed_edits(self):
        return knock_knock.outcome.outcomes_containing_pegRNA_programmed_edits(self.editing_strategy,
                                                                               self.outcome_fractions(),
                                                                              )
    @memoized_property
    def pegRNA_conversion_fractions(self):
        fs = {}

        for edit_name, outcomes in self.outcomes_containing_pegRNA_programmed_edits.items():
            fs[edit_name] = self.outcome_fractions().loc[outcomes].sum()

        if len(fs) > 0:
            fs_df = pd.DataFrame.from_dict(fs, orient='index')

            fs_df.columns.names = self.full_condition_keys
            fs_df.index.name = 'edit_name'

        else:
            fs_df = None

        return fs_df

    @memoized_property
    def pegRNA_conversion_fractions_by_edit_description(self):
        if self.editing_strategy.pegRNA is None or self.pegRNA_conversion_fractions is None:
            return None
        else:
            def name_to_description(name):
                return self.editing_strategy.pegRNA_programmed_edit_name_to_description.get(name, name)

            df = self.pegRNA_conversion_fractions.copy()

            df.index = df.index.map(name_to_description)

            return df

    def write_pegRNA_conversion_fractions(self):
        ''' Note that this writes transposed. '''
        if self.editing_strategy.pegRNA is not None and self.pegRNA_conversion_fractions is not None:
            self.pegRNA_conversion_fractions_by_edit_description.T.to_csv(self.fns['pegRNA_conversion_fractions'])

    @memoized_with_kwargs
    def deletion_boundaries(self, *, include_simple_deletions=True, include_edit_plus_deletions=False):
        return knock_knock.outcome.extract_deletion_boundaries(self.editing_strategy,
                                                               self.outcome_fractions(),
                                                               include_simple_deletions=include_simple_deletions,
                                                               include_edit_plus_deletions=include_edit_plus_deletions,
                                                              )

    def explore(self, **kwargs):
        import knock_knock.explore
        explorer = knock_knock.explore.ArrayedGroupExplorer(self, **kwargs)
        return explorer.layout

class ArrayedIlluminaExperimentGroup(ArrayedExperimentGroup):
    Experiment = ArrayedIlluminaExperiment

class ArrayedPacbioExperimentGroup(ArrayedExperimentGroup):
    Experiment = ArrayedPacbioExperiment

def sanitize_and_validate_sample_sheet(sample_sheet_fn):
    sample_sheet_df = knock_knock.utilities.read_and_sanitize_csv(sample_sheet_fn)

    # Default to hg38 if genome column isn't present.
    if 'genome' not in sample_sheet_df.columns:
        sample_sheet_df['genome'] = 'hg38'

    if 'extra_sequences' not in sample_sheet_df.columns:
        sample_sheet_df['extra_sequences'] = ''

    if 'donor' not in sample_sheet_df.columns:
        sample_sheet_df['donor'] = ''

    if 'genome_source' not in sample_sheet_df.columns:
        sample_sheet_df['genome_source'] = sample_sheet_df['genome']

    if 'platform' not in sample_sheet_df.columns:
        sample_sheet_df['platform'] = 'illumina'

    # Confirm mandatory columns are present.

    mandatory_columns = [
        'sample_name',
        'amplicon_primers',
        'sgRNAs',
        'genome',
        'genome_source',
        'extra_sequences',
        'donor',
    ]
    
    missing_columns = [col for col in mandatory_columns if col not in sample_sheet_df]
    if len(missing_columns) > 0:
        raise ValueError(f'{missing_columns} column(s) not found in sample sheet')

    if not sample_sheet_df['sample_name'].is_unique:
        counts = sample_sheet_df['sample_name'].value_counts()
        bad_names = ', '.join(f'{name} ({count})' for name, count in counts[counts > 1].items())
        raise ValueError(f'Sample names are not unique: {bad_names}')

    # Since only the final path component of R1 and R2 files will be retained,
    # ensure that these are unique to avoid clobbering.

    if 'R1' in sample_sheet_df and not sample_sheet_df['R1'].apply(lambda fn: Path(fn).name).is_unique:
        raise ValueError(f'R1 files do not have unique names')

    if 'R2' in sample_sheet_df and not sample_sheet_df['R1'].apply(lambda fn: Path(fn).name).is_unique:
        raise ValueError(f'R2 files do not have unique names')
    
    return sample_sheet_df

def make_default_strategy_name(amplicon_primers, genome, genome_source, extra_sequences='', donor='', platform='illumina'):
    strategy_name = f'{amplicon_primers}_{genome}'

    if genome_source != genome:
        strategy_name = f'{strategy_name}_{genome_source}'

    if extra_sequences != '':
        strategy_name = f'{strategy_name}_{extra_sequences}'

    if donor != '':
        strategy_name = f'{strategy_name}_{donor}'

    strategy_name = f'{strategy_name}_{platform}'

    # Names can't contain a forward slash since they are a path component.
    strategy_name = strategy_name.replace('/', '_SLASH_')

    return strategy_name

def make_strategies(base_dir, sample_sheet_df):
    valid_supplemental_indices = set(knock_knock.editing_strategy.locate_supplemental_indices(base_dir))

    strategies = {}

    grouped = sample_sheet_df.groupby(['amplicon_primers', 'genome', 'genome_source', 'donor', 'extra_sequences', 'platform'])

    for (amplicon_primers, genome, genome_source, donor, extra_sequences, platform), rows in grouped:
        all_sgRNAs = set()
        for sgRNAs in rows['sgRNAs']:
            if sgRNAs != '':
                all_sgRNAs.update(sgRNAs.split(';'))

        strategy_name = make_default_strategy_name(amplicon_primers, genome, genome_source, extra_sequences, donor, platform)

        extra_sequences = ';'.join(set(extra_sequences.split(';')) - valid_supplemental_indices)

        strategies[strategy_name] = {
            'genome': genome,
            'genome_source': genome_source,
            'amplicon_primers': amplicon_primers,
            'sgRNAs': ';'.join(all_sgRNAs),
            'extra_sequences': extra_sequences,
            'donor': donor,
        }

    strategies_df = pd.DataFrame.from_dict(strategies, orient='index')
    strategies_df.index.name = 'name'

    strategies_dir = knock_knock.editing_strategy.get_strategies_dir(base_dir)
    strategies_dir.mkdir(parents=True, exist_ok=True)

    strategies_csv_fn = strategies_dir / 'strategies.csv'
    strategies_df.to_csv(strategies_csv_fn)

    knock_knock.build_strategies.build_editing_strategies_from_csv(base_dir)

def detect_sequencing_primers(base_dir, batch_name, sample_sheet_df):

    opposite_side = {
        'R1': 'R2',
        'R2': 'R1',
    }

    expected_adapter_prefixes = {}

    prefix_length = 13

    for adapter_type in ['truseq', 'nextera']:
        expected_adapter_prefixes[adapter_type] = {}

        for side in ['R1', 'R2']:
            adapter_sequence = adapters.primers[adapter_type][opposite_side[side]]
            expected_adapter_prefixes[adapter_type][side] = utilities.reverse_complement(adapter_sequence)[:prefix_length]

    base_dir = Path(base_dir)

    sequencing_adapters = {}
    
    for _, row in sample_sheet_df.iterrows():
        adapter_types = {'R1': 'truseq', 'R2': 'truseq'}

        for which_read in ['R1', 'R2']:
            if which_read in row:
                fastq_fn = base_dir / 'data' / batch_name / Path(row[which_read]).name
                reads = fastq.reads(fastq_fn)

                max_reads_to_check = 10000

                reads = itertools.islice(reads, max_reads_to_check)

                prefix_counts = Counter()

                for read in reads:
                    for adapter_type, prefixes in expected_adapter_prefixes.items():
                        if prefixes[which_read] in read.seq:
                            prefix_counts[adapter_type] += 1

                    if max(prefix_counts.values(), default=0) >= 500:
                        break

                if len(prefix_counts) == 0:
                    adapter_type = 'truseq'
                else:
                    adapter_type, _ = prefix_counts.most_common()[0]

                adapter_types[opposite_side[which_read]] = adapter_type

        sequencing_adapters[row['sample_name']] = f"{adapter_types['R1']};{adapter_types['R2']}" 

    return sequencing_adapters

def detect_sequencing_start_feature_names(base_dir, batch_name, sample_sheet_df):
    base_dir = Path(base_dir)

    sequencing_start_feature_names = {}
    
    for _, row in sample_sheet_df.iterrows():
        R1_fn = base_dir / 'data' / batch_name / Path(row['R1']).name

        if row.get('R2', '') != '':
            R2_fn = base_dir / 'data' / batch_name / Path(row['R2']).name

            # Redundant with stitching code in illumina_experiment, but not sure
            # how to avoid this.
            sequencing_primers = row.get('sequencing_primers', 'truseq')

            if ';' in sequencing_primers:
                R1, R2 = sequencing_primers.split(';')
            else:
                R1, R2 = sequencing_primers, sequencing_primers

            sequencing_primers = {'R1': R1, 'R2': R2}

            reverse_complement = bool(row.get('reverse_complement', False))

            def stitched_reads():
                before_R1 = adapters.primers[sequencing_primers['R1']]['R1']
                before_R2 = adapters.primers[sequencing_primers['R2']]['R2']

                for R1, R2 in fastq.read_pairs(R1_fn, R2_fn, up_to_space=True):
                    if R1.name != R2.name:
                        raise ValueError

                    stitched = sw.stitch_read_pair(R1, R2, before_R1, before_R2, indel_penalty=-1000)

                    if reverse_complement:
                        stitched = stitched.reverse_complement()

                    yield stitched

            reads = stitched_reads()
        
        else:
            reads = fastq.reads(R1_fn)

        strategy_name = make_default_strategy_name(row['amplicon_primers'],
                                                         row['genome'],
                                                         row['genome_source'],
                                                         row['extra_sequences'],
                                                         row['donor'],
                                                        )
        strat = knock_knock.editing_strategy.EditingStrategy(base_dir, strategy_name) 

        primer_sequences = {name: strat.feature_sequence(strat.target, name).upper() for name in strat.primers}

        primer_prefix_length = 6
        read_length_to_examine = 30
        max_reads_to_check = 10000

        primer_prefixes = {name: seq[:primer_prefix_length] for name, seq in primer_sequences.items()}

        reads = itertools.islice(reads, max_reads_to_check)

        prefix_counts = Counter()

        for read in reads:
            for name, prefix in primer_prefixes.items():
                if prefix in read.seq[:read_length_to_examine]:
                    prefix_counts[name] += 1

            if max(prefix_counts.values(), default=0) >= 500:
                break

        if len(prefix_counts) == 0:
            logger.warning(f"Unable to detect sequencing orientation for {row['sample_name']}")
        else:
            feature_name, _ = prefix_counts.most_common()[0]
        
            sequencing_start_feature_names[row['sample_name']] = feature_name
        
    return sequencing_start_feature_names

def get_sanitized_group_name(group_i):
    return f'group{group_i:05d}'

def get_sanitized_sample_name(sample_i):
    return f'sample{sample_i:05d}'

def make_group_descriptions_and_sample_sheet(base_dir, sample_sheet_df, batch_name=None):
    base_dir = Path(base_dir)
    sample_sheet_df = sample_sheet_df.copy()

    if batch_name is None:
        fn_parents = {Path(fn).parent for fn in sample_sheet_df['R1']}

        batch_names = {fn_parent.parts[3] for fn_parent in fn_parents}
        if len(batch_names) > 1:
            raise ValueError(batch_names)
        else:
            batch_name = batch_names.pop()

    batch_dir = base_dir / 'data' / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    sequencing_primers = detect_sequencing_primers(base_dir, batch_name, sample_sheet_df)
    sample_sheet_df['sequencing_primers'] = sample_sheet_df['sample_name'].map(sequencing_primers)

    if 'R1' in sample_sheet_df.columns:
        sequencing_start_feature_names = detect_sequencing_start_feature_names(base_dir, batch_name, sample_sheet_df)
        sample_sheet_df['sequencing_start_feature_name'] = sample_sheet_df['sample_name'].map(sequencing_start_feature_names).fillna('')

    # For each set of editing strategy parameter values, assign the most common
    # sequencing_start_feature_name to all samples.

    group_keys = [
        'amplicon_primers',
        'genome',
        'genome_source',
        'sgRNAs',
        'donor',
        'extra_sequences',
    ]

    grouped = sample_sheet_df.groupby(group_keys)

    keys_to_feature_name = {}

    for keys, rows in grouped:
        if 'sequencing_start_feature_name' not in rows.columns:
            orientations = set()
        else:
            orientations = rows['sequencing_start_feature_name'].value_counts().drop('', errors='ignore')

        if len(orientations) == 0:
            row = rows.iloc[0]

            strategy_name = make_default_strategy_name(row['amplicon_primers'],
                                                       row['genome'],
                                                       row['genome_source'],
                                                       row['extra_sequences'],
                                                       row['donor'],
                                                       row['platform'],
                                                      )

            strat = knock_knock.editing_strategy.EditingStrategy(base_dir, strategy_name) 

            feature_name = sorted(strat.primers)[0]

            logger.warning(f'No sequencing orientations detected for {keys}, arbitrarily choosing {feature_name}')

        else:
            feature_name = orientations.index[0]

        keys_to_feature_name[keys] = feature_name

    for i in sample_sheet_df.index:
        sample_sheet_df.loc[i, 'sequencing_start_feature_name'] = keys_to_feature_name[tuple(sample_sheet_df.loc[i, group_keys])]
            
    # If unedited controls are annotated, make virtual samples.

    if 'is_unedited_control' in sample_sheet_df:
        sample_sheet_df = sample_sheet_df.rename(columns={'is_unedited_control': 'condition:is_unedited_control'})

    if 'condition:is_unedited_control' in sample_sheet_df:
        sample_sheet_df['condition:is_unedited_control'] = sample_sheet_df['condition:is_unedited_control'] == 'unedited'

        amplicon_keys = ['amplicon_primers', 'genome']
        strategy_keys = ['sgRNAs', 'donor', 'extra_sequences']

        grouped_by_amplicon = sample_sheet_df.groupby(amplicon_keys)

        new_rows = []

        for amplicon_strategy, amplicon_rows in grouped_by_amplicon:
            grouped_by_editing_strategy = amplicon_rows.groupby(strategy_keys)

            for strategy_i, (strategy, strategy_rows) in enumerate(grouped_by_editing_strategy):
                for _, row in amplicon_rows.query('`condition:is_unedited_control`').iterrows():
                    augmented_sample_name = f"{row['sample_name']}_{'_'.join(strategy)}"
                    existing_name = Path(row['R1']).name
                    existing_name_stem, extension = existing_name.split('.', 1)
                    link_name = f'{existing_name_stem}_{strategy_i}.{extension}'

                    existing_path = batch_dir / existing_name
                    link = batch_dir / link_name

                    if link.exists():
                        if not link.is_symlink():
                            raise ValueError(link)

                        link.unlink()

                    link.symlink_to(existing_path)

                    new_row = row.copy()
                    new_row['R1'] = str(link)
                    new_row['sample_name'] = augmented_sample_name

                    for key, value in zip(strategy_keys, strategy):
                        new_row[key] = value

                    new_rows.append(new_row)

        new_rows = pd.DataFrame(new_rows)        

        existing_unedited_idxs = sample_sheet_df.query('`condition:is_unedited_control`').index
        sample_sheet_df = pd.concat([sample_sheet_df.drop(existing_unedited_idxs), new_rows])

    valid_supplemental_indices = set(knock_knock.editing_strategy.locate_supplemental_indices(base_dir))

    groups = {}
    samples = {}

    condition_columns = [column for column in sample_sheet_df.columns if column.lower().startswith('condition:')]
    shortened_condition_columns = [column[len('condition:'):] for column in condition_columns]
    if 'group' in shortened_condition_columns:
        raise ValueError('"group" is a reserved column name and can\'t be a condition')

    group_keys = [
        'amplicon_primers',
        'genome',
        'genome_source',
        'sgRNAs',
        'donor',
        'extra_sequences',
        'sequencing_start_feature_name',
        'platform',
    ]

    optional_keys = [
        'trim_to_max_length',
        'UMI_key',
        'sequencing_primers',
        'reverse_complement',
        'max_reads',
        'plate',
        'row',
        'column',
    ]

    grouped = sample_sheet_df.groupby(group_keys)

    for group_i, ((amplicon_primers, genome, genome_source, sgRNAs, donor, extra_sequences, sequencing_start_feature_name, platform), group_rows) in enumerate(grouped):
        strategy_name = make_default_strategy_name(amplicon_primers, genome, genome_source, extra_sequences, donor, platform)

        group_name = f'{strategy_name}_{sgRNAs}'
        group_name = group_name.replace(';', '+')

        sanitized_group_name = get_sanitized_group_name(group_i)

        sanitized_sample_names = {}
        for sample_i, (_, row) in enumerate(group_rows.iterrows()):
            sample_name = row['sample_name']
            sanitized_sample_name = get_sanitized_sample_name(sample_i)

            sanitized_sample_names[sample_name] = sanitized_sample_name

        strat = knock_knock.editing_strategy.EditingStrategy(base_dir, strategy_name, sgRNAs=sgRNAs)

        if 'experiment_type' in group_rows.columns:
            experiment_types = set(group_rows['experiment_type'])

            if len(experiment_types) != 1:
                raise ValueError('more than one experiment type specified')
            else:
                experiment_type = list(experiment_types)[0]

            if knock_knock.utilities.is_one_sided(experiment_type) and strat.pegRNA_names is not None and len(strat.pegRNA_names) == 2:
                experiment_type = f'{experiment_type}_dual_flap'

        elif len(strat.pegRNA_names) == 0 and strat.donor is not None:
            experiment_type = 'HDR'

        elif strat.pegRNA_names is None or len(strat.pegRNA_names) <= 1:
            experiment_type = 'single_flap'

        elif len(strat.pegRNA_names) == 2:
            if donor == '':
                experiment_type = 'dual_flap'
            else:
                experiment_type = 'Bxb1_dual_flap'

        else:
            raise ValueError

        if 'min_relevant_length' in group_rows.columns:
            min_relevant_lengths = set(group_rows['min_relevant_length'])
            if len(min_relevant_lengths) != 1:
                raise ValueError('more than one minimum length specified')
            else:
                min_relevant_length = list(min_relevant_lengths)[0]
        else:
            min_relevant_length = 100

        baseline_condition = ';'.join(map(str, tuple(group_rows[condition_columns].iloc[0])))

        supplemental_indices = set()

        for name in [genome, genome_source] + extra_sequences.split(';'):
            if name in valid_supplemental_indices:
                supplemental_indices.add(name)

        if len(supplemental_indices) == 0:
            supplemental_indices.add('hg38')

        supplemental_indices.add('phiX')

        supplemental_indices = supplemental_indices & valid_supplemental_indices

        groups[group_name] = {
            'sanitized_group_name': sanitized_group_name,
            'supplemental_indices': ';'.join(supplemental_indices),
            'experiment_type': experiment_type,
            'editing_strategy': strategy_name,
            'sequencing_start_feature_name': sequencing_start_feature_name,
            'genome': genome,
            'sgRNAs': sgRNAs,
            'donor': donor,
            'platform': platform,
            'min_relevant_length': min_relevant_length,
            'condition_keys': ';'.join(shortened_condition_columns),
            'baseline_condition': baseline_condition,
        }

        if len(condition_columns) > 0:
            for condition_i, (condition, condition_rows) in enumerate(group_rows.groupby(condition_columns)):
                for rep_i, (_, row) in enumerate(condition_rows.iterrows(), 1):
                    sample_name = row['sample_name']

                    samples[sample_name] = {
                        'sanitized_sample_name': sanitized_sample_names[sample_name],
                        'R1': Path(row['R1']).name,
                        'group': group_name,
                        'replicate': rep_i,
                        'color': (condition_i * len(groups) + group_i), 
                    }

                    for k in ['R2', 'I1', 'I2']:
                        if k in row:
                            samples[sample_name][k] = Path(row[k]).name

                    for k in optional_keys:
                        if k in row:
                            samples[sample_name][k] = row[k]

                    for full, short in zip(condition_columns, shortened_condition_columns):
                        samples[sample_name][short] = row[full]

        else:
            for rep_i, (_, row) in enumerate(group_rows.iterrows(), 1):
                sample_name = row['sample_name']

                samples[sample_name] = {
                    'sanitized_sample_name': sanitized_sample_names[sample_name],
                    'group': group_name,
                    'replicate': rep_i,
                    'color': group_i + 1,
                }

                for k in ['R1', 'R2', 'I1', 'I2', 'CCS_fastq_fn']:
                    if k in row:
                        samples[sample_name][k] = Path(row[k]).name

                for k in optional_keys:
                    if k in row:
                        samples[sample_name][k] = row[k]

    groups_df = pd.DataFrame.from_dict(groups, orient='index')
    groups_df.index.name = 'group'

    groups_csv_fn = batch_dir / 'group_descriptions.csv'
    groups_df.to_csv(groups_csv_fn)

    samples_df = pd.DataFrame.from_dict(samples, orient='index')
    samples_df.index.name = 'sample_name'

    samples_csv_fn = batch_dir / 'sample_sheet.csv'
    samples_df.to_csv(samples_csv_fn)

    return samples_df, groups_df

def setup_from_metadata(base_dir, batch_name, download=False):
    base_dir = Path(base_dir)

    batch_metadata_dir = get_metadata_dir(base_dir) / batch_name

    strategies_dir = knock_knock.editing_strategy.get_strategies_dir(base_dir)
    strategies_dir.mkdir(exist_ok=True)

    sample_sheet_fn = batch_metadata_dir / 'sample_sheet.csv'
    sample_sheet_df = sanitize_and_validate_sample_sheet(sample_sheet_fn)

    extensions_to_copy = {
        '.fasta',
        '.fa',
        '.genbank',
        '.gb',
    }

    fns_to_copy = [fn.name for fn in batch_metadata_dir.iterdir() if fn.suffix in extensions_to_copy]

    for name in ['amplicon_primers.csv', 'sgRNAs.csv', 'donors.csv'] + fns_to_copy:
        src = batch_metadata_dir / name
        dest = strategies_dir / name

        if src.exists():
            shutil.copyfile(src, dest)
    
    if download:
        S3_fns = []

        for k in ['R1', 'R2', 'I1', 'I2']:
            if k in sample_sheet_df.columns:
                S3_fns.extend(list(sample_sheet_df[k]))

        aws_utils.download_s3_paths(base_dir, batch_name, S3_fns)

    make_strategies(base_dir, sample_sheet_df)
    make_group_descriptions_and_sample_sheet(base_dir, sample_sheet_df, batch_name=batch_name)
