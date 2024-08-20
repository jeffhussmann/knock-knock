import gzip
import itertools
import logging
import shutil
import warnings

from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import seaborn as sns

import knock_knock.build_targets
import knock_knock.experiment_group
import knock_knock.outcome
import knock_knock.illumina_experiment
import knock_knock.prime_editing_layout
import knock_knock.twin_prime_layout
import knock_knock.Bxb1_layout
import knock_knock.target_info
import knock_knock.utilities

from hits import adapters, fastq, sw, utilities

import knock_knock.TECseq_layout
import knock_knock.seeseq_layout

memoized_property = utilities.memoized_property
memoized_with_kwargs = utilities.memoized_with_kwargs

class Batch:
    def __init__(self,
                 base_dir,
                 batch_name,
                 category_groupings=None,
                 baseline_condition=None,
                 add_pseudocount=False,
                 only_edited=False,
                 progress=None,
                ):

        self.base_dir = Path(base_dir)
        self.batch_name = batch_name

        self.data_dir = self.base_dir / 'data' / self.batch_name
        self.results_dir = self.base_dir / 'results' / self.batch_name

        if progress is None or getattr(progress, '_silent', False):
            def ignore_kwargs(x, **kwargs):
                return x
            progress = ignore_kwargs

        self.progress = progress

        self.category_groupings = category_groupings
        self.baseline_condition = baseline_condition
        self.add_pseudocount = add_pseudocount
        self.only_edited = only_edited

        self.sample_sheet_fn = self.data_dir / 'sample_sheet.csv'
        # Note: set_index after construction is necessary to force dtype=str for the index.
        self.sample_sheet = pd.read_csv(self.sample_sheet_fn, dtype=str).set_index('sample_name').fillna('')

        self.group_descriptions_fn = self.data_dir / 'group_descriptions.csv'
        self.group_descriptions = pd.read_csv(self.group_descriptions_fn, index_col='group').replace({np.nan: None})

        self.fns = {
            'group_name_to_sanitized_group_name': self.results_dir / 'group_name_to_sanitized_group_name.csv',
        }

    def __repr__(self):
        return f'Batch: {self.batch_name}, base_dir={self.base_dir}'

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

    def group(self, group_name):
        return ArrayedExperimentGroup(self.base_dir,
                                      self.batch_name,
                                      group_name,
                                      category_groupings=self.category_groupings,
                                      baseline_condition=self.baseline_condition,
                                      add_pseudocount=self.add_pseudocount,
                                      only_edited=self.only_edited,
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

    def copy_snapshot(self, new_base_dir,
                      new_batch_name=None,
                      groups_to_include=None,
                      include_target_infos=True,
                     ):
        if new_batch_name is None:
            new_batch_name = self.batch_name

        if groups_to_include is None:
            groups_to_include = {group_name: group_name for group_name in self.groups}

        new_base_dir = Path(new_base_dir)

        # Out of paranoia, make sure that new_base_dir is different
        # than this pool's base_dir since existing dirs will be deleted.
        if str(new_base_dir) == str(self.base_dir):
            raise ValueError('Attempted to copy to same base dir.')

        new_results_dir = new_base_dir / 'results' / new_batch_name
        new_data_dir = new_base_dir / 'data' / new_batch_name

        for new_dir in [new_results_dir, new_data_dir]:
            if new_dir.exists():
                shutil.rmtree(new_dir)
            new_dir.mkdir()

        for new_group_name in groups_to_include.values():
            (new_results_dir / new_group_name).mkdir()

        # Copy relevant results files.
        fns_to_copy = [
            'outcome_counts',
            'total_outcome_counts',
        ]

        for old_group_name, new_group_name in groups_to_include.items():
            old_group = self.groups[old_group_name]
            for fn_key in fns_to_copy:
                old_fn = old_group.fns[fn_key]
                new_fn = new_results_dir / new_group_name / old_fn.name
                shutil.copy(old_fn, new_fn)

        # Copy group descriptions.
        new_group_descriptions = self.group_descriptions.loc[sorted(groups_to_include)].copy()
        new_group_descriptions.index = [groups_to_include[name] for name in new_group_descriptions.index]
        new_group_descriptions.index.name = 'group'

        # Convoluted way of blanking supplmental_indices - '' will be parsed as nan, then coverted to None,
        # then converted to [].
        new_group_descriptions['supplemental_indices'] = ''

        new_group_descriptions.to_csv(new_data_dir / 'group_descriptions.csv')

        # Copy sample sheet.
        new_sample_sheet_fn = new_data_dir / 'sample_sheet.csv'
        new_sample_sheet = self.sample_sheet.query('group in @groups_to_include').copy()
        new_sample_sheet['group'] = new_sample_sheet['group'].replace(groups_to_include)
        new_sample_sheet.to_csv(new_sample_sheet_fn)

        ## Copy the pool sample sheet, wiping any value of supplemental_indices.
        #sample_sheet = copy.deepcopy(self.sample_sheet)
        #sample_sheet['supplemental_indices'] = []
        #new_sample_sheet_fn = new_snapshot_dir / self.sample_sheet_fn.name
        #new_sample_sheet_fn.write_text(yaml.safe_dump(sample_sheet, default_flow_style=False))

        if include_target_infos:
            for old_group_name in groups_to_include:
                old_group = self.groups[old_group_name]

                new_target_info_dir = new_base_dir / 'targets' / old_group.target_info.name

                if new_target_info_dir.exists():
                    shutil.rmtree(new_target_info_dir)

                shutil.copytree(old_group.target_info.dir, new_target_info_dir)

    @memoized_property
    def category_fractions(self):
        fs = {}
        for gn, group in self.groups.items():
            try:
                fs[gn] = group.category_fractions
            except:
                logging.warning(f'No category fractions for {gn}')

        fs = pd.concat(fs, axis='columns').fillna(0).sort_index()
        fs.columns.names = ['group'] + fs.columns.names[1:]

        return fs

    @memoized_property
    def subcategory_fractions(self):
        fs = {gn: group.subcategory_fractions for gn, group in self.groups.items()}
        fs = pd.concat(fs, axis='columns').fillna(0).sort_index()
        fs.columns.names = ['group'] + fs.columns.names[1:]
        return fs

    @memoized_property
    def total_valid_reads(self):
        counts = {}
        for gn, group in self.groups.items():
            try:
                counts[gn] = group.total_valid_reads
            except:
                logging.warning(f'No read counts for {gn}')

        counts = pd.concat(counts).fillna(0).sort_index()
        counts.index.names = ['group'] + counts.index.names[1:]
        
        return counts
    
def get_batch(base_dir, batch_name, progress=None, **kwargs):
    group_dir = Path(base_dir) / 'data' / batch_name
    group_descriptions_fn = group_dir / 'group_descriptions.csv'

    if group_descriptions_fn.exists():
        batch = Batch(base_dir, batch_name, progress=progress, **kwargs)
    else:
        batch = None

    return batch

def get_all_batches(base_dir=Path.home() / 'projects' / 'knock_knock', progress=None, **kwargs):
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

            exps[batch_name, group_name, sample_name] = group.sample_name_to_experiment(sample_name)

    return exps

class ArrayedExperimentGroup(knock_knock.experiment_group.ExperimentGroup):
    def __init__(self,
                 base_dir,
                 batch_name,
                 group_name,
                 category_groupings=None,
                 progress=None,
                 baseline_condition=None,
                 add_pseudocount=None,
                 only_edited=False,
                ):

        self.base_dir = Path(base_dir)
        self.batch_name = batch_name
        self.group_name = group_name

        self.category_groupings = category_groupings
        self.add_pseudocount = add_pseudocount
        self.only_edited = only_edited

        self.group_args = (base_dir, batch_name, group_name)

        if progress is None or getattr(progress, '_silent', False):
            def ignore_kwargs(x, **kwargs):
                return x
            progress = ignore_kwargs

        self.silent = True

        self.progress = progress

        self.batch = Batch(self.base_dir, self.batch_name)

        self.sample_sheet = self.batch.sample_sheet.query('group == @self.group_name').copy()

        self.description = self.batch.group_descriptions.loc[self.group_name].copy()

        self.sanitized_group_name = self.description.get('sanitized_group_name', self.group_name)

        if self.description.get('condition_keys') is None:
            self.condition_keys = []
        else:
            self.condition_keys = self.description['condition_keys'].split(';')
        self.full_condition_keys = tuple(self.condition_keys + ['replicate'])

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
            return tuple(str(row[key]) if key != 'replicate' else int(row[key]) for key in self.full_condition_keys)

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
            self.conditions = sorted(set(c[:-1] for c in self.full_conditions))

        # Indexing breaks if it is a length 1 tuple.
        if len(self.condition_keys) == 1:
            self.baseline_condition = self.baseline_condition[0]
            self.conditions = [c[0] for c in self.conditions]

        self.sample_names = sorted(self.sample_sheet.index)

        self.condition_to_sample_names = defaultdict(list)
        for sample_name, row in self.sample_sheet.iterrows():
            condition = condition_from_row(row)
            self.condition_to_sample_names[condition].append(sample_name)

        super().__init__()

    def __repr__(self):
        return f'ArrayedExperimentGroup: batch={self.batch_name}, group={self.group_name}, type={self.experiment_type}, base_dir={self.base_dir}'

    @memoized_property
    def data_dir(self):
        d = self.base_dir / 'data' / self.batch_name
        return d

    @memoized_property
    def results_dir(self):
        d = self.base_dir / 'results' / self.batch_name / self.sanitized_group_name
        return d

    def experiments(self, no_progress=False):
        for sample_name in self.sample_names:
            yield self.sample_name_to_experiment(sample_name, no_progress=no_progress)

    @memoized_property
    def first_experiment(self):
        return next(self.experiments())

    @property
    def preprocessed_read_type(self):
        return self.first_experiment.preprocessed_read_type

    @property
    def categorizer(self):
        return self.first_experiment.categorizer

    @property
    def layout_mode(self):
        return self.first_experiment.layout_mode

    @property
    def target_info(self):
        return self.first_experiment.target_info

    def common_sequence_chunk_exp_from_name(self, chunk_name):
        chunk_exp = ArrayedCommonSequencesExperiment(self, chunk_name)
        return chunk_exp

    @memoized_property
    def num_experiments(self):
        return len(self.sample_sheet)

    def condition_replicates(self, condition):
        sample_names = self.condition_to_sample_names[condition]
        return [self.sample_name_to_experiment(sample_name) for sample_name in sample_names]

    def condition_query(self, query_string):
        conditions_df = pd.DataFrame(self.full_conditions, index=self.full_conditions, columns=self.full_condition_keys)
        return conditions_df.query(query_string).index

    def sample_name_to_experiment(self, sample_name, no_progress=False):
        if no_progress:
            progress = None
        else:
            progress = self.progress

        exp = ArrayedExperiment(self.base_dir,
                                self.batch_name,
                                self.group_name,
                                sample_name,
                                experiment_group=self,
                                progress=progress,
                               )
        return exp

    @memoized_property
    def experiments_by_sanitized_name(self):
        return {exp.description['sanitized_sample_name']: exp for exp in self.experiments()}

    @memoized_property
    def full_condition_to_experiment(self):
        return {full_condition: self.sample_name_to_experiment(sample_name) for full_condition, sample_name in self.full_condition_to_sample_name.items()}

    @memoized_with_kwargs
    def condition_colors(self, *, palette='husl', unique=False):
        if unique:
            condition_colors = {condition: f'C{i}' for i, condition in enumerate(self.full_conditions)}
        else:
            colors = sns.color_palette(palette, n_colors=len(self.conditions))
            condition_colors = dict(zip(self.conditions, colors))

            for full_condition in self.full_conditions:
                condition = full_condition[:-1]

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

        for condition in self.full_conditions:
            sample_name = self.full_condition_to_sample_name[condition]
            if len(condition) > 1:
                partial_label = ', '.join(condition[:-1]) + ', '
            else:
                partial_label = ''

            label = f'{partial_label}rep. {condition[-1]} ({sample_name}, {self.total_valid_reads[condition]:,} total reads)'

            condition_labels[condition] = label

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

        for c_i, row in enumerate(np.array(self.conditions).T):
            if len(set(row)) > 1:
                informative_condition_idxs.append(c_i)

        for full_condition in self.full_conditions:
            if len(full_condition) > 1:
                partial_label = ', '.join(f'{self.condition_keys[c_i]}: {full_condition[c_i]}' for c_i in informative_condition_idxs) + ', '
            else:
                partial_label = ''

            label = f'{partial_label}rep: {full_condition[-1]}'

            condition_labels[full_condition] = label

        for condition in self.conditions:
            if isinstance(condition, str):
                label = condition
            else:
                label = ', '.join(f'{self.condition_keys[c_i]}: {condition[c_i]}' for c_i in informative_condition_idxs)

            condition_labels[condition] = label

        return condition_labels

    def extract_genomic_insertion_length_distributions(self):
        length_distributions = {}
        
        for condition, exp in self.progress(self.full_condition_to_experiment.items()):
            for organism in ['hg19', 'bosTau7']:
                key = (*condition, organism)
                length_distributions[key] = np.zeros(1600)

            for outcome in exp.outcome_iter():
                if outcome.category == 'genomic insertion':
                    organism = outcome.subcategory
                    
                    lti  = knock_knock.outcome.LongTemplatedInsertionOutcome.from_string(outcome.details)
                    key = (*condition, organism)
                    length_distributions[key][lti.insertion_length()] += 1

        length_distributions_df = pd.DataFrame(length_distributions).T

        length_distributions_df.index.names = list(self.outcome_column_levels) + ['organism']

        # Normalize to number of valid reads in each sample.
        length_distributions_df = length_distributions_df.div(self.total_valid_reads, axis=0)

        length_distributions_df = length_distributions_df.reorder_levels(['organism'] + list(self.outcome_column_levels))

        length_distributions_df.to_csv(self.fns['genomic_insertion_length_distributions'])

    @memoized_property
    def genomic_insertion_length_distributions(self):
        num_index_cols = len(self.outcome_column_levels) + 1
        df = pd.read_csv(self.fns['genomic_insertion_length_distributions'], index_col=list(range(num_index_cols)))
        df.columns = [int(c) for c in df.columns]
        return df

    @memoized_property
    def outcome_counts(self):
        # Ignore nonspecific amplification products in denominator of any outcome fraction calculations.
        to_drop = [
            'nonspecific amplification',
            #'bad sequence',
        ]

        # Empirically, overall editing rates can vary considerably across arrayed 
        # experiments, presumably due to nucleofection efficiency. If self.only_edited
        # is true, exclude unedited reads from outcome counting.
        if self.only_edited:
            to_drop.append('wild type')

        outcome_counts = self.outcome_counts_df(False).drop(to_drop, errors='ignore')
        
        # Sort columns to avoid annoying pandas PerformanceWarnings.
        outcome_counts = outcome_counts.sort_index(axis='columns')

        return outcome_counts

    @memoized_property
    def outcome_counts_with_bad(self):
        outcome_counts = self.outcome_counts_df(False)
        
        # Sort columns to avoid annoying pandas PerformanceWarnings.
        outcome_counts = outcome_counts.sort_index(axis='columns')

        return outcome_counts

    @memoized_property
    def total_valid_reads(self):
        return self.outcome_counts.sum()

    @memoized_property
    def total_reads(self):
        return self.outcome_counts_with_bad.sum()

    @memoized_property
    def outcome_fractions(self):
        fractions = self.outcome_counts / self.total_valid_reads
        order = fractions[self.baseline_condition].mean(axis='columns').sort_values(ascending=False).index
        fractions = fractions.loc[order]
        return fractions

    @memoized_property
    def outcome_fractions_with_bad(self):
        return self.outcome_counts / self.outcome_counts.sum()

    @memoized_property
    def outcome_fraction_condition_means(self):
        if len(self.condition_keys) == 0:
            return self.outcome_fractions.mean(axis='columns')
        else:
            return self.outcome_fractions.groupby(axis='columns', level=self.condition_keys).mean()

    @memoized_property
    def outcome_fraction_baseline_means(self):
        return self.outcome_fraction_condition_means[self.baseline_condition]

    @memoized_property
    def outcome_fraction_condition_stds(self):
        return self.outcome_fractions.groupby(axis='columns', level=self.condition_keys).std()

    @memoized_property
    def outcomes_by_baseline_frequency(self):
        return self.outcome_fraction_baseline_means.sort_values(ascending=False).index.values

    @memoized_property
    def outcome_fraction_differences(self):
        return self.outcome_fractions.sub(self.outcome_fraction_baseline_means, axis=0)

    @memoized_property
    def outcome_fraction_difference_condition_means(self):
        return self.outcome_fraction_differences.groupby(axis='columns', level=self.condition_keys).mean()

    @memoized_property
    def outcome_fraction_difference_condition_stds(self):
        return self.outcome_fraction_differences.groupby(axis='columns', level=self.condition_keys).std()

    @memoized_property
    def log2_fold_changes(self):
        # Using the warnings context manager doesn't work here, maybe because of pandas multithreading?
        warnings.filterwarnings('ignore')

        fold_changes = self.outcome_fractions.div(self.outcome_fraction_baseline_means, axis=0)
        log2_fold_changes = np.log2(fold_changes)

        warnings.resetwarnings()

        return log2_fold_changes

    @memoized_property
    def log2_fold_change_condition_means(self):
        return self.log2_fold_changes.groupby(axis='columns', level=self.condition_keys).mean()

    @memoized_property
    def log2_fold_change_condition_stds(self):
        return self.log2_fold_changes.groupby(axis='columns', level=self.condition_keys).std()

    @memoized_property
    def category_fractions(self):
        fs = self.outcome_fractions.groupby(level='category').sum()

        if self.category_groupings is not None:
            only_relevant_cats = pd.Index.difference(fs.index, self.category_groupings['not_relevant'])
            relevant_but_not_specific_cats = pd.Index.difference(only_relevant_cats, self.category_groupings['specific'])

            only_relevant = fs.loc[only_relevant_cats]

            only_relevant_normalized = only_relevant / only_relevant.sum()

            relevant_but_not_specific = only_relevant_normalized.loc[relevant_but_not_specific_cats].sum()

            grouped = only_relevant_normalized.loc[self.category_groupings['specific']]
            grouped.loc['all others'] = relevant_but_not_specific

            fs = grouped

            if self.add_pseudocount:
                reads_per_sample = self.outcome_counts.drop(self.category_groupings['not_relevant'], errors='ignore').sum()
                counts = fs * reads_per_sample
                counts += 1
                fs = counts / counts.sum()

        return fs

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

    def group_by_condition(self, df):
        if len(self.condition_keys) == 0:
            # Supplying a constant function to by means
            # all columns will be grouped together. Making
            # this constant value 'all' means that will be
            # the name of eventual aggregated column. 
            kwargs = dict(by=lambda x: 'all')
        else:
            kwargs = dict(level=self.condition_keys)

        return df.T.groupby(**kwargs)

    @memoized_property
    def category_fraction_condition_means(self):
        return self.group_by_condition(self.category_fractions).mean().T

    @memoized_property
    def category_fraction_baseline_means(self):
        return self.category_fraction_condition_means[self.baseline_condition]

    @memoized_property
    def category_fraction_condition_stds(self):
        return self.group_by_condition(self.category_fractions).std().T

    @memoized_property
    def categories_by_baseline_frequency(self):
        return self.category_fraction_baseline_means.sort_values(ascending=False).index.values

    @memoized_property
    def category_fraction_differences(self):
        return self.category_fractions.sub(self.category_fraction_baseline_means, axis=0)

    @memoized_property
    def category_fraction_difference_condition_means(self):
        return self.group_by_condition(self.category_fraction_differences).mean().T

    @memoized_property
    def category_fraction_difference_condition_stds(self):
        return self.group_by_condition(self.category_fraction_differences).std().T

    @memoized_property
    def category_log2_fold_changes(self):
        # Using the warnings context manager doesn't work here, maybe because of pandas multithreading?
        warnings.filterwarnings('ignore')

        fold_changes = self.category_fractions.div(self.category_fraction_baseline_means, axis=0)
        log2_fold_changes = np.log2(fold_changes)

        warnings.resetwarnings()

        return log2_fold_changes

    @memoized_property
    def category_log2_fold_change_condition_means(self):
        # calculate mean in linear space, not log space
        fold_changes = self.category_fraction_condition_means.div(self.category_fraction_baseline_means, axis=0)
        return np.log2(fold_changes)

    @memoized_property
    def category_log2_fold_change_condition_stds(self):
        # calculate effective log2 fold change of mean +/- std in linear space
        means = self.category_fraction_condition_means
        stds = self.category_fraction_condition_stds
        baseline_means = self.category_fraction_baseline_means
        return {
            'lower': np.log2((means - stds).div(baseline_means, axis=0)),
            'upper': np.log2((means + stds).div(baseline_means, axis=0)),
        }

    # TODO: figure out how to avoid this hideous code duplication.

    @memoized_property
    def subcategory_fractions(self):
        return self.outcome_fractions.groupby(level=['category', 'subcategory']).sum()

    @memoized_property
    def subcategory_fraction_condition_means(self):
        return self.group_by_condition(self.subcategory_fractions).mean().T

    @memoized_property
    def subcategory_fraction_baseline_means(self):
        return self.subcategory_fraction_condition_means[self.baseline_condition]

    @memoized_property
    def subcategory_fraction_condition_stds(self):
        return self.group_by_condition(self.subcategory_fractions).std().T

    @memoized_property
    def subcategories_by_baseline_frequency(self):
        return self.subcategory_fraction_baseline_means.sort_values(ascending=False).index.values

    @memoized_property
    def subcategory_fraction_differences(self):
        return self.subcategory_fractions.sub(self.subcategory_fraction_baseline_means, axis=0)

    @memoized_property
    def subcategory_fraction_difference_condition_means(self):
        return self.group_by_condition(self.subcategory_fraction_differences).mean().T

    @memoized_property
    def subcategory_fraction_difference_condition_stds(self):
        return self.group_by_condition(self.subcategory_fraction_differences).std().T

    @memoized_property
    def subcategory_log2_fold_changes(self):
        # Using the warnings context manager doesn't work here, maybe because of pandas multithreading?
        warnings.filterwarnings('ignore')

        fold_changes = self.subcategory_fractions.div(self.subcategory_fraction_baseline_means, axis=0)
        log2_fold_changes = np.log2(fold_changes)

        warnings.resetwarnings()

        return log2_fold_changes

    @memoized_property
    def subcategory_log2_fold_change_condition_means(self):
        # calculate mean in linear space, not log space
        fold_changes = self.subcategory_fraction_condition_means.div(self.subcategory_fraction_baseline_means, axis=0)
        return np.log2(fold_changes)

    @memoized_property
    def subcategory_log2_fold_change_condition_stds(self):
        # calculate effective log2 fold change of mean +/- std in linear space
        means = self.subcategory_fraction_condition_means
        stds = self.subcategory_fraction_condition_stds
        baseline_means = self.subcategory_fraction_baseline_means
        return {
            'lower': np.log2((means - stds).div(baseline_means, axis=0)),
            'upper': np.log2((means + stds).div(baseline_means, axis=0)),
        }

    # Duplication of code in pooled_screen
    def donor_outcomes_containing_SNV(self, SNV_name):
        ti = self.target_info
        SNV_index = sorted(ti.donor_SNVs['target']).index(SNV_name)
        donor_base = ti.donor_SNVs['donor'][SNV_name]['base']
        nt_fracs = self.outcome_fraction_baseline_means
        outcomes = [(c, s, d) for c, s, d in nt_fracs.index.values if c == 'donor' and d[SNV_index] == donor_base]
        return outcomes

    @memoized_property
    def conversion_fractions(self):
        conversion_fractions = {}

        SNVs = self.target_info.donor_SNVs['target']

        outcome_fractions = self.outcome_fractions

        for SNV_name in SNVs:
            outcomes = self.donor_outcomes_containing_SNV(SNV_name)
            fractions = outcome_fractions.loc[outcomes].sum()
            conversion_fractions[SNV_name] = fractions

        conversion_fractions = pd.DataFrame.from_dict(conversion_fractions, orient='index').sort_index()
        
        return conversion_fractions

    @memoized_property
    def outcomes_containing_pegRNA_programmed_edits(self):
        outcomes_containing_pegRNA_programmed_edits = {}
        if self.target_info.pegRNA_SNVs is not None:
            SNVs = self.target_info.pegRNA_SNVs[self.target_info.target]
            # Note: sorting SNVs is critical here to match the order in outcome.SNV_read_bases.
            SNV_order = sorted(SNVs)

            for SNV_name in SNVs:
                outcomes_containing_pegRNA_programmed_edits[SNV_name] = []

        else:
            SNVs = None
        
        if self.target_info.pegRNA_programmed_insertion is not None:
            insertion = self.target_info.pegRNA_programmed_insertion

            outcomes_containing_pegRNA_programmed_edits[str(insertion)] = []

        else:
            insertion = None

        if self.target_info.pegRNA_programmed_deletion is not None:
            deletion = self.target_info.pegRNA_programmed_deletion

            outcomes_containing_pegRNA_programmed_edits[str(deletion)] = []

        else:
            deletion = None
            
        for c, s, d  in self.outcome_fractions.index:
            if c in {'intended edit', 'partial replacement', 'partial edit'}:
                outcome = knock_knock.outcome.ProgrammedEditOutcome.from_string(d).undo_anchor_shift(self.target_info.anchor)

                if SNVs is not None:
                    for SNV_name, read_base in zip(SNV_order, outcome.SNV_read_bases):
                        SNV = SNVs[SNV_name]
                        if read_base == SNV['alternative_base']:
                            outcomes_containing_pegRNA_programmed_edits[SNV_name].append((c, s, d))

                if insertion is not None and insertion in outcome.insertions:
                    outcomes_containing_pegRNA_programmed_edits[str(insertion)].append((c, s, d))

                if deletion is not None and deletion in outcome.deletions:
                    outcomes_containing_pegRNA_programmed_edits[str(deletion)].append((c, s, d))

        return outcomes_containing_pegRNA_programmed_edits

    @memoized_property
    def pegRNA_conversion_fractions(self):
        fs = {}

        for edit_name, outcomes in self.outcomes_containing_pegRNA_programmed_edits.items():
            fs[edit_name] = self.outcome_fractions.loc[outcomes].sum()

        if len(fs) > 0:
            fs_df = pd.DataFrame.from_dict(fs, orient='index')

            fs_df.columns.names = self.full_condition_keys
            fs_df.index.name = 'edit_name'

        else:
            fs_df = None

        return fs_df

    @memoized_property
    def pegRNA_conversion_fractions_by_edit_description(self):
        if self.target_info.pegRNA is None or self.pegRNA_conversion_fractions is None:
            return None
        else:
            def name_to_description(name):
                return self.target_info.pegRNA_programmed_edit_name_to_description.get(name, name)

            df = self.pegRNA_conversion_fractions.copy()

            df.index = df.index.map(name_to_description)

            new_tuples = [(self.full_condition_to_sample_name[t],) + t for t in df.columns.values]
            new_columns = pd.MultiIndex.from_tuples(new_tuples, names=['sample'] + df.columns.names)
            df.columns = new_columns

            return df

    def write_pegRNA_conversion_fractions(self):
        if self.target_info.pegRNA is not None and self.pegRNA_conversion_fractions is not None:
            self.pegRNA_conversion_fractions_by_edit_description.T.to_csv(self.fns['pegRNA_conversion_fractions'])

    @memoized_with_kwargs
    def deletion_boundaries(self, *, include_simple_deletions=True, include_edit_plus_deletions=False):
        ti = self.target_info

        deletions = [
            (c, s, d) for c, s, d in self.outcome_fractions.index
            if (include_simple_deletions and c == 'deletion')
            or (include_edit_plus_deletions and (c, s) == ('edit + indel', 'deletion'))
        ]

        deletion_fractions = self.outcome_fractions.loc[deletions]
        index = np.arange(len(ti.target_sequence))
        columns = deletion_fractions.columns

        fraction_removed = np.zeros((len(index), len(columns)))
        starts = np.zeros_like(fraction_removed)
        stops = np.zeros_like(fraction_removed)

        for (c, s, d), row in deletion_fractions.iterrows():
            # Undo anchor shift to make coordinates relative to full target sequence.
            if c == 'deletion':
                deletion = knock_knock.outcome.DeletionOutcome.from_string(d).undo_anchor_shift(ti.anchor).deletion
            elif c == 'edit + indel':
                deletions = knock_knock.outcome.ProgrammedEditOutcome.from_string(d).undo_anchor_shift(ti.anchor).deletions
                if len(deletions) != 1:
                    raise NotImplementedError
                else:
                    deletion = deletions[0]
            else:
                raise ValueError
            
            per_possible_start = row.values / len(deletion.starts_ats)
            
            for start, stop in zip(deletion.starts_ats, deletion.ends_ats):
                deletion_slice = slice(start, stop + 1)

                fraction_removed[deletion_slice] += per_possible_start
                starts[start] += per_possible_start
                stops[stop] += per_possible_start

        fraction_removed = pd.DataFrame(fraction_removed, index=index, columns=columns)
        starts = pd.DataFrame(starts, index=index, columns=columns)
        stops = pd.DataFrame(stops, index=index, columns=columns)

        deletion_boundaries = {
            'fraction_removed': fraction_removed,
            'starts': starts,
            'stops': stops,
        }
        return deletion_boundaries

    def explore(self, **kwargs):
        import knock_knock.explore
        explorer = knock_knock.explore.ArrayedGroupExplorer(self, **kwargs)
        return explorer.layout

class ArrayedExperiment(knock_knock.illumina_experiment.IlluminaExperiment):
    def __init__(self, base_dir, batch_name, group_name, sample_name, experiment_group=None, **kwargs):
        if experiment_group is None:
            experiment_group = ArrayedExperimentGroup(base_dir, batch_name, group_name)

        self.experiment_group = experiment_group
        self.experiment_type = experiment_group.experiment_type

        super().__init__(base_dir, (batch_name, group_name), sample_name, **kwargs)

        self.batch_name = batch_name
        self.group_name = group_name
        self.sample_name = sample_name

        self.read_types.add(self.uncommon_read_type)

    @property
    def uncommon_read_type(self):
        return f'{self.preprocessed_read_type}_uncommon'

    @memoized_property
    def categorizer(self):
        experiment_type_to_categorizer = {
            'prime_editing': knock_knock.prime_editing_layout.Layout,
            'twin_prime': knock_knock.twin_prime_layout.Layout,
            'Bxb1_twin_prime': knock_knock.Bxb1_layout.Layout,
            'TECseq': knock_knock.TECseq_layout.Layout,
            'seeseq': knock_knock.seeseq_layout.Layout,
            'seeseq_dual_flap': knock_knock.seeseq_layout.DualFlapLayout,
        }

        aliases = {
            'single_flap': 'prime_editing',
            'dual_flap': 'twin_prime',
            'Bxb1_dual_flap': 'Bxb1_twin_prime',
        }

        for alias, original_name in aliases.items():
            experiment_type_to_categorizer[alias] = experiment_type_to_categorizer[original_name]

        return experiment_type_to_categorizer[self.experiment_type]

    def load_description(self):
        description = {
            **self.experiment_group.description,
            **self.experiment_group.sample_sheet.loc[self.sample_name],
        }
        return description

    @memoized_property
    def data_dir(self):
        return self.experiment_group.data_dir

    @memoized_property
    def results_dir(self):
        sanitized_sample_name = self.description.get('sanitized_sample_name', self.sample_name)
        return self.experiment_group.results_dir / sanitized_sample_name

    def extract_reads_with_uncommon_sequences(self):
        # Extract reads with sequences that weren't seen more than once across the group.
        fn = self.fns_by_read_type['fastq'][self.uncommon_read_type]
        with gzip.open(fn, 'wt', compresslevel=1) as fh:
            for read in self.reads_by_type(self.preprocessed_read_type):
                if read.seq not in self.experiment_group.common_sequence_to_outcome:
                    fh.write(str(read))

    @memoized_property
    def common_sequence_to_outcome(self):
        return self.experiment_group.common_sequence_to_outcome

    @memoized_property
    def common_sequence_to_alignments(self):
        return self.experiment_group.common_sequence_to_alignments

class ArrayedCommonSequencesExperiment(knock_knock.common_sequences.CommonSequencesExperiment, ArrayedExperiment):
    def __init__(self, experiment_group, chunk_name):
        ArrayedExperiment.__init__(self,
                                   experiment_group.base_dir,
                                   experiment_group.batch_name,
                                   experiment_group.group_name,
                                   chunk_name,
                                   experiment_group=experiment_group,
                                  )

    @memoized_property
    def results_dir(self):
        return self.experiment_group.fns['common_sequences_dir'] / self.sample_name

    def load_description(self):
        return self.experiment_group.description

def sanitize_and_validate_sample_sheet(sample_sheet_fn):
    sample_sheet_df = knock_knock.utilities.read_and_sanitize_csv(sample_sheet_fn)

    # Default to hg38 if genome column isn't present.
    if 'genome' not in sample_sheet_df.columns:
        sample_sheet_df['genome'] = 'hg38'

    if 'extra_sequences' not in sample_sheet_df.columns:
        sample_sheet_df['extra_sequences'] = ''

    if 'donor' not in sample_sheet_df.columns:
        sample_sheet_df['donor'] = ''

    # Confirm mandatory columns are present.

    mandatory_columns = [
        'sample_name',
        'R1',
        'amplicon_primers',
        'sgRNAs',
        'genome',
        'extra_sequences',
        'donor',
    ]
    
    missing_columns = [col for col in mandatory_columns if col not in sample_sheet_df.columns]
    if len(missing_columns) > 0:
        raise ValueError(f'{missing_columns} column(s) not found in sample sheet')

    if not sample_sheet_df['sample_name'].is_unique:
        counts = sample_sheet_df['sample_name'].value_counts()
        bad_names = ', '.join(f'{name} ({count})' for name, count in counts[counts > 1].items())
        raise ValueError(f'Sample names are not unique: {bad_names}')

    # Since only the final path component of R1 and R2 files will be retained,
    # ensure that these are unique to avoid clobbering.

    if not sample_sheet_df['R1'].apply(lambda fn: Path(fn).name).is_unique:
        raise ValueError(f'R1 files do not have unique names')

    if 'R2' in sample_sheet_df and not sample_sheet_df['R1'].apply(lambda fn: Path(fn).name).is_unique:
        raise ValueError(f'R2 files do not have unique names')
    
    return sample_sheet_df

def make_default_target_info_name(amplicon_primers, genome, extra_sequences):
    target_info_name = f'{amplicon_primers}_{genome}'

    if extra_sequences != '':
        target_info_name = f'{target_info_name}_{extra_sequences}'

    # Names can't contain a forward slash since they are a path component.
    target_info_name = target_info_name.replace('/', '_SLASH_')

    return target_info_name

def make_targets(base_dir, sample_sheet_df):
    valid_supplemental_indices = set(knock_knock.target_info.locate_supplemental_indices(base_dir))

    targets = {}

    grouped = sample_sheet_df.groupby(['amplicon_primers', 'genome', 'extra_sequences'])

    for (amplicon_primers, genome, extra_sequences), rows in grouped:
        all_sgRNAs = set()
        for sgRNAs in rows['sgRNAs']:
            if sgRNAs != '':
                all_sgRNAs.update(sgRNAs.split(';'))

        target_info_name = make_default_target_info_name(amplicon_primers, genome, extra_sequences)

        extra_sequences = ';'.join(set(extra_sequences.split(';')) - valid_supplemental_indices)

        targets[target_info_name] = {
            'genome': genome,
            'amplicon_primers': amplicon_primers,
            'sgRNAs': ';'.join(all_sgRNAs),
            'extra_sequences': extra_sequences,
        }

    targets_df = pd.DataFrame.from_dict(targets, orient='index')
    targets_df.index.name = 'name'

    targets_dir = Path(base_dir) / 'targets'
    targets_dir.mkdir(parents=True, exist_ok=True)

    targets_csv_fn = targets_dir / 'targets.csv'
    targets_df.to_csv(targets_csv_fn)

    knock_knock.build_targets.build_target_infos_from_csv(base_dir)

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

        target_info_name = make_default_target_info_name(row['amplicon_primers'], row['genome'], row['extra_sequences'])
        ti = knock_knock.target_info.TargetInfo(base_dir, target_info_name) 

        primer_sequences = {name: ti.feature_sequence(ti.target, name).upper() for name in ti.primers}

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
            logging.warning(f"Unable to detect sequencing orientation for {row['sample_name']}")
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

    sequencing_start_feature_names = detect_sequencing_start_feature_names(base_dir, batch_name, sample_sheet_df)
    sample_sheet_df['sequencing_start_feature_name'] = sample_sheet_df['sample_name'].map(sequencing_start_feature_names).fillna('')

    # For each set of target info parameter values, assign the most common
    # sequencing_start_feature_name to all samples.

    group_keys = [
        'amplicon_primers',
        'genome',
        'sgRNAs',
        'donor',
        'extra_sequences',
    ]

    grouped = sample_sheet_df.groupby(group_keys)

    keys_to_feature_name = {}

    for keys, rows in grouped:
        orientations = rows['sequencing_start_feature_name'].value_counts().drop('', errors='ignore')
        
        if len(orientations) == 0:
            raise ValueError(f'No sequencing orientations detected for {keys}')
        else:
            feature_name = orientations.index[0]

            keys_to_feature_name[keys] = feature_name

    for i in sample_sheet_df.index:
        sample_sheet_df.loc[i, 'sequencing_start_feature_name'] = keys_to_feature_name[tuple(sample_sheet_df.loc[i, group_keys])]
            
    # If unedited controls are annotated, make virtual samples.

    if 'is_unedited_control' in sample_sheet_df:
        sample_sheet_df['is_unedited_control'] = sample_sheet_df['is_unedited_control'] == 'unedited'

        amplicon_keys = ['amplicon_primers', 'genome']
        strategy_keys = ['sgRNAs', 'donor', 'extra_sequences']

        grouped_by_amplicon = sample_sheet_df.groupby(amplicon_keys)

        new_rows = []

        for _, amplicon_rows in grouped_by_amplicon:
            grouped_by_editing_strategy = amplicon_rows.groupby(strategy_keys)

            for strategy_i, (strategy, strategy_rows) in enumerate(grouped_by_editing_strategy):
                for _, row in amplicon_rows.query('is_unedited_control').iterrows():
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

        existing_unedited_idxs = sample_sheet_df.query('is_unedited_control').index
        sample_sheet_df = pd.concat([sample_sheet_df.drop(existing_unedited_idxs), new_rows])

    valid_supplemental_indices = set(knock_knock.target_info.locate_supplemental_indices(base_dir))

    groups = {}
    samples = {}

    condition_columns = [column for column in sample_sheet_df.columns if column.startswith('condition:')]
    shortened_condition_columns = [column[len('condition:'):] for column in condition_columns]
    if 'group' in shortened_condition_columns:
        raise ValueError('"group" is a reserved column name and can\'t be a condition')

    group_keys = [
        'amplicon_primers',
        'genome',
        'sgRNAs',
        'donor',
        'extra_sequences',
        'sequencing_start_feature_name',
    ]

    grouped = sample_sheet_df.groupby(group_keys)

    for group_i, ((amplicon_primers, genome, sgRNAs, donor, extra_sequences, sequencing_start_feature_name), group_rows) in enumerate(grouped):
        target_info_name = make_default_target_info_name(amplicon_primers, genome, extra_sequences)

        group_name = f'{target_info_name}_{sgRNAs}_{donor}'
        group_name = group_name.replace(';', '+')

        sanitized_group_name = get_sanitized_group_name(group_i)

        sanitized_sample_names = {}
        for sample_i, (_, row) in enumerate(group_rows.iterrows()):
            sample_name = row['sample_name']
            sanitized_sample_name = get_sanitized_sample_name(sample_i)

            sanitized_sample_names[sample_name] = sanitized_sample_name

        ti = knock_knock.target_info.TargetInfo(base_dir, target_info_name, sgRNAs=sgRNAs)

        if 'experiment_type' in group_rows.columns:
            experiment_types = set(group_rows['experiment_type'])

            if len(experiment_types) != 1:
                raise ValueError('more than one experiment type specified')
            else:
                experiment_type = list(experiment_types)[0]

            if experiment_type == 'seeseq' and ti.pegRNA_names is not None and len(ti.pegRNA_names) == 2:
                experiment_type = 'seeseq_dual_flap'

        elif ti.pegRNA_names is None or len(ti.pegRNA_names) <= 1:
            experiment_type = 'single_flap'

        elif len(ti.pegRNA_names) == 2:
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

        if experiment_type in {'seeseq', 'TECseq'}:
            min_relevant_length = 0

        baseline_condition = ';'.join(map(str, tuple(group_rows[condition_columns].iloc[0])))

        supplemental_indices = set()

        for name in [genome] + extra_sequences.split(';'):
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
            'target_info': target_info_name,
            'sequencing_start_feature_name': sequencing_start_feature_name,
            'sgRNAs': sgRNAs,
            'donor': donor,
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

                    for k in ['trim_to_max_length', 'UMI_key', 'sequencing_primers', 'reverse_complement']:
                        if k in row:
                            samples[sample_name][k] = row[k]

                    for full, short in zip(condition_columns, shortened_condition_columns):
                        samples[sample_name][short] = row[full]

        else:
            for rep_i, (_, row) in enumerate(group_rows.iterrows(), 1):
                sample_name = row['sample_name']

                samples[sample_name] = {
                    'sanitized_sample_name': sanitized_sample_names[sample_name],
                    'R1': Path(row['R1']).name,
                    'group': group_name,
                    'replicate': rep_i,
                    'color': group_i + 1,
                }

                for k in ['R2', 'I1', 'I2']:
                    if k in row:
                        samples[sample_name][k] = Path(row[k]).name

                for k in ['trim_to_max_length', 'UMI_key', 'sequencing_primers', 'reverse_complement']:
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