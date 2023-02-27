import gzip
import itertools
import shutil
import time
import warnings
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import pysam

import knock_knock.build_targets
import knock_knock.experiment_group
import knock_knock.outcome
import knock_knock.target_info

from hits import utilities, sam

memoized_property = utilities.memoized_property

class Batch:
    def __init__(self, base_dir, batch,
                 category_groupings=None,
                 baseline_condition=None,
                 add_pseudocount=False,
                 only_edited=False,
                 progress=None,
                ):
        self.base_dir = Path(base_dir)
        self.batch = batch
        self.data_dir = self.base_dir / 'data' / batch

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
        self.sample_sheet = pd.read_csv(self.sample_sheet_fn, index_col='sample_name')
        self.sample_sheet.index = self.sample_sheet.index.astype(str)

        self.group_descriptions_fn = self.data_dir / 'group_descriptions.csv'
        self.group_descriptions = pd.read_csv(self.group_descriptions_fn, index_col='group').replace({np.nan: None})

        self.condition_colors_fn = self.data_dir / 'condition_colors.csv'
        if self.condition_colors_fn.exists():
            self.condition_colors = pd.read_csv(self.condition_colors_fn, index_col='perturbation').squeeze()
        else:
            self.condition_colors = None

    def __repr__(self):
        return f'Batch: {self.batch}, base_dir={self.base_dir}'

    @property
    def group_names(self):
        return self.sample_sheet['group'].unique()

    def group(self, group_name):
        return ArrayedExperimentGroup(self.base_dir, self.batch, group_name,
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
            new_batch_name = self.batch

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
        fs = {gn: group.category_fractions for gn, group in self.groups.items()}
        fs = pd.concat(fs, axis='columns').fillna(0).sort_index()
        fs.columns.names = ['group'] + fs.columns.names[1:]
        return fs

    @memoized_property
    def subcategory_fractions(self):
        fs = {gn: group.subcategory_fractions for gn, group in self.groups.items()}
        fs = pd.concat(fs, axis='columns').fillna(0).sort_index()
        fs.columns.names = ['group'] + fs.columns.names[1:]
        return fs

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
            if 'group' in conditions and row['group'] not in conditions['group']:
                continue

            group = batch.groups[row['group']]

            if 'experiment_type' in conditions and group.experiment_type not in conditions['experiment_type']:
                continue

            exps[batch_name, group.group, sample_name] = group.sample_name_to_experiment(sample_name)

    return exps

class ArrayedExperimentGroup(knock_knock.experiment_group.ExperimentGroup):
    def __init__(self, base_dir, batch, group,
                 category_groupings=None,
                 progress=None,
                 baseline_condition=None,
                 add_pseudocount=None,
                 only_edited=False,
                ):
        self.base_dir = Path(base_dir)
        self.batch = batch
        self.group = group

        self.category_groupings = category_groupings
        self.add_pseudocount = add_pseudocount
        self.only_edited = only_edited

        self.group_args = (base_dir, batch, group)

        super().__init__()

        if progress is None or getattr(progress, '_silent', False):
            def ignore_kwargs(x, **kwargs):
                return x
            progress = ignore_kwargs

        self.silent = True

        self.progress = progress

        self.Batch = Batch(self.base_dir, self.batch)

        self.batch_sample_sheet = self.Batch.sample_sheet
        self.sample_sheet = self.batch_sample_sheet.query('group == @self.group').copy()

        self.description = self.Batch.group_descriptions.loc[self.group].copy()

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

        self.ExperimentType, self.CommonSequencesExperimentType = arrayed_specialized_experiment_factory(self.experiment_type)

        self.outcome_index_levels = ('category', 'subcategory', 'details')
        self.outcome_column_levels = self.full_condition_keys

        def condition_from_row(row):
            condition = tuple(str(row[key]) for key in self.condition_keys)
            if len(condition) == 1:
                condition = condition[0]
            return condition

        def full_condition_from_row(row):
            return tuple(str(row[key]) for key in self.full_condition_keys)

        self.full_conditions = [full_condition_from_row(row) for _, row in self.sample_sheet.iterrows()]

        conditions_are_unique = len(set(self.full_conditions)) == len(self.full_conditions)
        if not conditions_are_unique:
            print(f'{self}\nconditions are not unique:')
            for k, v in Counter(self.full_conditions).most_common():
                print(k, v)
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

    def __repr__(self):
        return f'ArrayedExperimentGroup: batch={self.batch}, group={self.group}, base_dir={self.base_dir}'

    @memoized_property
    def data_dir(self):
        return self.base_dir / 'data' / self.batch

    @memoized_property
    def results_dir(self):
        return self.base_dir / 'results' / self.batch / self.group

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
        chunk_exp = self.CommonSequencesExperimentType(self.base_dir, self.batch, self.group, chunk_name,
                                                       experiment_group=self,
                                                       description=self.description,
                                                      )
        return chunk_exp

    @memoized_property
    def num_experiments(self):
        return len(self.sample_sheet)

    def condition_replicates(self, condition):
        sample_names = self.condition_to_sample_names[condition]
        return [self.sample_name_to_experiment(sample_name) for sample_name in sample_names]

    def sample_name_to_experiment(self, sample_name, no_progress=False):
        if no_progress:
            progress = None
        else:
            progress = self.progress

        exp = self.ExperimentType(self.base_dir, self.batch, self.group, sample_name, experiment_group=self, progress=progress)
        return exp

    @memoized_property
    def full_condition_to_experiment(self):
        return {full_condition: self.sample_name_to_experiment(sample_name) for full_condition, sample_name in self.full_condition_to_sample_name.items()}

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
        # is true, exlcude unedited reads from outcome counting.
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

    def group_by_condition(self, df):
        if len(self.condition_keys) == 0:
            # Supplying a constant function to buy means
            # all columns will be grouped together. Making
            # this constant value 'all' means that will be
            # the name of eventual aggregated column. 
            kwargs = dict(axis='columns', by=lambda x: 'all')
        else:
            kwargs = dict(axis='columns', level=self.condition_keys)

        return df.groupby(**kwargs)

    @memoized_property
    def category_fraction_condition_means(self):
        return self.group_by_condition(self.category_fractions).mean()

    @memoized_property
    def category_fraction_baseline_means(self):
        return self.category_fraction_condition_means[self.baseline_condition]

    @memoized_property
    def category_fraction_condition_stds(self):
        return self.group_by_condition(self.category_fractions).std()

    @memoized_property
    def categories_by_baseline_frequency(self):
        return self.category_fraction_baseline_means.sort_values(ascending=False).index.values

    @memoized_property
    def category_fraction_differences(self):
        return self.category_fractions.sub(self.category_fraction_baseline_means, axis=0)

    @memoized_property
    def category_fraction_difference_condition_means(self):
        return self.group_by_condition(self.category_fraction_differences).mean()

    @memoized_property
    def category_fraction_difference_condition_stds(self):
        return self.group_by_condition(self.category_fraction_differences).std()

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
        return self.group_by_condition(self.subcategory_fractions).mean()

    @memoized_property
    def subcategory_fraction_baseline_means(self):
        return self.subcategory_fraction_condition_means[self.baseline_condition]

    @memoized_property
    def subcategory_fraction_condition_stds(self):
        return self.group_by_condition(self.subcategory_fractions).std()

    @memoized_property
    def subcategories_by_baseline_frequency(self):
        return self.subcategory_fraction_baseline_means.sort_values(ascending=False).index.values

    @memoized_property
    def subcategory_fraction_differences(self):
        return self.subcategory_fractions.sub(self.subcategory_fraction_baseline_means, axis=0)

    @memoized_property
    def subcategory_fraction_difference_condition_means(self):
        return self.group_by_condition(self.subcategory_fraction_differences).mean()

    @memoized_property
    def subcategory_fraction_difference_condition_stds(self):
        return self.group_by_condition(self.subcategory_fraction_differences).std()

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

    def explore(self, **kwargs):
        import knock_knock.explore
        explorer = knock_knock.explore.ArrayedGroupExplorer(self, **kwargs)
        return explorer.layout

class ArrayedExperiment:
    def __init__(self, base_dir, batch, group, sample_name, experiment_group=None):
        if experiment_group is None:
            experiment_group = ArrayedExperimentGroup(base_dir, batch, group)

        self.base_dir = Path(base_dir)
        self.batch = batch
        self.group = group
        self.sample_name = sample_name
        self.experiment_group = experiment_group

        self.has_UMIs = False

    @property
    def default_read_type(self):
        # None required to trigger check for common sequence in alignment_groups
        return None

    def load_description(self):
        description = self.experiment_group.sample_sheet.loc[self.sample_name].to_dict()
        for key, value in self.experiment_group.description.items():
            description[key] = value
        return description

    @memoized_property
    def data_dir(self):
        return self.experiment_group.data_dir

    def make_nonredundant_sequence_fastq(self):
        # Extract reads with sequences that weren't seen more than once across the group.
        fn = self.fns_by_read_type['fastq']['nonredundant']
        with gzip.open(fn, 'wt', compresslevel=1) as fh:
            for read in self.reads_by_type(self.preprocessed_read_type):
                if read.seq not in self.experiment_group.common_sequence_to_outcome:
                    fh.write(str(read))

    @memoized_property
    def results_dir(self):
        return self.experiment_group.results_dir / self.sample_name

    @memoized_property
    def seq_to_outcome(self):
        seq_to_outcome = self.experiment_group.common_sequence_to_outcome
        for seq, outcome in seq_to_outcome.items():
            outcome.special_alignment = self.experiment_group.common_name_to_special_alignment.get(outcome.query_name)
        return seq_to_outcome

    @memoized_property
    def seq_to_alignments(self):
        return self.experiment_group.common_sequence_to_alignments

    @memoized_property
    def combined_header(self):
        return sam.get_header(self.fns_by_read_type['bam_by_name']['nonredundant'])

    def alignment_groups(self, fn_key='bam_by_name', outcome=None, read_type=None):
        if read_type is None:
            nonredundant_alignment_groups = super().alignment_groups(read_type='nonredundant', outcome=outcome)
            reads = self.reads_by_type(self.preprocessed_read_type)

            if outcome is None:
                outcome_records = itertools.repeat(None)
            else:
                outcome_records = self.outcome_iter()

            for read, outcome_record in zip(reads, outcome_records):
                if outcome is None or outcome_record.category == outcome or (outcome_record.category, outcome_record.subcategory) == outcome:
                    if read.seq in self.seq_to_alignments:
                        name = read.name
                        als = self.seq_to_alignments[read.seq]

                    else:
                        name, als = next(nonredundant_alignment_groups)

                        if name != read.name:
                            raise ValueError('iters out of sync', name, read.name)

                    yield name, als
        else:
            yield from super().alignment_groups(fn_key=fn_key, outcome=outcome, read_type=read_type)

    def categorize_outcomes(self, max_reads=None):
        # Record how long each categorization takes.
        times_taken = []

        if self.fns['outcomes_dir'].is_dir():
            shutil.rmtree(str(self.fns['outcomes_dir']))

        self.fns['outcomes_dir'].mkdir()

        outcome_to_qnames = defaultdict(list)

        bam_read_type = 'nonredundant'

        # iter wrap since tqdm objects are not iterators
        alignment_groups = iter(self.alignment_groups())

        if max_reads is not None:
            alignment_groups = itertools.islice(alignment_groups, max_reads)

        special_als = defaultdict(list)

        with self.fns['outcome_list'].open('w') as outcome_fh:

            for name, als in self.progress(alignment_groups, desc='Categorizing reads'):
                seq = als[0].get_forward_sequence()

                # Special handling of empty sequence.
                if seq is None:
                    seq = ''

                if seq in self.seq_to_outcome:
                    layout = self.seq_to_outcome[seq]
                    layout.query_name = name

                else:
                    layout = self.categorizer(als, self.target_info,
                                              error_corrected=self.has_UMIs,
                                              mode=self.layout_mode,
                                             )

                    try:
                        layout.categorize()
                    except:
                        print()
                        print(self.sample_name, name)
                        raise
                
                if layout.special_alignment is not None:
                    special_als[layout.category, layout.subcategory].append(layout.special_alignment)

                outcome_to_qnames[layout.category, layout.subcategory].append(name)

                try:
                    outcome = self.final_Outcome.from_layout(layout)
                except:
                    print()
                    print(self.sample_name, name)
                    raise

                outcome_fh.write(f'{outcome}\n')

                times_taken.append(time.monotonic())

        # To make plotting easier, for each outcome, make a file listing all of
        # qnames for the outcome and a bam file (sorted by name) with all of the
        # alignments for these qnames.

        qname_to_outcome = {}

        bam_fn = self.fns_by_read_type['bam_by_name'][bam_read_type]
        header = sam.get_header(bam_fn)

        alignment_sorters = sam.multiple_AlignmentSorters(header, by_name=True)

        for outcome, qnames in outcome_to_qnames.items():
            outcome_fns = self.outcome_fns(outcome)
            outcome_fns['dir'].mkdir()

            alignment_sorters[outcome] = outcome_fns['bam_by_name'][bam_read_type]
            
            with outcome_fns['query_names'].open('w') as fh:
                for qname in qnames:
                    qname_to_outcome[qname] = outcome
                    fh.write(qname + '\n')
            
        with alignment_sorters:
            saved_verbosity = pysam.set_verbosity(0)
            with pysam.AlignmentFile(bam_fn) as full_bam_fh:
                for al in self.progress(full_bam_fh, desc='Making outcome-specific bams'):
                    if al.query_name in qname_to_outcome:
                        outcome = qname_to_outcome[al.query_name]
                        alignment_sorters[outcome].write(al)
            pysam.set_verbosity(saved_verbosity)

        # Make special alignments bams.
        for outcome, als in self.progress(special_als.items(), desc='Making special alignments bams'):
            outcome_fns = self.outcome_fns(outcome)
            bam_fn = outcome_fns['special_alignments']
            sorter = sam.AlignmentSorter(bam_fn, header)
            with sorter:
                for al in als:
                    sorter.write(al)

        return np.array(times_taken)

def arrayed_specialized_experiment_factory(experiment_kind):
    from knock_knock.illumina_experiment import IlluminaExperiment
    from knock_knock.prime_editing_experiment import PrimeEditingExperiment
    from knock_knock.prime_editing_experiment import TwinPrimeExperiment
    from knock_knock.prime_editing_experiment import Bxb1TwinPrimeExperiment

    experiment_kind_to_class = {
        'illumina': IlluminaExperiment,
        'prime_editing': PrimeEditingExperiment,
        'twin_prime': TwinPrimeExperiment,
        'Bxb1_twin_prime': Bxb1TwinPrimeExperiment,
    }

    SpecializedExperiment = experiment_kind_to_class[experiment_kind]

    class ArrayedSpecializedExperiment(ArrayedExperiment, SpecializedExperiment):
        def __init__(self, base_dir, batch, group, sample_name, experiment_group=None, **kwargs):
            ArrayedExperiment.__init__(self, base_dir, batch, group, sample_name, experiment_group=experiment_group)
            SpecializedExperiment.__init__(self, base_dir, (batch, group), sample_name, **kwargs)

        def __repr__(self):
            # 22.06.03: TODO: this doesn't actually call the SpecializedExperiment form of __repr__.
            return f'Arrayed{SpecializedExperiment.__repr__(self)}'
    
    class ArrayedSpecializedCommonSequencesExperiment(knock_knock.experiment_group.CommonSequencesExperiment, ArrayedExperiment, SpecializedExperiment):
        def __init__(self, base_dir, batch, group, sample_name, experiment_group=None, **kwargs):
            knock_knock.experiment_group.CommonSequencesExperiment.__init__(self)
            ArrayedExperiment.__init__(self, base_dir, batch, group, sample_name, experiment_group=experiment_group)
            SpecializedExperiment.__init__(self, base_dir, (batch, group), sample_name, **kwargs)
    
    return ArrayedSpecializedExperiment, ArrayedSpecializedCommonSequencesExperiment

def sanitize_and_validate_input(df):
    # Remove any rows or columns that are entirely nan (e.g. because excel exported
    # unwanted empty rows into a csv), then replace any remaining nans with an empty string.
    df = df.dropna(axis='index', how='all').dropna(axis='columns', how='all').fillna('')

    # Confirm mandatory columns are present.

    mandatory_columns = [
        'sample_name',
        'R1',
        'amplicon_primers',
        'sgRNAs',
    ]
    
    missing_columns = [col for col in mandatory_columns if col not in df.columns]
    if len(missing_columns) > 0:
        raise ValueError(f'{missing_columns} column(s) not found in sample sheet')

    if not df['sample_name'].is_unique:
        counts = df['sample_name'].value_counts()
        bad_names = ', '.join(f'{name} ({count})' for name, count in counts[counts > 1].items())
        raise ValueError(f'Sample name are not unique: {bad_names}')

    return df

def make_targets(base_dir, df, extra_genbanks=None):
    if extra_genbanks is None:
        extra_genbanks = []

    targets = {}

    for amplicon_primers, rows in df.groupby('amplicon_primers'):
        all_sgRNAs = set()
        for sgRNAs in rows['sgRNAs']:
            if sgRNAs != '':
                all_sgRNAs.update(sgRNAs.split(';'))

        targets[amplicon_primers] = {
            'genome': 'hg38',
            'amplicon_primers': amplicon_primers,
            'sgRNAs': ';'.join(all_sgRNAs),
            'extra_genbanks': ';'.join(extra_genbanks),
        }

    targets_df = pd.DataFrame.from_dict(targets, orient='index')
    targets_df.index.name = 'name'

    targets_dir = Path(base_dir) / 'targets'
    targets_dir.mkdir(parents=True, exist_ok=True)

    targets_csv_fn = targets_dir / 'targets.csv'
    targets_df.to_csv(targets_csv_fn)

    knock_knock.build_targets.build_target_infos_from_csv(base_dir)

def make_group_descriptions_and_sample_sheet(base_dir, df, batch_name=None):
    df = df.copy()

    if 'donor' not in df.columns:
        df['donor'] = ''

    groups = {}
    samples = {}

    condition_columns = [column for column in df.columns if column.startswith('condition:')]
    shortened_condition_columns = [column[len('condition:'):] for column in condition_columns]

    grouped = df.groupby(['amplicon_primers', 'sgRNAs', 'donor'])

    for group_i, ((amplicon_primers, sgRNAs, donor), group_rows) in enumerate(grouped, 1):
        group_name = f'{amplicon_primers}_{sgRNAs}_{donor}'
        group_name = group_name.replace(';', '+')

        target_info_name = amplicon_primers

        ti = knock_knock.target_info.TargetInfo(base_dir, target_info_name, sgRNAs=sgRNAs)
        
        if ti.pegRNA_names is None or len(ti.pegRNA_names) <= 1:
            experiment_type = 'prime_editing'
        elif len(ti.pegRNA_names) == 2:
            if donor == '':
                experiment_type = 'twin_prime'
            else:
                experiment_type = 'Bxb1_twin_prime'
        else:
            raise ValueError

        baseline_condition = ';'.join(map(str, tuple(group_rows[condition_columns].iloc[0])))

        groups[group_name] = {
            'supplemental_indices': 'hg38',
            'experiment_type': experiment_type,
            'target_info': target_info_name,
            'sgRNAs': sgRNAs,
            'donor': donor,
            'min_relevant_length': 100,
            'condition_keys': ';'.join(shortened_condition_columns),
            'baseline_condition': baseline_condition,
        }

        if len(condition_columns) > 0:
            for condition_i, (condition, condition_rows) in enumerate(group_rows.groupby(condition_columns), 1):
                for rep_i, (_, row) in enumerate(condition_rows.iterrows(), 1):
                    samples[row['sample_name']] = {
                        'R1': Path(row['R1']).name,
                        'group': group_name,
                        'replicate': rep_i,
                        'color': (condition_i * 10 + group_i), 
                    }
                    if 'R2' in row:
                        samples[row['sample_name']]['R2'] = Path(row['R2']).name

                    for full, short in zip(condition_columns, shortened_condition_columns):
                        samples[row['sample_name']][short] = row[full]

        else:
            for rep_i, (_, row) in enumerate(group_rows.iterrows(), 1):
                samples[row['sample_name']] = {
                    'R1': Path(row['R1']).name,
                    'group': group_name,
                    'replicate': rep_i,
                    'color': group_i,
                }
                if 'R2' in row:
                    samples[row['sample_name']]['R2'] = Path(row['R2']).name

    if batch_name is None:
        fn_parents = {Path(fn).parent for fn in df['R1']}

        batch_names = {fn_parent.parts[3] for fn_parent in fn_parents}
        if len(batch_names) > 1:
            raise ValueError(batch_names)
        else:
            batch_name = batch_names.pop()

    batch_dir = Path(base_dir) / 'data' / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    groups_df = pd.DataFrame.from_dict(groups, orient='index')
    groups_df.index.name = 'group'

    groups_csv_fn = batch_dir / 'group_descriptions.csv'
    groups_df.to_csv(groups_csv_fn)

    samples_df = pd.DataFrame.from_dict(samples, orient='index')
    samples_df.index.name = 'sample_name'

    samples_csv_fn = batch_dir / 'sample_sheet.csv'
    samples_df.to_csv(samples_csv_fn)

    return samples_df, groups_df