from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import hits.utilities
memoized_property = hits.utilities.memoized_property
memoized_with_kwargs = hits.utilities.memoized_with_kwargs

def cumulative_from_end(array):
    return np.cumsum(array[::-1])[::-1]

class OutcomeStratifiedLengths:
    def __init__(self,
                 outcome_iter,
                 min_relevant_length,
                 max_relevant_length,
                 length_to_store_unknown,
                 non_relevant_categories,
                ):

        self.min_relevant_length = min_relevant_length
        self.max_relevant_length = max_relevant_length
        self.length_to_store_unknown = length_to_store_unknown
        self.max_length = max(max_relevant_length, length_to_store_unknown)
        self.non_relevant_categories = non_relevant_categories

        if self.length_to_store_unknown <= self.max_relevant_length:
            raise ValueError

        subcategory_lengths = defaultdict(Counter)

        for outcome in outcome_iter:
            subcategory_lengths[outcome.category, outcome.subcategory][outcome.inferred_amplicon_length] += 1

        index = pd.MultiIndex.from_tuples(sorted(subcategory_lengths), names=['category', 'subcategory'])
        columns = pd.RangeIndex(self.min_relevant_length, self.length_to_store_unknown + 1, name='length')

        self.subcategory_lengths_df = pd.DataFrame(0, index=index, columns=columns, dtype=int)

        for (cat, subcat), counts in subcategory_lengths.items():
            for length, value in counts.items():
                if length is None:
                    length = self.length_to_store_unknown

                elif length <= self.min_relevant_length:
                    length = self.min_relevant_length

                elif length >= self.max_relevant_length:
                    length = self.max_relevant_length

                self.subcategory_lengths_df.loc[(cat, subcat), length] += value

    def to_file(self, fn):
        with pd.HDFStore(fn, mode='w') as store:
            store['subcategory_lengths_df'] = self.subcategory_lengths_df

            attrs = store.get_storer('subcategory_lengths_df').attrs
            attrs.min_relevant_length = self.min_relevant_length
            attrs.max_relevant_length = self.max_relevant_length
            attrs.length_to_store_unknown = self.length_to_store_unknown
            attrs.non_relevant_categories = sorted(self.non_relevant_categories)

    @classmethod
    def from_file(cls, fn):
        with pd.HDFStore(fn) as store:
            attrs = store.get_storer('subcategory_lengths_df').attrs
            lengths = cls([],
                          attrs.min_relevant_length,
                          attrs.max_relevant_length,
                          attrs.length_to_store_unknown,
                          attrs.non_relevant_categories,
                         )

            lengths.subcategory_lengths_df = store.subcategory_lengths_df

        return lengths

    @classmethod
    def combine(cls, osl_list):
        min_relevant_lengths = set(osl.min_relevant_length for osl in osl_list)
        if len(min_relevant_lengths) > 1:
            raise ValueError
        else:
            min_relevant_length = list(min_relevant_lengths)[0]

        max_relevant_lengths = set(osl.max_relevant_length for osl in osl_list)
        if len(max_relevant_lengths) > 1:
            raise ValueError
        else:
            max_relevant_length = list(max_relevant_lengths)[0]

        length_to_store_unknowns = set(osl.length_to_store_unknown for osl in osl_list)
        if len(length_to_store_unknowns) > 1:
            raise ValueError
        else:
            length_to_store_unknown = list(length_to_store_unknowns)[0]

        non_relevant_categories = set()
        for osl in osl_list:
            non_relevant_categories.update(osl.non_relevant_categories)

        combined = cls([],
                       min_relevant_length,
                       max_relevant_length,
                       length_to_store_unknown,
                       non_relevant_categories,
                      )

        concatenated = pd.concat({i: osl.subcategory_lengths_df for i, osl in enumerate(osl_list)}, axis=0, names=['sample'])
        levels_to_keep = [level for level in concatenated.index.names if level != 'sample']
        summed = concatenated.groupby(levels_to_keep).sum()

        combined.subcategory_lengths_df = summed

        return combined

    def truncate(self, new_min_length, new_max_length):
        if (new_min_length <= self.min_relevant_length) and (new_max_length >= self.max_relevant_length):
            truncated = self

        else:
            new_length_to_store_unknown = int(1.05 * new_max_length)

            truncated = OutcomeStratifiedLengths([],
                                                 new_min_length,
                                                 new_max_length,
                                                 new_length_to_store_unknown,
                                                 self.non_relevant_categories,
                                                )


            old_df = self.subcategory_lengths_df

            index = old_df.index
            columns = pd.RangeIndex(new_min_length, new_length_to_store_unknown + 1, name='length')
            new_df = pd.DataFrame(0, index=index, columns=columns)

            # Copy over counts strictly between new_min_length and new_max_length.
            new_df.loc[:, new_min_length + 1:new_max_length - 1] = old_df.loc[:, new_min_length + 1:new_max_length - 1]

            new_df[new_length_to_store_unknown] = old_df[self.length_to_store_unknown]

            new_df[new_min_length] = old_df.loc[:, self.min_relevant_length:new_min_length].sum(axis=1)

            new_df[new_max_length] = old_df.loc[:, new_max_length:self.length_to_store_unknown - 1].sum(axis=1)

            assert (new_df.sum(axis=1) == old_df.sum(axis=1)).all()

            truncated.subcategory_lengths_df = new_df

        return truncated

    def truncate_to_max_observed_length(self, only_relevant=False):
        ls = self.lengths_for_all_outcomes(only_relevant=only_relevant)
        max_observed_length = max((l for l in ls[ls > 0].index if l != self.length_to_store_unknown), default=self.max_relevant_length)

        # Add a 5% buffer, then round up to the nearest multiple of 500.
        new_max_length = int(np.ceil((max_observed_length * 1.05) / 500) * 500)

        return self.truncate(0, new_max_length)

    @memoized_with_kwargs
    def lengths_df(self, *, level='subcategory', only_relevant=False):
        df = self.subcategory_lengths_df

        if level == 'subcategory':
            pass
        elif level == 'category': 
            df = df.groupby('category').sum()

        if only_relevant:
            df = df.query('category not in @self.non_relevant_categories')

        return df

    def by_outcome(self, outcome=None, only_relevant=False):
        if outcome is None:
            lengths = self.lengths_for_all_outcomes(only_relevant=only_relevant)

        else:
            if isinstance(outcome, str):
                level = 'category'
            else:
                level = 'subcategory'

            lengths = self.lengths_df(level=level).loc[outcome]

        return lengths

    @memoized_with_kwargs
    def cumulative_lengths_from_end(self, *, level='subcategory'):
        return self.lengths_df(level=level).iloc[:, ::-1].cumsum(axis=1).iloc[:, ::-1]

    @memoized_with_kwargs
    def lengths_for_all_outcomes(self, *, only_relevant=False):
        lengths = self.lengths_df(only_relevant=only_relevant).sum()

        return lengths

    @memoized_with_kwargs
    def total_reads(self, *, only_relevant=False):
        return self.lengths_for_all_outcomes(only_relevant=only_relevant).sum()

    @memoized_with_kwargs
    def highest_points(self, *, level='subcategory', only_relevant=False, smooth_window=0):
        ''' Dictionary of {(category, subcategory): maximum of that subcategory's read length frequency distribution} '''

        highest_points = {}

        df = self.lengths_df(level=level, only_relevant=only_relevant)

        for outcome, lengths in df.iterrows():
            window = smooth_window * 2 + 1
            smoothed_lengths = lengths.rolling(window=window, center=True, min_periods=1).sum()
            highest_points[outcome] = max(smoothed_lengths / self.total_reads(only_relevant=only_relevant) * 100)

        return highest_points

    @memoized_with_kwargs
    def outcome_to_color(self, *, only_relevant=False, smooth_window=0):
        # To minimize the chance that a color will be used more than once in the same panel in 
        # outcome_stratified_lengths plots, sort color order by highest point.

        outcome_to_color = {}

        for level in ['category', 'subcategory']:
            highest_points = self.highest_points(level=level, only_relevant=only_relevant, smooth_window=smooth_window)
            color_order = sorted(highest_points, key=highest_points.get, reverse=True)
            for i, outcome in enumerate(color_order):
                outcome_to_color[outcome] = f'C{i % 10}'

        return outcome_to_color

    @memoized_property
    def cumulative_lengths_from_end_for_all_reads(self):
        return cumulative_from_end(self.lengths_for_all_reads)

    @memoized_property
    def cumulative_lengths_from_end_for_relevant_reads(self):
        return cumulative_from_end(self.lengths_for_relevant_reads)

    @memoized_with_kwargs
    def cumulative_frequencies_from_end(self, *, level='subcategory'):
        numerator = self.cumulative_lengths_from_end(level=level)
        denominator = np.maximum(self.cumulative_lengths_from_end_for_relevant_reads, 1)

        return numerator / denominator