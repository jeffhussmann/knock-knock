from collections import Counter, defaultdict

import h5py
import numpy as np
import pandas as pd

import hits.utilities
memoized_property = hits.utilities.memoized_property
memoized_with_kwargs = hits.utilities.memoized_with_kwargs

def cumulative_from_end(array):
    return np.cumsum(array[::-1])[::-1]

class OutcomeStratifiedLengths:
    def __init__(self, outcome_iter, max_relevant_length, length_to_store_unknown):
        self.max_relevant_length = max_relevant_length
        self.length_to_store_unknown = length_to_store_unknown
        self.max_length = max(max_relevant_length, length_to_store_unknown)

        outcome_lengths = defaultdict(Counter)

        for outcome in outcome_iter:
            outcome_lengths[outcome.category, outcome.subcategory][outcome.inferred_amplicon_length] += 1

        self.outcome_length_arrays = {}
        for (cat, subcat), counts in outcome_lengths.items():
            array = np.zeros(self.max_length + 1, dtype=int)
            for length, value in counts.items():
                if length == -1:
                    array[self.length_to_store_unknown] = value
                elif length >= self.max_relevant_length:
                    array[self.max_relevant_length] += value
                else:
                    array[length] = value

            self.outcome_length_arrays[cat, subcat] = array

    def to_file(self, fn):
        with h5py.File(fn, 'w') as fh:
            fh.attrs['max_relevant_length'] = self.max_relevant_length
            fh.attrs['length_to_store_unknown'] = self.length_to_store_unknown

            for (cat, subcat), counts in self.outcome_length_arrays.items():
                # object names can't have '/' characters in them.
                subcat = subcat.replace('/', 'SANITIZED_SLASH')
                fh.create_dataset(f'{cat}/{subcat}', data=counts)

    @classmethod
    def from_file(cls, fn):
        with h5py.File(fn, 'r') as fh:
            lengths = cls([], fh.attrs['max_relevant_length'], fh.attrs['length_to_store_unknown'])

            with h5py.File(fn, 'r') as fh:
                for cat, group in fh.items():
                    for subcat, dataset in group.items():
                        # Undo the sanitization of '/' characters.
                        subcat = subcat.replace('SANITIZED_SLASH', '/')
                        lengths.outcome_length_arrays[cat, subcat] = dataset[()]

        return lengths

    @memoized_property
    def lengths_for_all_outcomes(self):
        if len(self.outcome_length_arrays) == 0:
            lengths = np.array([], dtype=int)
        else:
            lengths = sum(self.outcome_length_arrays.values())
        return lengths

    @memoized_property
    def total_reads(self):
        return self.lengths_for_all_outcomes.sum()

    @memoized_with_kwargs
    def outcome_highest_points(self, *, smooth_window=0):
        ''' Dictionary of {outcome: maximum of that outcome's read length frequency distribution} '''
        highest_points = {}

        for outcome, lengths in self.outcome_length_arrays.items():
            window = smooth_window * 2 + 1
            smoothed_lengths = pd.Series(lengths).rolling(window=window, center=True, min_periods=1).sum()
            highest_points[outcome] = max(smoothed_lengths / self.total_reads * 100)

        return highest_points

    @memoized_with_kwargs
    def outcome_to_color(self, *, smooth_window=0):
        # To minimize the chance that a color will be used more than once in the same panel in 
        # outcome_stratified_lengths plots, sort color order by highest point.
        highest_points = self.outcome_highest_points(smooth_window=smooth_window)
        color_order = sorted(highest_points, key=highest_points.get, reverse=True)
        return {outcome: f'C{i % 10}' for i, outcome in enumerate(color_order)}

    def truncate_to_max_length(self, new_max_length):
        if new_max_length >= self.max_relevant_length:
            truncated = self
        else:
            new_length_to_store_unknown = int(1.05 * new_max_length)
            truncated = OutcomeStratifiedLengths([], new_max_length, new_length_to_store_unknown)

            for (cat, subcat), counts in self.outcome_length_arrays.items():
                new_counts = np.zeros(new_length_to_store_unknown + 1, dtype=int)
                new_counts[new_length_to_store_unknown] = counts[self.length_to_store_unknown]
                new_counts[new_max_length] = counts[new_max_length:self.max_relevant_length + 1].sum()
                new_counts[:new_max_length] = counts[:new_max_length]
                truncated.outcome_length_arrays[cat, subcat] = new_counts

        return truncated

    @memoized_property
    def cumulative_lengths_for_all_outcomes(self):
        return cumulative_from_end(self.lengths_for_all_outcomes)

    @memoized_property
    def cumulative_frequencies_from_end(self):
        cumulative_fs = {}

        for (cat, subcat), counts in self.outcome_length_arrays.items():
            cumulative_fs[cat, subcat] = cumulative_from_end(counts) / self.cumulative_lengths_for_all_outcomes

        df = pd.DataFrame(cumulative_fs).T
        df.columns.name = 'length'
        df.index.names = ['category', 'subcategory']

        return df
