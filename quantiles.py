import random
from collections import Counter, defaultdict
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
import numpy as np
import h5py
import tqdm

from knockin import coherence, experiment
from sequencing import utilities

progress = tqdm.tqdm

def get_quantile(leq, negative_geq, q):
    if q <= 0.5:
        # largest value such that no more than q * total are less than or equal to value '''
        total = leq[-1]
        quantile = np.searchsorted(leq, total * q)
        
    else:
        # smallest value such that no more than (1 - q) * total are greater than or equal to value '''
        total = negative_geq[0]
        quantile = np.searchsorted(negative_geq, total * (1 - q), side='right')
            
    return quantile

def quantiles_to_record(num_samples):
    to_record = {
        'median': 0.5,
    }

    for exponent in range(1, int(np.log10(num_samples)) + 1):
        to_record['down_{}'.format(exponent)] = 10**-exponent
        to_record['up_{}'.format(exponent)] = 1 - 10**-exponent

    return to_record

def sample_negative_controls(guide_outcomes, num_cells, num_samples):
    progress = utilities.identity
    print(num_cells)

    counts_by_sample = []
    outcome_names = set()
    
    qs = quantiles_to_record(num_samples)

    guides = list(guide_outcomes)

    for _ in range(num_samples):
        cells = []

        while len(cells) < num_cells:
            guide = random.choice(guides)
            cells.extend(guide_outcomes[guide])

        cells = random.sample(cells, num_cells)
        outcomes = ((outcome.category, outcome.subcategory, outcome.details) for outcome in cells)
        counts = Counter(outcomes)

        counts_by_sample.append(counts)
        outcome_names.update(counts)

    outcome_groups = {
        'deletion': {outcome for outcome in outcome_names if outcome[:2] == ('indel', 'deletion')},
        'insertion': {outcome for outcome in outcome_names if outcome[:2] == ('indel', 'insertion')},
        'partial_donor': {outcome for outcome in outcome_names if outcome[:2] == ('no indel', 'other')},
    }

    outcome_to_groups = defaultdict(set)
    for outcome_group_name, outcome_group in outcome_groups.items():
        for outcome in outcome_group:
            outcome_to_groups[outcome].add(outcome_group_name)

    filled_counts = {k: np.zeros(num_samples, int) for k in list(outcome_groups) + ['not_observed']}

    results = {}

    def process_frequencies(frequencies):
        full_frequencies = np.zeros(num_cells + 1, int)
        full_frequencies[:len(frequencies)] = frequencies

        leq = np.cumsum(full_frequencies)
        geq = leq[-1] - leq + full_frequencies
        negative_geq = -geq
        
        results = {
            'frequencies': frequencies,
            'quantiles': {}
        }
        
        for key, q in qs.items():
            results['quantiles'][key] = get_quantile(leq, negative_geq, q)

        return results

    for outcome in progress(outcome_names):
        filled = np.array([counts[outcome] for counts in counts_by_sample])
        frequencies = utilities.counts_to_array(Counter(filled))

        results[outcome] = process_frequencies(frequencies)

        for outcome_group_name in outcome_to_groups[outcome]:
            filled_counts[outcome_group_name] += filled
        
    for outcome, filled in filled_counts.items():
        frequencies = utilities.counts_to_array(Counter(filled))
        results[outcome] = process_frequencies(frequencies)
            
    return results

if __name__ == '__main__':
    base_dir = Path('/home/jah/projects/britt')
    group_name = '2018_09_07_rep1'
    pool = experiment.PooledExperiment(base_dir, group_name)

    num_cells_list = np.concatenate([np.arange(10, 100, 5),
                                     np.arange(100, 1000, 50),
                                     np.arange(1000, 5000, 100),
                                     np.arange(5000, 20000, 1000),
                                     np.arange(20000, 40000, 5000),
                                    ],
    )
    num_samples = 100000
    guide_outcomes = pool.non_targeting_outcomes

    args_list = [(guide_outcomes, num_cells, num_samples) for num_cells in num_cells_list]

    with Pool(processes=20) as process_pool:
        all_results = process_pool.starmap(sample_negative_controls, args_list, chunksize=1)
    
    #all_results = [sample_negative_controls(*args) for args in progress(args_list)]

    by_num_cells = dict(zip(num_cells_list, all_results))

    by_outcome = {}

    all_outcome_names = set()
    for num_cells in num_cells_list:
        all_outcome_names.update(by_num_cells[num_cells])
        
    all_outcome_names.remove('not_observed')

    for outcome_name in all_outcome_names:
        by_outcome[outcome_name] = {
            'quantiles': defaultdict(list),
            'frequencies': {},
        }

    for num_cells in num_cells_list:
        results = by_num_cells[num_cells]
        for outcome_name in all_outcome_names:
            outcome_results = results.get(outcome_name, results['not_observed'])
            
            for q, v in outcome_results['quantiles'].items():
                by_outcome[outcome_name]['quantiles'][q].append(v)
            
            by_outcome[outcome_name]['frequencies'][num_cells] = outcome_results['frequencies']

    hdf5_fn = pool.fns['quantiles']

    with h5py.File(hdf5_fn, 'w') as f:
        f.create_dataset('num_cells', data=num_cells_list)

        f.attrs['num_samples'] = num_samples

        for outcome, results in progress(by_outcome.items()):
            if isinstance(outcome, tuple):
                name = '_'.join(outcome)
            else:
                name = outcome

            group = f.create_group(name)

            frequencies = group.create_group('frequencies')
            for num_cells, vs in results['frequencies'].items():
                frequencies.create_dataset(str(num_cells), data=vs)

            quantiles = group.create_group('quantiles')
            for q, vs in results['quantiles'].items():
                quantiles.create_dataset(str(q), data=vs)
