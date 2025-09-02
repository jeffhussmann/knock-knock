import shutil
from pathlib import Path

import pandas as pd

import hits.utilities

import knock_knock.test

import knock_knock.test.partial_incorporation

memoized_property = hits.utilities.memoized_property

parent = Path(__file__).resolve().parent

class Comparison(knock_knock.test.Extractor):
    base_dir = parent
    sample_name = 'sample'

    def __init__(self, name):
        self.name = name

    @memoized_property
    def expected_results_experiment(self):
        return self.experiment('expected')

    @memoized_property
    def actual_results_experiment(self):
        exp = self.experiment('actual')

        if exp.results_dir.exists():
            shutil.rmtree(exp.results_dir)

        return exp

    def extract_test_from_existing_editing_strategy(self, editing_strategy):
        self.data_dir.mkdir(exist_ok=True, parents=True)

        self.copy_editing_strategy(editing_strategy)
        self.generate_fastqs(editing_strategy)
        self.generate_sample_sheet(editing_strategy)

    def generate_fastqs(self, editing_strategy):
        reads, expected_categorizations = knock_knock.test.partial_incorporation.generate_simulated_reads(editing_strategy, reads_per_sequence=100)

        with open(self.data_dir / 'R1.fastq', 'w') as R1_fh, open(self.data_dir / 'R2.fastq', 'w') as R2_fh:
            for R1, R2 in reads:
                # Note: name already has prefix
                R1_fh.write(str(R1))
                R2_fh.write(str(R2))

    def generate_sample_sheet(self, editing_strategy):
        sample_sheet_fn = self.data_dir / 'sample_sheet.csv'

        sample_sheet = {
            'sample_name': type(self).sample_name,
            'R1': 'R1.fastq',
            'R2': 'R2.fastq',
            'experiment_type': 'prime_editing' if len(editing_strategy.pegRNA_names) == 1 else 'twin_prime',
            'editing_strategy': self.name,
            'sequencing_start_feature_name': editing_strategy.sequencing_start_feature_name,
            'sgRNAs': ';'.join(editing_strategy.pegRNA_names),
            'platform': 'illumina',
        }
        
        sample_sheet = pd.Series(sample_sheet).to_frame().T.set_index('sample_name')

        sample_sheet.to_csv(sample_sheet_fn)

    def process(self):
        self.actual_results_experiment.process(stage='preprocess')
        self.actual_results_experiment.process(stage='align')
        self.actual_results_experiment.process(stage='categorize')

def get_all_comparisons():
    dirs = (parent / 'data').iterdir()
    
    comparisons = {}

    for d in dirs:
        name = d.name
        comparison = Comparison(name)
        comparisons[name] = comparison

    return comparisons

def test_outcome_counts(comparison):
    expected = comparison.expected_results_experiment.outcome_counts
    actual = comparison.actual_results_experiment.outcome_counts
    assert (expected == actual).all()