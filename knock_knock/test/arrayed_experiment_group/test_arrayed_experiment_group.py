import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import hits.utilities

import knock_knock.arrayed_experiment_group

import knock_knock.test
import knock_knock.test.partial_incorporation

memoized_property = hits.utilities.memoized_property

parent = Path(__file__).resolve().parent

@dataclass(frozen=True)
class GroupIdentifier(knock_knock.arrayed_experiment_group.GroupIdentifier):
    results_prefix: str

class Group(knock_knock.arrayed_experiment_group.ArrayedIlluminaExperimentGroup):
    def __init__(self, identifier):
        self.results_prefix = identifier.results_prefix
        super().__init__(identifier)

    @memoized_property
    def results_dir(self):
        d = super().results_dir

        return d.parent / f'{self.results_prefix}_{d.name}'

class Comparison:
    base_dir = parent
    sample_name_template = 'sample_{:03d}'

    def __init__(self, name):
        self.name = name

        self.metadata_dir = knock_knock.arrayed_experiment_group.get_metadata_dir(type(self).base_dir) / self.name
        self.data_dir = type(self).base_dir / 'data' / name
        self.strategy_dir = type(self).base_dir / 'strategies' / name

        self.sample_names = [type(self).sample_name_template.format(sample_i) for sample_i in range(1, 4)]
        self.fastq_fns = {name: {which: self.data_dir / f'{name}_{which}.fastq' for which in ['R1', 'R2']} for name in self.sample_names}

        self.batch_id = knock_knock.arrayed_experiment_group.BatchIdentifier(type(self).base_dir,
                                                                             self.name,
                                                                            )

    def batch(self):
        return knock_knock.arrayed_experiment_group.Batch(self.batch_id)

    def group(self, results_prefix):
        batch = self.batch()

        if len(batch.group_names) != 1:
            raise ValueError

        group_name = batch.group_names[0]

        group_id = GroupIdentifier(self.batch_id,
                                   group_name,
                                   results_prefix,
                                  )

        return Group(group_id)

    @memoized_property
    def expected_results_group(self):
        return self.group('expected')

    @memoized_property
    def actual_results_group(self):
        group = self.group('actual')

        if group.results_dir.exists():
            shutil.rmtree(group.results_dir)

        return group

    def extract_test_from_existing_editing_strategy(self, editing_strategy):
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.metadata_dir.mkdir(exist_ok=True, parents=True)

        self.copy_metadata_sequences(editing_strategy)

        sample_sheet = self.generate_sample_sheet(editing_strategy)
        sample_sheet.to_csv(self.metadata_dir / 'sample_sheet.csv')

        knock_knock.arrayed_experiment_group.setup_from_metadata(type(self).base_dir, self.name)

        self.generate_fastqs(editing_strategy)

    def copy_metadata_sequences(self, editing_strategy):
        strat = editing_strategy

        for source in strat.sources:
            shutil.copy(strat.dir / (source + '.gb'), self.metadata_dir / (source + '.gb'))

        shutil.copy(strat.dir / 'sgRNAs.csv', self.metadata_dir)
        shutil.copy(strat.dir.parent / 'amplicon_primers.csv', self.metadata_dir)

    def generate_fastqs(self, editing_strategy):
        for sample_i, sample_name in enumerate(self.sample_names):
            reads, _ = knock_knock.test.partial_incorporation.generate_simulated_reads(editing_strategy,
                                                                                       reads_per_sequence=10 * sample_i,
                                                                                      )

            with open(self.fastq_fns[sample_name]['R1'], 'w') as R1_fh, open(self.fastq_fns[sample_name]['R2'], 'w') as R2_fh:
                for R1, R2 in reads:
                    # Note: name already has prefix
                    R1_fh.write(str(R1))
                    R2_fh.write(str(R2))

    def generate_sample_sheet(self, editing_strategy):

        sample_sheet = {}
        
        for sample_name in self.sample_names:
            sample_sheet[sample_name] = {
                'R1': self.fastq_fns[sample_name]['R1'].name,
                'R2': self.fastq_fns[sample_name]['R2'].name,
                'genome': editing_strategy.target,
                'amplicon_primers': editing_strategy.target,
                'sgRNAs': ';'.join(editing_strategy.pegRNA_names),
            }
        
        sample_sheet = pd.DataFrame.from_dict(sample_sheet, orient='index')
        sample_sheet.index.name = 'sample_name'

        return sample_sheet

    def process(self):
        self.actual_results_group.process(num_processes=1, generate_summary_figures=False)

def get_all_comparisons():
    dirs = (parent / 'data').iterdir()
    
    comparisons = {}

    for d in dirs:
        name = d.name
        comparison = Comparison(name)
        comparisons[name] = comparison

    return comparisons

def test_outcome_counts(comparison):
    expected = comparison.expected_results_group.outcome_counts()
    actual = comparison.actual_results_group.outcome_counts()
    assert (expected == actual).all(axis=None)
