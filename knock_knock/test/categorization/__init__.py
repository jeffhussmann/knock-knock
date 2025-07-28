from pathlib import Path
import shutil
import textwrap

import pandas as pd
import yaml

import hits.utilities

memoized_property = hits.utilities.memoized_property

import knock_knock.illumina_experiment
import knock_knock.pegRNAs

parent = Path(__file__).resolve().parent

class Experiment(knock_knock.illumina_experiment.IlluminaExperiment):
    def __init__(self, base_dir, batch_name, sample_name, results_prefix):
        self.results_prefix = results_prefix
        super().__init__(base_dir, batch_name, sample_name)

    @memoized_property
    def results_dir(self):
        d = super().results_dir

        return d.parent / f'{self.results_prefix}_{d.name}'

class ReadSet:
    ''' data contains fastq and expected_categorizations.yaml
    alignment_results has outputs of execeuting preprocess and align
    results gets rmtree'ed, then preprocess, align, categorize
    categorization_results gets rmtree'ed, copied from alignment, then categorized
    '''

    read_prefix = 'test:'

    sample_name = 'extracted'

    base_dir = parent

    def __init__(self, name):
        self.name = name

        self.data_dir = type(self).base_dir / 'data' / name
        self.strategy_dir = type(self).base_dir / 'strategies' / name

        self.fastq_fn = self.data_dir / f'{type(self).sample_name}.fastq'
        self.expected_categorizations_fn = self.data_dir / 'expected_categorizations.yaml'

    def experiment(self, results_prefix):
        return Experiment(type(self).base_dir, self.name, type(self).sample_name, results_prefix)

    @memoized_property
    def expected_categorizations(self):
        return yaml.safe_load(self.expected_categorizations_fn.read_text())

    @memoized_property
    def alignment_results_experiment(self):
        return self.experiment('alignment')

    @memoized_property
    def categorization_results_experiment(self):
        exp = self.experiment('categorization')

        if exp.results_dir.exists():
            shutil.rmtree(exp.results_dir)

        shutil.copytree(self.alignment_results_experiment.results_dir, exp.results_dir)

        return exp

    @memoized_property
    def full_results_experiment(self):
        exp = self.experiment('full')

        if exp.results_dir.exists():
            shutil.rmtree(exp.results_dir)

        return exp

    def copy_editing_strategy(self, editing_strategy):
        strat = editing_strategy

        self.strategy_dir.mkdir(exist_ok=True, parents=True)

        manifest_fn = self.strategy_dir / 'manifest.yaml'

        manifest_fn.write_text(yaml.safe_dump(strat.manifest))

        for source in strat.sources:
            shutil.copy(strat.dir / (source + '.gb'), self.strategy_dir / (source + '.gb'))

        sgRNAs = knock_knock.pegRNAs.read_csv(strat.fns['sgRNAs'], process=False).loc[strat.sgRNAs]

        sgRNAs.to_csv(self.strategy_dir / 'sgRNAs.csv')

    def copy_sample_sheet(self, existing_exp):
        sample_sheet_fn = self.data_dir / 'sample_sheet.csv'

        sample_sheet = {
            'sample_name': type(self).sample_name,
            'R1': self.fastq_fn.name,
            'experiment_type': existing_exp.experiment_type,
            'editing_strategy': self.name,
            'sequencing_start_feature_name': existing_exp.editing_strategy.sequencing_start_feature_name,
            'sgRNAs': ';'.join(existing_exp.editing_strategy.sgRNAs),
        }
        
        sample_sheet = pd.Series(sample_sheet).to_frame().T.set_index('sample_name')

        sample_sheet.to_csv(sample_sheet_fn)

    def extract_fastq(self, existing_exp):

        reads = [read for read in existing_exp.reads if read.name in self.expected_categorizations]

        if len(reads) != len(self.expected_categorizations):
            raise ValueError
        
        with open(self.fastq_fn, 'w') as fh:
            for read in reads:
                read.name = f'{ReadSet.read_prefix}{read.name}'
                fh.write(str(read))

    def process_alignment_results_experiment(self):
        exp = self.alignment_results_experiment

        if exp.results_dir.exists():
            shutil.rmtree(exp.results_dir)

        exp.process(stage='preprocess')
        exp.process(stage='align')

    def extract_test_from_existing_experiment(self, existing_experiment):
        self.data_dir.mkdir(exist_ok=True, parents=True)

        self.copy_editing_strategy(existing_experiment.editing_strategy)
        self.copy_sample_sheet(existing_experiment)
        self.extract_fastq(existing_experiment)
        self.process_alignment_results_experiment()

    @memoized_property
    def categorized_from_fastq(self):
        exp = self.full_results_experiment

        for stage in ['preprocess', 'align', 'categorize']:
            exp.process(stage=stage)

        categorizations = {outcome.query_name[len(type(self).read_prefix):]: outcome for outcome in exp.outcome_iter()}

        return categorizations

    @memoized_property
    def categorized_from_alignments(self):
        exp = self.categorization_results_experiment

        exp.process(stage='categorize')

        categorizations = {outcome.query_name[len(ReadSet.read_prefix):]: outcome for outcome in exp.outcome_iter()}

        return categorizations

    def compare_to_expected(self, read_name, source_name='fastq'):
        if source_name == 'fastq':
            source = self.categorized_from_fastq
        elif source_name == 'alignments':
            source = self.categorized_from_alignments
        else:
            raise ValueError(source)

        if read_name not in source:
            raise ValueError(f'{read_name} not found in output from {source_name}, {source}')

        actual = source[read_name]

        expected = self.expected_categorizations[read_name]

        matches = (
            actual.category == expected['category'] and \
            actual.subcategory == expected['subcategory'] and \
            ('details' not in expected or (str(actual.details) == expected['details']))
        )

        diagnostic_message = textwrap.dedent(f'''
            query name: {read_name}
            expected: ({expected['category']}, {expected['subcategory']}, {expected.get('details')})
            actual: ({actual.category}, {actual.subcategory}, {actual.details})
            note: {expected.get('note', '')}
        ''')

        return matches, diagnostic_message

def get_all_read_sets():
    set_dirs = (parent / 'data').iterdir()
    
    read_sets = {}

    for set_dir in set_dirs:
        set_name = set_dir.name
        read_set = ReadSet(set_name)
        read_sets[set_name] = read_set

    return read_sets
