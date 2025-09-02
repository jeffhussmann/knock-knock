from pathlib import Path
import shutil
import textwrap

import pandas as pd
import yaml

import hits.utilities

memoized_property = hits.utilities.memoized_property

import knock_knock.experiment
import knock_knock.illumina_experiment
import knock_knock.pacbio_experiment
import knock_knock.test

parent = Path(__file__).resolve().parent

class ReadSet(knock_knock.test.Extractor):
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

        self.R1_fastq_fn = self.data_dir / f'{type(self).sample_name}_R1.fastq'
        self.R2_fastq_fn = self.data_dir / f'{type(self).sample_name}_R2.fastq'

        self.CCS_fastq_fn = self.data_dir / f'{type(self).sample_name}_CCS.fastq'

        self.expected_categorizations_fn = self.data_dir / 'expected_categorizations.yaml'

    def __str__(self):
        return self.name

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

    def copy_sample_sheet(self, existing_exp):

        sample_sheet = {
            'sample_name': type(self).sample_name,
            'experiment_type': existing_exp.experiment_type,
            'editing_strategy': self.name,
            'sequencing_start_feature_name': existing_exp.editing_strategy.sequencing_start_feature_name,
            'sgRNAs': ';'.join(existing_exp.editing_strategy.sgRNAs),
        }

        if isinstance(existing_exp, knock_knock.illumina_experiment.IlluminaExperiment):
            sample_sheet['platform'] = 'illumina'

            sample_sheet['R1'] = self.R1_fastq_fn.name

            if existing_exp.paired_end:
                sample_sheet['R2'] = self.R2_fastq_fn.name

        elif isinstance(existing_exp, knock_knock.pacbio_experiment.PacbioExperiment):
            sample_sheet['platform'] = 'pacbio'

            sample_sheet['CCS_fastq_fn'] = self.CCS_fastq_fn.name

        else:
            raise ValueError

        sample_sheet = pd.Series(sample_sheet).to_frame().T.set_index('sample_name')

        sample_sheet.to_csv(self.sample_sheet_fn)

    def extract_fastq(self, existing_exp):
        reads = [read for read in existing_exp.reads if read.name in self.expected_categorizations]

        if len(reads) != len(self.expected_categorizations):
            raise ValueError
        
        if isinstance(existing_exp, knock_knock.illumina_experiment.IlluminaExperiment):
            fastq_fn = self.R1_fastq_fn
        elif isinstance(existing_exp, knock_knock.pacbio_experiment.PacbioExperiment):
            fastq_fn = self.CCS_fastq_fn

        with open(fastq_fn, 'w') as fh:
            for read in reads:
                read.name = f'{ReadSet.read_prefix}{read.name}'
                fh.write(str(read))

    def extract_paired_fastqs(self, existing_exp):
        read_pairs = [(R1, R2) for R1, R2 in existing_exp.read_pairs if R1.name in self.expected_categorizations]

        if len(read_pairs) != len(self.expected_categorizations):
            raise ValueError
        
        with open(self.R1_fastq_fn, 'w') as R1_fh, open(self.R2_fastq_fn, 'w') as R2_fh:
            for R1, R2 in read_pairs:
                R1.name = f'{ReadSet.read_prefix}{R1.name}'
                R1_fh.write(str(R1))

                R2.name = f'{ReadSet.read_prefix}{R2.name}'
                R2_fh.write(str(R2))

    def process_alignment_results_experiment(self):
        exp = self.alignment_results_experiment

        if exp.results_dir.exists():
            shutil.rmtree(exp.results_dir)

        exp.process(stage='preprocess')
        exp.process(stage='align')

    def extract_test_from_existing_experiment(self, existing_exp):
        self.data_dir.mkdir(exist_ok=True, parents=True)

        self.copy_editing_strategy(existing_exp.editing_strategy)
        self.copy_sample_sheet(existing_exp)

        if isinstance(existing_exp, knock_knock.illumina_experiment.IlluminaExperiment) and existing_exp.paired_end:
            self.extract_paired_fastqs(existing_exp)
        else:
            self.extract_fastq(existing_exp)

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
