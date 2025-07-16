import random
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

import hits.fastq
import hits.utilities

import knock_knock.outcome
import knock_knock.illumina_experiment

import knock_knock.experiment

parent = Path(__file__).parent

def generate_simulated_reads(strat, reads_per_sequence=1):
    subs = strat.pegRNA_substitutions[strat.target]

    reads = []

    total_read_i = 0

    expected_categorizations = {}
    
    for subs_included in hits.utilities.powerset(subs):
        edited_target_sequence_list = list(strat.target_sequence)
        
        for sub_name in subs_included:
            edited_target_sequence_list[subs[sub_name]['position']] = subs[sub_name]['alternative_base']
            
        edited_target_sequence = ''.join(edited_target_sequence_list)
        
        amplicon = edited_target_sequence[strat.amplicon_interval.start:strat.amplicon_interval.end + 1]
        
        if strat.sequencing_direction == '-':
            amplicon = hits.utilities.reverse_complement(amplicon)

        qual = hits.fastq.unambiguous_sanger_Q40(len(amplicon))
    
        programmed_substitution_read_bases = ''.join(subs[sub_name]['alternative_base'] if sub_name in subs_included else '_' for sub_name in sorted(subs))

        if len(subs_included) == 0:
            category = 'wild type'
            subcategory = 'clean'
            details = knock_knock.outcome.Details()

        else:
            if len(subs_included) < len(subs):
                category = 'partial edit'
                subcategory = 'partial incorporation'
            else:
                category = 'intended edit'
                subcategory = 'substitution'

            details = knock_knock.outcome.Details(programmed_substitution_read_bases=programmed_substitution_read_bases)

        for read_i in range(reads_per_sequence):
            name = f'simulated:{total_read_i:06d}'

            total_read_i += 1
        
            read = hits.fastq.Read(name, amplicon, qual)
            reads.append(read)

            expected_categorizations[name] = {
                'category': category,
                'subcategory': subcategory,
                'details': str(details),
            }

    return reads, expected_categorizations

def generate_test(existing_base_dir, existing_strategy_name, pegRNA, prefix, test_base_dir, reads_per_sequence=1):
    test_base_dir = Path(test_base_dir)

    # Copy relevant editing strategy into test directory structure

    existing_strat = knock_knock.editing_strategy.EditingStrategy(existing_base_dir, existing_strategy_name, sgRNAs=[pegRNA])

    targets_dir = test_base_dir / 'targets'

    prefixed_name = f'{prefix}_{pegRNA}'

    new_dir = targets_dir / prefixed_name
    new_dir.mkdir(exist_ok=True)

    manifest_fn = new_dir / 'manifest.yaml'

    manifest_fn.write_text(yaml.safe_dump(existing_strat.manifest))

    for source in existing_strat.sources:
        shutil.copy(existing_strat.dir / (source + '.gb'), new_dir / (source + '.gb'))

    sgRNAs = knock_knock.pegRNAs.read_csv(existing_strat.fns['sgRNAs'], process=False).loc[[pegRNA]]

    sgRNAs.to_csv(new_dir / 'sgRNAs.csv')
    
    # Setup sample sheet

    data_dir = test_base_dir / 'data' / prefixed_name
    data_dir.mkdir(exist_ok=True)

    sample_sheet_fn = data_dir / 'sample_sheet.csv'

    sample_sheet = {
        'sample_name': 'simulated',
        'R1': 'simulated.fastq',
        'experiment_type': 'prime_editing',
        'editing_strategy': prefixed_name,
        'sgRNAs': pegRNA,
    }
    
    sample_sheet = pd.Series(sample_sheet).to_frame().T.set_index('sample_name')

    sample_sheet.to_csv(sample_sheet_fn)

    # Write fastq

    fastq_fn = data_dir / 'simulated.fastq'
    expected_categorizations_fn = data_dir / 'expected_categorizations.yaml'
    
    new_strat = knock_knock.editing_strategy.EditingStrategy(test_base_dir, prefixed_name, sgRNAs=[pegRNA])
    
    reads, expected_categorizations = generate_simulated_reads(new_strat, reads_per_sequence=reads_per_sequence)

    random.shuffle(reads)

    with open(fastq_fn, 'w') as fh:
        for read in reads:
            fh.write(str(read))

    expected_categorizations_fn.write_text(yaml.safe_dump(expected_categorizations))

@pytest.mark.parametrize('prefixed_name', knock_knock.experiment.get_all_batch_names(parent))
def test_partial_incorporation_categorization(prefixed_name):
    exp = knock_knock.illumina_experiment.IlluminaExperiment(parent, prefixed_name, 'simulated')

    if exp.results_dir.is_dir():
        shutil.rmtree(exp.results_dir)

    for stage in ['preprocess', 'align', 'categorize']:
        exp.process(stage=stage)

    expected_categorizations_fn = exp.data_dir / 'expected_categorizations.yaml'
    expected_categorizations = yaml.safe_load(expected_categorizations_fn.read_text())

    for outcome in exp.outcome_iter():
        expected_categorization = expected_categorizations[outcome.query_name] 
        assert outcome.category == expected_categorization['category']
        assert outcome.subcategory == expected_categorization['subcategory']
        assert str(outcome.details) == expected_categorization['details']

    shutil.rmtree(exp.results_dir)
