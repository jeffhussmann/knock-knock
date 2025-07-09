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

test_base_dir = Path(__file__).parent

def generate_simulated_reads(strat):
    subs = strat.pegRNA_substitutions[strat.target]

    reads = []
    
    for subs_included in hits.utilities.powerset(subs):
        edited_target_sequence_list = list(strat.target_sequence)
        
        for sub_name in subs_included:
            edited_target_sequence_list[subs[sub_name]['position']] = subs[sub_name]['alternative_base']
            
        edited_target_sequence = ''.join(edited_target_sequence_list)
        
        amplicon = edited_target_sequence[strat.amplicon_interval.start:strat.amplicon_interval.end + 1]
        
        if strat.sequencing_direction == '-':
            amplicon = hits.utilities.reverse_complement(amplicon)
    
        programmed_substitution_read_bases = ''.join(subs[sub_name]['alternative_base'] if sub_name in subs_included else '_' for sub_name in sorted(subs))
        qual = hits.fastq.unambiguous_sanger_Q40(len(amplicon))
        
        name = f'simulated:{knock_knock.outcome.Details(programmed_substitution_read_bases=programmed_substitution_read_bases)}'
    
        read = hits.fastq.Read(name, amplicon, qual)
        reads.append(read)

    return reads

def generate_test(base_dir, target_name, pegRNA, prefix):
    # Copy relevant editing strategy into test directory structure

    existing_strat = knock_knock.target_info.TargetInfo(base_dir, target_name, sgRNAs=[pegRNA])

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
        'target_info': prefixed_name,
        'sgRNAs': pegRNA,
    }
    
    sample_sheet = pd.Series(sample_sheet).to_frame().T.set_index('sample_name')

    sample_sheet.to_csv(sample_sheet_fn)

    # Write fastq

    fastq_fn = data_dir / 'simulated.fastq'
    
    new_strat = knock_knock.target_info.TargetInfo(test_base_dir, prefixed_name, sgRNAs=[pegRNA])
    
    reads = generate_simulated_reads(new_strat)

    with open(fastq_fn, 'w') as fh:
        for read in reads:
            fh.write(str(read))

@pytest.mark.parametrize('prefixed_name', knock_knock.experiment.get_all_batch_names(test_base_dir))
def test_partial_incorporation_categorization(prefixed_name):
    exp = knock_knock.illumina_experiment.IlluminaExperiment(test_base_dir, prefixed_name, 'simulated')

    exp.process(stage='preprocess')
    exp.process(stage='align')
    exp.process(stage='categorize')

    for outcome in exp.outcome_iter():
        
        expected_details = knock_knock.outcome.Details.from_string(outcome.query_name.lstrip('simulated:'))
        
        if set(expected_details.programmed_substitution_read_bases) == {'_'}:
            expected_category = 'wild type'
            expected_details = knock_knock.outcome.Details()
        
        else:
            if '_' not in set(expected_details.programmed_substitution_read_bases):
                expected_category = 'intended edit'
            else:
                expected_category = 'partial edit'
                
        assert outcome.category == expected_category
        assert outcome.details == expected_details

    shutil.rmtree(exp.results_dir)
