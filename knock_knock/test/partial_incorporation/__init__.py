import random
from pathlib import Path

import pandas as pd
import yaml

import hits.adapters
import hits.fastq
import hits.utilities

import knock_knock.outcome

import knock_knock.test.categorization

parent = Path(__file__).resolve().parent

class ReadSet(knock_knock.test.categorization.ReadSet):
    read_prefix = ''

    sample_name = 'simulated'

    base_dir = parent

    def extract_test_from_existing_editing_strategy(self, editing_strategy):
        self.data_dir.mkdir(exist_ok=True, parents=True)

        self.copy_editing_strategy(editing_strategy)
        self.generate_fastq_and_expected_categorizations(editing_strategy)
        self.generate_sample_sheet(editing_strategy)
        self.process_alignment_results_experiment()

    def generate_fastq_and_expected_categorizations(self, editing_strategy):
        reads, expected_categorizations = generate_simulated_reads(editing_strategy)

        with open(self.fastq_fn, 'w') as fh:
            for R1, R2 in reads:
                # Note: name already has prefix
                fh.write(str(R1))

        self.expected_categorizations_fn.write_text(yaml.safe_dump(expected_categorizations))

    def generate_sample_sheet(self, editing_strategy):
        sample_sheet_fn = self.data_dir / 'sample_sheet.csv'

        sample_sheet = {
            'sample_name': type(self).sample_name,
            'R1': self.fastq_fn.name,
            'experiment_type': 'prime_editing' if len(editing_strategy.pegRNA_names) == 1 else 'twin_prime',
            'editing_strategy': self.name,
            'sequencing_start_feature_name': editing_strategy.sequencing_start_feature_name,
            'sgRNAs': ';'.join(editing_strategy.pegRNA_names),
            'platform': 'illumina',
        }
        
        sample_sheet = pd.Series(sample_sheet).to_frame().T.set_index('sample_name')

        sample_sheet.to_csv(sample_sheet_fn)

def generate_simulated_reads(editing_strategy, reads_per_sequence=1, max_sequences=1000):
    strat = editing_strategy

    primers = hits.adapters.primers['truseq']

    subs = strat.pegRNA_substitutions[strat.target]

    reads = []

    total_read_i = 0

    expected_categorizations = {}
    
    subs_included_list = list(hits.utilities.powerset(subs))

    if len(subs_included_list) > max_sequences:
        subs_included_list = random.sample(subs_included_list, max_sequences)

    for subs_included in subs_included_list:
        edited_target_sequence_list = list(strat.target_sequence)
        
        for sub_name in subs_included:
            edited_target_sequence_list[subs[sub_name]['position']] = subs[sub_name]['alternative_base']
            
        edited_target_sequence = ''.join(edited_target_sequence_list)
        
        amplicon = edited_target_sequence[strat.amplicon_interval.start:strat.amplicon_interval.end + 1]
        
        if strat.sequencing_direction == '-':
            amplicon = hits.utilities.reverse_complement(amplicon)

        amplicon = primers['R1'] + amplicon + hits.utilities.reverse_complement(primers['R2'])

        qual = hits.fastq.unambiguous_sanger_Q40(len(amplicon))
    
        programmed_substitution_read_bases = ''.join(subs[sub_name]['alternative_base'] if sub_name in subs_included else '_' for sub_name in sorted(subs))

        if len(editing_strategy.pegRNA_names) == 2:

            if len(subs_included) == 0:
                category = 'wild type'
                subcategory = 'clean'
                details = knock_knock.outcome.Details()

            else:
                if len(subs_included) < len(subs):
                    category = 'partial replacement'

                    pegRNAs_that_explain_all_substitutions = set()

                    for pegRNA_name in editing_strategy.pegRNA_names:
                        if all(substitution_name in strat.pegRNA_substitutions[pegRNA_name] for substitution_name in subs_included):
                            pegRNAs_that_explain_all_substitutions.add(pegRNA_name)

                    if len(pegRNAs_that_explain_all_substitutions) == 0:
                        subcategory = 'both pegRNAs'
                    elif len(pegRNAs_that_explain_all_substitutions) == 2:
                        subcategory = 'single pegRNA (ambiguous)'
                    elif editing_strategy.pegRNA_names_by_side_of_read['left'] in pegRNAs_that_explain_all_substitutions:
                        subcategory = 'left pegRNA'
                    elif editing_strategy.pegRNA_names_by_side_of_read['right'] in pegRNAs_that_explain_all_substitutions:
                        subcategory = 'right pegRNA'

                else:
                    category = 'intended edit'
                    subcategory = 'replacement'

                details = knock_knock.outcome.Details(programmed_substitution_read_bases=programmed_substitution_read_bases)

        else:

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
        
            R1_seq = amplicon[len(primers['R1']):]
            R1 = hits.fastq.Read(name, R1_seq, qual[:len(R1_seq)])

            R2_seq = hits.utilities.reverse_complement(amplicon)[len(primers['R2']):]
            R2 = hits.fastq.Read(name, R2_seq, qual[:len(R2_seq)])

            reads.append((R1, R2))

            expected_categorizations[name] = {
                'category': category,
                'subcategory': subcategory,
                'details': str(details),
            }

    return reads, expected_categorizations

def get_all_read_sets():
    set_dirs = (parent / 'data').iterdir()
    
    read_sets = {}

    for set_dir in set_dirs:
        set_name = set_dir.name
        read_set = ReadSet(set_name)
        read_sets[set_name] = read_set

    return read_sets