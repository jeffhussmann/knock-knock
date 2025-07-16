'''
Extracting alignments for manually identified read+expected outcome pairs
and regression testing these categorizations.
'''

import shutil
from pathlib import Path

import yaml

import hits.utilities
import hits.sam

import knock_knock.architecture
import knock_knock.arrayed_experiment_group
import knock_knock.editing_strategy

memoized_property = hits.utilities.memoized_property

base_dir = Path(__file__).parent

def populate_source_dir(source_dir):
    if source_dir is None:
        source_dir = base_dir
    return Path(source_dir)

class ReadSet:
    def __init__(self, name, source_dir=None):
        self.name = name

        self.source_dir = populate_source_dir(source_dir)

        self.dir = self.source_dir / 'read_sets' / self.name
        
        self.bam_fn =  self.dir / 'alignments.bam'
        self.expected_values_fn = self.dir / 'expected_values.yaml'

    def __repr__(self):
        return f'ReadSet {self.name} ({self.source_dir})'

    @memoized_property
    def details(self):
        return yaml.safe_load(self.expected_values_fn.read_text())

    @memoized_property
    def expected_values(self):
        return self.details['expected_values']

    @memoized_property
    def alignments(self):
        return dict(hits.sam.grouped_by_name(self.bam_fn))

    @memoized_property
    def qnames(self):
        return sorted(self.alignments)

    def get_read_architecture(self, read_id):
        architecture = None

        for qname, als in hits.sam.grouped_by_name(self.bam_fn):
            if qname == read_id:
                architecture = self.categorizer(als, self.editing_strategy)
                break

        if architecture is None:
            raise ValueError(read_id)
        
        return architecture

    @memoized_property
    def editing_strategy(self):
        strategy_name = self.details['editing_strategy']
        supplemental_index_names = ['hg38', 'hg19', 'bosTau7', 'e_coli']
        supplemental_indices = knock_knock.editing_strategy.locate_supplemental_indices(base_dir)
        supplemental_indices = {name: supplemental_indices[name] for name in supplemental_index_names}
        editing_strategy = knock_knock.editing_strategy.EditingStrategy(self.source_dir,
                                                                        strategy_name,
                                                                        supplemental_indices=supplemental_indices,
                                                                        **self.details.get('strategy_kwargs', {}),
                                                                       )

        return editing_strategy

    @memoized_property
    def categorizer(self):
        return knock_knock.architecture.experiment_type_to_categorizer(self.details['experiment_type'])

    def compare_to_expected(self, qname):
        architecture = self.categorizer(self.alignments[qname], self.editing_strategy)

        try:
            architecture.categorize()

        except:
            architecture.category = 'error'
            architecture.subcategory = 'error'
            architecture.details = 'error'

        expected = self.expected_values[qname]

        expected_tuple = (
            expected['category'],
            expected['subcategory'],
            expected.get('details', architecture.details),
        )

        observed_tuple = (
            architecture.category,
            architecture.subcategory,
            architecture.details,
        )

        return observed_tuple == expected_tuple, architecture, expected

    def process(self):
        tested_architectures = {
            True: [],
            False: [],
        }

        for qname in self.qnames:
            agrees_with_expected, architecture, expected = self.compare_to_expected(qname)

            tested_architectures[agrees_with_expected].append((architecture, expected))

        num_passed = len(tested_architectures[True]) 
        num_failed = len(tested_architectures[False])

        print(f'Tested {num_passed + num_failed: >3d} sequences ({num_passed} passed, {num_failed} failed) for {self.name}.')

        return tested_architectures

def get_all_read_sets(source_dir=None):
    source_dir = populate_source_dir(source_dir)

    read_set_names = sorted([d.name for d in (source_dir / 'read_sets').iterdir() if d.is_dir()])
    read_sets = {name: ReadSet(name, source_dir=source_dir) for name in read_set_names}

    return read_sets

def build_all_pooled_screen_read_sets(only_new=False):
    src_read_sets_dir = base_dir / 'read_set_specifications' / 'pooled_screens'
    read_set_fns = src_read_sets_dir.glob('*.yaml')

    for read_set_fn in read_set_fns:
        set_name = read_set_fn.stem

        read_set = ReadSet(set_name)

        if read_set.dir.is_dir() and only_new:
            continue
        else:
            build_pooled_screen_read_set(set_name)

def build_pooled_screen_read_set(set_name):
    import repair_seq.pooled_screen

    read_set_fn = base_dir / 'read_set_specifications' / 'pooled_screens' / f'{set_name}.yaml'
    manual_details = yaml.safe_load(read_set_fn.read_text())

    read_set = ReadSet(set_name)

    read_set.dir.mkdir(exist_ok=True, parents=True)
    
    pool = repair_seq.pooled_screen.get_pool(manual_details['base_dir'], manual_details['pool_name'])
    exp = pool.single_guide_experiment(manual_details['fixed_guide'], manual_details['variable_guide'])

    new_strategy_dir = base_dir / 'targets' / pool.editing_strategy.name
    existing_strategy_dir = pool.editing_strategy.dir

    if new_strategy_dir.is_dir():
        approved_deletion = input(f'Deleting target directory {new_strategy_dir}, proceed? ') == 'y'

        if approved_deletion:
            shutil.rmtree(new_strategy_dir)
        else:
            raise ValueError

    shutil.copytree(existing_strategy_dir, new_strategy_dir)

    # Pool experiments specify specialized values for some editing_strategy
    # parameters that need to be passed along.
    possible_strategy_kwargs_keys = [
        'sgRNAs',
        'sequencing_start_feature_name',
        'primer_names',
    ]

    strategy_kwargs = {
        key: exp.description[key]
        for key in possible_strategy_kwargs_keys
        if key in exp.description
    }

    alignment_sorter = hits.sam.AlignmentSorter(read_set.bam_fn, exp.combined_header, by_name=True)

    read_info = {
        'experiment_type': pool.sample_sheet['experiment_type'],
        'editing_strategy': pool.editing_strategy.name,
        'strategy_kwargs': strategy_kwargs,
        'expected_values': {},
    }

    with alignment_sorter:
        for read_id, details in manual_details['expected_values'].items():
            als = exp.get_read_alignments(read_id)
            for al in als:
                # Overwrite potential common sequence query_name. 
                al.query_name = read_id
                alignment_sorter.write(al)
                
            read_info['expected_values'][read_id] = details

    read_set.expected_values_fn.write_text(yaml.safe_dump(read_info, sort_keys=False))
        
def build_all_arrayed_group_read_sets(only_new=False):
    src_read_sets_dir = base_dir / 'read_set_specifications' / 'arrayed_groups'
    read_set_fns = src_read_sets_dir.glob('*.yaml')

    for read_set_fn in read_set_fns:
        set_name = read_set_fn.stem

        read_set = ReadSet(set_name)

        if read_set.dir.is_dir() and only_new:
            continue
        else:
            build_arrayed_group_read_set(set_name)

def build_arrayed_group_read_set(set_name, source_dir=None, prompt=True):
    source_dir = populate_source_dir(source_dir)

    read_set_fn = source_dir / 'read_set_specifications' / 'arrayed_groups' / f'{set_name}.yaml'
    manual_details = yaml.safe_load(read_set_fn.read_text())

    read_set = ReadSet(set_name, source_dir=source_dir)

    read_set.dir.mkdir(exist_ok=True, parents=True)

    exps = knock_knock.arrayed_experiment_group.get_all_experiments(manual_details['base_dir'])
    exp = exps[manual_details['batch_name'], manual_details['group_name'], manual_details['exp_name']]
    exp_type = exp.experiment_group.description['experiment_type']

    # Experiments may specify specialized values for some editing_strategy
    # parameters that need to be passed along.
    possible_strategy_kwargs_keys = [
        'sgRNAs',
        'sequencing_start_feature_name',
        'primer_names',
    ]

    strategy_kwargs = {
        key: exp.description[key]
        for key in possible_strategy_kwargs_keys
        if key in exp.description
    }

    new_strategy_name = exp.editing_strategy.name
    if 'strategy_prefix_to_add' in manual_details:
        prefix = manual_details['strategy_prefix_to_add']
        new_strategy_name = f'{prefix}_{new_strategy_name}'

    new_strategy_dir = source_dir / 'targets' / new_strategy_name
    existing_strategy_dir = exp.editing_strategy.dir

    if new_strategy_dir.is_dir():
        if prompt:
            approved_deletion = input(f'Deleting target directory {new_strategy_dir}, proceed? ') == 'y'
        else:
            approved_deletion = True

        if approved_deletion:
            shutil.rmtree(new_strategy_dir)
        else:
            raise ValueError

    shutil.copytree(existing_strategy_dir, new_strategy_dir)

    alignment_sorter = hits.sam.AlignmentSorter(read_set.bam_fn, exp.combined_header, by_name=True)
    
    read_info = {
        'experiment_type': exp_type,
        'editing_strategy': new_strategy_name,
        'strategy_kwargs': strategy_kwargs,
        'expected_values': {},
    }

    with alignment_sorter:
        for read_id, details in manual_details['expected_values'].items():
            als = exp.get_read_alignments(read_id)
            for al in als:
                # Overwrite potential common sequence query_name. 
                al.query_name = read_id
                alignment_sorter.write(al)
                
            read_info['expected_values'][read_id] = details

    read_set.expected_values_fn.write_text(yaml.safe_dump(read_info, sort_keys=False))

def process_read_set(set_name, source_dir=None):
    read_set = ReadSet(set_name, source_dir=source_dir)
    return read_set.process()