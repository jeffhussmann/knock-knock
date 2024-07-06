'''
Extracting alignments for manually identified read+expected outcome pairs
and regression testing these categorizations.
'''

import shutil
from pathlib import Path

import yaml

import hits.utilities
import hits.sam
import knock_knock.arrayed_experiment_group
import knock_knock.target_info

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

    def get_read_layout(self, read_id):
        layout = None

        for qname, als in hits.sam.grouped_by_name(self.bam_fn):
            if qname == read_id:
                layout = self.categorizer(als, self.target_info)
                break

        if layout is None:
            raise ValueError(read_id)
        
        return layout

    @memoized_property
    def target_info(self):
        target_info_name = self.details['target_info']
        supplemental_index_names = ['hg38', 'hg19', 'bosTau7', 'e_coli']
        supplemental_indices = knock_knock.target_info.locate_supplemental_indices(base_dir)
        supplemental_indices = {name: supplemental_indices[name] for name in supplemental_index_names}
        target_info = knock_knock.target_info.TargetInfo(self.source_dir,
                                                         target_info_name,
                                                         supplemental_indices=supplemental_indices,
                                                         **self.details.get('target_info_kwargs', {}),
                                                        )

        return target_info

    @memoized_property
    def categorizer(self):
        if self.details['experiment_type'] in ['twin_prime', 'dual_flap']:
            from knock_knock.twin_prime_layout import Layout
        elif self.details['experiment_type'] in ['prime_editing', 'prime_editing_layout', 'single_flap']:
            from knock_knock.prime_editing_layout import Layout
        elif self.details['experiment_type'] == 'TECseq':
            from knock_knock.TECseq_layout import Layout
        else:
            raise NotImplementedError

        categorizer = Layout

        return categorizer

    def compare_to_expected(self, qname):
        try:
            layout = self.categorizer(self.alignments[qname], self.target_info)
            layout.categorize()

        except:
            layout.category = 'error'
            layout.subcategory = 'error'
            layout.details = 'error'

        expected = self.expected_values[qname]

        expected_tuple = (
            expected['category'],
            expected['subcategory'],
            expected.get('details', layout.details),
        )

        observed_tuple = (
            layout.category,
            layout.subcategory,
            layout.details,
        )

        return observed_tuple == expected_tuple, layout, expected

    def process(self):
        tested_layouts = {
            True: [],
            False: [],
        }

        for qname in self.qnames:
            agrees_with_expected, layout, expected = self.compare_to_expected(qname)

            tested_layouts[agrees_with_expected].append((layout, expected))

        num_passed = len(tested_layouts[True]) 
        num_failed = len(tested_layouts[False])

        print(f'Tested {num_passed + num_failed: >3d} sequences ({num_passed} passed, {num_failed} failed) for {self.name}.')

        return tested_layouts

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

    new_target_info_dir = base_dir / 'targets' / pool.target_info.name
    existing_target_info_dir = pool.target_info.dir

    if new_target_info_dir.is_dir():
        approved_deletion = input(f'Deleting target directory {new_target_info_dir}, proceed? ') == 'y'

        if approved_deletion:
            shutil.rmtree(new_target_info_dir)
        else:
            raise ValueError

    shutil.copytree(existing_target_info_dir, new_target_info_dir)

    # Pool experiments specify specialized values for some target_info
    # parameters that need to be passed along.
    possible_target_info_kwargs_keys = [
        'sgRNAs',
        'sequencing_start_feature_name',
        'primer_names',
    ]

    target_info_kwargs = {
        key: exp.description[key]
        for key in possible_target_info_kwargs_keys
        if key in exp.description
    }

    alignment_sorter = hits.sam.AlignmentSorter(read_set.bam_fn, exp.combined_header, by_name=True)

    read_info = {
        'experiment_type': pool.sample_sheet['layout_module'],
        'target_info': pool.target_info.name,
        'target_info_kwargs': target_info_kwargs,
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

    # Experiments may specify specialized values for some target_info
    # parameters that need to be passed along.
    possible_target_info_kwargs_keys = [
        'sgRNAs',
        'sequencing_start_feature_name',
        'primer_names',
    ]

    target_info_kwargs = {
        key: exp.description[key]
        for key in possible_target_info_kwargs_keys
        if key in exp.description
    }

    new_target_info_name = exp.target_info.name
    if 'target_info_prefix_to_add' in manual_details:
        prefix = manual_details['target_info_prefix_to_add']
        new_target_info_name = f'{prefix}_{new_target_info_name}'

    new_target_info_dir = source_dir / 'targets' / new_target_info_name
    existing_target_info_dir = exp.target_info.dir

    if new_target_info_dir.is_dir():
        if prompt:
            approved_deletion = input(f'Deleting target directory {new_target_info_dir}, proceed? ') == 'y'
        else:
            approved_deletion = True

        if approved_deletion:
            shutil.rmtree(new_target_info_dir)
        else:
            raise ValueError

    shutil.copytree(existing_target_info_dir, new_target_info_dir)

    alignment_sorter = hits.sam.AlignmentSorter(read_set.bam_fn, exp.combined_header, by_name=True)
    
    read_info = {
        'experiment_type': exp_type,
        'target_info': new_target_info_name,
        'target_info_kwargs': target_info_kwargs,
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