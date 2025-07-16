import multiprocessing
from pathlib import Path

import pytest

import knock_knock.test.read_sets
import knock_knock.test.test_pegRNAs
from knock_knock.target_info import DegenerateDeletion

@pytest.fixture(scope='session', autouse=True)
def always_spawn():
    multiprocessing.set_start_method('fork')

def pytest_assertrepr_compare(op, left, right):
    if isinstance(left, DegenerateDeletion) and isinstance(right, DegenerateDeletion) and op == '==':
        return [
            f'Comparing DegenerateDeletion instances:',
            f'   starts_ats: {left.starts_ats} {right.starts_ats}',
            f'   lengths: {left.length} {right.length}'
        ]

def pytest_generate_tests(metafunc, source_dir=None):
    if 'read_set' in metafunc.fixturenames and 'qname' in metafunc.fixturenames:
        read_sets = knock_knock.test.read_sets.get_all_read_sets(source_dir=source_dir)

        params = []
        
        for read_set in read_sets.values():
            for qname in read_set.qnames:
                if 'expected_failure' in read_set.expected_values[qname]:
                    marks = [pytest.mark.xfail(strict=True)]
                else:
                    marks = []

                param = pytest.param(read_set, qname, marks=marks, id=f'{read_set.name} {qname}')

                params.append(param)

        metafunc.parametrize(['read_set', 'qname'], params)

    elif 'flap_sequence' in metafunc.fixturenames and 'downstream_genomic_sequence' in metafunc.fixturenames and 'expected_coordinates' in metafunc.fixturenames:
        RTT_alignments = knock_knock.test.test_pegRNAs.load_RTT_alignments(source_dir=source_dir)

        params = []
        
        for pair_name, details_tuple in RTT_alignments.items():
            param = pytest.param(*details_tuple, id=pair_name)
            params.append(param)

        metafunc.parametrize(['flap_sequence', 'downstream_genomic_sequence', 'expected_coordinates'], params)