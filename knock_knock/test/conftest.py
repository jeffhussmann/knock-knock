import pytest

import knock_knock.test.read_sets
from knock_knock.target_info import DegenerateDeletion

def pytest_assertrepr_compare(op, left, right):
    if isinstance(left, DegenerateDeletion) and isinstance(right, DegenerateDeletion) and op == "==":
        return [
            f'Comparing DegenerateDeletion instances:',
            f'   starts_ats: {left.starts_ats} {right.starts_ats}',
            f'   lengths: {left.length} {right.length}'
        ]

def pytest_generate_tests(metafunc):
    if 'read_set' in metafunc.fixturenames and 'qname' in metafunc.fixturenames:
        read_sets = knock_knock.test.read_sets.get_all_read_sets()

        params = []
        
        for read_set in read_sets.values():
            for qname in read_set.qnames:
                if 'expected_failure' in read_set.expected_values[qname]:
                    marks = [pytest.mark.xfail]
                else:
                    marks = []

                param = pytest.param(read_set, qname, marks=marks)

                params.append(param)

        metafunc.parametrize(['read_set', 'qname'], params)