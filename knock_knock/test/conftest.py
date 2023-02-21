from knock_knock.target_info import DegenerateDeletion

def pytest_assertrepr_compare(op, left, right):
    if isinstance(left, DegenerateDeletion) and isinstance(right, DegenerateDeletion) and op == "==":
        return [
            f'Comparing DegenerateDeletion instances:',
            f'   starts_ats: {left.starts_ats} {right.starts_ats}',
            f'   lengths: {left.length} {right.length}'
        ]