import pandas as pd
import numpy as np
from pymatgen.core import Composition
from pathlib import Path

ROOT = Path('.')
RESULTS_DIR = ROOT / 'results'

mat = pd.read_csv(RESULTS_DIR / 'matminer_for_sisso.csv')
sys_df = pd.read_excel(ROOT / 'sysTEm_dataset' / 'sysTEm_dataset.xlsx')

print(f'mat rows: {len(mat)}')
print(f'sys_df rows: {len(sys_df)}')

# Method 1: get_matminer_fold_maes.py's approach (positional slice)
def canonical_formula(formula):
    if pd.isna(formula):
        raise ValueError('empty')
    return Composition(str(formula)).reduced_formula


def make_group_labels(formulas):
    labels = []
    for formula in formulas.astype(str):
        text = formula.strip()
        if not text or text.lower() in {'nan', 'none'}:
            labels.append('nan')
            continue
        try:
            labels.append(canonical_formula(text))
        except Exception:
            labels.append(text)
    return np.array(labels, dtype=object)

n_rows = len(mat)
groups_method1 = make_group_labels(sys_df.iloc[:n_rows]['Pretty Formula'])

# Method 2: fair_comparison_final.py's approach (skip invalid, accumulate to EXPECTED_ROWS)
groups_method2 = []
for formula in sys_df['Pretty Formula'].tolist():
    try:
        groups_method2.append(canonical_formula(formula))
    except Exception:
        continue
    if len(groups_method2) >= n_rows:
        break
groups_method2 = np.array(groups_method2, dtype=object)

print(f'\nMethod 1 (positional slice) - unique groups: {len(set(groups_method1))}')
print(f'Method 2 (skip-invalid accumulate) - unique groups: {len(set(groups_method2))}')

# Are they identical arrays?
if len(groups_method1) == len(groups_method2):
    identical = np.array_equal(groups_method1, groups_method2)
    print(f'\nArrays identical: {identical}')
    if not identical:
        n_diff = np.sum(groups_method1 != groups_method2)
        print(f'Number of positions where they differ: {n_diff} / {len(groups_method1)}')
        diffs = np.where(groups_method1 != groups_method2)[0]
        print('First 10 differing positions:')
        for i in diffs[:10]:
            print(f'  Position {i}: method1={groups_method1[i]!r}, method2={groups_method2[i]!r}')
else:
    print(f'\nLENGTH MISMATCH: method1 has {len(groups_method1)}, method2 has {len(groups_method2)}')

# Also check: does sys_df have any invalid/unparseable formulas in the first n_rows?
invalid_count = 0
for formula in sys_df.iloc[:n_rows]['Pretty Formula'].astype(str):
    try:
        canonical_formula(formula)
    except Exception:
        invalid_count += 1
print(f'\nInvalid/unparseable formulas in first {n_rows} rows of sys_df: {invalid_count}')
