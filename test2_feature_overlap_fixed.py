#!/usr/bin/env python3
"""Test 2 ONLY - Feature Overlap Check with FIXED get_optimal_descriptors() method."""

import pickle
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.impute import SimpleImputer
from modnet.preprocessing import MODData

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / 'results'

TARGET_COLUMN = 'target'
EXPECTED_ROWS = 7594
FAIR_SPLITS_PATH = RESULTS_DIR / 'fair_splits.pkl'
FEATURE_OVERLAP_CSV = RESULTS_DIR / 'FEATURE_OVERLAP_BY_FOLD_FIXED.csv'


def safe_print(*args, **kwargs):
    print(*args, flush=True, **kwargs)


def canonical_formula(formula_text):
    if pd.isna(formula_text):
        raise ValueError('Formula is missing')
    text = str(formula_text).strip()
    if not text:
        raise ValueError('Formula is empty')
    cleaned = text.split('_', 1)[0].strip()
    return Composition(cleaned).reduced_formula


def build_groups_from_sysem():
    sys_path = ROOT / 'sysTEm_dataset' / 'sysTEm_dataset.xlsx'
    if not sys_path.exists():
        raise FileNotFoundError(f'Missing SysTEm dataset: {sys_path}')
    sys_df = pd.read_excel(sys_path)
    if 'Pretty Formula' not in sys_df.columns:
        raise ValueError('SysTEm dataset missing Pretty Formula column')

    groups = []
    for formula in sys_df['Pretty Formula'].tolist():
        try:
            groups.append(canonical_formula(formula))
        except Exception:
            continue
        if len(groups) >= EXPECTED_ROWS:
            break
    if len(groups) != EXPECTED_ROWS:
        raise ValueError(f'Could not build {EXPECTED_ROWS} canonical groups from SysTEm dataset; found {len(groups)}')
    return pd.Series(groups, dtype=str)


def load_splits():
    if not FAIR_SPLITS_PATH.exists():
        raise FileNotFoundError(f'Expected split file not found: {FAIR_SPLITS_PATH}')
    with open(FAIR_SPLITS_PATH, 'rb') as handle:
        splits = pickle.load(handle)
    if len(splits) != 5:
        raise ValueError(f'Expected 5 splits in {FAIR_SPLITS_PATH}, found {len(splits)}')
    safe_print(f'Loaded {len(splits)} pre-existing splits from {FAIR_SPLITS_PATH} ✅\n')
    return splits


def load_structure_features(groups):
    path = RESULTS_DIR / 'matminer_structure_features.csv'
    struct_df = pd.read_csv(path)
    if 'composition' not in struct_df.columns:
        raise ValueError('Structure features file must contain composition column')

    struct_cols = [col for col in struct_df.columns if col != 'composition']
    lookup = {}
    for _, row in struct_df.iterrows():
        raw_formula = row['composition']
        try:
            canonical = canonical_formula(raw_formula)
        except Exception:
            continue
        if canonical not in lookup:
            lookup[canonical] = row[struct_cols].to_dict()

    struct_features = pd.DataFrame(
        [lookup.get(group, {col: np.nan for col in struct_cols}) for group in groups],
        columns=struct_cols,
    )

    for col in struct_features.columns:
        if struct_features[col].dtype == object:
            codes, _ = pd.factorize(struct_features[col], sort=True)
            struct_features[col] = codes.astype(float)
        elif pd.api.types.is_bool_dtype(struct_features[col]):
            struct_features[col] = struct_features[col].astype(float)
        else:
            struct_features[col] = pd.to_numeric(struct_features[col], errors='coerce')

    matched = int(struct_features.notna().any(axis=1).sum())
    safe_print(f'Structure features matched: {matched}/{EXPECTED_ROWS}')
    return struct_features


def main():
    safe_print('\n' + '╔' + '═' * 68 + '╗')
    safe_print('║ TEST 2 - FEATURE OVERLAP CHECK (FIXED with get_optimal_descriptors)║')
    safe_print('╚' + '═' * 68 + '╝\n')

    # Load data
    mat_path = RESULTS_DIR / 'matminer_for_sisso.csv'
    mat_df = pd.read_csv(mat_path)
    X_base = mat_df.drop(columns=[TARGET_COLUMN])
    y = mat_df[TARGET_COLUMN].astype(float)
    groups = build_groups_from_sysem()
    splits = load_splits()
    struct_features = load_structure_features(groups)

    # Prepare data
    X_base_clean = X_base.reset_index(drop=True)
    y_clean = y.reset_index(drop=True)
    groups_clean = pd.Series(groups, dtype=str).reset_index(drop=True)

    imputer_a = SimpleImputer(strategy='mean')
    X_base_imp = pd.DataFrame(imputer_a.fit_transform(X_base_clean), columns=X_base_clean.columns)
    X_base_imp = X_base_imp.loc[:, X_base_imp.nunique(dropna=True) > 1]

    X_B = pd.concat([X_base.reset_index(drop=True), struct_features.reset_index(drop=True)], axis=1)
    X_B = X_B.loc[:, X_B.nunique(dropna=True) > 1]

    imputer_b = SimpleImputer(strategy='mean')
    X_b_imp = pd.DataFrame(imputer_b.fit_transform(X_B), columns=X_B.columns)
    X_b_imp = X_b_imp.loc[:, X_b_imp.nunique(dropna=True) > 1]

    safe_print(f'X_base_imp shape: {X_base_imp.shape}')
    safe_print(f'X_b_imp shape: {X_b_imp.shape}\n')

    # Run feature overlap analysis
    if FEATURE_OVERLAP_CSV.exists():
        FEATURE_OVERLAP_CSV.unlink()

    overlap_results = []

    safe_print('Computing feature overlap at n_feat=50...\n')

    for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train_a = X_base_imp.iloc[train_idx].reset_index(drop=True)
        X_train_b = X_b_imp.iloc[train_idx].reset_index(drop=True)

        # Model A feature selection
        try:
            train_data_a = MODData(
                materials=list(range(len(X_train_a))),
                targets=[[float(v)] for v in y_clean.iloc[train_idx].values],
                target_names=[TARGET_COLUMN],
            )
            train_data_a.df_featurized = X_train_a
            train_data_a.feature_selection(n=min(50, X_train_a.shape[1]))
            # FIXED: Use get_optimal_descriptors() to get ACTUAL selected features
            selected_a = set(train_data_a.get_optimal_descriptors())
        except Exception as e:
            safe_print(f'WARNING: Could not extract Model A features for fold {fold_num}: {e}')
            selected_a = set()

        # Model B feature selection
        try:
            train_data_b = MODData(
                materials=list(range(len(X_train_b))),
                targets=[[float(v)] for v in y_clean.iloc[train_idx].values],
                target_names=[TARGET_COLUMN],
            )
            train_data_b.df_featurized = X_train_b
            train_data_b.feature_selection(n=min(50, X_train_b.shape[1]))
            # FIXED: Use get_optimal_descriptors() to get ACTUAL selected features
            selected_b = set(train_data_b.get_optimal_descriptors())
        except Exception as e:
            safe_print(f'WARNING: Could not extract Model B features for fold {fold_num}: {e}')
            selected_b = set()

        overlap = selected_a.intersection(selected_b)
        pct_overlap = 100 * len(overlap) / len(selected_a) if selected_a else 0
        
        # VERIFICATION: Check that we're actually getting ~50 selected features, not the full pool
        if len(selected_a) > 60:
            safe_print(f'❌ WARNING FOLD {fold_num}: Model A selected {len(selected_a)} features (expected ~50)!')
            safe_print(f'   → get_optimal_descriptors() may not be working correctly.')
            safe_print(f'   → This overlap measurement is UNRELIABLE.')
        if len(selected_b) > 70:
            safe_print(f'❌ WARNING FOLD {fold_num}: Model B selected {len(selected_b)} features (expected ~50)!')
            safe_print(f'   → get_optimal_descriptors() may not be working correctly.')
            safe_print(f'   → This overlap measurement is UNRELIABLE.')
        
        dropped = selected_a - selected_b
        added = selected_b - selected_a

        overlap_results.append({
            'fold': fold_num,
            'n_features_A': len(selected_a),
            'n_features_B': len(selected_b),
            'n_overlap': len(overlap),
            'pct_overlap': pct_overlap,
            'features_dropped_from_A': ';'.join(sorted(list(dropped))[:5]),  # Top 5
            'features_added_in_B': ';'.join(sorted(list(added))[:5]),  # Top 5
        })

        safe_print(f'Fold {fold_num}: Model A={len(selected_a):3d}, Model B={len(selected_b):3d}, Overlap={len(overlap):3d}/{len(selected_a):3d} ({pct_overlap:5.1f}%)')

    # Save results
    df_test2 = pd.DataFrame(overlap_results)
    df_test2.to_csv(FEATURE_OVERLAP_CSV, index=False)
    safe_print(f'\n✓ TEST 2 results saved to: {FEATURE_OVERLAP_CSV}\n')

    # Print summary
    safe_print('SUMMARY:')
    safe_print('='*68)
    safe_print(df_test2.to_string(index=False))
    safe_print('='*68)
    
    avg_overlap = np.mean(df_test2['pct_overlap'])
    safe_print(f'\nAverage feature overlap: {avg_overlap:.1f}%')
    
    if avg_overlap < 50:
        safe_print('→ Significant feature displacement detected!')
    elif avg_overlap > 80:
        safe_print('→ Features largely overlap; displacement is minimal.')
    else:
        safe_print('→ Moderate displacement observed.')


if __name__ == '__main__':
    main()
