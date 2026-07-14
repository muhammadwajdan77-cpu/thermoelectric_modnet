#!/usr/bin/env python3
"""n_feat sensitivity analysis: Does increasing feature budget close the MAE gap?

This script tests whether the small MAE gap between Model A (composition-only)
and Model B (composition+structure) is caused by MODNet's feature-selection
budget constraint.

HYPOTHESIS: With n_feat=50, adding structure features changes which 50 features
get selected via mutual-information ranking, potentially displacing useful
composition features. If we increase n_feat to 100, 150, composition features
should no longer need to "compete" with structure features for slots, and
Model B's MAE should approach or beat Model A's.
"""

import os
import pickle
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from modnet.models import MODNetModel
from modnet.preprocessing import MODData

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = 'target'
EXPECTED_ROWS = 7594
FAIR_SPLITS_PATH = RESULTS_DIR / 'fair_splits.pkl'
NFEAT_RESULTS_CSV = RESULTS_DIR / 'NFEAT_SENSITIVITY_COMPARISON.csv'
SEED = 42
np.random.seed(SEED)

# Define n_feat values to test
N_FEAT_VALUES = [50, 100, 150]


def safe_print(*args, **kwargs):
    print(*args, flush=True, **kwargs)


def canonical_formula(formula_text):
    if pd.isna(formula_text):
        raise ValueError('Formula is missing')
    text = str(formula_text).strip()
    if not text:
        raise ValueError('Formula is empty')
    # Remove trailing metadata tokens, keeping the chemical formula only.
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


def verify_preflight():
    required_files = [
        RESULTS_DIR / 'matminer_for_sisso.csv',
        RESULTS_DIR / 'matminer_structure_features.csv',
        ROOT / 'sysTEm_dataset' / 'sysTEm_dataset.xlsx',
        FAIR_SPLITS_PATH,  # Must have pre-existing splits
    ]

    safe_print('\n' + '=' * 70)
    safe_print('PRE-FLIGHT CHECKS')
    safe_print('=' * 70)

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f'Required file not found: {path}')
        size_mb = path.stat().st_size / (1024 * 1024)
        safe_print(f'  {path}  ({size_mb:.2f} MB)')

    mat_path = RESULTS_DIR / 'matminer_for_sisso.csv'
    mat_df = pd.read_csv(mat_path)
    for col in [TARGET_COLUMN, 'Temperature_K']:
        if col not in mat_df.columns:
            raise ValueError(f"Column '{col}' missing from {mat_path}")

    if len(mat_df) != EXPECTED_ROWS:
        raise ValueError(f'Unexpected matminer_for_sisso.csv row count: {len(mat_df)} (expected {EXPECTED_ROWS})')

    if 'Temperature_K' not in mat_df.columns:
        raise ValueError('Temperature_K missing from master feature matrix')

    groups = build_groups_from_sysem()
    if len(groups) != EXPECTED_ROWS:
        raise ValueError(f'Group count mismatch: {len(groups)} (expected {EXPECTED_ROWS})')

    safe_print('Temperature_K included: YES')
    safe_print('Master rows: 7594 ✅')
    safe_print(f'Fair splits file present: {FAIR_SPLITS_PATH} ✅')
    safe_print('PRE-FLIGHT CHECKS PASSED')
    safe_print('=' * 70 + '\n')

    return mat_df, groups


def load_splits():
    """Load pre-existing GroupKFold splits from fair_comparison_final.py"""
    if not FAIR_SPLITS_PATH.exists():
        raise FileNotFoundError(
            f'Expected split file not found: {FAIR_SPLITS_PATH}\n'
            f'Please run fair_comparison_final.py first to generate splits.'
        )
    with open(FAIR_SPLITS_PATH, 'rb') as handle:
        splits = pickle.load(handle)
    if len(splits) != 5:
        raise ValueError(f'Expected 5 splits in {FAIR_SPLITS_PATH}, found {len(splits)}')
    safe_print(f'Loaded {len(splits)} pre-existing splits from {FAIR_SPLITS_PATH} ✅\n')
    return splits


def train_fold(X_train, y_train, X_test, y_test, fold_num, model_label, n_feat):
    """
    Train a single MODNet fold with specified n_feat.
    """
    try:
        train_data = MODData(
            materials=list(range(len(X_train))),
            targets=[[float(v)] for v in y_train.values],
            target_names=[TARGET_COLUMN],
        )
        train_data.df_featurized = X_train
        # Select top n_feat features using mutual information
        train_data.feature_selection(n=min(n_feat, X_train.shape[1]))

        model = MODNetModel(
            [[[TARGET_COLUMN]]],
            weights={TARGET_COLUMN: 1},
            n_feat=min(n_feat, X_train.shape[1]),
        )
        model.fit(
            train_data,
            val_fraction=0.1,
            lr=0.001,
            batch_size=64,
            loss='mae',
            epochs=100,
            verbose=0,
        )

        test_data = MODData(
            materials=list(range(len(X_test))),
            targets=[[0.0]] * len(X_test),
            target_names=[TARGET_COLUMN],
        )
        test_data.df_featurized = X_test
        y_pred = model.predict(test_data)[TARGET_COLUMN].values
        fallback_used = False
    except Exception as exc:  # pylint: disable=broad-except
        safe_print(f'WARNING: {model_label} fold {fold_num} MODNet (n_feat={n_feat}) failed: {exc}')
        safe_print(f'WARNING: Falling back to RandomForestRegressor for fold {fold_num}')
        fallback_used = True
        rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
        rf.fit(X_train, y_train.values)
        y_pred = rf.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    safe_print(f'{model_label} - Fold {fold_num}/5: n_feat={n_feat:3d}, MAE={mae:.4f} R²={r2:.4f}')
    return mae, rmse, r2, y_pred, fallback_used


def train_model(X, y, groups, splits, model_label, n_feat):
    """
    Train model across all folds with specified n_feat.
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    groups = pd.Series(groups, dtype=str).reset_index(drop=True)

    imputer = SimpleImputer(strategy='mean')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_imp = X_imp.loc[:, X_imp.nunique(dropna=True) > 1]

    fold_maes = []
    fold_rmses = []
    fold_r2s = []
    all_y_true = []
    all_y_pred = []
    any_fallback = False

    for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise ValueError(f'Leakage in fold {fold_num}: {len(overlap)} overlapping groups')

        X_train = X_imp.iloc[train_idx].reset_index(drop=True)
        X_test = X_imp.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        mae, rmse, r2, y_pred, fallback_used = train_fold(
            X_train, y_train, X_test, y_test, fold_num, model_label, n_feat
        )
        any_fallback = any_fallback or fallback_used

        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_r2s.append(r2)
        all_y_true.extend(y_test.values.tolist())
        all_y_pred.extend(y_pred.tolist())

    metrics = {
        'MAE_mean': np.mean(fold_maes),
        'MAE_std': np.std(fold_maes),
        'RMSE_mean': np.mean(fold_rmses),
        'RMSE_std': np.std(fold_rmses),
        'R2_mean': np.mean(fold_r2s),
        'R2_std': np.std(fold_r2s),
        'fallback_used': any_fallback,
    }

    return metrics, np.array(all_y_true), np.array(all_y_pred)


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


def build_master_dataset():
    mat_path = RESULTS_DIR / 'matminer_for_sisso.csv'
    mat_df = pd.read_csv(mat_path)
    if TARGET_COLUMN not in mat_df.columns:
        raise ValueError(f"Missing {TARGET_COLUMN} column in {mat_path}")
    if 'Temperature_K' not in mat_df.columns:
        raise ValueError('Temperature_K missing from master dataset')
    if len(mat_df) != EXPECTED_ROWS:
        raise ValueError(f'Master dataset row count mismatch: {len(mat_df)} (expected {EXPECTED_ROWS})')

    X_base = mat_df.drop(columns=[TARGET_COLUMN])
    if 'Temperature_K' not in X_base.columns:
        raise ValueError('Temperature_K not found in X_base after dropping target')

    groups = build_groups_from_sysem()
    if len(groups) != EXPECTED_ROWS:
        raise ValueError(f'Built groups length mismatch: {len(groups)}')

    safe_print('Temperature_K included: YES')
    safe_print('Master rows: 7594 ✅')
    return mat_df, X_base, mat_df[TARGET_COLUMN].astype(float), groups


def append_result_row(n_feat, model_label, metrics):
    """Append a single result row to the CSV file."""
    row = {
        'n_feat': n_feat,
        'model': model_label,
        'MAE_mean': metrics['MAE_mean'],
        'MAE_std': metrics['MAE_std'],
        'RMSE_mean': metrics['RMSE_mean'],
        'RMSE_std': metrics['RMSE_std'],
        'R2_mean': metrics['R2_mean'],
        'R2_std': metrics['R2_std'],
    }

    if NFEAT_RESULTS_CSV.exists():
        df = pd.read_csv(NFEAT_RESULTS_CSV)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(NFEAT_RESULTS_CSV, index=False)


def print_comparison_table(results_data):
    """Print a formatted comparison table of MAE gaps across n_feat values."""
    safe_print('\n' + '=' * 80)
    safe_print('n_feat SENSITIVITY ANALYSIS - MAE GAP COMPARISON')
    safe_print('=' * 80)
    safe_print(f'{"n_feat":>8} | {"Model A (MAE ± std)":>20} | {"Model B (MAE ± std)":>20} | {"Gap (B-A)":>12}')
    safe_print('-' * 80)

    for n_feat in N_FEAT_VALUES:
        rows_a = [r for r in results_data if r['n_feat'] == n_feat and 'Composition' in r['model']]
        rows_b = [r for r in results_data if r['n_feat'] == n_feat and 'Composition' in r['model'] and 'Structure' in r['model']]

        if rows_a and rows_b:
            mae_a = rows_a[0]['MAE_mean']
            std_a = rows_a[0]['MAE_std']
            mae_b = rows_b[0]['MAE_mean']
            std_b = rows_b[0]['MAE_std']
            gap = mae_b - mae_a

            safe_print(
                f'{n_feat:>8} | {mae_a:.4f} ± {std_a:.4f}         | '
                f'{mae_b:.4f} ± {std_b:.4f}         | {gap:>+.4f}'
            )

    safe_print('=' * 80)

    # Compute gap trends
    gaps = {}
    for n_feat in N_FEAT_VALUES:
        rows_a = [r for r in results_data if r['n_feat'] == n_feat and 'Composition' in r['model']]
        rows_b = [r for r in results_data if r['n_feat'] == n_feat and 'Composition' in r['model'] and 'Structure' in r['model']]
        if rows_a and rows_b:
            gap = rows_b[0]['MAE_mean'] - rows_a[0]['MAE_mean']
            gaps[n_feat] = gap

    if len(gaps) > 1:
        gap_50 = gaps.get(50)
        gap_150 = gaps.get(150)
        if gap_50 is not None and gap_150 is not None:
            gap_change = gap_150 - gap_50
            safe_print(f'\nGap change from n_feat=50 to n_feat=150: {gap_change:+.4f}')

            if abs(gap_change) < 0.001:
                safe_print('\n✓ INTERPRETATION: Gap remains essentially constant.')
                safe_print('  → Feature-selection-budget explanation is UNLIKELY.')
                safe_print('  → Noise-floor explanation is more plausible.')
            elif gap_change < -0.002:
                safe_print('\n✓ INTERPRETATION: Gap SHRINKS as n_feat increases.')
                safe_print('  → Feature-selection-budget explanation is SUPPORTED.')
                safe_print('  → Adding structure features improves with more budget.')
            elif gap_change > 0.002:
                safe_print('\n✓ INTERPRETATION: Gap GROWS as n_feat increases.')
                safe_print('  → Feature-selection-budget explanation is CONTRADICTED.')
                safe_print('  → Structure features may be noisy or uninformative.')
            else:
                safe_print('\n✓ INTERPRETATION: Gap changes are within noise.')
                safe_print('  → Inconclusive; more n_feat values may be needed.')

    safe_print('=' * 80 + '\n')


def main():
    mat_df, X_base, y, groups = build_master_dataset()
    splits = load_splits()

    # Load structure features once
    struct_features = load_structure_features(groups)

    # Clear or create results file
    if NFEAT_RESULTS_CSV.exists():
        NFEAT_RESULTS_CSV.unlink()

    results_data = []

    for n_feat in N_FEAT_VALUES:
        safe_print(f'\n{"=" * 70}')
        safe_print(f'TESTING n_feat = {n_feat}')
        safe_print(f'{"=" * 70}')

        # Model A: Composition + Temperature
        safe_print(f'\nModel A: Composition + Temperature (n_feat={n_feat})')
        safe_print('-' * 70)
        results_a, y_true_a, y_pred_a = train_model(X_base, y, groups, splits, 'Model A', n_feat)

        result_row_a = {
            'n_feat': n_feat,
            'model': 'A: Composition + Temperature',
            'MAE_mean': results_a['MAE_mean'],
            'MAE_std': results_a['MAE_std'],
            'RMSE_mean': results_a['RMSE_mean'],
            'RMSE_std': results_a['RMSE_std'],
            'R2_mean': results_a['R2_mean'],
            'R2_std': results_a['R2_std'],
        }
        results_data.append(result_row_a)
        append_result_row(n_feat, 'A: Composition + Temperature', results_a)

        # Model B: Composition + Temperature + Structure
        safe_print(f'\nModel B: Composition + Temperature + Structure (n_feat={n_feat})')
        safe_print('-' * 70)
        X_B = pd.concat([X_base.reset_index(drop=True), struct_features.reset_index(drop=True)], axis=1)
        X_B = X_B.loc[:, X_B.nunique(dropna=True) > 1]
        if len(X_B) != EXPECTED_ROWS:
            raise ValueError(f'X_B row count mismatch: {len(X_B)} (expected {EXPECTED_ROWS})')

        results_b, y_true_b, y_pred_b = train_model(X_B, y, groups, splits, 'Model B', n_feat)

        result_row_b = {
            'n_feat': n_feat,
            'model': 'B: Composition + Temperature + Structure',
            'MAE_mean': results_b['MAE_mean'],
            'MAE_std': results_b['MAE_std'],
            'RMSE_mean': results_b['RMSE_mean'],
            'RMSE_std': results_b['RMSE_std'],
            'R2_mean': results_b['R2_mean'],
            'R2_std': results_b['R2_std'],
        }
        results_data.append(result_row_b)
        append_result_row(n_feat, 'B: Composition + Temperature + Structure', results_b)

    safe_print(f'\nResults saved to: {NFEAT_RESULTS_CSV}')
    print_comparison_table(results_data)


# Run pre-flight checks on import, without starting training.
PRE_FLIGHT_DATA = verify_preflight()

if __name__ == '__main__':
    main()
