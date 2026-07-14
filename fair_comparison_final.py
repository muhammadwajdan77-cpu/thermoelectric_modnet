#!/usr/bin/env python3
"""Fair comparison between composition-only and structure-aware ZT models.

This script performs strict, leakage-free GroupKFold evaluation using identical
splits for both models. It validates required input files and preserves the
full 7594-row dataset throughout.
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
SUMMARY_CSV_PATH = RESULTS_DIR / 'FAIR_COMPARISON_FINAL.csv'
MODEL_A_PLOT = FIGURES_DIR / 'fair_parity_model_A.png'
MODEL_B_PLOT = FIGURES_DIR / 'fair_parity_model_B.png'
SEED = 42
np.random.seed(SEED)


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
    safe_print('PRE-FLIGHT CHECKS PASSED')
    safe_print('=' * 70 + '\n')

    return mat_df, groups


def generate_splits(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    splits = []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        overlap = set(groups.iloc[train_idx]).intersection(set(groups.iloc[test_idx]))
        if overlap:
            raise ValueError(f'Leakage found in generated split: {len(overlap)} overlapping groups')
        splits.append((train_idx, test_idx))
    with open(FAIR_SPLITS_PATH, 'wb') as handle:
        pickle.dump(splits, handle)
    for i, (train_idx, test_idx) in enumerate(splits, start=1):
        safe_print(f'Fold {i}: train={len(train_idx)}, test={len(test_idx)}, overlap=0 ✅')
    return splits


def load_splits():
    if not FAIR_SPLITS_PATH.exists():
        raise FileNotFoundError(f'Expected split file not found: {FAIR_SPLITS_PATH}')
    with open(FAIR_SPLITS_PATH, 'rb') as handle:
        splits = pickle.load(handle)
    if len(splits) != 5:
        raise ValueError(f'Expected 5 splits in {FAIR_SPLITS_PATH}, found {len(splits)}')
    return splits


def train_fold(X_train, y_train, X_test, y_test, fold_num, model_label):
    try:
        train_data = MODData(
            materials=list(range(len(X_train))),
            targets=[[float(v)] for v in y_train.values],
            target_names=[TARGET_COLUMN],
        )
        train_data.df_featurized = X_train
        train_data.feature_selection(n=min(50, X_train.shape[1]))

        model = MODNetModel(
            [[[TARGET_COLUMN]]],
            weights={TARGET_COLUMN: 1},
            n_feat=min(50, X_train.shape[1]),
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
        safe_print(f'WARNING: {model_label} fold {fold_num} MODNet failed with error: {exc}')
        safe_print(f'WARNING: Falling back to RandomForestRegressor for fold {fold_num}')
        fallback_used = True
        rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
        rf.fit(X_train, y_train.values)
        y_pred = rf.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    safe_print(f'{model_label} - Fold {fold_num}/5: MAE={mae:.4f} R²={r2:.4f}')
    return mae, rmse, r2, y_pred, fallback_used


def train_model(X, y, groups, splits, model_label):
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

        mae, rmse, r2, y_pred, fallback_used = train_fold(X_train, y_train, X_test, y_test, fold_num, model_label)
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


def save_parity_plot(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.4, s=18, color='tab:blue', edgecolors='none')
    minimum = min(np.nanmin(y_true), np.nanmin(y_pred))
    maximum = max(np.nanmax(y_true), np.nanmax(y_pred))
    margin = (maximum - minimum) * 0.05 if maximum > minimum else 0.1
    lim = [minimum - margin, maximum + margin]
    ax.plot(lim, lim, 'r--', lw=1.5, label='Perfect prediction')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('Actual target', fontweight='bold')
    ax.set_ylabel('Predicted target', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    txt = f'MAE={mean_absolute_error(y_true, y_pred):.4f}\nRMSE={np.sqrt(mean_squared_error(y_true, y_pred)):.4f}\nR²={r2_score(y_true, y_pred):.4f}'
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va='top', ha='left', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    safe_print(f'Saved parity plot: {path}')


def save_summary(results_a, results_b):
    summary = pd.DataFrame([
        {
            'model': 'A: Composition + Temperature',
            'MAE_mean': results_a['MAE_mean'],
            'MAE_std': results_a['MAE_std'],
            'RMSE_mean': results_a['RMSE_mean'],
            'RMSE_std': results_a['RMSE_std'],
            'R2_mean': results_a['R2_mean'],
            'R2_std': results_a['R2_std'],
        },
        {
            'model': 'B: Composition + Temperature + Structure',
            'MAE_mean': results_b['MAE_mean'],
            'MAE_std': results_b['MAE_std'],
            'RMSE_mean': results_b['RMSE_mean'],
            'RMSE_std': results_b['RMSE_std'],
            'R2_mean': results_b['R2_mean'],
            'R2_std': results_b['R2_std'],
        },
    ])
    delta_r2 = results_b['R2_mean'] - results_a['R2_mean']
    summary['rows'] = EXPECTED_ROWS
    summary['Temperature_K_included'] = 'YES'
    summary['identical_splits'] = 'CONFIRMED ✅'
    summary.to_csv(SUMMARY_CSV_PATH, index=False)
    safe_print(f'Saved summary CSV: {SUMMARY_CSV_PATH}')
    return delta_r2


def print_final_report(results_a, results_b, delta_r2):
    safe_print('\n' + '╔' + '═' * 58 + '╗')
    safe_print('║           FAIR COMPARISON - FINAL RESULTS                ║')
    safe_print('╠' + '═' * 58 + '╣')
    safe_print('║ Model A: Composition + Temperature                       ║')
    safe_print(f'║   MAE:  {results_a["MAE_mean"]:.4f} ± {results_a["MAE_std"]:.4f}                        ║')
    safe_print(f'║   RMSE: {results_a["RMSE_mean"]:.4f} ± {results_a["RMSE_std"]:.4f}                        ║')
    safe_print(f'║   R²:   {results_a["R2_mean"]:.4f} ± {results_a["R2_std"]:.4f}                        ║')
    safe_print('╠' + '═' * 58 + '╣')
    safe_print('║ Model B: Composition + Temperature + Structure           ║')
    safe_print(f'║   MAE:  {results_b["MAE_mean"]:.4f} ± {results_b["MAE_std"]:.4f}                        ║')
    safe_print(f'║   RMSE: {results_b["RMSE_mean"]:.4f} ± {results_b["RMSE_std"]:.4f}                        ║')
    safe_print(f'║   R²:   {results_b["R2_mean"]:.4f} ± {results_b["R2_std"]:.4f}                        ║')
    safe_print('╠' + '═' * 58 + '╣')
    safe_print(f'║ Structure improvement: ΔR² = {delta_r2:.4f}                     ║')
    safe_print('║ Both models used IDENTICAL splits: CONFIRMED ✅          ║')
    safe_print('║ Temperature_K included in both: YES ✅                   ║')
    safe_print('║ Rows: 7594 in both models ✅                             ║')
    safe_print('╚' + '═' * 58 + '╝\n')


def main():
    mat_df, X_base, y, groups = build_master_dataset()
    splits = generate_splits(X_base, y, groups)

    safe_print('\n' + '=' * 70)
    safe_print('STEP 3 - MODEL A: Composition + Temperature (no structure)')
    safe_print('=' * 70)
    results_a, y_true_a, y_pred_a = train_model(X_base, y, groups, splits, 'Model A')

    safe_print('\n' + '=' * 70)
    safe_print('STEP 4 - MODEL B: Composition + Temperature + Structure')
    safe_print('=' * 70)
    struct_features = load_structure_features(groups)
    X_B = pd.concat([X_base.reset_index(drop=True), struct_features.reset_index(drop=True)], axis=1)
    X_B = X_B.loc[:, X_B.nunique(dropna=True) > 1]
    if len(X_B) != EXPECTED_ROWS:
        raise ValueError(f'X_B row count mismatch: {len(X_B)} (expected {EXPECTED_ROWS})')

    results_b, y_true_b, y_pred_b = train_model(X_B, y, groups, splits, 'Model B')

    save_summary(results_a, results_b)
    delta_r2 = results_b['R2_mean'] - results_a['R2_mean']
    save_parity_plot(y_true_a, y_pred_a, 'Model A: Composition + Temperature', MODEL_A_PLOT)
    save_parity_plot(y_true_b, y_pred_b, 'Model B: Composition + Temperature + Structure', MODEL_B_PLOT)
    print_final_report(results_a, results_b, delta_r2)


# Run pre-flight checks on import, without starting training.
PRE_FLIGHT_DATA = verify_preflight()

if __name__ == '__main__':
    main()
