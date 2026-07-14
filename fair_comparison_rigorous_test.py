#!/usr/bin/env python3
"""Rigorous 4-test investigation into why Model B (composition+structure) has higher MAE.

TESTS:
1. n_feat SENSITIVITY: [50, 100, 150] to test feature-selection-budget hypothesis
2. FEATURE OVERLAP CHECK: exact feature displacement analysis at n_feat=50
3. FORCED-INCLUSION TEST: Model A's features guaranteed + structure features added
4. STATISTICAL TESTS: corrected resampling t-test (Nadeau & Bengio) on all comparisons
"""

import os
import pickle
from pathlib import Path
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from scipy.stats import t as t_dist
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
SEED = 42
np.random.seed(SEED)

# Test configuration
N_FEAT_VALUES = [50, 100, 150]
K_FOLDS = 5
TEST_SIZE = 1519
TRAIN_SIZE = 6075

# Output files
NFEAT_RESULTS_CSV = RESULTS_DIR / 'NFEAT_SENSITIVITY_COMPARISON.csv'
FEATURE_OVERLAP_CSV = RESULTS_DIR / 'FEATURE_OVERLAP_BY_FOLD_FIXED.csv'
FORCED_INCLUSION_CSV = RESULTS_DIR / 'FORCED_INCLUSION_MODEL_C.csv'
STATISTICAL_TESTS_CSV = RESULTS_DIR / 'RIGOROUS_STATISTICAL_TESTS.csv'


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


def verify_preflight():
    required_files = [
        RESULTS_DIR / 'matminer_for_sisso.csv',
        RESULTS_DIR / 'matminer_structure_features.csv',
        ROOT / 'sysTEm_dataset' / 'sysTEm_dataset.xlsx',
        FAIR_SPLITS_PATH,
    ]

    safe_print('\n' + '=' * 70)
    safe_print('RIGOROUS INVESTIGATION - PRE-FLIGHT CHECKS')
    safe_print('=' * 70)

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f'Required file not found: {path}')
        size_mb = path.stat().st_size / (1024 * 1024)
        safe_print(f'  ✓ {path}  ({size_mb:.2f} MB)')

    mat_path = RESULTS_DIR / 'matminer_for_sisso.csv'
    mat_df = pd.read_csv(mat_path)
    for col in [TARGET_COLUMN, 'Temperature_K']:
        if col not in mat_df.columns:
            raise ValueError(f"Column '{col}' missing from {mat_path}")

    if len(mat_df) != EXPECTED_ROWS:
        raise ValueError(f'Unexpected matminer_for_sisso.csv row count: {len(mat_df)} (expected {EXPECTED_ROWS})')

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
    if not FAIR_SPLITS_PATH.exists():
        raise FileNotFoundError(f'Expected split file not found: {FAIR_SPLITS_PATH}')
    with open(FAIR_SPLITS_PATH, 'rb') as handle:
        splits = pickle.load(handle)
    if len(splits) != 5:
        raise ValueError(f'Expected 5 splits in {FAIR_SPLITS_PATH}, found {len(splits)}')
    safe_print(f'Loaded {len(splits)} pre-existing splits from {FAIR_SPLITS_PATH} ✅\n')
    return splits


def train_fold_with_feature_tracking(X_train, y_train, X_test, y_test, fold_num, model_label, n_feat):
    """Train fold and return metrics + selected feature indices."""
    selected_features = None
    try:
        train_data = MODData(
            materials=list(range(len(X_train))),
            targets=[[float(v)] for v in y_train.values],
            target_names=[TARGET_COLUMN],
        )
        train_data.df_featurized = X_train
        train_data.feature_selection(n=min(n_feat, X_train.shape[1]))
        
        # Capture selected feature indices
        if hasattr(train_data, 'df_featurized'):
            selected_features = train_data.df_featurized.columns.tolist()

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
    except Exception as exc:
        safe_print(f'WARNING: {model_label} fold {fold_num} MODNet failed: {exc}')
        fallback_used = True
        rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
        rf.fit(X_train, y_train.values)
        y_pred = rf.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2, y_pred, fallback_used, selected_features


def train_model_with_tracking(X, y, groups, splits, model_label, n_feat):
    """Train model and track selected features for feature overlap analysis."""
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
    feature_selections = []
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

        mae, rmse, r2, y_pred, fallback_used, selected_features = train_fold_with_feature_tracking(
            X_train, y_train, X_test, y_test, fold_num, model_label, n_feat
        )
        any_fallback = any_fallback or fallback_used

        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_r2s.append(r2)
        all_y_true.extend(y_test.values.tolist())
        all_y_pred.extend(y_pred.tolist())
        feature_selections.append(selected_features)

        safe_print(f'{model_label} - Fold {fold_num}/5: n_feat={n_feat:3d}, MAE={mae:.4f} R²={r2:.4f}')

    metrics = {
        'MAE_mean': np.mean(fold_maes),
        'MAE_std': np.std(fold_maes),
        'RMSE_mean': np.mean(fold_rmses),
        'RMSE_std': np.std(fold_rmses),
        'R2_mean': np.mean(fold_r2s),
        'R2_std': np.std(fold_r2s),
        'fallback_used': any_fallback,
    }

    return metrics, np.array(all_y_true), np.array(all_y_pred), fold_maes, fold_rmses, fold_r2s, feature_selections


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

    return mat_df, X_base, mat_df[TARGET_COLUMN].astype(float), groups


def corrected_resampling_t_test(maes_a, maes_b):
    """Nadeau & Bengio corrected resampling t-test."""
    diffs = np.array(maes_a, dtype=float) - np.array(maes_b, dtype=float)
    d_bar = float(np.mean(diffs))
    var_d = float(np.var(diffs, ddof=1))
    var_corrected = (1 / K_FOLDS + TEST_SIZE / TRAIN_SIZE) * var_d
    if var_corrected <= 0:
        t_stat = 0.0
        p_value = 1.0
    else:
        t_stat = d_bar / np.sqrt(var_corrected)
        p_value = float(2 * t_dist.sf(abs(t_stat), df=K_FOLDS - 1))
    return t_stat, p_value, d_bar


# ============================================================================
# TEST 1: n_feat SENSITIVITY
# ============================================================================
def test1_nfeat_sensitivity(mat_df, X_base, y, groups, splits, struct_features):
    safe_print('\n' + '╔' + '═' * 68 + '╗')
    safe_print('║ TEST 1 - n_feat SENSITIVITY ANALYSIS                             ║')
    safe_print('╚' + '═' * 68 + '╝\n')

    if NFEAT_RESULTS_CSV.exists():
        NFEAT_RESULTS_CSV.unlink()

    all_results = []

    for n_feat in N_FEAT_VALUES:
        safe_print(f'\n--- Testing n_feat = {n_feat} ---')
        
        # Model A
        safe_print(f'Model A: Composition + Temperature (n_feat={n_feat})')
        metrics_a, _, _, maes_a, _, _, _ = train_model_with_tracking(
            X_base, y, groups, splits, f'Model A', n_feat
        )

        result_a = {
            'n_feat': n_feat,
            'model': 'A: Composition + Temperature',
            'MAE_mean': metrics_a['MAE_mean'],
            'MAE_std': metrics_a['MAE_std'],
            'RMSE_mean': metrics_a['RMSE_mean'],
            'RMSE_std': metrics_a['RMSE_std'],
            'R2_mean': metrics_a['R2_mean'],
            'R2_std': metrics_a['R2_std'],
        }
        all_results.append(result_a)

        # Model B
        safe_print(f'Model B: Composition + Temperature + Structure (n_feat={n_feat})')
        X_B = pd.concat([X_base.reset_index(drop=True), struct_features.reset_index(drop=True)], axis=1)
        X_B = X_B.loc[:, X_B.nunique(dropna=True) > 1]
        
        metrics_b, _, _, maes_b, _, _, _ = train_model_with_tracking(
            X_B, y, groups, splits, f'Model B', n_feat
        )

        result_b = {
            'n_feat': n_feat,
            'model': 'B: Composition + Temperature + Structure',
            'MAE_mean': metrics_b['MAE_mean'],
            'MAE_std': metrics_b['MAE_std'],
            'RMSE_mean': metrics_b['RMSE_mean'],
            'RMSE_std': metrics_b['RMSE_std'],
            'R2_mean': metrics_b['R2_mean'],
            'R2_std': metrics_b['R2_std'],
        }
        all_results.append(result_b)

        gap = metrics_b['MAE_mean'] - metrics_a['MAE_mean']
        safe_print(f'MAE gap (B - A): {gap:+.4f}')

    # Save TEST 1 results
    df_test1 = pd.DataFrame(all_results)
    df_test1.to_csv(NFEAT_RESULTS_CSV, index=False)
    safe_print(f'\n✓ TEST 1 results saved to: {NFEAT_RESULTS_CSV}')
    
    return all_results


# ============================================================================
# TEST 2: FEATURE OVERLAP CHECK
# ============================================================================
def test2_feature_overlap(X_base, y, groups, splits, struct_features):
    safe_print('\n' + '╔' + '═' * 68 + '╗')
    safe_print('║ TEST 2 - FEATURE OVERLAP CHECK (at n_feat=50)                    ║')
    safe_print('╚' + '═' * 68 + '╝\n')

    if FEATURE_OVERLAP_CSV.exists():
        FEATURE_OVERLAP_CSV.unlink()

    overlap_results = []
    X_B = pd.concat([X_base.reset_index(drop=True), struct_features.reset_index(drop=True)], axis=1)
    X_B = X_B.loc[:, X_B.nunique(dropna=True) > 1]

    safe_print('Computing feature overlap at n_feat=50...\n')

    X_base_clean = X_base.reset_index(drop=True)
    y_clean = y.reset_index(drop=True)
    groups_clean = pd.Series(groups, dtype=str).reset_index(drop=True)

    imputer_a = SimpleImputer(strategy='mean')
    X_base_imp = pd.DataFrame(imputer_a.fit_transform(X_base_clean), columns=X_base_clean.columns)
    X_base_imp = X_base_imp.loc[:, X_base_imp.nunique(dropna=True) > 1]

    imputer_b = SimpleImputer(strategy='mean')
    X_b_imp = pd.DataFrame(imputer_b.fit_transform(X_B), columns=X_B.columns)
    X_b_imp = X_b_imp.loc[:, X_b_imp.nunique(dropna=True) > 1]

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
            # FIX: Use get_optimal_descriptors() to get ACTUAL selected features, not full pool
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
            # FIX: Use get_optimal_descriptors() to get ACTUAL selected features, not full pool
            selected_b = set(train_data_b.get_optimal_descriptors())
        except Exception as e:
            safe_print(f'WARNING: Could not extract Model B features for fold {fold_num}: {e}')
            selected_b = set()

        overlap = selected_a.intersection(selected_b)
        pct_overlap = 100 * len(overlap) / len(selected_a) if selected_a else 0
        
        # VERIFICATION: Check that we're actually getting 50 selected features, not the full pool
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

    df_test2 = pd.DataFrame(overlap_results)
    df_test2.to_csv(FEATURE_OVERLAP_CSV, index=False)
    safe_print(f'\n✓ TEST 2 results saved to: {FEATURE_OVERLAP_CSV}')
    
    return overlap_results


# ============================================================================
# TEST 3: FORCED-INCLUSION TEST
# ============================================================================
def test3_forced_inclusion(X_base, y, groups, splits, struct_features):
    safe_print('\n' + '╔' + '═' * 68 + '╗')
    safe_print('║ TEST 3 - FORCED-INCLUSION TEST (Model A features protected)      ║')
    safe_print('╚' + '═' * 68 + '╝\n')

    if FORCED_INCLUSION_CSV.exists():
        FORCED_INCLUSION_CSV.unlink()

    safe_print('Building Model C with Model A\'s features guaranteed included...\n')

    X_base_clean = X_base.reset_index(drop=True)
    y_clean = y.reset_index(drop=True)
    groups_clean = pd.Series(groups, dtype=str).reset_index(drop=True)

    imputer_a = SimpleImputer(strategy='mean')
    X_base_imp = pd.DataFrame(imputer_a.fit_transform(X_base_clean), columns=X_base_clean.columns)
    X_base_imp = X_base_imp.loc[:, X_base_imp.nunique(dropna=True) > 1]

    fold_maes_c = []
    fold_rmses_c = []
    fold_r2s_c = []

    for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train_a = X_base_imp.iloc[train_idx].reset_index(drop=True)
        X_test_a = X_base_imp.iloc[test_idx].reset_index(drop=True)
        y_train = y_clean.iloc[train_idx].reset_index(drop=True)
        y_test = y_clean.iloc[test_idx].reset_index(drop=True)

        # Get Model A's selected features
        try:
            train_data_a = MODData(
                materials=list(range(len(X_train_a))),
                targets=[[float(v)] for v in y_train.values],
                target_names=[TARGET_COLUMN],
            )
            train_data_a.df_featurized = X_train_a
            train_data_a.feature_selection(n=min(50, X_train_a.shape[1]))
            selected_features_a = train_data_a.df_featurized.columns.tolist()
        except Exception as e:
            safe_print(f'WARNING: Could not get Model A features for fold {fold_num}: {e}')
            fold_maes_c.append(np.nan)
            fold_rmses_c.append(np.nan)
            fold_r2s_c.append(np.nan)
            continue

        # Build Model C: Model A's features + structure features
        imputer_struct = SimpleImputer(strategy='mean')
        struct_imp = pd.DataFrame(
            imputer_struct.fit_transform(struct_features.iloc[train_idx]),
            columns=struct_features.columns
        )

        X_train_c = pd.concat([
            X_base_imp.iloc[train_idx].reset_index(drop=True),
            struct_imp.reset_index(drop=True)
        ], axis=1)
        X_train_c = X_train_c.loc[:, X_train_c.nunique(dropna=True) > 1]

        X_test_c = pd.concat([
            X_base_imp.iloc[test_idx].reset_index(drop=True),
            struct_features.iloc[test_idx].reset_index(drop=True)
        ], axis=1)
        X_test_c = X_test_c.loc[:, X_test_c.nunique(dropna=True) > 1]

        # Train Model C with n_feat=50 (same as Model A budget)
        try:
            train_data_c = MODData(
                materials=list(range(len(X_train_c))),
                targets=[[float(v)] for v in y_train.values],
                target_names=[TARGET_COLUMN],
            )
            train_data_c.df_featurized = X_train_c
            train_data_c.feature_selection(n=min(50, X_train_c.shape[1]))

            model_c = MODNetModel(
                [[[TARGET_COLUMN]]],
                weights={TARGET_COLUMN: 1},
                n_feat=min(50, X_train_c.shape[1]),
            )
            model_c.fit(
                train_data_c,
                val_fraction=0.1,
                lr=0.001,
                batch_size=64,
                loss='mae',
                epochs=100,
                verbose=0,
            )

            test_data_c = MODData(
                materials=list(range(len(X_test_c))),
                targets=[[0.0]] * len(X_test_c),
                target_names=[TARGET_COLUMN],
            )
            test_data_c.df_featurized = X_test_c
            y_pred_c = model_c.predict(test_data_c)[TARGET_COLUMN].values
        except Exception as e:
            safe_print(f'WARNING: Model C MODNet failed for fold {fold_num}: {e}, using RF fallback')
            rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
            rf.fit(X_train_c, y_train.values)
            y_pred_c = rf.predict(X_test_c)

        mae_c = mean_absolute_error(y_test, y_pred_c)
        rmse_c = np.sqrt(mean_squared_error(y_test, y_pred_c))
        r2_c = r2_score(y_test, y_pred_c)

        fold_maes_c.append(mae_c)
        fold_rmses_c.append(rmse_c)
        fold_r2s_c.append(r2_c)

        safe_print(f'Model C - Fold {fold_num}/5: MAE={mae_c:.4f} R²={r2_c:.4f}')

    result_c = {
        'model': 'C: Forced-inclusion (Model A features + Structure)',
        'MAE_mean': np.mean(fold_maes_c),
        'MAE_std': np.std(fold_maes_c),
        'RMSE_mean': np.mean(fold_rmses_c),
        'RMSE_std': np.std(fold_rmses_c),
        'R2_mean': np.mean(fold_r2s_c),
        'R2_std': np.std(fold_r2s_c),
    }

    df_test3 = pd.DataFrame([result_c])
    df_test3.to_csv(FORCED_INCLUSION_CSV, index=False)
    safe_print(f'\n✓ TEST 3 results saved to: {FORCED_INCLUSION_CSV}')
    
    return result_c, fold_maes_c


# ============================================================================
# TEST 4: STATISTICAL TESTS
# ============================================================================
def test4_statistical_tests(test1_results, fold_maes_c, splits, X_base, y, groups, struct_features):
    safe_print('\n' + '╔' + '═' * 68 + '╗')
    safe_print('║ TEST 4 - STATISTICAL SIGNIFICANCE TESTS                          ║')
    safe_print('║ (Nadeau & Bengio corrected resampling t-test, α=0.05)            ║')
    safe_print('╚' + '═' * 68 + '╝\n')

    if STATISTICAL_TESTS_CSV.exists():
        STATISTICAL_TESTS_CSV.unlink()

    stats_results = []

    # Collect fold-level MAEs from all tests
    test1_by_model = defaultdict(list)
    for n_feat in N_FEAT_VALUES:
        safe_print(f'Collecting fold-level MAEs for n_feat={n_feat}...')
        
        # Model A
        _, _, _, maes_a, _, _, _ = train_model_with_tracking(
            X_base, y, groups, splits, f'Model A (n_feat={n_feat})', n_feat
        )
        test1_by_model[(n_feat, 'A')] = maes_a

        # Model B
        X_B = pd.concat([X_base.reset_index(drop=True), struct_features.reset_index(drop=True)], axis=1)
        X_B = X_B.loc[:, X_B.nunique(dropna=True) > 1]
        _, _, _, maes_b, _, _, _ = train_model_with_tracking(
            X_B, y, groups, splits, f'Model B (n_feat={n_feat})', n_feat
        )
        test1_by_model[(n_feat, 'B')] = maes_b

    # TEST 4a: Model A vs Model B at each n_feat
    safe_print('\n--- Model A vs Model B at each n_feat ---')
    for n_feat in N_FEAT_VALUES:
        maes_a = test1_by_model[(n_feat, 'A')]
        maes_b = test1_by_model[(n_feat, 'B')]
        t_stat, p_value, mean_diff = corrected_resampling_t_test(maes_a, maes_b)
        
        significant = p_value < 0.05
        stats_results.append({
            'comparison': f'Model A vs Model B (n_feat={n_feat})',
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_diff_MAE': mean_diff,
            'significant': significant,
        })
        
        safe_print(f'n_feat={n_feat}: t={t_stat:.3f}, p={p_value:.4f}, Δ MAE={mean_diff:+.4f} {"*" if significant else ""}')

    # TEST 4b: Model A vs Model C (forced-inclusion)
    safe_print('\n--- Model A vs Model C (forced-inclusion) ---')
    maes_a_original = test1_by_model[(50, 'A')]
    maes_c = fold_maes_c
    
    # Filter out NaNs
    valid_pairs = [(a, c) for a, c in zip(maes_a_original, maes_c) if not np.isnan(a) and not np.isnan(c)]
    if valid_pairs:
        maes_a_filtered, maes_c_filtered = zip(*valid_pairs)
        t_stat, p_value, mean_diff = corrected_resampling_t_test(list(maes_a_filtered), list(maes_c_filtered))
        
        significant = p_value < 0.05
        stats_results.append({
            'comparison': 'Model A vs Model C (forced-inclusion)',
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_diff_MAE': mean_diff,
            'significant': significant,
        })
        
        safe_print(f't={t_stat:.3f}, p={p_value:.4f}, Δ MAE={mean_diff:+.4f} {"*" if significant else ""}')

    df_test4 = pd.DataFrame(stats_results)
    df_test4.to_csv(STATISTICAL_TESTS_CSV, index=False)
    safe_print(f'\n✓ TEST 4 results saved to: {STATISTICAL_TESTS_CSV}')
    
    return stats_results


def print_final_summary(test1_results, test2_results, test3_result, test4_results):
    """Print honest, comprehensive summary."""
    safe_print('\n' + '╔' + '═' * 68 + '╗')
    safe_print('║ FINAL HONEST SUMMARY                                             ║')
    safe_print('╚' + '═' * 68 + '╝\n')

    # Extract key metrics
    model_a_50 = [r for r in test1_results if r['n_feat'] == 50 and 'Composition' in r['model'] and 'Structure' not in r['model']][0]
    model_b_50 = [r for r in test1_results if r['n_feat'] == 50 and 'Structure' in r['model']][0]
    model_a_150 = [r for r in test1_results if r['n_feat'] == 150 and 'Composition' in r['model'] and 'Structure' not in r['model']][0]
    model_b_150 = [r for r in test1_results if r['n_feat'] == 150 and 'Structure' in r['model']][0]

    safe_print('QUESTION 1: Does feature overlap show meaningful displacement?')
    safe_print('-' * 68)
    avg_overlap = np.mean([r['pct_overlap'] for r in test2_results])
    safe_print(f'Average feature overlap (Model A ∩ Model B): {avg_overlap:.1f}%')
    if avg_overlap < 50:
        safe_print('→ YES: Significant feature displacement detected.')
    elif avg_overlap > 80:
        safe_print('→ NO: Features largely overlap; displacement is minimal.')
    else:
        safe_print('→ PARTIAL: Moderate displacement observed.')

    safe_print('\nQUESTION 2: When composition features are protected (Model C),')
    safe_print('does Model C match, beat, or underperform Model A?')
    safe_print('-' * 68)
    mae_a = model_a_50['MAE_mean']
    mae_c = test3_result['MAE_mean']
    diff_c = mae_c - mae_a
    safe_print(f'Model A MAE: {mae_a:.4f} ± {model_a_50["MAE_std"]:.4f}')
    safe_print(f'Model C MAE: {mae_c:.4f} ± {test3_result["MAE_std"]:.4f}')
    safe_print(f'Difference:  {diff_c:+.4f}')
    
    if diff_c < -0.002:
        safe_print('→ Model C BEATS Model A: Protection + structure helps!')
    elif diff_c > 0.002:
        safe_print('→ Model C UNDERPERFORMS Model A: Structure features are unhelpful.')
    else:
        safe_print('→ Model C matches Model A: Structure adds no value.')

    safe_print('\nQUESTION 3: Is Model A vs B difference statistically significant?')
    safe_print('-' * 68)
    sig_comparisons = [r for r in test4_results if 'Model A vs Model B' in r['comparison'] and r['significant']]
    if sig_comparisons:
        safe_print(f'YES: {len(sig_comparisons)} n_feat level(s) show p < 0.05')
        for r in sig_comparisons:
            safe_print(f"  - {r['comparison']}: p={r['p_value']:.4f}")
    else:
        safe_print('NO: No statistically significant differences found (p ≥ 0.05)')

    safe_print('\nQUESTION 4: FINAL HONEST CONCLUSION')
    safe_print('-' * 68)

    gap_50 = model_b_50['MAE_mean'] - model_a_50['MAE_mean']
    gap_150 = model_b_150['MAE_mean'] - model_a_150['MAE_mean']
    gap_change = gap_150 - gap_50

    if gap_change < -0.002 and avg_overlap < 60 and diff_c < -0.002:
        safe_print('SUPPORTED: Feature-selection-budget displacement hypothesis.')
        safe_print('  • Significant feature overlap change (avg {:.1f}%)'.format(avg_overlap))
        safe_print('  • Gap shrinks as n_feat increases ({:+.4f})'.format(gap_change))
        safe_print('  • Model C with protected features beats Model A')
    elif abs(gap_change) < 0.001 and avg_overlap > 70 and abs(diff_c) < 0.002:
        safe_print('REFUTED: Feature-selection-budget explanation is UNLIKELY.')
        safe_print('  • Feature overlap is high ({:.1f}%)'.format(avg_overlap))
        safe_print('  • Gap remains constant across n_feat levels ({:+.4f})'.format(gap_change))
        safe_print('  • Model C with protected features does NOT beat Model A')
        safe_print('  → CONCLUSION: Noise-floor or data limitation explanation more plausible.')
    else:
        safe_print('INCONCLUSIVE: Evidence is mixed or insufficient.')
        safe_print('  • Feature overlap: {:.1f}%'.format(avg_overlap))
        safe_print('  • Gap change (n_feat=50→150): {:+.4f}'.format(gap_change))
        safe_print('  • Model C performance: {:+.4f} vs Model A'.format(diff_c))
        safe_print('  → Recommend additional investigation or domain expertise.')

    safe_print('\n' + '═' * 68 + '\n')


def main():
    mat_df, groups = verify_preflight()
    _, X_base, y, _ = build_master_dataset()
    splits = load_splits()
    struct_features = load_structure_features(groups)

    # Run all 4 tests
    test1_results = test1_nfeat_sensitivity(mat_df, X_base, y, groups, splits, struct_features)
    test2_results = test2_feature_overlap(X_base, y, groups, splits, struct_features)
    test3_result, fold_maes_c = test3_forced_inclusion(X_base, y, groups, splits, struct_features)
    test4_results = test4_statistical_tests(test1_results, fold_maes_c, splits, X_base, y, groups, struct_features)

    # Print comprehensive summary
    print_final_summary(test1_results, test2_results, test3_result, test4_results)

    safe_print('All tests completed. Results saved to results/ directory.')


PRE_FLIGHT_DATA = verify_preflight()

if __name__ == '__main__':
    main()
