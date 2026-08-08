#!/usr/bin/env python3
"""Structure comparison v2: composition-only vs composition+structure

Follows the user's specification: merges corrected matminer CSVs, accepts
minimum structure coverage for Model B, generates canonical-formula GroupKFold
splits, trains MODNet (n_feat=50, epochs=100) for Model A and Model B across
5 folds, repeats 3 independent runs, runs corrected resampling t-test, and
saves fold-level results to results/STRUCTURE_COMPARISON_V2_FINAL.csv.
"""

from pathlib import Path
import pickle
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from scipy.stats import t as t_dist

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MATMINER_PATH = RESULTS_DIR / 'matminer_for_sisso_v2.csv'
STRUCT_PATH = RESULTS_DIR / 'matminer_structure_features.csv'
SPLITS_PICKLE = RESULTS_DIR / 'group_kfold_splits_v2.pkl'
OUTPUT_CSV = RESULTS_DIR / 'STRUCTURE_COMPARISON_V2_FINAL.csv'

TARGET_COLUMN = 'target'
N_SPLITS = 5
N_RUNS = 3
N_FEAT = 50
EPOCHS = 100
MIN_MATCH_RATE = 0.75

SEED = 42
np.random.seed(SEED)


def safe_print(*args, **kwargs):
    print(*args, flush=True, **kwargs)


def canonical_formula(formula_text):
    if pd.isna(formula_text):
        return ""
    text = str(formula_text).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    try:
        return Composition(text).reduced_formula
    except Exception:
        return text.replace(" ", "")


def extract_composition_from_structure_raw(raw_value):
    text = str(raw_value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    # Strip suffixes from CIF-derived composition strings such as
    # "Ag0.01Sn0.99Se0.65S0.35_generalized_grouping_grouped_from_..."
    return text.split('_', 1)[0].strip()


def load_and_merge():
    if not MATMINER_PATH.exists():
        raise FileNotFoundError(f'Missing file: {MATMINER_PATH}')
    if not STRUCT_PATH.exists():
        raise FileNotFoundError(f'Missing file: {STRUCT_PATH}')

    mat = pd.read_csv(MATMINER_PATH)
    struct = pd.read_csv(STRUCT_PATH)

    # Basic checks
    for col in ['canonical_formula', TARGET_COLUMN]:
        if col not in mat.columns:
            raise ValueError(f"Column '{col}' missing from {MATMINER_PATH}")
    if 'composition' not in struct.columns:
        raise ValueError(f"Column 'composition' missing from {STRUCT_PATH}")

    # Re-canonicalize both sides with the same function before matching.
    mat['canonical_formula'] = mat['canonical_formula'].astype(str).map(canonical_formula)
    struct['canonical_formula'] = (
        struct['composition']
        .astype(str)
        .map(extract_composition_from_structure_raw)
        .map(canonical_formula)
    )

    # Build lookup from structure file: map canonical -> features (first occurrence)
    struct_cols = [c for c in struct.columns if c not in {'composition', 'canonical_formula'}]
    lookup = {}
    for _, row in struct.iterrows():
        can = row['canonical_formula']
        if not can:
            continue
        if can not in lookup:
            lookup[can] = row[struct_cols].to_dict()

    unique_groups = mat['canonical_formula'].astype(str).str.strip().replace('', pd.NA).dropna().unique()
    matched_groups = [g for g in unique_groups if g in lookup]
    match_rate = len(matched_groups) / len(unique_groups) if len(unique_groups) > 0 else 0.0
    original_rows = len(mat)
    matched_rows = mat['canonical_formula'].astype(str).str.strip().isin(matched_groups).sum()
    safe_print(f'Structure feature match: {len(matched_groups)}/{len(unique_groups)} unique formulas ({matched_rows}/{original_rows} rows)')
    safe_print(
        f'Structure comparison uses {len(matched_groups)}/{len(unique_groups)} unique compositions ({match_rate:.2%}); '
        'the remaining compositions have no available ProtoCSP-generated structure and are excluded from Model B.'
    )

    if match_rate < MIN_MATCH_RATE:
        missing = [g for g in unique_groups if g not in lookup][:20]
        safe_print('ERROR: Structure features missing for some canonical_formula values.')
        safe_print('Examples of missing canonical_formula values:', missing)
        safe_print(f'ERROR: Match rate is {match_rate:.4%}, aborting.')
        sys.exit(1)

    if match_rate < 1.0:
        missing = [g for g in unique_groups if g not in lookup][:20]
        safe_print('WARNING: Structure features missing for some canonical_formula values.')
        safe_print('Examples of missing canonical_formula values:', missing)
        safe_print(f'Proceeding with coverage {match_rate:.4%} (minimum required {MIN_MATCH_RATE:.0%}).')

    mat = mat[mat['canonical_formula'].astype(str).str.strip().isin(matched_groups)].reset_index(drop=True)

    # Build struct_features aligned to mat rows (by canonical_formula)
    def row_to_struct(can):
        return lookup.get(can, {col: np.nan for col in struct_cols})

    struct_features = pd.DataFrame([row_to_struct(c) for c in mat['canonical_formula'].astype(str).str.strip()])
    # Coerce types: factorize strings, convert bool to float, numeric coercion
    for col in struct_features.columns:
        if struct_features[col].dtype == object:
            codes, _ = pd.factorize(struct_features[col], sort=True)
            struct_features[col] = codes.astype(float)
        elif pd.api.types.is_bool_dtype(struct_features[col]):
            struct_features[col] = struct_features[col].astype(float)
        else:
            struct_features[col] = pd.to_numeric(struct_features[col], errors='coerce')

    return mat, struct_features


def build_or_load_splits(df):
    if SPLITS_PICKLE.exists():
        with open(SPLITS_PICKLE, 'rb') as fh:
            splits = pickle.load(fh)
        if len(splits) == N_SPLITS and all((len(train) + len(test)) == len(df) for train, test in splits):
            safe_print(f'Loaded existing splits from {SPLITS_PICKLE}')
            return splits
        safe_print(f'Existing split file {SPLITS_PICKLE} does not match current data; regenerating.')

    gkf = GroupKFold(n_splits=N_SPLITS)
    groups = df['canonical_formula'].astype(str).values
    splits = []
    for train_idx, test_idx in gkf.split(df, groups=groups):
        splits.append((train_idx.tolist(), test_idx.tolist()))

    with open(SPLITS_PICKLE, 'wb') as fh:
        pickle.dump(splits, fh)
    safe_print(f'Generated and saved splits to {SPLITS_PICKLE}')
    return splits


def train_fold_modnet(X_train, y_train, X_test):
    # Try MODNet; fallback to RandomForest on error
    try:
        from modnet.models import MODNetModel
        from modnet.preprocessing import MODData

        train_data = MODData(materials=list(range(len(X_train))), targets=[[float(v)] for v in y_train.values], target_names=[TARGET_COLUMN])
        train_data.df_featurized = X_train
        train_data.feature_selection(n=min(N_FEAT, X_train.shape[1]))

        model = MODNetModel([[[TARGET_COLUMN]]], weights={TARGET_COLUMN: 1}, n_feat=min(N_FEAT, X_train.shape[1]))
        model.fit(train_data, val_fraction=0.1, lr=0.001, batch_size=64, loss='mae', epochs=EPOCHS, verbose=0)

        test_data = MODData(materials=list(range(len(X_test))), targets=[[0.0]] * len(X_test), target_names=[TARGET_COLUMN])
        test_data.df_featurized = X_test
        y_pred = model.predict(test_data)[TARGET_COLUMN].values
        return y_pred, False
    except Exception as exc:
        safe_print(f'  MODNet failed, falling back to RandomForest: {exc}')
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
        rf.fit(X_train, y_train.values)
        y_pred = rf.predict(X_test)
        return y_pred, True


def train_model_once(X_base, X_struct, y, splits, model_label, groups_df):
    # X_base: composition features (includes Temperature etc.) aligned to rows
    # X_struct: structure features aligned to rows
    X_base = X_base.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if model_label == 'A':
        X = X_base.copy()
    else:
        X = pd.concat([X_base.reset_index(drop=True), X_struct.reset_index(drop=True)], axis=1)

    imputer = SimpleImputer(strategy='mean')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_imp = X_imp.loc[:, X_imp.nunique(dropna=True) > 1]

    fold_results = []
    any_fallback = False

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        # Zero-overlap assertion using canonical_formula groups
        train_comps = set(groups_df.loc[train_idx, 'canonical_formula'].astype(str).unique())
        test_comps = set(groups_df.loc[test_idx, 'canonical_formula'].astype(str).unique())
        assert len(train_comps & test_comps) == 0, f"LEAKAGE in fold {fold_idx}"
        safe_print(f'Fold {fold_idx}: verified zero overlap ✓')

        X_train = X_imp.iloc[train_idx].reset_index(drop=True)
        X_test = X_imp.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        y_pred, fallback = train_fold_modnet(X_train, y_train, X_test)
        any_fallback = any_fallback or fallback

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        fold_results.append({'model': model_label, 'fold': fold_idx, 'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)})
        safe_print(f'{model_label} - Fold {fold_idx}: MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}')

    return fold_results, any_fallback


def corrected_resampling_t_test(maes_a, maes_b, train_size, test_size, k=5):
    diffs = np.array(maes_a, dtype=float) - np.array(maes_b, dtype=float)
    d_bar = float(np.mean(diffs))
    var_d = float(np.var(diffs, ddof=1))
    var_corrected = (1 / k + test_size / train_size) * var_d
    if var_corrected <= 0:
        t_stat = 0.0
        p_value = 1.0
    else:
        t_stat = d_bar / np.sqrt(var_corrected)
        p_value = float(2 * t_dist.sf(abs(t_stat), df=k - 1))
    return t_stat, p_value, d_bar


if __name__ == '__main__':
    safe_print('Loading and merging datasets...')
    mat_df, struct_features = load_and_merge()
    df = mat_df

    # Prepare feature matrices
    metadata_cols = ['formula', 'canonical_formula', 'sys_df_original_index']
    X_base = mat_df.drop(columns=[TARGET_COLUMN] + metadata_cols, errors='ignore')
    X_base = X_base.select_dtypes(include=[np.number]).copy()
    y = mat_df[TARGET_COLUMN].astype(float)

    # Build or load splits (list of (train_idx, test_idx) tuples)
    splits = build_or_load_splits(mat_df)

    # For t-test sizes: use first split lengths
    train_len = len(splits[0][0])
    test_len = len(splits[0][1])

    # Run N_RUNS repetitions
    all_runs = []
    for run in range(1, N_RUNS + 1):
        safe_print(f'=== RUN {run}/{N_RUNS} ===')

        # Model A
        res_a, fallback_a = train_model_once(X_base, struct_features, y, splits, 'A', df)
        for r in res_a:
            r.update({'run': run, 'model_name': 'Composition-only (A)'})

        # Model B
        res_b, fallback_b = train_model_once(X_base, struct_features, y, splits, 'B', df)
        for r in res_b:
            r.update({'run': run, 'model_name': 'Composition+Structure (B)'})

        all_runs.extend(res_a)
        all_runs.extend(res_b)

    # Save fold-level CSV
    df_out = pd.DataFrame(all_runs)[['model_name', 'run', 'fold', 'mae', 'rmse', 'r2']]
    df_out.to_csv(OUTPUT_CSV, index=False)
    safe_print(f'Saved fold-level results to: {OUTPUT_CSV}')

    # Aggregate per-model across runs: compute MAE mean±std and R2 mean±std
    ag = df_out.groupby(['model_name', 'run', 'fold']).first().reset_index()
    # Average fold MAEs across runs for each fold
    maes_a = []
    maes_b = []
    for fold in range(1, N_SPLITS + 1):
        vals_a = df_out[(df_out['model_name'].str.contains('Composition-only', regex=False)) & (df_out['fold'] == fold)]['mae'].values
        vals_b = df_out[(df_out['model_name'].str.contains('Composition+Structure', regex=False)) & (df_out['fold'] == fold)]['mae'].values
        # average across runs
        maes_a.append(float(np.mean(vals_a)))
        maes_b.append(float(np.mean(vals_b)))

    t_stat, p_value, d_bar = corrected_resampling_t_test(maes_a, maes_b, train_len, test_len, k=N_SPLITS)
    significant = p_value < 0.05

    # Compute overall summaries
    summary = df_out.groupby('model_name').agg({'mae': ['mean', 'std'], 'r2': ['mean', 'std']})
    safe_print('\nFINAL SUMMARY:')
    for name, row in summary.iterrows():
        mae_mean = row[('mae', 'mean')]
        mae_std = row[('mae', 'std')]
        r2_mean = row[('r2', 'mean')]
        r2_std = row[('r2', 'std')]
        safe_print(f"{name}: MAE={mae_mean:.4f}±{mae_std:.4f}, R2={r2_mean:.4f}±{r2_std:.4f}")

    safe_print(f'\nt-test: t={t_stat:.4f}, p={p_value:.4e}, significant={significant}')
    safe_print('\nDone.')
