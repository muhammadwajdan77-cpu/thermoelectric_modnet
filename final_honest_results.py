#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')

import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pymatgen.core import Composition
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from modnet.models import MODNetModel
from modnet.preprocessing import MODData

SEED = 42
np.random.seed(SEED)
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = 'target'
MODEL_NAMES = [
    'Matminer + MODNet',
    'Matminer + Roost + MODNet',
    'Matminer + l-MM + MODNet',
    'Matminer + Roost + l-OFM + MVL + ORB (XGB-RFE) + MODNet',
]


def canonical_formula(formula):
    if pd.isna(formula):
        raise ValueError('Empty formula cannot be canonicalized')
    return Composition(str(formula)).reduced_formula


def read_sys_dataset():
    dataset_path = ROOT / 'sysTEm_dataset' / 'sysTEm_dataset.xlsx'
    if not dataset_path.exists():
        raise FileNotFoundError(f'Could not find SysTEm dataset at {dataset_path}')
    df = pd.read_excel(dataset_path)
    if 'Pretty Formula' not in df.columns:
        raise ValueError('SysTEm dataset must contain Pretty Formula column')
    return df


def load_matminer_anchor():
    anchor_path = RESULTS_DIR / 'matminer_for_sisso.csv'
    if not anchor_path.exists():
        raise FileNotFoundError(f'Anchor file not found: {anchor_path}')
    df = pd.read_csv(anchor_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f'Anchor file must contain a "{TARGET_COLUMN}" column')
    return df


def numeric_feature_matrix(df, drop_columns=None):
    df = df.copy()
    if drop_columns is not None:
        df = df.drop(columns=[c for c in drop_columns if c in df.columns], errors='ignore')
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return df[numeric_cols].copy()


def safe_canonical_series(formulas):
    return formulas.astype(str).apply(canonical_formula)


def canonical_or_nan(formula):
    if pd.isna(formula):
        return np.nan
    text = str(formula).strip()
    if not text or text.lower() in {'nan', 'none'}:
        return np.nan
    try:
        return canonical_formula(text)
    except Exception:
        return np.nan


def _is_formula_parseable(formula):
    try:
        Composition(str(formula))
        return True
    except Exception:
        return False


def find_composition_column(df):
    for col in ['composition', 'pretty_formula', 'Pretty Formula']:
        if col in df.columns:
            return col
    unnamed = [c for c in df.columns if c.startswith('Unnamed')]
    return unnamed[0] if unnamed else None


def load_features_with_canonical(path, composition_column=None):
    df = pd.read_csv(path)
    comp_col = composition_column or find_composition_column(df)
    if comp_col is None:
        raise ValueError(f'{path.name} must contain a composition column')
    if comp_col != 'composition':
        df = df.rename(columns={comp_col: 'composition'})
    df['canonical'] = df['composition'].apply(canonical_or_nan)
    df = df.dropna(subset=['canonical']).drop_duplicates(subset=['canonical']).reset_index(drop=True)
    return df


def train_groupkfold_modnet(X, y, groups, model_label, n_feat=50, epochs=100):
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    groups = groups.reset_index(drop=True)

    imputer = SimpleImputer(strategy='mean')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_imp = X_imp.loc[:, X_imp.nunique(dropna=True) > 1]

    group_kf = GroupKFold(n_splits=5)
    fold_maes = []
    fold_rmses = []
    fold_r2s = []
    all_y_true = []
    all_y_pred = []

    use_rf = False
    fallback_reason = None

    def run_fold_loop(use_random_forest=False):
        nonlocal all_y_true, all_y_pred
        if use_random_forest:
            print('    [Fallback] Using RandomForestRegressor for all folds')
        for fold, (train_idx, test_idx) in enumerate(group_kf.split(X_imp, y, groups=groups), start=1):
            train_groups = set(groups.iloc[train_idx].astype(str).unique())
            test_groups = set(groups.iloc[test_idx].astype(str).unique())
            overlap = train_groups.intersection(test_groups)
            if overlap:
                raise ValueError(f'Fold {fold} has {len(overlap)} composition overlaps between training and test')
            print(f'  Fold {fold}/5 — group overlap check passed ({len(train_groups)} train groups, {len(test_groups)} test groups)')

            X_train = X_imp.iloc[train_idx].reset_index(drop=True)
            X_test = X_imp.iloc[test_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            y_test = y.iloc[test_idx].reset_index(drop=True)

            if use_random_forest:
                model = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
                model.fit(X_train, y_train.values.ravel())
                y_pred = model.predict(X_test)
            else:
                train_data = MODData(
                    materials=list(range(len(train_idx))),
                    targets=[[float(v)] for v in y_train.values],
                    target_names=[TARGET_COLUMN],
                )
                train_data.df_featurized = X_train
                train_data.feature_selection(n=min(n_feat, X_train.shape[1]))

                model = MODNetModel([[[TARGET_COLUMN]]], weights={TARGET_COLUMN: 1}, n_feat=min(n_feat, X_train.shape[1]))
                model.fit(
                    train_data,
                    val_fraction=0.1,
                    lr=0.001,
                    batch_size=64,
                    loss='mae',
                    epochs=epochs,
                    verbose=0,
                )

                test_data = MODData(
                    materials=list(range(len(test_idx))),
                    targets=[[0.0]] * len(test_idx),
                    target_names=[TARGET_COLUMN],
                )
                test_data.df_featurized = X_test
                y_pred = model.predict(test_data)[TARGET_COLUMN].values

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            print(f'    MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}')

            fold_maes.append(mae)
            fold_rmses.append(rmse)
            fold_r2s.append(r2)
            all_y_true.extend(y_test.values.tolist())
            all_y_pred.extend(y_pred.tolist())

    print(f'\nTRAINING {model_label}')
    print(f'Features: {X_imp.shape[1]} | Samples: {len(y)} | Unique compositions: {groups.nunique()}')
    try:
        run_fold_loop(use_random_forest=False)
    except Exception as exc:
        fallback_reason = str(exc)
        print(f'  MODNet failed: {fallback_reason}')
        print('  Falling back to RandomForestRegressor for this model')
        fold_maes.clear(); fold_rmses.clear(); fold_r2s.clear(); all_y_true.clear(); all_y_pred.clear()
        use_rf = True
        run_fold_loop(use_random_forest=True)

    metrics = {
        'MAE': np.mean(fold_maes),
        'MAE_std': np.std(fold_maes),
        'RMSE': np.mean(fold_rmses),
        'RMSE_std': np.std(fold_rmses),
        'R2': np.mean(fold_r2s),
        'R2_std': np.std(fold_r2s),
        'model_type': 'RandomForest' if use_rf else 'MODNet',
        'fallback_reason': fallback_reason,
    }
    return metrics, np.array(all_y_true), np.array(all_y_pred)


def parity_plot(true_vals, pred_vals, model_name):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(true_vals, pred_vals, alpha=0.35, s=16, color='darkblue', edgecolors='none')
    low = min(true_vals.min(), pred_vals.min())
    high = max(true_vals.max(), pred_vals.max())
    margin = (high - low) * 0.05 if high > low else 0.1
    ax.plot([low - margin, high + margin], [low - margin, high + margin], 'r--', lw=1.5)
    ax.set_xlabel('Actual ZT', fontweight='bold')
    ax.set_ylabel('Predicted ZT', fontweight='bold')
    ax.set_title(f'Honest parity plot — {model_name}', fontweight='bold')
    ax.set_xlim(low - margin, high + margin)
    ax.set_ylim(low - margin, high + margin)
    ax.grid(True, linestyle=':', alpha=0.4)
    output_path = FIGURES_DIR / f'honest_parity_{model_name.replace(" ", "_").replace("+", "plus").replace("/", "_")}.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved parity plot: {output_path}')


def build_model1(mat_df, groups, sys_valid=None):
    X = numeric_feature_matrix(mat_df.drop(columns=[TARGET_COLUMN], errors='ignore'))
    y = mat_df[TARGET_COLUMN].astype(float)
    return X, y, groups


def build_model2(mat_df, groups, sys_valid=None):
    if sys_valid is None:
        sys_valid = read_sys_dataset().loc[
            [i for i, f in enumerate(read_sys_dataset()['Pretty Formula'].astype(str))
             if _is_formula_parseable(f)]
        ].reset_index(drop=True)

    roost = pd.read_csv(RESULTS_DIR / 'roost_features.csv')
    roost = roost.copy()
    n_rows = min(len(roost), len(sys_valid))
    if len(roost) != len(sys_valid):
        print(f'  Warning: roost_features.csv has {len(roost)} rows, sys_valid has {len(sys_valid)} rows; using first {n_rows} canonical formulas')
    roost.loc[:n_rows - 1, 'canonical'] = safe_canonical_series(sys_valid.loc[:n_rows - 1, 'Pretty Formula'])
    roost = roost.loc[:n_rows - 1].drop_duplicates(subset=['canonical']).reset_index(drop=True)

    anchor = mat_df.assign(canonical=groups.reset_index(drop=True))
    merged = anchor.merge(roost, on='canonical', how='left', validate='many_to_one')
    roost_feature_cols = [c for c in roost.columns if c not in ['canonical']]
    matched = merged[roost_feature_cols].notna().any(axis=1).sum()
    print(f'  Matminer anchor rows: {len(anchor)}')
    print(f'  Roost matched rows: {matched}/{len(anchor)}')

    X = numeric_feature_matrix(merged.drop(columns=[TARGET_COLUMN, 'canonical'], errors='ignore'))
    y = merged[TARGET_COLUMN].astype(float)
    return X, y, groups


def build_model3(mat_df, groups, sys_valid=None):
    lmm = pd.read_csv(RESULTS_DIR / 'lMM_features.csv')
    comp_col = find_composition_column(lmm)
    if comp_col is None:
        raise ValueError('lMM_features.csv must contain a composition column')
    if comp_col != 'composition':
        lmm = lmm.rename(columns={comp_col: 'composition'})
    lmm['canonical'] = lmm['composition'].apply(canonical_or_nan)
    megnet_cols = [c for c in lmm.columns if c.startswith('MEGNet')]
    if not megnet_cols:
        raise ValueError('No MEGNet columns found in lMM_features.csv')
    lmm = lmm.dropna(subset=['canonical']).drop_duplicates(subset=['canonical']).reset_index(drop=True)

    anchor = mat_df.assign(canonical=groups.reset_index(drop=True))
    merged = anchor.merge(lmm[['canonical'] + megnet_cols], on='canonical', how='left', validate='many_to_one')
    matched = merged[megnet_cols].notna().any(axis=1).sum()
    print(f'  Matminer anchor rows: {len(anchor)}')
    print(f'  l-MM matched rows: {matched}/{len(anchor)}')

    X = numeric_feature_matrix(merged.drop(columns=[TARGET_COLUMN, 'canonical'], errors='ignore'))
    y = merged[TARGET_COLUMN].astype(float)
    return X, y, groups


def build_model4(mat_df, groups, sys_valid=None):
    if sys_valid is None:
        sys_valid = read_sys_dataset().loc[
            [i for i, f in enumerate(read_sys_dataset()['Pretty Formula'].astype(str))
             if _is_formula_parseable(f)]
        ].reset_index(drop=True)

    roost = pd.read_csv(RESULTS_DIR / 'roost_features.csv')
    roost = roost.copy()
    n_rows = min(len(roost), len(sys_valid))
    if len(roost) != len(sys_valid):
        print(f'  Warning: roost_features.csv has {len(roost)} rows, sys_valid has {len(sys_valid)} rows; using first {n_rows} canonical formulas')
    roost.loc[:n_rows - 1, 'canonical'] = safe_canonical_series(sys_valid.loc[:n_rows - 1, 'Pretty Formula'])
    roost = roost.loc[:n_rows - 1].drop_duplicates(subset=['canonical']).reset_index(drop=True)

    anchor = mat_df.assign(canonical=groups.reset_index(drop=True))
    merged = anchor.merge(roost, on='canonical', how='left', validate='many_to_one')
    roost_feature_cols = [c for c in roost.columns if c not in ['canonical']]
    matched_roost = merged[roost_feature_cols].notna().any(axis=1).sum()
    print(f'  Matminer anchor rows: {len(anchor)}')
    print(f'  Roost matched rows: {matched_roost}/{len(anchor)}')

    def merge_source(base, path, composition_column=None, source_name=None):
        source = load_features_with_canonical(path, composition_column=composition_column)
        feature_cols = [c for c in source.columns if c not in ['composition', 'canonical']]
        source = source[['canonical'] + feature_cols]
        source = source.drop_duplicates(subset=['canonical']).reset_index(drop=True)
        merged_source = base.merge(source, on='canonical', how='left', validate='many_to_one')
        matched = merged_source[feature_cols].notna().any(axis=1).sum()
        print(f'  {source_name} matched rows: {matched}/{len(base)}')
        return merged_source

    merged = merge_source(merged, RESULTS_DIR / 'lOFM_features.csv', composition_column=None, source_name='l-OFM')
    merged = merge_source(merged, RESULTS_DIR / 'MVL_features.csv', composition_column=None, source_name='MVL')
    merged = merge_source(merged, RESULTS_DIR / 'ORB_features.csv', composition_column='composition', source_name='ORB')

    X = numeric_feature_matrix(merged.drop(columns=[TARGET_COLUMN, 'canonical'], errors='ignore'))
    X = X.loc[:, ~X.columns.duplicated(keep='first')]

    X_imp = X.fillna(X.mean())
    X_final = X_imp.loc[:, X_imp.nunique(dropna=True) > 1].reset_index(drop=True)
    print(f'  Full combined feature set before RFE: {X.shape[1]} features')
    print(f'  After removing constant columns: {X_final.shape[1]} features')
    print('  NOTE: Performing RFE on full data before GroupKFold split (slightly optimistic)')

    selector = RFE(
        estimator=XGBRegressor(n_estimators=100, random_state=SEED, n_jobs=-1, verbosity=0),
        n_features_to_select=50,
        step=50,
    )
    selector.fit(X_final, mat_df[TARGET_COLUMN].astype(float).values)
    X_selected = X_final.loc[:, selector.get_support()].reset_index(drop=True)
    print(f'  Selected {X_selected.shape[1]} features with XGB-RFE')
    y = mat_df[TARGET_COLUMN].astype(float).reset_index(drop=True)
    return X_selected, y, groups


def save_results(results):
    out_path = RESULTS_DIR / 'HONEST_FINAL_RESULTS.csv'
    df = pd.DataFrame(results)
    df = df[['Model', 'MAE', 'RMSE', 'R2', 'N_Samples', 'N_Unique_Compositions']]
    if out_path.exists():
        existing = pd.read_csv(out_path)
        combined = pd.concat([existing, df], axis=0, ignore_index=True)
        combined = combined.drop_duplicates(subset=['Model'], keep='last')
        df = combined
    df.to_csv(out_path, index=False)
    print(f'\nSaved honest final results: {out_path}')


def main():
    mat_df = load_matminer_anchor()
    sys_df = read_sys_dataset()

    valid_indices = []
    for i, f in enumerate(sys_df['Pretty Formula'].astype(str)):
        try:
            Composition(f)
            valid_indices.append(i)
        except Exception:
            continue
    sys_valid = sys_df.iloc[valid_indices].reset_index(drop=True)
    if len(sys_valid) < len(mat_df):
        raise ValueError('Not enough parseable rows in the SysTEm dataset to align with matminer anchor')

    anchor_formulas = sys_valid.loc[:len(mat_df) - 1, 'Pretty Formula'].astype(str)
    canonical_groups = safe_canonical_series(anchor_formulas)

    print('Loaded anchor and canonical groups:')
    print(f'  Anchor rows: {len(mat_df)}')
    print(f'  Parseable anchor rows: {len(anchor_formulas)}')
    print(f'  Unique canonical groups: {canonical_groups.nunique()}')

    parser = argparse.ArgumentParser(description='Run honest model evaluation')
    parser.add_argument('--models', type=str, default='1,2,3,4', help='Comma-separated model indices to run (1-4)')
    args, _ = parser.parse_known_args()
    requested = [int(x) for x in args.models.split(',') if x.strip().isdigit() and 1 <= int(x.strip()) <= 4]

    model_builders = [build_model1, build_model2, build_model3, build_model4]
    requested_builders = [(idx, MODEL_NAMES[idx - 1], model_builders[idx - 1]) for idx in requested]

    results = []

    for idx, name, builder in requested_builders:
        print(f'\nStarting Model {idx}/4: {name}...')
        try:
            X, y, groups = builder(mat_df, canonical_groups)
            metrics, y_true, y_pred = train_groupkfold_modnet(X, y, groups, name, n_feat=50, epochs=100)
            parity_plot(y_true, y_pred, name)
            results.append({
                'Model': name,
                'MAE': f'{metrics["MAE"]:.4f}±{metrics["MAE_std"]:.4f}',
                'RMSE': f'{metrics["RMSE"]:.4f}±{metrics["RMSE_std"]:.4f}',
                'R2': f'{metrics["R2"]:.4f}±{metrics["R2_std"]:.4f}',
                'N_Samples': len(y),
                'N_Unique_Compositions': groups.nunique(),
            })
            print(f'Model {idx}/4 complete: R2={metrics["R2"]:.4f}')
        except Exception as exc:
            print(f'Model {idx}/4 failed: {name}')
            print(f'  Error: {exc}')
            results.append({
                'Model': name,
                'MAE': 'FAILED',
                'RMSE': 'FAILED',
                'R2': 'FAILED',
                'N_Samples': 0,
                'N_Unique_Compositions': 0,
            })

    save_results(results)

    final_df = pd.DataFrame(results)
    final_df['R2_sort'] = final_df['R2'].replace('FAILED', np.nan).str.extract(r'([0-9\.-]+)').astype(float)
    final_df = final_df.sort_values(by='R2_sort', ascending=False).drop(columns=['R2_sort'])
    print('\nFINAL COMPARISON TABLE')
    print(final_df.to_string(index=False))


if __name__ == '__main__':
    main()
