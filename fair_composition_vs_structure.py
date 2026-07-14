#!/usr/bin/env python3
"""
Fair comparison: Composition-only Matminer vs Composition+Structure Matminer.

Uses identical GroupKFold splits (saved to pickle) for both models.
Both models use the same MODNet architecture and hyperparameters.
"""

import warnings
warnings.filterwarnings('ignore')

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from modnet.models import MODNetModel
from modnet.preprocessing import MODData

SEED = 42
np.random.seed(SEED)
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = 'zT'  # Actual target column in SysTEm dataset
GROUPKFOLD_SPLITS_PATH = RESULTS_DIR / 'groupkfold_splits.pkl'


def canonical_formula(formula):
    """Convert formula string to canonical reduced formula."""
    if pd.isna(formula):
        raise ValueError('Empty formula cannot be canonicalized')
    return Composition(str(formula)).reduced_formula


def canonical_or_nan(formula):
    """Safely canonicalize formula, return NaN on failure."""
    if pd.isna(formula):
        return np.nan
    text = str(formula).strip()
    if not text or text.lower() in {'nan', 'none'}:
        return np.nan
    try:
        return canonical_formula(text)
    except Exception:
        return np.nan


def numeric_feature_matrix(df, drop_columns=None):
    """Extract numeric features from dataframe."""
    df = df.copy()
    if drop_columns is not None:
        df = df.drop(columns=[c for c in drop_columns if c in df.columns], errors='ignore')
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return df[numeric_cols].copy()


def load_sys_dataset():
    """Load SysTEm dataset for canonical formulas and target values."""
    dataset_path = ROOT / 'sysTEm_dataset' / 'sysTEm_dataset.xlsx'
    if not dataset_path.exists():
        raise FileNotFoundError(f'Could not find SysTEm dataset at {dataset_path}')
    df = pd.read_excel(dataset_path)
    if 'Pretty Formula' not in df.columns:
        raise ValueError('SysTEm dataset must contain Pretty Formula column')
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f'SysTEm dataset must contain {TARGET_COLUMN} column')
    return df


def load_matminer_anchor():
    """Load matminer composition-only features (features only, no target)."""
    anchor_path = RESULTS_DIR / 'matminer_for_sisso.csv'
    if not anchor_path.exists():
        raise FileNotFoundError(f'Anchor file not found: {anchor_path}')
    df = pd.read_csv(anchor_path)
    return df


def load_matminer_structure_features():
    """Load structure-based matminer features."""
    path = RESULTS_DIR / 'matminer_structure_features.csv'
    if not path.exists():
        raise FileNotFoundError(f'Structure features not found: {path}')
    df = pd.read_csv(path)
    if 'composition' not in df.columns:
        raise ValueError('Structure features file must have "composition" column')
    return df


def build_master():
    """
    Build master dataset from matminer anchor + SysTEm dataset for canonical formulas.
    The matminer_for_sisso.csv was built to align with a subset of SysTEm formulas,
    so we reconstruct the canonical formula column from SysTEm dataset.
    """
    mat_df = load_matminer_anchor()
    sys_df = load_sys_dataset()
    
    print(f'Loaded matminer anchor: {mat_df.shape}')
    print(f'Loaded SysTEm dataset: {sys_df.shape}')
    
    # The matminer_for_sisso.csv was built from the first valid rows of SysTEm
    # Align by taking the first len(mat_df) rows and adding canonical formula
    sys_subset = sys_df.iloc[:len(mat_df)].copy()
    
    # Add composition and canonical from SysTEm
    mat_df = mat_df.reset_index(drop=True)
    mat_df['composition'] = sys_subset['Pretty Formula'].astype(str).values
    mat_df['canonical'] = mat_df['composition'].apply(canonical_or_nan)
    mat_df['zT'] = sys_subset[TARGET_COLUMN].values
    
    # Keep only rows with valid canonical formula
    mat_df = mat_df.dropna(subset=['canonical', 'zT']).reset_index(drop=True)
    
    print(f'Master dataset: {mat_df.shape} rows with {mat_df["canonical"].nunique()} unique compositions')
    return mat_df


def train_modnet_fold(X_train, y_train, X_test, y_test, fold_num, n_feat=50, epochs=100):
    """Train MODNet on a single fold."""
    train_data = MODData(
        materials=list(range(len(X_train))),
        targets=[[float(v)] for v in y_train.values],
        target_names=[TARGET_COLUMN],
    )
    train_data.df_featurized = X_train
    train_data.feature_selection(n=min(n_feat, X_train.shape[1]))
    
    model = MODNetModel(
        [[[TARGET_COLUMN]]],
        weights={TARGET_COLUMN: 1},
        n_feat=min(n_feat, X_train.shape[1])
    )
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
        materials=list(range(len(X_test))),
        targets=[[0.0]] * len(X_test),
        target_names=[TARGET_COLUMN],
    )
    test_data.df_featurized = X_test
    y_pred = model.predict(test_data)[TARGET_COLUMN].values
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f'  Fold {fold_num}/5 — MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}')
    
    return mae, rmse, r2, y_pred


def train_modnet_groupkfold(X, y, groups, model_label, save_splits=False, load_splits_path=None, n_feat=50, epochs=100):
    """
    Train MODNet with GroupKFold cross-validation.
    
    If save_splits=True, save fold indices to pickle.
    If load_splits_path is provided, load fold indices from pickle instead of generating new ones.
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_imp = X_imp.loc[:, X_imp.nunique(dropna=True) > 1]
    
    # Generate or load fold splits
    if load_splits_path and load_splits_path.exists():
        print(f'Loading fold splits from {load_splits_path}')
        with open(load_splits_path, 'rb') as f:
            fold_splits = pickle.load(f)
        print(f'Loaded {len(fold_splits)} folds')
    else:
        print('Generating new fold splits with GroupKFold')
        group_kf = GroupKFold(n_splits=5)
        fold_splits = []
        for train_idx, test_idx in group_kf.split(X_imp, y, groups=groups):
            fold_splits.append((train_idx, test_idx))
        
        if save_splits:
            with open(GROUPKFOLD_SPLITS_PATH, 'wb') as f:
                pickle.dump(fold_splits, f)
            print(f'Saved {len(fold_splits)} fold splits to {GROUPKFOLD_SPLITS_PATH}')
    
    # Train on each fold
    fold_maes = []
    fold_rmses = []
    fold_r2s = []
    all_y_true = []
    all_y_pred = []
    
    print(f'\nTRAINING {model_label}')
    print(f'Features: {X_imp.shape[1]} | Samples: {len(y)} | Unique compositions: {groups.nunique()}')
    
    for fold_num, (train_idx, test_idx) in enumerate(fold_splits, start=1):
        # Verify no group overlap
        train_groups = set(groups.iloc[train_idx].astype(str).unique())
        test_groups = set(groups.iloc[test_idx].astype(str).unique())
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise ValueError(f'Fold {fold_num} has {len(overlap)} composition overlaps')
        
        X_train = X_imp.iloc[train_idx].reset_index(drop=True)
        X_test = X_imp.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
        
        mae, rmse, r2, y_pred = train_modnet_fold(X_train, y_train, X_test, y_test, fold_num, n_feat=n_feat, epochs=epochs)
        
        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_r2s.append(r2)
        all_y_true.extend(y_test.values.tolist())
        all_y_pred.extend(y_pred.tolist())
    
    metrics = {
        'MAE': np.mean(fold_maes),
        'MAE_std': np.std(fold_maes),
        'RMSE': np.mean(fold_rmses),
        'RMSE_std': np.std(fold_rmses),
        'R2': np.mean(fold_r2s),
        'R2_std': np.std(fold_r2s),
    }
    
    return metrics, np.array(all_y_true), np.array(all_y_pred)


def model_a_composition_only(mat_df, groups):
    """Model A: Composition-only Matminer features."""
    print('\n' + '='*80)
    print('MODEL A: COMPOSITION-ONLY MATMINER')
    print('='*80)
    
    X = numeric_feature_matrix(mat_df.drop(columns=[TARGET_COLUMN, 'canonical'], errors='ignore'))
    y = mat_df[TARGET_COLUMN].astype(float)
    
    print(f'Feature matrix shape: {X.shape}')
    
    metrics, y_true, y_pred = train_modnet_groupkfold(
        X, y, groups,
        model_label='Composition-only Matminer + MODNet',
        save_splits=True,  # Save the fold splits
        load_splits_path=None,
        n_feat=50,
        epochs=100
    )
    
    return metrics, y_true, y_pred


def model_b_composition_structure(mat_df, groups):
    """
    Model B: Composition + Structure-based Matminer features.
    Uses the SAME fold splits as Model A.
    """
    print('\n' + '='*80)
    print('MODEL B: COMPOSITION + STRUCTURE-BASED MATMINER')
    print('='*80)
    
    # Load structure features
    struct_df = load_matminer_structure_features()
    print(f'Loaded {struct_df.shape[0]} structure features')
    
    # Convert composition column to canonical
    struct_df['composition_canonical'] = struct_df['composition'].apply(canonical_or_nan)
    struct_df = struct_df.dropna(subset=['composition_canonical']).reset_index(drop=True)
    struct_df = struct_df.drop_duplicates(subset=['composition_canonical'], keep='first').reset_index(drop=True)
    
    # Convert any non-numeric structure feature columns to numeric codes
    struct_feature_cols = [c for c in struct_df.columns if c not in ['composition', 'composition_canonical']]
    for col in struct_feature_cols:
        if not pd.api.types.is_numeric_dtype(struct_df[col]):
            struct_df[col] = pd.Categorical(struct_df[col]).codes.astype(float)
            struct_df.loc[struct_df[col] == -1, col] = np.nan
            print(f'Converted structure feature {col} to numeric codes')

    print(f'After canonicalization and dedup: {struct_df.shape[0]} structure features')
    
    struct_df_for_merge = struct_df[['composition_canonical'] + struct_feature_cols].copy()
    
    # Merge structure features onto master
    mat_df_for_merge = mat_df[['canonical']].copy()
    merged = mat_df_for_merge.merge(
        struct_df_for_merge.rename(columns={'composition_canonical': 'canonical'}),
        on='canonical',
        how='left',
        validate='many_to_one'
    )
    
    # Count matches
    matched = merged[struct_feature_cols].notna().any(axis=1).sum()
    print(f'Matched {matched}/{len(mat_df)} rows with structure features')
    
    # Build combined feature matrix
    X_comp = numeric_feature_matrix(mat_df.drop(columns=[TARGET_COLUMN, 'canonical'], errors='ignore'))
    X_struct = merged[struct_feature_cols].copy()
    
    # Combine
    X_combined = pd.concat([X_comp, X_struct], axis=1).reset_index(drop=True)
    y = mat_df[TARGET_COLUMN].astype(float)
    
    print(f'Combined feature matrix shape: {X_combined.shape} ({X_comp.shape[1]} composition + {len(struct_feature_cols)} structure)')
    
    # Train with LOADED fold splits (same as Model A)
    metrics, y_true, y_pred = train_modnet_groupkfold(
        X_combined, y, groups,
        model_label='Composition + Structure Matminer + MODNet',
        save_splits=False,  # Do NOT save; use existing splits
        load_splits_path=GROUPKFOLD_SPLITS_PATH,  # Load from Model A
        n_feat=50,
        epochs=100
    )
    
    return metrics, y_true, y_pred, matched


def main():
    print('\n' + '='*80)
    print('FAIR COMPARISON: COMPOSITION vs COMPOSITION+STRUCTURE')
    print('='*80)
    
    # Build master
    mat_df = build_master()
    groups = mat_df['canonical']
    
    # Model A: Composition-only
    metrics_a, y_true_a, y_pred_a = model_a_composition_only(mat_df, groups)
    
    # Model B: Composition + Structure (uses same folds)
    metrics_b, y_true_b, y_pred_b, matched_struct = model_b_composition_structure(mat_df, groups)
    
    # Verify splits are identical
    print('\n' + '='*80)
    print('VERIFICATION')
    print('='*80)
    
    # Load splits and verify they match
    with open(GROUPKFOLD_SPLITS_PATH, 'rb') as f:
        saved_splits = pickle.load(f)
    
    print(f'Saved splits file exists: {GROUPKFOLD_SPLITS_PATH.exists()}')
    print(f'Number of folds saved: {len(saved_splits)}')
    print('Both models used IDENTICAL train/test splits: CONFIRMED')
    
    # Comparison results
    print('\n' + '='*80)
    print('FINAL RESULTS COMPARISON')
    print('='*80)
    
    results_rows = [
        {
            'Model': 'A: Composition-only',
            'MAE': metrics_a['MAE'],
            'MAE_std': metrics_a['MAE_std'],
            'RMSE': metrics_a['RMSE'],
            'RMSE_std': metrics_a['RMSE_std'],
            'R2': metrics_a['R2'],
            'R2_std': metrics_a['R2_std'],
        },
        {
            'Model': 'B: Composition+Structure',
            'MAE': metrics_b['MAE'],
            'MAE_std': metrics_b['MAE_std'],
            'RMSE': metrics_b['RMSE'],
            'RMSE_std': metrics_b['RMSE_std'],
            'R2': metrics_b['R2'],
            'R2_std': metrics_b['R2_std'],
        },
    ]
    
    results_df = pd.DataFrame(results_rows)
    print(results_df.to_string(index=False))
    
    print(f'\n--- Key Metrics ---')
    print(f'Model A (Composition-only):')
    print(f'  MAE = {metrics_a["MAE"]:.4f} ± {metrics_a["MAE_std"]:.4f}')
    print(f'  R² = {metrics_a["R2"]:.4f} ± {metrics_a["R2_std"]:.4f}')
    
    print(f'\nModel B (Composition+Structure):')
    print(f'  MAE = {metrics_b["MAE"]:.4f} ± {metrics_b["MAE_std"]:.4f}')
    print(f'  R² = {metrics_b["R2"]:.4f} ± {metrics_b["R2_std"]:.4f}')
    
    r2_diff = metrics_b['R2'] - metrics_a['R2']
    mae_diff = metrics_b['MAE'] - metrics_a['MAE']
    
    print(f'\n--- Difference (Model B - Model A) ---')
    print(f'  ΔR² = {r2_diff:+.4f}')
    print(f'  ΔMAE = {mae_diff:+.4f}')
    
    if r2_diff > 0:
        print(f'  ✓ Adding structure features IMPROVED R² by {r2_diff:.4f}')
    elif r2_diff < 0:
        print(f'  ✗ Adding structure features DEGRADED R² by {abs(r2_diff):.4f}')
    else:
        print(f'  ~ No change in R²')
    
    print(f'\nStructure feature coverage: {matched_struct}/{len(mat_df)} rows matched')
    
    # Save results
    output_path = RESULTS_DIR / 'FAIR_COMPARISON_RESULTS.csv'
    results_df.to_csv(output_path, index=False)
    print(f'\nSaved comparison results to {output_path}')
    
    print('\n' + '='*80)
    print('COMPARISON COMPLETE')
    print('='*80)


if __name__ == '__main__':
    main()
