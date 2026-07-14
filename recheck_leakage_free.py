"""Leakage-free re-evaluation for all four MODNet model variants.

Uses GroupKFold grouped by canonical composition to ensure no composition
appears in both train and test within the same fold.

Variants included:
- Matminer + MODNet (baseline)
- Roost + MODNet
- Combined (Matminer + Roost) + MODNet
- Matminer + l-MM + MODNet
"""

import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

try:
    from modnet.models import MODNetModel
    from modnet.preprocessing import MODData
except Exception as exc:
    raise ImportError('MODNet import failed: {}'.format(exc))

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / 'FINAL_RESULTS_LEAKAGE_FREE.csv'

SEED = 42
N_SPLITS = 5
TARGET_NAME = 'ZT'

EXCLUDE_COLUMNS = {
    'canonical',
    'Pretty Formula',
    'pretty formula',
    'Pymatgen Composition',
    'reduced_compositions',
    'Type of Formula',
    'Source Paper',
    'Initial Dataset',
    'Formula',
    'formula',
    'target',
    'zT',
    'composition',
    'structure_id',
    'filename',
    '#',
}


def numeric_feature_matrix(df: pd.DataFrame, drop_temperature: bool = False) -> pd.DataFrame:
    df = df.copy()
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df_num = df[cols].copy()
    drop_cols = [c for c in EXCLUDE_COLUMNS if c in df_num.columns]
    if drop_temperature:
        for temp_col in ['Temperature (K)', 'Temperature_K']:
            if temp_col in df_num.columns:
                drop_cols.append(temp_col)
    df_num = df_num.drop(columns=drop_cols, errors='ignore')
    return df_num


def assert_canonical_alignment(*dfs):
    if len(dfs) < 2:
        return
    reference = dfs[0]["canonical"].astype(str).reset_index(drop=True)
    for idx, df in enumerate(dfs[1:], 1):
        if not reference.equals(df["canonical"].astype(str).reset_index(drop=True)):
            raise ValueError(f'Canonical ordering mismatch between dataset 0 and dataset {idx}')


def build_roost_features(roost_df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
    X_roost = numeric_feature_matrix(roost_df, drop_temperature=True)
    if 'Temperature_K' not in X_roost.columns:
        if 'Temperature_K' in master_df.columns:
            X_roost['Temperature_K'] = master_df['Temperature_K'].values
        elif 'Temperature (K)' in master_df.columns:
            X_roost['Temperature_K'] = master_df['Temperature (K)'].values
        else:
            raise ValueError('Cannot locate temperature column for Roost variant')
    return X_roost


def build_combined_matminer_roost(mat_df: pd.DataFrame, roost_df: pd.DataFrame) -> pd.DataFrame:
    X_mat = numeric_feature_matrix(mat_df, drop_temperature=False)
    X_roost = build_roost_features(roost_df, mat_df)
    X_combined = pd.concat([X_mat.reset_index(drop=True), X_roost.reset_index(drop=True)], axis=1)
    X_combined = X_combined.loc[:, ~X_combined.columns.duplicated(keep='first')]
    return X_combined


def build_matminer_lmm(mat_df: pd.DataFrame, lmm_df: pd.DataFrame) -> pd.DataFrame:
    X_mat = numeric_feature_matrix(mat_df, drop_temperature=False)
    X_lmm = numeric_feature_matrix(lmm_df, drop_temperature=True)
    if 'Temperature_K' in X_mat.columns:
        X_lmm['Temperature_K'] = X_mat['Temperature_K'].values
    elif 'Temperature (K)' in mat_df.columns:
        X_lmm['Temperature_K'] = mat_df['Temperature (K)'].values
    X_combined = pd.concat([X_mat.reset_index(drop=True), X_lmm.reset_index(drop=True)], axis=1)
    X_combined = X_combined.loc[:, ~X_combined.columns.duplicated(keep='first')]
    return X_combined


def train_grouped_modnet(X: pd.DataFrame, y: pd.Series, groups: pd.Series, model_label: str, n_feat: int, epochs: int):
    mask = y > 0
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    groups = groups.loc[mask].reset_index(drop=True)

    imputer = SimpleImputer(strategy='mean')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    group_kf = GroupKFold(n_splits=N_SPLITS)
    fold_maes = []
    fold_rmses = []
    fold_r2s = []

    print(f"\nTRAINING {model_label}")
    print(f"Features: {X_imp.shape[1]} | Samples: {len(y)} | Groups: {groups.nunique()}")

    for fold, (train_idx, test_idx) in enumerate(group_kf.split(X_imp, y, groups=groups), 1):
        print(f"  Fold {fold}/{N_SPLITS}")
        X_train = X_imp.iloc[train_idx].reset_index(drop=True)
        X_test = X_imp.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        train_data = MODData(
            materials=list(range(len(train_idx))),
            targets=[[v] for v in y_train.values],
            target_names=[TARGET_NAME],
        )
        train_data.df_featurized = X_train
        train_data.feature_selection(n=min(n_feat, X_train.shape[1]))

        model = MODNetModel([[[TARGET_NAME]]], weights={TARGET_NAME: 1}, n_feat=min(n_feat, X_train.shape[1]))
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
            targets=[[0]] * len(test_idx),
            target_names=[TARGET_NAME],
        )
        test_data.df_featurized = X_test
        y_pred = model.predict(test_data)[TARGET_NAME].values

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"    MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_r2s.append(r2)

    metrics = {
        'MAE': np.mean(fold_maes),
        'MAE_std': np.std(fold_maes),
        'RMSE': np.mean(fold_rmses),
        'RMSE_std': np.std(fold_rmses),
        'R2': np.mean(fold_r2s),
        'R2_std': np.std(fold_r2s),
    }
    print(f"  -> {model_label} summary: MAE={metrics['MAE']:.4f} ± {metrics['MAE_std']:.4f}, R²={metrics['R2']:.4f} ± {metrics['R2_std']:.4f}")
    return metrics


def save_results(path: Path, metrics: dict):
    df = pd.DataFrame([
        {
            'Model': model,
            'MAE': f"{metric['MAE']:.4f}±{metric['MAE_std']:.4f}",
            'RMSE': f"{metric['RMSE']:.4f}±{metric['RMSE_std']:.4f}",
            'R2': f"{metric['R2']:.4f}±{metric['R2_std']:.4f}",
        }
        for model, metric in metrics.items()
    ])
    df.to_csv(path, index=False)
    print(f"\nSaved leakage-free summary: {path}")


def main():
    mat_df = pd.read_csv(RESULTS_DIR / 'aligned_matminer.csv')
    roost_df = pd.read_csv(RESULTS_DIR / 'aligned_roost.csv')
    lmm_df = pd.read_csv(RESULTS_DIR / 'aligned_lMM.csv')

    assert_canonical_alignment(mat_df, roost_df, lmm_df)
    groups = mat_df['canonical'].astype(str)
    y = mat_df['zT'].astype(float)

    X_matminer = numeric_feature_matrix(mat_df, drop_temperature=False)
    X_roost = build_roost_features(roost_df, mat_df)
    X_combined = build_combined_matminer_roost(mat_df, roost_df)
    X_lmm = build_matminer_lmm(mat_df, lmm_df)

    metrics = {}
    metrics['Matminer + MODNet (Baseline)'] = train_grouped_modnet(
        X_matminer, y, groups, 'Matminer + MODNet (Baseline)', n_feat=50, epochs=100
    )
    metrics['MatterVial (Roost) + MODNet'] = train_grouped_modnet(
        X_roost, y, groups, 'MatterVial (Roost) + MODNet', n_feat=30, epochs=30
    )
    metrics['Combined (Matminer + Roost) + MODNet'] = train_grouped_modnet(
        X_combined, y, groups, 'Combined (Matminer + Roost) + MODNet', n_feat=30, epochs=30
    )
    metrics['Matminer + l-MM + MODNet'] = train_grouped_modnet(
        X_lmm, y, groups, 'Matminer + l-MM + MODNet', n_feat=50, epochs=100
    )

    save_results(OUTPUT_PATH, metrics)

    print('\nLEAKAGE-FREE RECHECK COMPLETE')
    for model, metric in metrics.items():
        print(f"{model}: R² = {metric['R2']:.4f} ± {metric['R2_std']:.4f}")


if __name__ == '__main__':
    main()
