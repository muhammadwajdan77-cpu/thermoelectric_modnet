#!/usr/bin/env python3
"""Hybrid training using Matminer features merged with CrabNet latent features (v3).

Loads `results/matminer_for_sisso_v2.csv` and `results/CRABNET_LATENT_FEATURES_V2.csv`,
merges on `canonical_formula`, and trains MODNet (or RF fallback) with GroupKFold.
Saves fold-level results to `results/HYBRID_V3_RESULTS.csv`.
"""

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd

RESULTS_DIR = Path('results')
MATMINER_V2 = RESULTS_DIR / 'matminer_for_sisso_v2.csv'
LATENT_V2 = RESULTS_DIR / 'CRABNET_LATENT_FEATURES_V2.csv'
OUTPUT = RESULTS_DIR / 'HYBRID_V3_RESULTS.csv'

SEED = 42
N_SPLITS = 5

def load_files():
    if not MATMINER_V2.exists():
        raise FileNotFoundError(f"Missing {MATMINER_V2}")
    if not LATENT_V2.exists():
        raise FileNotFoundError(f"Missing {LATENT_V2}")
    mat = pd.read_csv(MATMINER_V2)
    latent = pd.read_csv(LATENT_V2)
    return mat, latent


def train_modnet_or_rf(X_train, y_train, X_test):
    if X_train.shape[1] == 0:
        return np.zeros(len(X_test), dtype=float)
    try:
        from modnet.models import MODNetModel
        from modnet.preprocessing import MODData
        MODNET_AVAILABLE = True
    except Exception:
        MODNET_AVAILABLE = False

    if MODNET_AVAILABLE:
        from xgboost import XGBRegressor
        xgb_selector = XGBRegressor(n_estimators=100, random_state=SEED, n_jobs=-1, verbosity=0)
        xgb_selector.fit(X_train, y_train)
        importances = xgb_selector.feature_importances_
        top_idx = np.argsort(importances)[-50:]
        X_train_sel = X_train[:, top_idx]
        X_test_sel = X_test[:, top_idx]

        train_data = MODData(materials=list(range(len(X_train_sel))), targets=[[float(v)] for v in y_train.tolist()], target_names=['ZT'])
        import pandas as _pd
        train_data.df_featurized = _pd.DataFrame(X_train_sel)
        n_feat = min(50, X_train_sel.shape[1])
        train_data.feature_selection(n=n_feat)
        model = MODNetModel([[['ZT']]], weights={'ZT':1}, n_feat=n_feat)
        model.fit(train_data, val_fraction=0.1, lr=0.001, batch_size=64, loss='mae', epochs=100, verbose=False)
        test_data = MODData(materials=list(range(len(X_test_sel))), targets=[[0.0]]*len(X_test_sel), target_names=['ZT'])
        test_data.df_featurized = _pd.DataFrame(X_test_sel)
        preds = model.predict(test_data)['ZT'].values
        return preds
    # fallback
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf.predict(X_test)


def main():
    mat, latent = load_files()
    print(f"Matminer rows: {len(mat)}, Latent rows: {len(latent)}", flush=True)

    # Prefer exact mat_index-based merge if latent provides it
    if 'mat_index' in latent.columns:
        mat = mat.copy().reset_index(drop=False).rename(columns={'index':'mat_index'})
        merged = pd.merge(mat, latent, how='inner', on='mat_index', validate='one_to_one')
        print('Merged on mat_index for exact alignment', flush=True)
    else:
        if 'canonical_formula' not in mat.columns or 'canonical_formula' not in latent.columns:
            raise ValueError('Both matminer v2 and latent v2 must contain canonical_formula for fallback merge')
        merged = pd.merge(mat, latent, how='inner', on='canonical_formula')
        print('Merged on canonical_formula (fallback)', flush=True)

    print(f"After merge rows: {len(merged)}", flush=True)
    # normalize canonical_formula column if pandas added suffixes
    if 'canonical_formula' not in merged.columns:
        if 'canonical_formula_x' in merged.columns:
            merged['canonical_formula'] = merged['canonical_formula_x']
        elif 'canonical_formula_y' in merged.columns:
            merged['canonical_formula'] = merged['canonical_formula_y']
    if len(merged) != len(mat):
        raise RuntimeError(f"Row count changed after merge: mat {len(mat)} -> merged {len(merged)}")

    # prepare features
    exclude = ['formula','formula_T','canonical_formula','canonical','Temperature_K','target','zT']
    feature_cols = [c for c in merged.columns if c not in exclude and pd.api.types.is_numeric_dtype(merged[c])]
    if not feature_cols:
        raise RuntimeError('No numeric features found after merge')
    X = merged[feature_cols].astype(float)
    y = merged['zT'].astype(float)
    groups = merged['canonical_formula'].astype(str).values

    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.impute import SimpleImputer

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups & test_groups
        assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} compositions overlap in fold {fold}"
        print(f"Fold {fold}: verified zero overlap ✓", flush=True)

        X_train = X.iloc[train_idx].values
        X_test = X.iloc[test_idx].values
        y_train = y.iloc[train_idx].values
        y_test = y.iloc[test_idx].values

        imp = SimpleImputer(strategy='mean')
        X_train_imp = imp.fit_transform(X_train)
        X_test_imp = imp.transform(X_test)

        preds = train_modnet_or_rf(X_train_imp, pd.Series(y_train), X_test_imp)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        print(f"Fold {fold}: MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f}", flush=True)
        fold_results.append({'fold': fold, 'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)})

    mean_mae = np.mean([r['mae'] for r in fold_results])
    mean_rmse = np.mean([r['rmse'] for r in fold_results])
    mean_r2 = np.mean([r['r2'] for r in fold_results])
    fold_results.append({'fold':'overall','mae':float(mean_mae),'rmse':float(mean_rmse),'r2':float(mean_r2)})
    pd.DataFrame(fold_results).to_csv(OUTPUT, index=False)
    print(f"Saved results to {OUTPUT}", flush=True)


if __name__ == '__main__':
    main()
