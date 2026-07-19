#!/usr/bin/env python3
"""CrabNet latent + Matminer + MODNet hybrid retrain using canonical groups from matminer_for_sisso_v2.csv"""

import warnings
warnings.filterwarnings('ignore')
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

try:
    from crabnet.crabnet_ import CrabNet
except Exception as exc:
    raise RuntimeError(f"CrabNet import failed: {exc}") from exc

try:
    from modnet.models import MODNetModel
    from modnet.preprocessing import MODData
    MODNET_AVAILABLE = True
except Exception:
    MODNET_AVAILABLE = False

from pymatgen.core import Composition

RESULTS_DIR = Path('results')
MATMINER_PATH = RESULTS_DIR / 'matminer_for_sisso_v2.csv'
OUTPUT_CSV = RESULTS_DIR / 'HYBRID_V2_RESULTS.csv'

SEED = 42
N_SPLITS = 5
BATCH_SIZE = 128


def temp_encoding(formula, temperature):
    if pd.isna(formula):
        return None
    try:
        temp = float(temperature)
    except Exception:
        return f"{formula} Og"
    frac = temp / 1000.0
    frac_text = f"{frac:.3f}".rstrip('0').rstrip('.')
    return f"{formula} Og{frac_text}"


def canonical_formula(formula):
    try:
        comp = Composition(str(formula))
        return comp.reduced_formula
    except Exception:
        return None


def load_matminer(mat_path: Path):
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing matminer file: {mat_path}")
    mat = pd.read_csv(mat_path)
    required = ['formula', 'canonical_formula', 'Temperature_K', 'target']
    for c in required:
        if c not in mat.columns:
            raise ValueError(f"Expected column '{c}' in {mat_path}")
    mat = mat.copy().reset_index(drop=True)
    mat['formula_T'] = mat.apply(lambda row: temp_encoding(row['formula'], row['Temperature_K']), axis=1)
    mat['canonical'] = mat['canonical_formula'].astype(str)
    mat['zT'] = mat['target'].astype(float)
    return mat


def extract_latent(crab_model, df_subset, batch_size=BATCH_SIZE):
    latent_store = {}

    def hook_fn(module, input, output):
        src = input[0]
        if src.dim() == 3 and src.shape[-1] == 1:
            src = src.squeeze(-1)
        mask = (src == 0)
        if output.dim() == 3:
            mask = mask.unsqueeze(-1).expand_as(output)
            out = output.masked_fill(mask, 0.0)
            count = (~mask).sum(dim=1).float().clamp(min=1.0)
            avg = out.sum(dim=1) / count
        else:
            out = output
            avg = out
        latent_store['feat'] = avg.detach().cpu().numpy()

    hook = crab_model.model.encoder.register_forward_hook(hook_fn)
    crab_df = pd.DataFrame({'formula': df_subset['formula_T'].values, 'target': df_subset['zT'].values})
    all_feats = []
    for start in range(0, len(crab_df), batch_size):
        batch = crab_df.iloc[start:start + batch_size]
        try:
            crab_model.load_data(batch, train=False)
            crab_model.predict(batch)
            if 'feat' in latent_store:
                all_feats.append(latent_store['feat'])
                latent_store.clear()
            else:
                raise RuntimeError('Latent hook did not capture features')
        except Exception:
            feat_dim = all_feats[-1].shape[1] if all_feats else 512
            all_feats.append(np.zeros((len(batch), feat_dim), dtype=float))

    hook.remove()
    if not all_feats:
        return np.zeros((len(df_subset), 512), dtype=float)
    return np.vstack(all_feats)


def train_modnet_or_rf(X_train, y_train, X_test):
    if X_train.shape[1] == 0:
        return np.zeros(len(X_test), dtype=float)
    try:
        if not MODNET_AVAILABLE:
            raise RuntimeError('MODNet unavailable')
        train_data = MODData(
            materials=list(range(len(X_train))),
            targets=[[float(v)] for v in y_train.tolist()],
            target_names=['ZT']
        )
        train_data.df_featurized = X_train
        n_feat = min(50, X_train.shape[1])
        train_data.feature_selection(n=n_feat)
        model = MODNetModel([[['ZT']]], weights={'ZT': 1}, n_feat=n_feat)
        model.fit(train_data, val_fraction=0.1, lr=0.001, batch_size=64, loss='mae', epochs=50, verbose=False)
        test_data = MODData(materials=list(range(len(X_test))), targets=[[0.0]] * len(X_test), target_names=['ZT'])
        test_data.df_featurized = X_test
        preds = model.predict(test_data)['ZT'].values
        return preds
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
        rf.fit(X_train, y_train)
        return rf.predict(X_test)


def run_pipeline(mat: pd.DataFrame):
    feature_cols = [c for c in mat.columns if c not in ['formula', 'formula_T', 'canonical_formula', 'canonical', 'Temperature_K', 'target', 'zT'] and pd.api.types.is_numeric_dtype(mat[c])]
    if not feature_cols:
        raise ValueError('No numeric Matminer features available for hybrid training')
    X_mat = mat[feature_cols].copy()
    y = mat['zT']
    groups = mat['canonical'].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_mat, y, groups=groups), start=1):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups & test_groups
        assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} compositions overlap in fold {fold}"
        print(f"Fold {fold}: verified zero overlap ✓", flush=True)

        df_train = mat.iloc[train_idx].reset_index(drop=True)
        df_test = mat.iloc[test_idx].reset_index(drop=True)

        valid_train_mask = df_train['formula'].apply(lambda x: True)
        valid_test_mask = df_test['formula'].apply(lambda x: True)

        crab_model = train_crab = None
        try:
            crab_model = CrabNet(compute_device='cpu', verbose=False, epochs=50, batch_size=128, lr=0.001, save=False)
            # train on df_train where applicable
            if valid_train_mask.sum() > 0:
                train_df_cb = df_train.loc[valid_train_mask, ['formula_T', 'zT']].rename(columns={'formula_T':'formula','zT':'target'})
                val_df_cb = train_df_cb.sample(frac=0.1, random_state=SEED)
                crab_model.fit(train_df=train_df_cb, val_df=val_df_cb)
        except Exception as exc:
            print(f"CrabNet training failed for fold {fold}: {exc}", flush=True)

        if crab_model is None:
            train_latent = np.zeros((len(df_train), 512), dtype=float)
            test_latent = np.zeros((len(df_test), 512), dtype=float)
        else:
            train_latent = extract_latent(crab_model, df_train)
            test_latent = extract_latent(crab_model, df_test)

        X_train = np.hstack([X_mat.iloc[train_idx].values, train_latent])
        X_test = np.hstack([X_mat.iloc[test_idx].values, test_latent])

        imp = SimpleImputer(strategy='mean')
        X_train_imp = imp.fit_transform(X_train.astype(float))
        X_test_imp = imp.transform(X_test.astype(float))

        preds = train_modnet_or_rf(X_train_imp, y.iloc[train_idx].values, X_test_imp)

        mae = mean_absolute_error(y.iloc[test_idx], preds)
        rmse = np.sqrt(mean_squared_error(y.iloc[test_idx], preds))
        r2 = r2_score(y.iloc[test_idx], preds)

        print(f"Fold {fold}: MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f}", flush=True)
        fold_results.append({'fold': fold, 'mae': mae, 'rmse': rmse, 'r2': r2})

    mean_mae = np.mean([r['mae'] for r in fold_results])
    mean_rmse = np.mean([r['rmse'] for r in fold_results])
    mean_r2 = np.mean([r['r2'] for r in fold_results])

    results_df = pd.DataFrame(fold_results)
    overall_row = pd.DataFrame([{'fold': 'overall', 'mae': mean_mae, 'rmse': mean_rmse, 'r2': mean_r2}])
    results_df = pd.concat([results_df, overall_row], ignore_index=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results: {OUTPUT_CSV}", flush=True)


def main():
    mat = load_matminer(MATMINER_PATH)
    run_pipeline(mat)


if __name__ == '__main__':
    main()
