"""Extract CrabNet latent features and combine with Matminer features for MODNet training."""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pymatgen.core import Composition

try:
    from crabnet.crabnet_ import CrabNet
    from crabnet.utils.composition import parse_formula
except Exception as exc:
    raise ImportError(f"Unable to import CrabNet or parser from crabnet: {exc}")

try:
    from modnet.models import MODNetModel
    from modnet.preprocessing import MODData
    MODNET_AVAILABLE = True
except Exception:
    MODNET_AVAILABLE = False

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SPLITS = 5
BATCH_SIZE = 128
DPI = 300
np.random.seed(SEED)

MATMINER_PATH = RESULTS_DIR / 'matminer_for_sisso.csv'
DATASET_PATH = PROJECT_DIR / 'sysTEm_dataset' / 'sysTEm_dataset.xlsx'
OUTPUT_CSV = RESULTS_DIR / 'CRABNET_LATENT_MODNET_RESULTS.csv'
OUTPUT_PLOT = FIGURES_DIR / 'parity_crabnet_latent_modnet.png'

BASELINE_MAE = 0.1347
BASELINE_R2 = 0.7002
CRABNET_ONLY_MAE = 0.1234
CRABNET_ONLY_R2 = 0.7509


def canonical_formula(formula):
    try:
        comp = Composition(str(formula))
        return comp.reduced_formula
    except Exception:
        return None


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


def load_data():
    if not MATMINER_PATH.exists():
        raise FileNotFoundError(f"Missing matminer file: {MATMINER_PATH}")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset file: {DATASET_PATH}")

    mat = pd.read_csv(MATMINER_PATH)
    if 'target' not in mat.columns:
        raise ValueError('matminer_for_sisso.csv must contain a target column')

    print(f"Loaded Matminer: {mat.shape}", flush=True)
    X_mat = mat.drop(columns=['target'], errors='ignore')
    y_mat = mat['target'].astype(float).reset_index(drop=True)

    df = pd.read_excel(DATASET_PATH)
    df = df[['Pretty Formula', 'zT', 'Temperature (K)']].copy()
    df = df.dropna(subset=['Pretty Formula', 'zT', 'Temperature (K)']).reset_index(drop=True)

    valid_formula = ~df['Pretty Formula'].astype(str).str.contains('wt%|vol%|wt |vol ', case=False, na=False)
    valid_zt = df['zT'] > 0
    df = df[valid_formula & valid_zt].reset_index(drop=True)

    if len(df) < len(mat):
        raise ValueError(f"Filtered dataset has {len(df)} rows but expected at least {len(mat)}")

    df = df.iloc[:len(mat)].reset_index(drop=True)
    df['formula_T'] = df.apply(lambda row: temp_encoding(row['Pretty Formula'], row['Temperature (K)']), axis=1)
    df['canonical'] = df['Pretty Formula'].apply(canonical_formula)
    missing_canonical = df['canonical'].isna().sum()
    if missing_canonical > 0:
        print(f"  Warning: {missing_canonical} formulas could not be canonicalized. Using raw formula as group fallback.", flush=True)
        df['canonical'] = df.apply(
            lambda row: row['canonical'] if pd.notna(row['canonical']) else str(row['Pretty Formula']),
            axis=1,
        )

    if len(df) != len(mat):
        raise ValueError(f"Aligned dataset length {len(df)} does not match matminer length {len(mat)}")

    print(f"Aligned dataset: {len(df)} rows, unique canonical/group labels: {df['canonical'].nunique()}", flush=True)
    return X_mat, y_mat, df


def extract_latent_features(crab_model, df_subset):
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

    for start in range(0, len(crab_df), BATCH_SIZE):
        batch = crab_df.iloc[start:start + BATCH_SIZE]
        try:
            crab_model.load_data(batch, train=False)
            crab_model.predict(batch)
            if 'feat' in latent_store:
                all_feats.append(latent_store['feat'])
                latent_store.clear()
            else:
                raise RuntimeError('Latent hook did not capture features')
        except Exception as exc:
            print(f"  Batch {start // BATCH_SIZE} error: {exc}", flush=True)
            feat_dim = all_feats[-1].shape[1] if all_feats else 512
            all_feats.append(np.zeros((len(batch), feat_dim), dtype=float))

    hook.remove()
    if not all_feats:
        return np.zeros((len(df_subset), 512), dtype=float)
    return np.vstack(all_feats)


def is_crabnet_valid(formula):
    if pd.isna(formula):
        return False
    formula_str = str(formula)
    if re.search(r'wt%|vol%|%|\bwt\b|\bvol\b', formula_str, re.IGNORECASE):
        return False
    if '+' in formula_str:
        return False
    try:
        base_form = formula_str.split(' Og')[0]
        Composition(base_form)
        return True
    except Exception:
        return False


def train_crabnet_model(df_train):
    if len(df_train) == 0:
        return None
    train_df, val_df = train_test_split(
        df_train[['formula_T', 'zT']].rename(columns={'formula_T': 'formula', 'zT': 'target'}),
        test_size=0.10,
        random_state=SEED,
    )
    model = CrabNet(compute_device='cpu', verbose=False, epochs=50, batch_size=128, lr=0.001, checkin=20, save=False)
    try:
        model.fit(train_df, val_df)
        return model
    except Exception as exc:
        print(f"  CrabNet fit error: {exc}", flush=True)
        try:
            import torch
            from torch.optim import SGD
            model.optimizer = SGD(model.model.parameters(), lr=0.001)
            model.fit(train_df, val_df)
            return model
        except Exception as exc2:
            print(f"  CrabNet fit retry failed: {exc2}", flush=True)
            return None


def train_modnet_or_rf(X_train, y_train, X_test):
    if X_train.shape[1] == 0:
        print('  No features available for MODNet/RF training; returning zeros', flush=True)
        return np.zeros(len(X_test), dtype=float)

    try:
        if not MODNET_AVAILABLE:
            raise RuntimeError('MODNet unavailable')

        # Fast feature selection via XGBoost importance before MODNet
        xgb_selector = XGBRegressor(n_estimators=100, random_state=SEED, n_jobs=-1, verbosity=0)
        xgb_selector.fit(X_train, y_train)
        importances = xgb_selector.feature_importances_
        top_idx = np.argsort(importances)[-50:]
        X_train = X_train[:, top_idx]
        X_test = X_test[:, top_idx]
        print(f"    Selected top {X_train.shape[1]} features via XGBoost", flush=True)

        train_data = MODData(
            materials=list(range(len(X_train))),
            targets=[[float(v)] for v in y_train.tolist()],
            target_names=['ZT']
        )
        train_data.df_featurized = X_train
        n_feat = min(50, X_train.shape[1])
        train_data.feature_selection(n=n_feat)

        model = MODNetModel([[['ZT']]], weights={'ZT': 1}, n_feat=n_feat)
        model.fit(train_data, val_fraction=0.1, lr=0.001,
                  batch_size=64, loss='mae', epochs=50, verbose=False)

        test_data = MODData(
            materials=list(range(len(X_test))),
            targets=[[0.0]] * len(X_test),
            target_names=['ZT']
        )
        test_data.df_featurized = X_test
        preds = model.predict(test_data)['ZT'].values
        return preds
    except Exception as exc:
        print(f"  MODNet failed: {exc}", flush=True)
        print('  Falling back to RandomForestRegressor', flush=True)
        rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
        rf.fit(X_train, y_train)
        return rf.predict(X_test)


def build_feature_dataframe(X_mat, latent, prefix='CrabLatent'):
    latent_cols = [f'{prefix}_{i}' for i in range(latent.shape[1])]
    return pd.DataFrame(np.hstack([X_mat.values, latent]), columns=list(X_mat.columns) + latent_cols)


def main():
    print('\n' + '='*70)
    print('CRABNET LATENT + MATMINER + MODNET PIPELINE')
    print('='*70)

    X_mat, y_mat, df_aligned = load_data()

    if len(df_aligned) != len(X_mat):
        raise ValueError('Aligned record counts do not match')

    groups = df_aligned['canonical'].astype(str).values
    group_kf = GroupKFold(n_splits=N_SPLITS)

    fold_results = []
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(group_kf.split(X_mat, y_mat, groups), start=1):
        print(f"\n--- Fold {fold}/{N_SPLITS} ---", flush=True)
        print(f"  Fold {fold}: Testing CrabNet import...", flush=True)
        try:
            from crabnet.crabnet_ import CrabNet
            print(f"  CrabNet import OK", flush=True)
        except Exception as e:
            print(f"  CrabNet import FAILED: {e}", flush=True)

        df_train = df_aligned.iloc[train_idx].copy().reset_index(drop=True)
        df_test = df_aligned.iloc[test_idx].copy().reset_index(drop=True)

        valid_mask = df_train['Pretty Formula'].apply(is_crabnet_valid)
        fold_data_clean = df_train[valid_mask].reset_index(drop=True)
        print(f"    CrabNet valid formulas (train): {len(fold_data_clean)}/{len(df_train)}", flush=True)

        valid_test_mask = df_test['Pretty Formula'].apply(is_crabnet_valid)
        test_data_clean = df_test[valid_test_mask].reset_index(drop=True)
        print(f"    CrabNet valid formulas (test): {len(test_data_clean)}/{len(df_test)}", flush=True)

        train_df_fold = fold_data_clean[['formula_T', 'zT']].copy()
        train_df_fold.columns = ['formula', 'target']

        test_df_fold = test_data_clean[['formula_T', 'zT']].copy()
        test_df_fold.columns = ['formula', 'target']

        crab_model = train_crabnet_model(fold_data_clean)

        if crab_model is None:
            print('  CrabNet training failed; using zero latent features', flush=True)
            train_latent = np.zeros((len(df_train), 512), dtype=float)
            test_latent = np.zeros((len(df_test), 512), dtype=float)
        else:
            train_latent_valid = extract_latent_features(crab_model, fold_data_clean)
            mean_train_latent = train_latent_valid.mean(axis=0, keepdims=True) if len(train_latent_valid) > 0 else np.zeros((1, 512), dtype=float)
            train_latent = np.repeat(mean_train_latent, len(df_train), axis=0)
            train_latent[valid_mask.values] = train_latent_valid

            if len(test_data_clean) > 0:
                test_latent_valid = extract_latent_features(crab_model, test_data_clean)
                mean_test_latent = test_latent_valid.mean(axis=0, keepdims=True)
            else:
                test_latent_valid = np.zeros((0, 512), dtype=float)
                mean_test_latent = np.zeros((1, 512), dtype=float)

            test_latent = np.repeat(mean_test_latent, len(df_test), axis=0)
            if len(test_data_clean) > 0:
                test_latent[valid_test_mask.values] = test_latent_valid

        print(f"  train latent shape: {train_latent.shape}", flush=True)
        print(f"  test latent shape:  {test_latent.shape}", flush=True)

        X_train = build_feature_dataframe(X_mat.iloc[train_idx].reset_index(drop=True), train_latent)
        X_test = build_feature_dataframe(X_mat.iloc[test_idx].reset_index(drop=True), test_latent)

        imp = SimpleImputer(strategy='mean')
        X_train_imp = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns)
        X_test_imp = pd.DataFrame(imp.transform(X_test), columns=X_test.columns)

        const_cols = X_train_imp.columns[X_train_imp.std(axis=0) == 0].tolist()
        if const_cols:
            print(f"  Removing {len(const_cols)} constant columns", flush=True)
            X_train_imp = X_train_imp.drop(columns=const_cols)
            X_test_imp = X_test_imp.drop(columns=const_cols, errors='ignore')

        X_test_imp = X_test_imp.reindex(columns=X_train_imp.columns, fill_value=0.0)
        print(f"  Combined feature count: {X_train_imp.shape[1]}", flush=True)

        y_train = y_mat.iloc[train_idx].reset_index(drop=True)
        y_test = y_mat.iloc[test_idx].reset_index(drop=True)

        preds = train_modnet_or_rf(X_train_imp, y_train, X_test_imp)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        print(f"  Fold {fold}/{N_SPLITS}: MAE={mae:.4f} R²={r2:.4f}", flush=True)

        fold_results.append({
            'fold': fold,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
        })
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(preds.tolist())

    mean_mae = np.mean([r['mae'] for r in fold_results])
    mean_rmse = np.mean([r['rmse'] for r in fold_results])
    mean_r2 = np.mean([r['r2'] for r in fold_results])

    print('\n' + '-'*70, flush=True)
    print('RESULTS SUMMARY', flush=True)
    print('-'*70, flush=True)
    print(f"Matminer + MODNet baseline:              MAE={BASELINE_MAE:.4f}  R²={BASELINE_R2:.4f}", flush=True)
    print(f"CrabNet + continuous temp encoding:      MAE={CRABNET_ONLY_MAE:.4f}  R²={CRABNET_ONLY_R2:.4f}", flush=True)
    print(f"Matminer + CrabNet latent + MODNet:      MAE={mean_mae:.4f}  R²={mean_r2:.4f}", flush=True)

    results_df = pd.DataFrame(fold_results)
    overall_row = pd.DataFrame([{ 'fold': 'overall', 'mae': mean_mae, 'rmse': mean_rmse, 'r2': mean_r2 }])
    results_df = pd.concat([results_df, overall_row], ignore_index=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results: {OUTPUT_CSV}", flush=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(all_y_true, all_y_pred, alpha=0.4, s=20, color='blue', edgecolors='none')
    lim = [min(min(all_y_true), min(all_y_pred)) - 0.05,
           max(max(all_y_true), max(all_y_pred)) + 0.05]
    ax.plot(lim, lim, 'r--', lw=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('Actual ZT', fontweight='bold')
    ax.set_ylabel('Predicted ZT', fontweight='bold')
    ax.set_title('Matminer + CrabNet Latent + MODNet Parity Plot', fontweight='bold')
    text = (f"MAE = {mean_mae:.4f}\n" f"RMSE = {mean_rmse:.4f}\n" f"R² = {mean_r2:.4f}")
    ax.text(0.05, 0.95, text, transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), family='monospace', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=DPI)
    plt.close()
    print(f"Saved parity plot: {OUTPUT_PLOT}", flush=True)

    print('\nPipeline complete.', flush=True)


if __name__ == '__main__':
    main()
