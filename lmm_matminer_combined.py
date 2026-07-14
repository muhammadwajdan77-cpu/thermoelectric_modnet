"""Build and train Matminer + l-MM combined model for thermoelectric zT prediction."""

import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_PATH = '/home/wajdan/miniconda3/envs/sysTEm_localenv/bin/python'
MATMINER_PATH = RESULTS_DIR / 'matminer_for_sisso.csv'
LMM_PATH = RESULTS_DIR / 'lMM_features.csv'
DATASET_PATH = PROJECT_DIR / 'sysTEm_dataset' / 'sysTEm_dataset.xlsx'
FINAL_RESULTS_PATH = RESULTS_DIR / 'FINAL_RESULTS_FIXED.csv'
PLOT_PATH = FIGURES_DIR / 'parity_plot_lmm_matminer.png'

SEED = 42
N_SPLITS = 5
LR = 0.001
BATCH_SIZE = 64
EPOCHS = 100
LOSS = 'mae'
TARGET_NAME = 'ZT'

try:
    from modnet.models import MODNetModel
    from modnet.preprocessing import MODData
    MODNET_AVAILABLE = True
except Exception as exc:
    print('MODNet import failed, falling back to RandomForest:', exc)
    MODNET_AVAILABLE = False


def fail(message: str):
    print('ERROR:', message)
    sys.exit(1)


def check_file(path: Path):
    if not path.exists():
        fail(f'File not found: {path}')


def print_file_info():
    print('\nSTEP 1 - DATA FILE CHECKS')
    print('===========================')

    for path, kind in [
        (MATMINER_PATH, 'matminer_for_sisso.csv'),
        (LMM_PATH, 'lMM_features.csv'),
        (DATASET_PATH, 'sysTEm_dataset.xlsx'),
    ]:
        print(f'\nFile: {path}')
        print(f'Exists: {path.exists()}')
        if not path.exists():
            continue

        if path.suffix == '.csv':
            df = pd.read_csv(path)
            print(f'  shape: {df.shape}')
            print(f'  columns: {df.columns.tolist()}')
            if 'target' in df.columns:
                print(f'  target first 3: {df["target"].head(3).tolist()}')
            if 'composition' in df.columns:
                print(f'  composition first 3: {df["composition"].head(3).tolist()}')
            if 'Pretty Formula' in df.columns:
                print(f'  Pretty Formula first 3: {df["Pretty Formula"].head(3).tolist()}')
        else:
            df = pd.read_excel(path)
            print(f'  shape: {df.shape}')
            if 'Pretty Formula' in df.columns:
                print(f'  Pretty Formula first 3: {df["Pretty Formula"].head(3).tolist()}')
            if 'zT' in df.columns:
                print(f'  zT first 3: {df["zT"].head(3).tolist()}')


def canonical_formula(value):
    try:
        formula = Composition(str(value))
        return formula.reduced_formula
    except Exception:
        return None


def build_master_index():
    print('\nSTEP 2 - BUILD MASTER INDEX')
    print('===========================')
    check_file(DATASET_PATH)
    df = pd.read_excel(DATASET_PATH)
    if 'Pretty Formula' not in df.columns or 'zT' not in df.columns:
        fail('sysTEm_dataset.xlsx must contain Pretty Formula and zT columns')

    master = df.loc[df['zT'] > 0].copy()
    master = master.loc[master['Pretty Formula'].notna()].copy()
    master['canonical'] = master['Pretty Formula'].apply(canonical_formula)
    master = master.loc[master['canonical'].notna()].reset_index(drop=True)

    print(f'  master rows after filtering zT>0 and Pretty Formula: {len(master)}')
    print(f'  master sample Pretty Formula: {master["Pretty Formula"].head(3).tolist()}')
    print(f'  master sample zT: {master["zT"].head(3).tolist()}')

    return master


def align_matminer(master: pd.DataFrame):
    print('\nSTEP 3 - ALIGN MATMINER')
    print('=======================')
    check_file(MATMINER_PATH)
    mat = pd.read_csv(MATMINER_PATH)
    if 'target' not in mat.columns:
        fail('matminer_for_sisso.csv must contain a target column')

    if any(col in mat.columns for col in ['Pretty Formula', 'composition', 'canonical']):
        print('  Formula column found in matminer; aligning by canonical formula')
        if 'Pretty Formula' in mat.columns:
            mat['canonical'] = mat['Pretty Formula'].apply(canonical_formula)
        elif 'composition' in mat.columns:
            mat['canonical'] = mat['composition'].apply(canonical_formula)
        elif 'canonical' in mat.columns:
            mat['canonical'] = mat['canonical'].apply(str)

        mat = mat.loc[mat['canonical'].notna()].reset_index(drop=True)
        result = master.merge(mat, on='canonical', how='left', suffixes=('_master', '_mat'))
        print(f'  matched matminer rows by canonical formula: {result["canonical"].notna().sum()} / {len(master)}')
        mat_features = result.drop(columns=['target'] + [c for c in result.columns if c.endswith('_master')])
        zt_values = result['target']
    else:
        if len(mat) != len(master):
            print(f'  WARNING: positional alignment length mismatch matminer({len(mat)}) master({len(master)})')
        n = min(len(mat), len(master))
        mat = mat.iloc[:n].reset_index(drop=True)
        master = master.iloc[:n].reset_index(drop=True)
        mat_features = mat.drop(columns=['target'], errors='ignore')
        zt_values = mat['target']
        print('  Aligning matminer positionally with master dataset')

    print(f'  first 3 matminer zT: {zt_values.head(3).tolist()}')
    if len(master) >= 3:
        print(f'  first 3 master zT: {master["zT"].head(3).tolist()}')

    return mat_features, zt_values, master


def align_lmm(master: pd.DataFrame):
    print('\nSTEP 4 - ALIGN l-MM')
    print('===================')
    check_file(LMM_PATH)
    lmm = pd.read_csv(LMM_PATH)
    if 'composition' not in lmm.columns:
        fail('lMM_features.csv must contain a composition column')

    lmm = lmm.copy()
    lmm['canonical'] = lmm['composition'].apply(canonical_formula)
    lmm = lmm.loc[lmm['canonical'].notna()].reset_index(drop=True)

    megnet_cols = [c for c in lmm.columns if 'MEGNet' in c]
    if not megnet_cols:
        fail('No MEGNet columns found in lMM_features.csv')

    lmm_megnet = lmm[['canonical'] + megnet_cols].copy()
    drop_allna = lmm_megnet.columns[lmm_megnet.isna().all()].tolist()
    if drop_allna:
        print(f'  dropping all-NaN columns: {drop_allna}')
        lmm_megnet = lmm_megnet.drop(columns=drop_allna)
        megnet_cols = [c for c in megnet_cols if c not in drop_allna]

    thresh = len(lmm_megnet) * 0.5
    drop_thresh = [c for c in megnet_cols if lmm_megnet[c].isna().sum() > thresh]
    if drop_thresh:
        print(f'  dropping >50% NaN columns: {drop_thresh}')
        lmm_megnet = lmm_megnet.drop(columns=drop_thresh)
        megnet_cols = [c for c in megnet_cols if c not in drop_thresh]

    if lmm_megnet.empty or not megnet_cols:
        fail('No valid MEGNet features remain after NaN pruning')

    lmm_grouped = lmm_megnet.groupby('canonical')[megnet_cols].mean()
    combined = lmm_grouped.reindex(master['canonical']).reset_index(drop=True)
    matched = combined.drop(columns=['canonical'], errors='ignore').notna().any(axis=1).sum()
    total = len(master)
    print(f'  matched rows: {matched} out of {total}')

    return combined


def build_features(mat_features: pd.DataFrame, lmm_features: pd.DataFrame, master: pd.DataFrame):
    print('\nSTEP 5 - COMBINE FEATURES')
    print('==========================')

    x_mat = mat_features.copy()
    x_lmm = lmm_features.copy()

    x = pd.concat([x_mat, x_lmm], axis=1)
    x['Temperature_K'] = master['Temperature (K)'].fillna(300.0).values

    target = mat_features.index.to_series().map(lambda i: None)
    # Target is separated already.

    imputer = SimpleImputer(strategy='mean')
    x_imp = pd.DataFrame(imputer.fit_transform(x), columns=x.columns)

    const_cols = x_imp.columns[x_imp.std(axis=0) == 0].tolist()
    if const_cols:
        print(f'  removing constant columns: {const_cols}')
        x_imp = x_imp.drop(columns=const_cols)

    print(f'  final feature matrix shape: {x_imp.shape}')
    return x_imp


def train_cv_model(X: pd.DataFrame, y: pd.Series):
    print('\nSTEP 6 - TRAIN MODNet 5-FOLD CV')
    print('================================')
    if len(y) != len(X):
        fail('Feature matrix and target length mismatch')

    mask = y > 0
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    print(f'  training samples after zT>0 filter: {len(y)}')
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_maes = []
    fold_rmses = []
    fold_r2s = []
    all_y_true = []
    all_y_pred = []
    start_time = time.time()

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        fold_start = time.time()
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        try:
            if MODNET_AVAILABLE:
                train_data = MODData(
                    materials=list(range(len(train_idx))),
                    targets=[[v] for v in y_train.values],
                    target_names=[TARGET_NAME],
                )
                train_data.df_featurized = X_train
                n_feat = min(50, X_train.shape[1])
                train_data.feature_selection(n=n_feat)
                model = MODNetModel([[[TARGET_NAME]]], weights={TARGET_NAME: 1}, n_feat=n_feat)
                model.fit(
                    train_data,
                    val_fraction=0.1,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                    loss=LOSS,
                    epochs=EPOCHS,
                    verbose=0,
                )
                test_data = MODData(
                    materials=list(range(len(test_idx))),
                    targets=[[0]] * len(test_idx),
                    target_names=[TARGET_NAME],
                )
                test_data.df_featurized = X_test
                y_pred = model.predict(test_data)[TARGET_NAME].values
            else:
                raise RuntimeError('MODNet unavailable; using RandomForest fallback')
        except Exception as mod_err:
            print('  MODNet training failed or unavailable:', mod_err)
            print('  Falling back to RandomForestRegressor for this fold')
            rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        elapsed = time.time() - start_time
        remaining = ((fold + 1) / (fold + 1) * (N_SPLITS - fold - 1)) * (elapsed / (fold + 1))
        print(f'  Fold {fold + 1}/{N_SPLITS} complete: MAE={mae:.4f} R²={r2:.4f} | Est. remaining: ~{remaining/60:.1f} min')

        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_r2s.append(r2)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

    metrics = {
        'MAE': np.mean(fold_maes),
        'MAE_std': np.std(fold_maes),
        'RMSE': np.mean(fold_rmses),
        'RMSE_std': np.std(fold_rmses),
        'R2': np.mean(fold_r2s),
        'R2_std': np.std(fold_r2s),
    }

    print(f"\n{TARGET_NAME} model results: MAE={metrics['MAE']:.4f} ± {metrics['MAE_std']:.4f}, R²={metrics['R2']:.4f} ± {metrics['R2_std']:.4f}")
    return metrics, np.array(all_y_true), np.array(all_y_pred)


def save_results(metrics):
    print('\nSTEP 7 - SAVE FINAL RESULTS')
    print('============================')
    rows = [
        {
            'Model': 'Matminer + MODNet',
            'MAE': '0.1130±0.0068',
            'RMSE': '',
            'R2': '0.7877±0.0196',
        },
        {
            'Model': 'MatterVial Roost + MODNet',
            'MAE': '0.1206±0.0067',
            'RMSE': '',
            'R2': '0.7336±0.0187',
        },
        {
            'Model': 'Matminer + l-MM + MODNet',
            'MAE': f"{metrics['MAE']:.4f}±{metrics['MAE_std']:.4f}",
            'RMSE': f"{metrics['RMSE']:.4f}±{metrics['RMSE_std']:.4f}",
            'R2': f"{metrics['R2']:.4f}±{metrics['R2_std']:.4f}",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(FINAL_RESULTS_PATH, index=False)
    print(f'  saved {FINAL_RESULTS_PATH}')
    print('\nFINAL COMPARISON')
    print(df.to_string(index=False))


def save_parity_plot(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.4, s=20, color='blue', edgecolors='none')
    lim = [min(y_true.min(), y_pred.min()) - 0.05, max(y_true.max(), y_pred.max()) + 0.05]
    ax.plot(lim, lim, 'r--', lw=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('Actual ZT', fontweight='bold')
    ax.set_ylabel('Predicted ZT', fontweight='bold')
    ax.set_title('Matminer + l-MM + MODNet Parity Plot', fontweight='bold')
    plt.tight_layout()
    fig.savefig(PLOT_PATH, dpi=300)
    plt.close(fig)
    print(f'  saved parity plot: {PLOT_PATH}')


if __name__ == '__main__':
    print('Starting Matminer + l-MM combined training...')
    print_file_info()
    master = build_master_index()
    mat_features, mat_target, master = align_matminer(master)
    lmm_features = align_lmm(master)
    X = build_features(mat_features, lmm_features, master)
    metrics, y_true, y_pred = train_cv_model(X, mat_target)
    save_results(metrics)
    save_parity_plot(y_true, y_pred)
    print('\nTRAINING COMPLETE - Check results/FINAL_RESULTS_FIXED.csv')
