"""xgboost_rfe_modnet.py

Build a mega feature matrix from Matminer, ROOST, l-OFM, and MVL features,
select the best 50 features with XGBoost RFE, then train MODNet on those
features and update FINAL_RESULTS_FIXED.csv.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from pymatgen.core import Composition

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    print('ERROR: xgboost is not installed. Install it with pip install xgboost')
    raise

try:
    from modnet.models import MODNetModel
    from modnet.preprocessing import MODData
except Exception as exc:
    print('ERROR: Could not import MODNet. Ensure modnet is installed in the environment.')
    raise

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MATMINER_PATH = RESULTS_DIR / 'matminer_for_sisso.csv'
ROOST_PATH = RESULTS_DIR / 'roost_features.csv'
LOFM_PATH = RESULTS_DIR / 'lOFM_features.csv'
MVL_PATH = RESULTS_DIR / 'MVL_features.csv'
DATASET_PATH = PROJECT_DIR / 'sysTEm_dataset' / 'sysTEm_dataset.xlsx'
FINAL_RESULTS_PATH = RESULTS_DIR / 'FINAL_RESULTS_FIXED.csv'
PARITY_PLOT_PATH = FIGURES_DIR / 'parity_plot_xgboost_rfe.png'

SEED = 42
N_SPLITS = 5
DPI = 300
np.random.seed(SEED)


def canonical_formula(value):
    try:
        return Composition(str(value)).reduced_formula
    except Exception:
        return None


def detect_formula_column(df: pd.DataFrame):
    for candidate in ['composition', 'Pretty Formula', 'pretty formula', 'Formula', 'formula']:
        if candidate in df.columns:
            return candidate
    # Fall back to the first string-like column if no explicit name is provided
    for col in df.columns:
        if df[col].dtype == object:
            return col
    return None


def load_master():
    print('\nSTEP 1 - BUILD MASTER DATASET')
    print('=' * 60)
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f'Missing dataset: {DATASET_PATH}')

    df = pd.read_excel(DATASET_PATH)
    if 'Pretty Formula' not in df.columns or 'zT' not in df.columns:
        raise RuntimeError('sysTEm_dataset.xlsx must contain Pretty Formula and zT columns')

    df = df.loc[df['zT'].notna() & (df['zT'] > 0)].copy()
    df['canonical'] = df['Pretty Formula'].astype(str).apply(canonical_formula)
    df = df.loc[df['canonical'].notna()].reset_index(drop=True)

    print(f'  master rows after zT>0 and valid formula: {len(df)}')
    if len(df) != 7594:
        print('  WARNING: master rows != 7594. Using available parsed rows.')

    return df[['Pretty Formula', 'Temperature (K)', 'zT', 'canonical']].copy()


def load_matminer(master):
    print('\nSTEP 2 - LOAD MATMINER ANCHOR')
    print('=' * 60)
    if not MATMINER_PATH.exists():
        raise FileNotFoundError(f'Missing matminer file: {MATMINER_PATH}')

    mat = pd.read_csv(MATMINER_PATH)
    if 'target' not in mat.columns:
        raise RuntimeError('matminer_for_sisso.csv must contain a target column')

    if len(mat) != len(master):
        print(f'  WARNING: Matminer rows {len(mat)} != master rows {len(master)}')

    n = min(len(mat), len(master))
    mat = mat.iloc[:n].reset_index(drop=True)
    master = master.iloc[:n].reset_index(drop=True)

    mat_features = mat.drop(columns=['target'], errors='ignore')
    target = mat['target'].iloc[:n].reset_index(drop=True)

    print(f'  Matminer features shape: {mat_features.shape}')
    print(f'  Matminer target sample: {target.head(3).tolist()}')

    return master, mat_features, target


def load_roost(master):
    print('\nSTEP 3 - LOAD ROOST FEATURES')
    print('=' * 60)
    if not ROOST_PATH.exists():
        raise FileNotFoundError(f'Missing ROOST file: {ROOST_PATH}')

    roost = pd.read_csv(ROOST_PATH)
    n = min(len(roost), len(master))
    if len(roost) != len(master):
        print(f'  WARNING: ROOST rows {len(roost)} != master rows {len(master)}; aligning by position on first {n} rows')
    roost = roost.iloc[:n].reset_index(drop=True)
    master = master.iloc[:n].reset_index(drop=True)
    print(f'  ROOST features shape: {roost.shape}')
    return master, roost


def load_merge_features(path: Path, label: str, master):
    print(f'\nSTEP 4 - LOAD {label.upper()} FEATURES')
    print('=' * 60)
    if not path.exists():
        raise FileNotFoundError(f'Missing file: {path}')

    df = pd.read_csv(path)
    formula_col = detect_formula_column(df)
    if formula_col is None:
        raise RuntimeError(f'No formula/composition column found in {path}')

    df = df.copy()
    if formula_col != 'composition':
        df = df.rename(columns={formula_col: 'composition'})
    df['canonical'] = df['composition'].astype(str).apply(canonical_formula)
    df = df.loc[df['canonical'].notna()].reset_index(drop=True)

    # Keep the first match for each canonical formula
    grouped = df.groupby('canonical', sort=False).first().reset_index()
    merged = master[['canonical']].merge(grouped, on='canonical', how='left')
    feature_cols = [c for c in merged.columns if c not in ['canonical']]
    matched = merged[feature_cols].notna().any(axis=1).sum()
    print(f'  matched {label} rows: {matched} / {len(master)}')

    merged = merged.drop(columns=['composition'], errors='ignore')
    return merged.drop(columns=['canonical'], errors='ignore')


def build_feature_matrix(mat_features, roost, lofm, mvl, master):
    print('\nSTEP 5 - BUILD MEGA FEATURE MATRIX')
    print('=' * 60)

    features = [mat_features.reset_index(drop=True),
                roost.reset_index(drop=True),
                lofm.reset_index(drop=True),
                mvl.reset_index(drop=True)]

    X = pd.concat(features, axis=1)
    print(f'  combined raw shape: {X.shape}')

    imputer = SimpleImputer(strategy='mean')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    const_cols = X_imp.columns[X_imp.std(axis=0, ddof=0) == 0].tolist()
    if const_cols:
        print(f'  removing constant columns: {len(const_cols)}')
        X_imp = X_imp.drop(columns=const_cols)
    print(f'  final feature matrix shape: {X_imp.shape}')
    return X_imp


def run_rfe(X, y):
    print('\nSTEP 6 - XGBOOST RFE FEATURE SELECTION')
    print('=' * 60)
    selector = RFE(
        estimator=XGBRegressor(
            n_estimators=100,
            random_state=SEED,
            n_jobs=-1,
            objective='reg:squarederror',
            verbosity=0
        ),
        n_features_to_select=50,
        step=10
    )
    selector.fit(X, y)
    selected = X.columns[selector.support_].tolist()
    print(f'  selected {len(selected)} features:')
    for name in selected:
        print(f'    {name}')
    return selected


def train_modnet(X, y, selected_features):
    print('\nSTEP 7 - TRAIN MODNET ON RFE-SELECTED FEATURES')
    print('=' * 60)

    X_sel = X[selected_features].copy()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_maes, fold_rmses, fold_r2s = [], [], []
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_sel), start=1):
        print(f'\n  Fold {fold}/{N_SPLITS} complete')
        X_tr = X_sel.iloc[train_idx].reset_index(drop=True)
        X_te = X_sel.iloc[test_idx].reset_index(drop=True)
        y_tr = y.iloc[train_idx].reset_index(drop=True)
        y_te = y.iloc[test_idx].reset_index(drop=True)

        train_data = MODData(
            materials=list(range(len(X_tr))),
            targets=[[v] for v in y_tr.values],
            target_names=['ZT']
        )
        train_data.df_featurized = X_tr
        n_feat = min(50, X_tr.shape[1])
        train_data.feature_selection(n=n_feat)

        model = MODNetModel([[['ZT']]], weights={'ZT': 1}, n_feat=n_feat)
        model.fit(
            train_data,
            val_fraction=0.1,
            lr=0.001,
            batch_size=64,
            loss='mae',
            epochs=100,
            verbose=0
        )

        test_data = MODData(
            materials=list(range(len(X_te))),
            targets=[[0]] * len(X_te),
            target_names=['ZT']
        )
        test_data.df_featurized = X_te
        y_pred = model.predict(test_data)['ZT'].values

        mae = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        r2 = r2_score(y_te, y_pred)
        print(f'    MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}')

        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_r2s.append(r2)
        all_y_true.extend(y_te.values)
        all_y_pred.extend(y_pred)

    return fold_maes, fold_rmses, fold_r2s, np.array(all_y_true), np.array(all_y_pred)


def save_parity_plot(y_true, y_pred, mae, rmse, r2):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.4, s=20, color='darkorange', edgecolors='none')
    lim = [min(y_true.min(), y_pred.min()) - 0.05,
           max(y_true.max(), y_pred.max()) + 0.05]
    ax.plot(lim, lim, 'r--', lw=2, label='Perfect prediction')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('Actual ZT', fontweight='bold')
    ax.set_ylabel('Predicted ZT', fontweight='bold')
    ax.set_title('Matminer+ROOST+l-OFM+MVL (XGBoost RFE) + MODNet', fontweight='bold')
    txt = (f'MAE  = {mae:.4f}\n'
           f'RMSE = {rmse:.4f}\n'
           f'R²   = {r2:.4f}')
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace', fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.savefig(PARITY_PLOT_PATH, dpi=DPI)
    plt.close()
    print(f'\nSaved parity plot: {PARITY_PLOT_PATH}')


def update_final_results(mae, rmse, r2, mae_std, rmse_std, r2_std):
    print('\nSTEP 8 - UPDATE FINAL RESULTS')
    print('=' * 60)
    df = pd.read_csv(FINAL_RESULTS_PATH)
    new_row = {
        'Model': 'Matminer+ROOST+l-OFM+MVL (XGBoost RFE) + MODNet',
        'MAE': f'{mae:.4f}±{mae_std:.4f}',
        'RMSE': f'{rmse:.4f}±{rmse_std:.4f}',
        'R2': f'{r2:.4f}±{r2_std:.4f}'
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(FINAL_RESULTS_PATH, index=False)
    print(f'Updated {FINAL_RESULTS_PATH}')


if __name__ == '__main__':
    master = load_master()
    master, mat_features, y = load_matminer(master)
    master, roost = load_roost(master)
    lofm = load_merge_features(LOFM_PATH, 'l-OFM', master)
    mvl = load_merge_features(MVL_PATH, 'MVL', master)

    X = build_feature_matrix(mat_features, roost, lofm, mvl, master)
    selected_features = run_rfe(X, y)

    fold_maes, fold_rmses, fold_r2s, all_y_true, all_y_pred = train_modnet(X, y, selected_features)
    mean_mae = np.mean(fold_maes)
    mean_rmse = np.mean(fold_rmses)
    mean_r2 = np.mean(fold_r2s)
    print('\n' + '=' * 60)
    print('CV SUMMARY')
    print(f'  MAE:  {mean_mae:.4f} ± {np.std(fold_maes):.4f}')
    print(f'  RMSE: {mean_rmse:.4f} ± {np.std(fold_rmses):.4f}')
    print(f'  R²:   {mean_r2:.4f} ± {np.std(fold_r2s):.4f}')

    save_parity_plot(all_y_true, all_y_pred, mean_mae, mean_rmse, mean_r2)
    update_final_results(
        mean_mae,
        mean_rmse,
        mean_r2,
        np.std(fold_maes),
        np.std(fold_rmses),
        np.std(fold_r2s)
    )
    print('\nANALYSIS COMPLETE')
