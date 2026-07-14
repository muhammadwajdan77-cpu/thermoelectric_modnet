"""
MODNet ZT Prediction — MatterVial (Roost) vs Matminer Comparison
Research Project: Prof. Gian-Marco Rignanese (UCLouvain)
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from modnet.models import MODNetModel
from modnet.preprocessing import MODData

PROJECT_DIR  = Path(__file__).parent
RESULTS_DIR  = PROJECT_DIR / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEED     = 42
N_SPLITS = 5
DPI      = 300
np.random.seed(SEED)

# ── Load aligned Roost features ────────────────────────────────────────────
print("\n" + "="*70)
print("LOADING ALIGNED MATTERVIAL (ROOST) FEATURES")
print("="*70)

roost_df = pd.read_csv(RESULTS_DIR / "aligned_roost.csv")
print(f"Aligned Roost features loaded: {roost_df.shape}")

if 'canonical' not in roost_df.columns or 'zT' not in roost_df.columns:
    raise ValueError('aligned_roost.csv must contain canonical and zT columns')

X = roost_df.drop(columns=['canonical', 'Pretty Formula'], errors='ignore')

y = roost_df['zT'].astype(float)
groups = roost_df['canonical'].astype(str)

if 'Temperature_K' not in X.columns and 'Temperature (K)' in roost_df.columns:
    X['Temperature_K'] = roost_df['Temperature (K)'].values

mask = y > 0
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)
groups = groups[mask].reset_index(drop=True)

print(f"Valid samples: {len(X)} | Features: {X.shape[1]} | Groups: {groups.nunique()}")
print(f"ZT range: {y.min():.4f} — {y.max():.4f}")

# ── 5-Fold CV with MODNet ─────────────────────────────────────────────────
print("\n" + "="*70)
print("5-FOLD CV - MODNet + MatterVial (Roost) Features")
print("="*70)

imp = SimpleImputer(strategy='mean')
X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

group_kf = GroupKFold(n_splits=N_SPLITS)
fold_maes, fold_rmses, fold_r2s = [], [], []
all_y_true, all_y_pred = [], []

for fold, (train_idx, test_idx) in enumerate(group_kf.split(X_imp, y, groups=groups), start=1):
    print(f"\n  Fold {fold}/{N_SPLITS}")

    X_tr = X_imp.iloc[train_idx].reset_index(drop=True)
    X_te = X_imp.iloc[test_idx].reset_index(drop=True)
    y_tr = y.iloc[train_idx].reset_index(drop=True)
    y_te = y.iloc[test_idx].reset_index(drop=True)

    train_data = MODData(
        materials=list(range(len(train_idx))),
        targets=[[v] for v in y_tr.values],
        target_names=["ZT"]
    )
    train_data.df_featurized = X_tr
    n_feat = min(30, X_tr.shape[1])
    train_data.feature_selection(n=n_feat)

    model = MODNetModel([[["ZT"]]], weights={"ZT": 1}, n_feat=n_feat)
    model.fit(train_data, val_fraction=0.1, lr=0.001,
              batch_size=64, loss="mae", epochs=30, verbose=1)

    test_data = MODData(
        materials=list(range(len(test_idx))),
        targets=[[0]]*len(test_idx),
        target_names=["ZT"]
    )
    test_data.df_featurized = X_te
    y_pred = model.predict(test_data)["ZT"].values

    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)
    print(f"    MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    fold_maes.append(mae); fold_rmses.append(rmse); fold_r2s.append(r2)
    all_y_true.extend(y_te.values); all_y_pred.extend(y_pred)

print(f"\n{'─'*70}")
print("MATTERVIAL (ROOST) + MODNet RESULTS")
print(f"{'─'*70}")
print(f"  MAE:  {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}")
print(f"  RMSE: {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}")
print(f"  R²:   {np.mean(fold_r2s):.4f} ± {np.std(fold_r2s):.4f}")

# ── Parity plot ───────────────────────────────────────────────────────────
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(all_y_true, all_y_pred, alpha=0.4, s=20, color='darkorange', edgecolors='none')
lim = [min(all_y_true.min(), all_y_pred.min())-0.05,
       max(all_y_true.max(), all_y_pred.max())+0.05]
ax.plot(lim, lim, 'r--', lw=2, label='Perfect prediction')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Actual ZT", fontweight='bold')
ax.set_ylabel("Predicted ZT", fontweight='bold')
ax.set_title("MatterVial (Roost) + MODNet — Parity Plot", fontweight='bold')
txt = (f"MAE  = {np.mean(fold_maes):.4f}\n"
       f"RMSE = {np.mean(fold_rmses):.4f}\n"
       f"R²   = {np.mean(fold_r2s):.4f}")
ax.text(0.05, 0.95, txt, transform=ax.transAxes, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        family='monospace', fontsize=11)
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "parity_plot_mattervial_roost.png", dpi=DPI)
plt.close()
print("\nSaved: parity_plot_mattervial_roost.png")

# ── Final comparison ──────────────────────────────────────────────────────
baseline = pd.read_csv(RESULTS_DIR / "results.csv")
b = baseline[baseline['Fold'] == 'Mean±Std'].iloc[0]

print(f"\n{'='*70}")
print("FINAL COMPARISON")
print(f"{'='*70}")
print(f"{'Model':<40} {'MAE':>20} {'R²':>15}")
print(f"{'─'*70}")
print(f"{'Matminer + MODNet (Baseline)':<40} {b['MAE']:>20} {b['R2']:>15}")
print(f"{'MatterVial (Roost) + MODNet':<40} "
      f"{np.mean(fold_maes):.4f}±{np.std(fold_maes):.4f}  "
      f"{np.mean(fold_r2s):.4f}±{np.std(fold_r2s):.4f}")

# Save results
new_rows = pd.DataFrame([{
    'Fold': f, 'MAE': m, 'RMSE': r, 'R2': r2v, 'Model': 'MatterVial (Roost) + MODNet'
} for f, m, r, r2v in zip(range(1, N_SPLITS+1), fold_maes, fold_rmses, fold_r2s)])
new_rows = pd.concat([new_rows, pd.DataFrame([{
    'Fold': 'Mean±Std',
    'MAE':  f"{np.mean(fold_maes):.4f}±{np.std(fold_maes):.4f}",
    'RMSE': f"{np.mean(fold_rmses):.4f}±{np.std(fold_rmses):.4f}",
    'R2':   f"{np.mean(fold_r2s):.4f}±{np.std(fold_r2s):.4f}",
    'Model': 'MatterVial (Roost) + MODNet'
}])])
pd.concat([baseline, new_rows]).to_csv(RESULTS_DIR / "results_complete.csv", index=False)
print("\nSaved: results_complete.csv")
print("\nANALYSIS COMPLETE")