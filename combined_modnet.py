"""
MODNet ZT Prediction — Combined (Matminer + MatterVial Roost)
Research Project: Prof. Gian-Marco Rignanese (UCLouvain)

Combines pre-computed Matminer and Roost features for improved zT prediction.
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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from pymatgen.core import Composition
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

# ── Load and combine features ─────────────────────────────────────────────
print("\n" + "="*70)
print("LOADING AND COMBINING FEATURES")
print("="*70)

matminer_df = pd.read_csv(RESULTS_DIR / "matminer_for_sisso.csv")
roost_df = pd.read_csv(RESULTS_DIR / "roost_features.csv")

print(f"Matminer features loaded: {matminer_df.shape}")
print(f"Roost features loaded: {roost_df.shape}")

# Extract only feature columns (drop target from matminer if present)
X_matminer = matminer_df.drop(columns=['target'], errors='ignore')
X_matminer = X_matminer.drop(columns=['Temperature_K'], errors='ignore')  # Will add back later

X_roost = roost_df.drop(columns=['Pretty Formula'], errors='ignore')

# Align to the smaller dataset
n = min(len(X_matminer), len(X_roost))
X_matminer = X_matminer.iloc[:n].reset_index(drop=True)
X_roost = X_roost.iloc[:n].reset_index(drop=True)

print(f"\nAligned to {n} samples")
print(f"Matminer features: {X_matminer.shape[1]}")
print(f"Roost features: {X_roost.shape[1]}")

# Combine features horizontally
X_combined = pd.concat([X_matminer, X_roost], axis=1)

# Check for duplicate columns and remove them
n_cols_before = X_combined.shape[1]
X_combined = X_combined.loc[:, ~X_combined.columns.duplicated(keep='first')]
n_cols_after = X_combined.shape[1]

if n_cols_before > n_cols_after:
    print(f"Removed {n_cols_before - n_cols_after} duplicate columns")

print(f"Combined features: {X_combined.shape[1]}")
print(f"Combined shape: {X_combined.shape}")

# Load zT from original dataset using same valid composition filter
dataset_path = PROJECT_DIR / "sysTEm_dataset" / "sysTEm_dataset.xlsx"
df_orig = pd.read_excel(dataset_path)

valid_mask = []
for f in df_orig['Pretty Formula'].astype(str):
    try:
        Composition(f)
        valid_mask.append(True)
    except:
        valid_mask.append(False)

df_valid = df_orig[valid_mask].reset_index(drop=True)

# Align to combined features length
n = min(len(X_combined), len(df_valid))
X_combined = X_combined.iloc[:n].reset_index(drop=True)
df_valid = df_valid.iloc[:n].reset_index(drop=True)

y = df_valid['zT'].astype(float)
X_combined['Temperature_K'] = df_valid['Temperature (K)'].fillna(300.0).values

# Filter positive ZT
mask = y > 0
X_combined = X_combined[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)

print(f"\nValid samples (zT > 0): {len(X_combined)}")
print(f"Final feature count: {X_combined.shape[1]}")
print(f"ZT range: {y.min():.4f} — {y.max():.4f}")

# ── 5-Fold CV with MODNet ─────────────────────────────────────────────────
print("\n" + "="*70)
print("5-FOLD CV - MODNet + Combined (Matminer + Roost) Features")
print("="*70)

imp = SimpleImputer(strategy='mean')
X_imp = pd.DataFrame(imp.fit_transform(X_combined), columns=X_combined.columns)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_maes, fold_rmses, fold_r2s = [], [], []
all_y_true, all_y_pred = [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_imp)):
    print(f"\n  Fold {fold+1}/{N_SPLITS}")

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

    fold_maes.append(mae)
    fold_rmses.append(rmse)
    fold_r2s.append(r2)
    all_y_true.extend(y_te.values)
    all_y_pred.extend(y_pred)

print(f"\n{'─'*70}")
print("COMBINED (MATMINER + ROOST) + MODNet RESULTS")
print(f"{'─'*70}")
print(f"  MAE:  {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}")
print(f"  RMSE: {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}")
print(f"  R²:   {np.mean(fold_r2s):.4f} ± {np.std(fold_r2s):.4f}")

# ── Parity plot ───────────────────────────────────────────────────────────
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(all_y_true, all_y_pred, alpha=0.4, s=20, color='green', edgecolors='none')
lim = [min(all_y_true.min(), all_y_pred.min())-0.05,
       max(all_y_true.max(), all_y_pred.max())+0.05]
ax.plot(lim, lim, 'r--', lw=2, label='Perfect prediction')
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("Actual ZT", fontweight='bold')
ax.set_ylabel("Predicted ZT", fontweight='bold')
ax.set_title("Combined (Matminer + Roost) + MODNet — Parity Plot", fontweight='bold')
txt = (f"MAE  = {np.mean(fold_maes):.4f}\n"
       f"RMSE = {np.mean(fold_rmses):.4f}\n"
       f"R²   = {np.mean(fold_r2s):.4f}")
ax.text(0.05, 0.95, txt, transform=ax.transAxes, va='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
        family='monospace', fontsize=11)
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "parity_plot_combined.png", dpi=DPI)
plt.close()
print("\nSaved: parity_plot_combined.png")

# ── Final comparison ──────────────────────────────────────────────────────
baseline = pd.read_csv(RESULTS_DIR / "results_complete.csv")

print(f"\n{'='*70}")
print("FINAL COMPARISON: ALL MODELS")
print(f"{'='*70}")

# Get baseline results (mean values)
matminer_baseline = baseline[baseline['Model'] == 'Matminer + MODNet (Baseline)']
if len(matminer_baseline) > 0:
    mm_mean = matminer_baseline[matminer_baseline['Fold'] == 'Mean±Std']
    if len(mm_mean) > 0:
        mm_mae = mm_mean['MAE'].values[0]
        mm_r2 = mm_mean['R2'].values[0]
    else:
        # Calculate from individual folds
        mm_mae_vals = matminer_baseline['MAE'].astype(float)
        mm_r2_vals = matminer_baseline['R2'].astype(float)
        mm_mae = f"{mm_mae_vals.mean():.4f}±{mm_mae_vals.std():.4f}"
        mm_r2 = f"{mm_r2_vals.mean():.4f}±{mm_r2_vals.std():.4f}"
else:
    mm_mae = "N/A"
    mm_r2 = "N/A"

roost_baseline = baseline[baseline['Model'] == 'MatterVial (Roost) + MODNet']
if len(roost_baseline) > 0:
    roost_mean = roost_baseline[roost_baseline['Fold'] == 'Mean±Std']
    if len(roost_mean) > 0:
        roost_mae = roost_mean['MAE'].values[0]
        roost_r2 = roost_mean['R2'].values[0]
    else:
        roost_mae_vals = roost_baseline['MAE'].astype(float)
        roost_r2_vals = roost_baseline['R2'].astype(float)
        roost_mae = f"{roost_mae_vals.mean():.4f}±{roost_mae_vals.std():.4f}"
        roost_r2 = f"{roost_r2_vals.mean():.4f}±{roost_r2_vals.std():.4f}"
else:
    roost_mae = "N/A"
    roost_r2 = "N/A"

print(f"{'Model':<40} {'MAE':>24} {'R²':>15}")
print(f"{'─'*70}")
print(f"{'Matminer + MODNet':<40} {str(mm_mae):>24} {str(mm_r2):>15}")
print(f"{'MatterVial (Roost) + MODNet':<40} {str(roost_mae):>24} {str(roost_r2):>15}")
print(f"{'Combined (Matminer + Roost) + MODNet':<40} "
      f"{np.mean(fold_maes):.4f}±{np.std(fold_maes):.4f}  "
      f"{np.mean(fold_r2s):.4f}±{np.std(fold_r2s):.4f}")

# Save results
new_rows = pd.DataFrame([{
    'Fold': f,
    'MAE': m,
    'RMSE': r,
    'R2': r2v,
    'Model': 'Combined (Matminer + Roost) + MODNet'
} for f, m, r, r2v in zip(range(1, N_SPLITS+1), fold_maes, fold_rmses, fold_r2s)])

new_rows = pd.concat([new_rows, pd.DataFrame([{
    'Fold': 'Mean±Std',
    'MAE': f"{np.mean(fold_maes):.4f}±{np.std(fold_maes):.4f}",
    'RMSE': f"{np.mean(fold_rmses):.4f}±{np.std(fold_rmses):.4f}",
    'R2': f"{np.mean(fold_r2s):.4f}±{np.std(fold_r2s):.4f}",
    'Model': 'Combined (Matminer + Roost) + MODNet'
}])], ignore_index=True)

pd.concat([baseline, new_rows], ignore_index=True).to_csv(RESULTS_DIR / "results_complete.csv", index=False)
print("\nSaved: results_complete.csv")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
