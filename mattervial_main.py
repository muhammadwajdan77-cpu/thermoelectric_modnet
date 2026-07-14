"""
MODNet ZT Prediction — MatterVial (SISSO) vs Matminer Comparison
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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent
DATASET_PATH = PROJECT_DIR / "sysTEm_dataset" / "sysTEm_dataset.xlsx"
RESULTS_DIR  = PROJECT_DIR / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEED     = 42
N_SPLITS = 5
DPI      = 300
np.random.seed(SEED)

COMP_COL   = "Pretty Formula"
TARGET_COL = "zT"
TEMP_COL   = "Temperature (K)"

# ── Step 1: Load data ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 1: LOADING DATASET")
print("="*70)

df = pd.read_excel(DATASET_PATH)
df = df[[COMP_COL, TARGET_COL, TEMP_COL]].dropna(subset=[TARGET_COL])
df = df[df[TARGET_COL] > 0].reset_index(drop=True)
print(f"Samples: {len(df)}")

# ── Step 2: Matminer featurization ────────────────────────────────────────
print("\n" + "="*70)
print("STEP 2: MATMINER FEATURIZATION")
print("="*70)

featurizer = ElementProperty.from_preset("magpie")
features_list, valid_idx, failed = [], [], 0

for i, row in df.iterrows():
    try:
        comp = Composition(str(row[COMP_COL]).strip())
        feat = featurizer.featurize(comp)
        if not np.isnan(feat).any():
            temp_val = float(row[TEMP_COL]) if pd.notna(row[TEMP_COL]) else 300.0
            features_list.append(list(feat) + [temp_val])
            valid_idx.append(i)
    except:
        failed += 1
    if (i + 1) % 1000 == 0:
        print(f"  Progress: {i+1}/{len(df)} | valid: {len(valid_idx)}")

feat_names = featurizer.feature_labels() + ["Temperature_K"]
X_matminer = pd.DataFrame(features_list, columns=feat_names).reset_index(drop=True)
y = df.loc[valid_idx, TARGET_COL].reset_index(drop=True)
print(f"Valid: {len(X_matminer)} | Features: {X_matminer.shape[1]} | Failed: {failed}")

# ── Step 3: MatterVial SISSO features ─────────────────────────────────────
print("\n" + "="*70)
print("STEP 3: MATTERVIAL SISSO FEATURIZATION")
print("="*70)

# Save Matminer features to CSV for SISSO (required format)
matminer_csv = PROJECT_DIR / "results" / "matminer_for_sisso.csv"
X_matminer_with_target = X_matminer.copy()
X_matminer_with_target['target'] = y.values
X_matminer_with_target.to_csv(matminer_csv, index=False)
print(f"Saved Matminer features for SISSO: {matminer_csv}")

try:
    from mattervial.featurizers import get_sisso_features
    print("Applying SISSO symbolic formulas to Matminer features...")
    sisso_df = get_sisso_features(
        input_csv_path=str(matminer_csv),
        type="SISSO_FORMULAS_v1"
    )
    print(f"SISSO features generated: {sisso_df.shape[1]} new features")

    # Combine Matminer + SISSO features = MatterVial
    X_mattervial = pd.concat([X_matminer.reset_index(drop=True),
                               sisso_df.reset_index(drop=True)], axis=1)
    print(f"Total MatterVial features: {X_mattervial.shape[1]}")

except Exception as e:
    print(f"SISSO error: {e}")
    print("Using Matminer features only as fallback")
    X_mattervial = X_matminer.copy()

# ── Step 4: 5-Fold CV with MODNet ─────────────────────────────────────────
print("\n" + "="*70)
print("STEP 4: 5-FOLD CV - MODNet + MatterVial (SISSO)")
print("="*70)

from modnet.models import MODNetModel
from modnet.preprocessing import MODData

imp_global = SimpleImputer(strategy='mean')
X_mv_imp = pd.DataFrame(
    imp_global.fit_transform(X_mattervial),
    columns=X_mattervial.columns
)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_maes, fold_rmses, fold_r2s = [], [], []
all_y_true, all_y_pred = [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_mv_imp)):
    print(f"\n  Fold {fold+1}/{N_SPLITS}")

    X_tr = X_mv_imp.iloc[train_idx].reset_index(drop=True)
    X_te = X_mv_imp.iloc[test_idx].reset_index(drop=True)
    y_tr = y.iloc[train_idx].reset_index(drop=True)
    y_te = y.iloc[test_idx].reset_index(drop=True)

    train_data = MODData(
        materials=list(range(len(train_idx))),
        targets=[[v] for v in y_tr.values],
        target_names=["ZT"]
    )
    train_data.df_featurized = X_tr
    n_feat = min(50, X_tr.shape[1])
    train_data.feature_selection(n=n_feat)

    model = MODNetModel([[["ZT"]]], weights={"ZT": 1}, n_feat=n_feat)
    model.fit(train_data, val_fraction=0.1, lr=0.001,
              batch_size=64, loss="mae", epochs=100, verbose=0)

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
print("MATTERVIAL RESULTS (Matminer + SISSO + MODNet)")
print(f"{'─'*70}")
print(f"  MAE:  {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}")
print(f"  RMSE: {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}")
print(f"  R²:   {np.mean(fold_r2s):.4f} ± {np.std(fold_r2s):.4f}")

# ── Step 5: Parity plot ───────────────────────────────────────────────────
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(all_y_true, all_y_pred, alpha=0.4, s=20,
           color='darkorange', edgecolors='none')
lim = [min(all_y_true.min(), all_y_pred.min()) - 0.05,
       max(all_y_true.max(), all_y_pred.max()) + 0.05]
ax.plot(lim, lim, 'r--', lw=2, label='Perfect prediction')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Actual ZT", fontweight='bold')
ax.set_ylabel("Predicted ZT", fontweight='bold')
ax.set_title("MatterVial + MODNet — Parity Plot", fontweight='bold')
txt = (f"MAE  = {np.mean(fold_maes):.4f}\n"
       f"RMSE = {np.mean(fold_rmses):.4f}\n"
       f"R²   = {np.mean(fold_r2s):.4f}")
ax.text(0.05, 0.95, txt, transform=ax.transAxes, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        family='monospace', fontsize=11)
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "parity_plot_mattervial.png", dpi=DPI)
plt.close()
print("\nSaved: parity_plot_mattervial.png")

# ── Step 6: Comparison table ──────────────────────────────────────────────
baseline = pd.read_csv(RESULTS_DIR / "results.csv")
b = baseline[baseline['Fold'] == 'Mean±Std'].iloc[0]

print(f"\n{'='*70}")
print("FINAL COMPARISON")
print(f"{'='*70}")
print(f"{'Model':<35} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
print(f"{'─'*70}")
print(f"{'Matminer + MODNet (Baseline)':<35} {b['MAE']:>10} {b['RMSE']:>10} {b['R2']:>10}")
print(f"{'MatterVial + MODNet':<35} "
      f"{np.mean(fold_maes):.4f}±{np.std(fold_maes):.4f}  "
      f"{np.mean(fold_rmses):.4f}±{np.std(fold_rmses):.4f}  "
      f"{np.mean(fold_r2s):.4f}±{np.std(fold_r2s):.4f}")

# Save combined results
new_rows = pd.DataFrame([{
    'Fold': f, 'MAE': m, 'RMSE': r, 'R2': r2,
    'Model': 'MatterVial + MODNet'
} for f, m, r, r2 in zip(
    range(1, N_SPLITS+1), fold_maes, fold_rmses, fold_r2s)])
new_rows = pd.concat([new_rows, pd.DataFrame([{
    'Fold': 'Mean±Std',
    'MAE': f"{np.mean(fold_maes):.4f}±{np.std(fold_maes):.4f}",
    'RMSE': f"{np.mean(fold_rmses):.4f}±{np.std(fold_rmses):.4f}",
    'R2': f"{np.mean(fold_r2s):.4f}±{np.std(fold_r2s):.4f}",
    'Model': 'MatterVial + MODNet'
}])])
pd.concat([baseline, new_rows]).to_csv(RESULTS_DIR / "results_complete.csv", index=False)
print("\nSaved: results_complete.csv")
print("\nANALYSIS COMPLETE")
