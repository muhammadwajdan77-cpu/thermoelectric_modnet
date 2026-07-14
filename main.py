"""
MODNet Thermoelectric ZT Prediction — Matminer Baseline
Research Project: Prof. Gian-Marco Rignanese (UCLouvain)
Dataset: SysTEm (Systematically Verified Experimental Thermoelectric Dataset)
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
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
import shap

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

# ── Step 1: Load data ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 1: LOADING DATASET")
print("="*70)

df = pd.read_excel(DATASET_PATH)
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

COMP_COL   = "Pretty Formula"
TARGET_COL = "zT"
TEMP_COL   = "Temperature (K)"

df = df[[COMP_COL, TARGET_COL, TEMP_COL]].dropna(subset=[TARGET_COL])
df = df[df[TARGET_COL] > 0].reset_index(drop=True)
print(f"After cleaning: {len(df)} samples")
print(f"ZT range: {df[TARGET_COL].min():.4f} — {df[TARGET_COL].max():.4f}")

# ── ZT distribution plot ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df[TARGET_COL], bins=40, color='steelblue', alpha=0.8, edgecolor='black')
axes[0].axvline(df[TARGET_COL].mean(), color='red', linestyle='--', lw=2,
                label=f"Mean: {df[TARGET_COL].mean():.3f}")
axes[0].axvline(df[TARGET_COL].median(), color='green', linestyle='--', lw=2,
                label=f"Median: {df[TARGET_COL].median():.3f}")
axes[0].set_xlabel("ZT", fontweight='bold')
axes[0].set_ylabel("Frequency", fontweight='bold')
axes[0].set_title("ZT Distribution", fontweight='bold')
axes[0].legend()
axes[1].boxplot(df[TARGET_COL], patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7))
axes[1].set_ylabel("ZT", fontweight='bold')
axes[1].set_title("ZT Box Plot", fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "zt_distribution.png", dpi=DPI)
plt.close()
print("Saved: zt_distribution.png")

# ── Step 2: Matminer featurization ────────────────────────────────────────
print("\n" + "="*70)
print("STEP 2: MATMINER FEATURIZATION (Magpie preset)")
print("="*70)

featurizer = ElementProperty.from_preset("magpie")
features_list = []
valid_idx     = []
failed        = 0

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
X = pd.DataFrame(features_list, columns=feat_names).reset_index(drop=True)
y = df.loc[valid_idx, TARGET_COL].reset_index(drop=True)
print(f"Valid samples: {len(X)}  |  Features: {X.shape[1]}  |  Failed: {failed}")

# ── Step 3: MODNet with pre-computed Matminer features ────────────────────
print("\n" + "="*70)
print("STEP 3: 5-FOLD CROSS-VALIDATION (MODNet + Matminer features)")
print("="*70)

from modnet.models import MODNetModel
from modnet.preprocessing import MODData

# Impute missing values once globally
imp_global = SimpleImputer(strategy='mean')
X_imp = pd.DataFrame(imp_global.fit_transform(X), columns=X.columns)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_maes, fold_rmses, fold_r2s = [], [], []
all_y_true, all_y_pred_list = [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_imp)):
    print(f"\n  Fold {fold+1}/{N_SPLITS}")

    X_tr = X_imp.iloc[train_idx].reset_index(drop=True)
    X_te = X_imp.iloc[test_idx].reset_index(drop=True)
    y_tr = y.iloc[train_idx].reset_index(drop=True)
    y_te = y.iloc[test_idx].reset_index(drop=True)

    # Feed pre-computed features directly into MODData
    train_data = MODData(
        materials=list(range(len(train_idx))),
        targets=[[v] for v in y_tr.values],
        target_names=["ZT"]
    )
    train_data.df_featurized = X_tr

    n_feat = min(50, X_tr.shape[1])
    train_data.feature_selection(n=n_feat)

    model = MODNetModel([[["ZT"]]], weights={"ZT": 1}, n_feat=n_feat)
    model.fit(
        train_data,
        val_fraction=0.1,
        lr=0.001,
        batch_size=64,
        loss="mae",
        epochs=100,
        verbose=0
    )

    test_data = MODData(
        materials=list(range(len(test_idx))),
        targets=[[0]] * len(test_idx),
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
    all_y_pred_list.extend(y_pred)

print(f"\n{'─'*70}")
print("CROSS-VALIDATION SUMMARY (Matminer + MODNet Baseline)")
print(f"{'─'*70}")
print(f"  MAE:  {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}")
print(f"  RMSE: {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}")
print(f"  R²:   {np.mean(fold_r2s):.4f} ± {np.std(fold_r2s):.4f}")

# ── Step 4: Parity plot ───────────────────────────────────────────────────
all_y_true      = np.array(all_y_true)
all_y_pred_arr  = np.array(all_y_pred_list)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(all_y_true, all_y_pred_arr, alpha=0.4, s=20,
           color='steelblue', edgecolors='none')
lim = [min(all_y_true.min(), all_y_pred_arr.min()) - 0.05,
       max(all_y_true.max(), all_y_pred_arr.max()) + 0.05]
ax.plot(lim, lim, 'r--', lw=2, label='Perfect prediction')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Actual ZT", fontweight='bold')
ax.set_ylabel("Predicted ZT", fontweight='bold')
ax.set_title("Matminer + MODNet — Parity Plot", fontweight='bold')
txt = (f"MAE  = {np.mean(fold_maes):.4f}\n"
       f"RMSE = {np.mean(fold_rmses):.4f}\n"
       f"R²   = {np.mean(fold_r2s):.4f}")
ax.text(0.05, 0.95, txt, transform=ax.transAxes, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        family='monospace', fontsize=11)
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "parity_plot_matminer.png", dpi=DPI)
plt.close()
print("\nSaved: parity_plot_matminer.png")

# ── Step 5: SHAP analysis ─────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 5: SHAP ANALYSIS")
print("="*70)

# Train final model on all data for SHAP
final_train = MODData(
    materials=list(range(len(X_imp))),
    targets=[[v] for v in y.values],
    target_names=["ZT"]
)
final_train.df_featurized = X_imp
n_feat = min(50, X_imp.shape[1])
final_train.feature_selection(n=n_feat)
selected_features = final_train.optimal_features[:n_feat]

final_model = MODNetModel([[["ZT"]]], weights={"ZT": 1}, n_feat=n_feat)
final_model.fit(final_train, val_fraction=0.1, lr=0.001,
                batch_size=64, loss="mae", epochs=100, verbose=0)

X_sel = X_imp[selected_features].values
bg    = X_sel[np.random.choice(len(X_sel), min(100, len(X_sel)), replace=False)]

def modnet_predict(x_arr):
    tmp = MODData(materials=list(range(len(x_arr))),
                  targets=[[0]]*len(x_arr), target_names=["ZT"])
    tmp.df_featurized = pd.DataFrame(x_arr, columns=selected_features)
    return final_model.predict(tmp)["ZT"].values

print("Computing SHAP values (this may take a few minutes)...")
explainer = shap.KernelExplainer(modnet_predict, bg)
samp_idx  = np.random.choice(len(X_sel), min(200, len(X_sel)), replace=False)
shap_vals = explainer.shap_values(X_sel[samp_idx])

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_vals, X_sel[samp_idx],
                  feature_names=selected_features, plot_type="dot", show=False)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=DPI)
plt.close()

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_vals, X_sel[samp_idx],
                  feature_names=selected_features, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_bar.png", dpi=DPI)
plt.close()
print("Saved: shap_summary.png and shap_bar.png")

imp_df = pd.DataFrame({
    'feature':    selected_features,
    'importance': np.abs(shap_vals).mean(axis=0)
}).sort_values('importance', ascending=False)
print("\nTop 10 features driving ZT:")
print(imp_df.head(10).to_string(index=False))
imp_df.to_csv(RESULTS_DIR / "shap_feature_importance.csv", index=False)

# ── Step 6: Save results ──────────────────────────────────────────────────
results_df = pd.DataFrame({
    'Fold':  list(range(1, N_SPLITS + 1)),
    'MAE':   fold_maes,
    'RMSE':  fold_rmses,
    'R2':    fold_r2s,
    'Model': 'Matminer + MODNet (Baseline)'
})
summary_row = pd.DataFrame([{
    'Fold':  'Mean±Std',
    'MAE':   f"{np.mean(fold_maes):.4f}±{np.std(fold_maes):.4f}",
    'RMSE':  f"{np.mean(fold_rmses):.4f}±{np.std(fold_rmses):.4f}",
    'R2':    f"{np.mean(fold_r2s):.4f}±{np.std(fold_r2s):.4f}",
    'Model': 'Matminer + MODNet (Baseline)'
}])
pd.concat([results_df, summary_row]).to_csv(RESULTS_DIR / "results.csv", index=False)
print("\nSaved: results.csv")

print("\n" + "="*70)
print("BASELINE ANALYSIS COMPLETE")
print(f"  Results → {RESULTS_DIR}/")
print(f"  Figures → {FIGURES_DIR}/")
print("="*70)
print("\nNOTE: MatterVial comparison pending — awaiting access from Prof. Rignanese")
