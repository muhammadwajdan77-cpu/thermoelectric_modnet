#!/usr/bin/env python3
"""
Final complete integration: Matminer + ROOST + l-OFM + MVL + ORB
+ XGBoost RFE (50 features) + MODNet training
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Imports
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from pymatgen.core.composition import Composition
import matplotlib.pyplot as plt

print("=" * 80)
print("FINAL COMPLETE INTEGRATION: ALL FEATURES + XGBoost RFE + MODNet")
print("=" * 80)

# Paths
RESULTS_DIR = Path("results")
DATASET_FILE = Path("sysTEm_dataset/sysTEm_dataset.xlsx")
OUTPUT_FILE = RESULTS_DIR / "FINAL_RESULTS_FIXED.csv"
FIGURE_DIR = RESULTS_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

print("\n" + "=" * 80)
print("STEP 1 - BUILD MASTER (Matminer anchor, 7594 rows)")
print("=" * 80)

# Load Matminer as anchor
df_matminer = pd.read_csv(RESULTS_DIR / "matminer_for_sisso.csv")
print(f"Matminer loaded: {df_matminer.shape}")
print(f"Columns: {df_matminer.columns.tolist()[:10]}...")

# Extract Matminer features (skip composition and target columns)
matminer_cols = [c for c in df_matminer.columns if c not in ['composition', 'pretty_formula', 'zT', 'Temperature_K']]
print(f"Matminer features: {len(matminer_cols)}")

# Load original dataset for zT and formula info
try:
    df_original = pd.read_excel(DATASET_FILE)
    print(f"Original dataset loaded: {df_original.shape}")
except Exception as e:
    print(f"Warning: Could not load original dataset: {e}")
    df_original = None

# Ensure composition and zT are present in matminer file; align by position if needed
if 'composition' not in df_matminer.columns or 'zT' not in df_matminer.columns:
    aligned_file = RESULTS_DIR / 'aligned_combined.csv'
    if aligned_file.exists():
        df_aligned = pd.read_csv(aligned_file)
        if len(df_aligned) == len(df_matminer):
            # aligned_combined.csv contains 'canonical' and 'target' columns
            if 'composition' not in df_matminer.columns and 'canonical' in df_aligned.columns:
                df_matminer['composition'] = df_aligned['canonical'].values
            if 'zT' not in df_matminer.columns and 'target' in df_aligned.columns:
                df_matminer['zT'] = df_aligned['target'].values
            if 'Temperature_K' not in df_matminer.columns:
                # aligned file may have 'Temperature (K)'
                if 'Temperature_K' in df_aligned.columns:
                    df_matminer['Temperature_K'] = df_aligned['Temperature_K'].values
                elif 'Temperature (K)' in df_aligned.columns:
                    df_matminer['Temperature_K'] = df_aligned['Temperature (K)'].values
            print("Aligned matminer rows with results/aligned_combined.csv by position")
        else:
            raise ValueError(f"Row mismatch: matminer {len(df_matminer)} vs aligned {len(df_aligned)}")
    else:
        raise FileNotFoundError(f"Aligned file not found: {aligned_file}")

# Initialize master dataframe
matminer_cols = [c for c in df_matminer.columns if c not in ['composition', 'pretty_formula', 'zT', 'Temperature_K']]
master = df_matminer[['composition', 'Temperature_K', 'zT'] + matminer_cols].copy()
master.rename(columns={c: f"Matminer_{c}" if c not in ['composition', 'Temperature_K', 'zT'] else c for c in master.columns}, inplace=True)
print(f"Master dataframe: {master.shape}")

print("\n" + "=" * 80)
print("STEP 2 - ALIGN ALL FEATURE SETS by canonical formula")
print("=" * 80)

# Helper function to get canonical formula
def get_canonical_formula(comp_str):
    try:
        return str(Composition(comp_str).reduced_formula)
    except:
        return None

# Add canonical formula to master
master['canonical_formula'] = master['composition'].apply(get_canonical_formula)
master_matched = master.dropna(subset=['canonical_formula'])
print(f"Master with canonical formula: {master_matched.shape}")

# ===== ROOST =====
print("\n--- ROOST alignment ---")
aligned_roost_path = RESULTS_DIR / "aligned_roost.csv"
if aligned_roost_path.exists():
    df_roost = pd.read_csv(aligned_roost_path)
    print(f"ROOST (aligned) loaded: {df_roost.shape}")
    # aligned_roost contains 'canonical' and typically 'Pretty Formula' or composition mapping
    if 'composition' not in df_roost.columns and 'canonical' in df_roost.columns:
        df_roost['composition'] = df_roost['canonical']
else:
    df_roost = pd.read_csv(RESULTS_DIR / "roost_features.csv")
    print(f"ROOST loaded: {df_roost.shape}")

roost_cols = [c for c in df_roost.columns if c not in ['composition', 'canonical', 'Pretty Formula']]
if 'composition' in df_roost.columns:
    df_roost['canonical_formula'] = df_roost['composition'].apply(get_canonical_formula)
elif 'canonical' in df_roost.columns:
    df_roost['canonical_formula'] = df_roost['canonical']
else:
    df_roost['canonical_formula'] = None

df_roost_matched = df_roost.dropna(subset=['canonical_formula'])
print(f"ROOST with canonical formula: {df_roost_matched.shape}")

# Merge by canonical formula
master_with_roost = master_matched.merge(
    df_roost_matched[['canonical_formula'] + roost_cols],
    on='canonical_formula',
    how='left'
)
roost_match_count = master_with_roost[roost_cols[0]].notna().sum()
print(f"ROOST matches: {roost_match_count}/{len(master_matched)}")

# ===== l-OFM =====
print("\n--- l-OFM alignment ---")
df_lofm = pd.read_csv(RESULTS_DIR / "lOFM_features.csv")
print(f"l-OFM loaded: {df_lofm.shape}")
# normalize composition column if unnamed index was used
if 'composition' not in df_lofm.columns:
    unnamed = [c for c in df_lofm.columns if c.startswith('Unnamed')]
    if unnamed:
        df_lofm.rename(columns={unnamed[0]: 'composition'}, inplace=True)

lofm_cols = [c for c in df_lofm.columns if c not in ['composition']]
if 'composition' in df_lofm.columns:
    df_lofm['canonical_formula'] = df_lofm['composition'].apply(get_canonical_formula)
else:
    df_lofm['canonical_formula'] = None
df_lofm_matched = df_lofm.dropna(subset=['canonical_formula'])
print(f"l-OFM with canonical formula: {df_lofm_matched.shape}")

# Merge
master_with_roost_lofm = master_with_roost.merge(
    df_lofm_matched[['canonical_formula'] + lofm_cols],
    on='canonical_formula',
    how='left'
)
lofm_match_count = master_with_roost_lofm[lofm_cols[0]].notna().sum()
print(f"l-OFM matches: {lofm_match_count}/{len(master_matched)}")

# ===== MVL =====
print("\n--- MVL alignment ---")
df_mvl = pd.read_csv(RESULTS_DIR / "MVL_features.csv")
print(f"MVL loaded: {df_mvl.shape}")
if 'composition' not in df_mvl.columns:
    unnamed = [c for c in df_mvl.columns if c.startswith('Unnamed')]
    if unnamed:
        df_mvl.rename(columns={unnamed[0]: 'composition'}, inplace=True)

mvl_cols = [c for c in df_mvl.columns if c not in ['composition']]
if 'composition' in df_mvl.columns:
    df_mvl['canonical_formula'] = df_mvl['composition'].apply(get_canonical_formula)
else:
    df_mvl['canonical_formula'] = None
df_mvl_matched = df_mvl.dropna(subset=['canonical_formula'])
print(f"MVL with canonical formula: {df_mvl_matched.shape}")

# Merge
master_all = master_with_roost_lofm.merge(
    df_mvl_matched[['canonical_formula'] + mvl_cols],
    on='canonical_formula',
    how='left'
)
mvl_match_count = master_all[mvl_cols[0]].notna().sum()
print(f"MVL matches: {mvl_match_count}/{len(master_matched)}")

# ===== ORB =====
print("\n--- ORB alignment ---")
df_orb = pd.read_csv(RESULTS_DIR / "ORB_features.csv")
print(f"ORB loaded: {df_orb.shape}")

orb_cols = [c for c in df_orb.columns if c not in ['composition', 'pretty_formula']]
df_orb['canonical_formula'] = df_orb['composition'].apply(get_canonical_formula)
df_orb_matched = df_orb.dropna(subset=['canonical_formula'])
print(f"ORB with canonical formula: {df_orb_matched.shape}")

# Merge
master_all_features = master_all.merge(
    df_orb_matched[['canonical_formula'] + orb_cols],
    on='canonical_formula',
    how='left'
)
orb_match_count = master_all_features[orb_cols[0]].notna().sum()
print(f"ORB matches: {orb_match_count}/{len(master_matched)}")

print(f"\nFinal master with all features: {master_all_features.shape}")

print("\n" + "=" * 80)
print("STEP 3 - BUILD MEGA FEATURE MATRIX")
print("=" * 80)


# Build mega dataframe and detect feature columns dynamically
feature_cols = [c for c in master_all_features.columns if c not in ['composition', 'canonical_formula']]

# Try to locate target and temperature columns with flexible names
def find_column(cols, candidates):
    for name in candidates:
        for c in cols:
            if c == name:
                return c
    for name in candidates:
        for c in cols:
            if name in c:
                return c
    return None

target_col = find_column(master_all_features.columns, ['zT', 'target', 'ZT'])
temp_col = find_column(master_all_features.columns, ['Temperature_K', 'Temperature (K)', 'Temperature_K_x'])
if target_col is None:
    raise KeyError('Could not find target column (zT/target) in merged master')
if temp_col is None:
    # fallback to creating Temperature_K filled with 300
    master_all_features['Temperature_K'] = 300.0
    temp_col = 'Temperature_K'

print(f"Total feature columns found in master: {len(feature_cols)-2}")
df_mega = master_all_features[['composition', 'canonical_formula', temp_col, target_col] + [c for c in feature_cols if c not in [temp_col, target_col]]].copy()
# normalize column names
df_mega.rename(columns={temp_col: 'Temperature_K', target_col: 'zT'}, inplace=True)

# Rename feature columns with source prefixes when possible
rename_dict = {}
for col in feature_cols:
    if col in matminer_cols and not col.startswith('Matminer_'):
        rename_dict[col] = f"Matminer_{col}"
    elif col in roost_cols and not col.startswith('ROOST_'):
        rename_dict[col] = f"ROOST_{col}"
    elif col in lofm_cols and not col.startswith('lOFM_'):
        rename_dict[col] = f"lOFM_{col}"
    elif col in mvl_cols and not col.startswith('MVL_'):
        rename_dict[col] = f"MVL_{col}"
    elif col in orb_cols and not col.startswith('ORB_'):
        rename_dict[col] = f"ORB_{col}"

df_mega.rename(columns=rename_dict, inplace=True)

# Extract feature matrix
X_cols = [c for c in df_mega.columns if c not in ['composition', 'canonical_formula', 'Temperature_K', 'zT']]
X_raw = df_mega[X_cols].copy()
y = df_mega['zT'].values
temp = df_mega['Temperature_K'].values

print(f"\nFeature matrix before imputation: {X_raw.shape}")
print(f"Missing values: {X_raw.isna().sum().sum()}")

# Impute NaN with mean
X_raw = X_raw.apply(pd.to_numeric, errors='coerce')
# Use pandas fillna with column means (safer across mixed dtypes)
X_imputed_df = X_raw.fillna(X_raw.mean())
# If any columns are entirely NaN, fill remaining NaNs with 0.0
X_imputed_df = X_imputed_df.fillna(0.0)
X_imputed = X_imputed_df.values
X_imputed = pd.DataFrame(X_imputed, columns=X_cols, index=X_raw.index)

# Remove constant columns
non_constant_cols = [c for c in X_cols if X_imputed[c].std() > 1e-10]
X_clean = X_imputed[non_constant_cols]

print(f"Feature matrix after imputation and cleaning: {X_clean.shape}")
print(f"Constant columns removed: {len(X_cols) - len(non_constant_cols)}")

print("\n" + "=" * 80)
print("STEP 4 - XGBOOST RFE FEATURE SELECTION (50 features)")
print("=" * 80)

print("Initializing XGBoost RFE...")
xgb_model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
selector = RFE(xgb_model, n_features_to_select=50, step=50)

print("Fitting RFE selector...")
selector.fit(X_clean, y)

selected_features = X_clean.columns[selector.support_].tolist()
print(f"\nSelected {len(selected_features)} features:")

# Group by source
source_counts = {}
for feat in selected_features:
    if feat.startswith("Matminer_"):
        source_counts["Matminer"] = source_counts.get("Matminer", 0) + 1
    elif feat.startswith("ROOST_"):
        source_counts["ROOST"] = source_counts.get("ROOST", 0) + 1
    elif feat.startswith("lOFM_"):
        source_counts["lOFM"] = source_counts.get("lOFM", 0) + 1
    elif feat.startswith("MVL_"):
        source_counts["MVL"] = source_counts.get("MVL", 0) + 1
    elif feat.startswith("ORB_"):
        source_counts["ORB"] = source_counts.get("ORB", 0) + 1

print("\nFeature source distribution:")
for source, count in sorted(source_counts.items()):
    print(f"  {source}: {count}")

print("\nSelected features:")
for i, feat in enumerate(selected_features[:10]):
    print(f"  {i+1}. {feat}")
print(f"  ... ({len(selected_features)} total)")

X_selected = X_clean[selected_features]

print("\n" + "=" * 80)
print("STEP 5 - TRAIN MODNet ON RFE-SELECTED 50 FEATURES")
print("=" * 80)

# Import MODNet
try:
    from modnet.models import MODNetModel
    from modnet.preprocessing import MODData
    print("Import check passed")
except ImportError as e:
    print(f"ERROR: Could not import MODNet: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("STEP 5 - TRAIN MODNet ON RFE-SELECTED 50 FEATURES")
print("=" * 80)

X_selected = X_clean[selected_features].copy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_selected), 1):
    print(f"\n--- Fold {fold} ---")

    X_tr = X_selected.iloc[train_idx].reset_index(drop=True)
    X_te = X_selected.iloc[test_idx].reset_index(drop=True)
    y_tr = pd.Series(y[train_idx]).reset_index(drop=True)
    y_te = pd.Series(y[test_idx]).reset_index(drop=True)

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

    mae = np.mean(np.abs(y_te.values - y_pred))
    rmse = np.sqrt(np.mean((y_te.values - y_pred) ** 2))
    ss_res = np.sum((y_te.values - y_pred) ** 2)
    ss_tot = np.sum((y_te.values - np.mean(y_te.values)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    fold_results.append({
        'fold': fold,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'y_test': y_te.values,
        'y_pred': y_pred
    })

    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")

print("\n" + "=" * 80)
print("FOLD SUMMARY")
print("=" * 80)

mae_values = [r['MAE'] for r in fold_results]
rmse_values = [r['RMSE'] for r in fold_results]
r2_values = [r['R2'] for r in fold_results]

print(f"MAE:  {np.mean(mae_values):.4f} ± {np.std(mae_values):.4f}")
print(f"RMSE: {np.mean(rmse_values):.4f} ± {np.std(rmse_values):.4f}")
print(f"R²:   {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")

print("\n" + "=" * 80)
print("STEP 6 - FINAL COMPARISON TABLE & PARITY PLOT")
print("=" * 80)

# Collect all predictions
all_y_test = np.concatenate([r['y_test'] for r in fold_results])
all_y_pred = np.concatenate([r['y_pred'] for r in fold_results])

# Create result entry
result_entry = {
    'Model': 'Matminer+ROOST+l-OFM+MVL+ORB (XGBoost RFE) + MODNet',
    'Features': 50,
    'Train_Set_Size': len(y),
    'MAE': np.mean(mae_values),
    'RMSE': np.mean(rmse_values),
    'R2': np.mean(r2_values),
    'MAE_Std': np.std(mae_values),
    'RMSE_Std': np.std(rmse_values),
    'R2_Std': np.std(r2_values),
}

# Load existing results or create new
if OUTPUT_FILE.exists():
    df_results = pd.read_csv(OUTPUT_FILE)
    df_results = pd.concat([df_results, pd.DataFrame([result_entry])], ignore_index=True)
else:
    df_results = pd.DataFrame([result_entry])

df_results.to_csv(OUTPUT_FILE, index=False)
print(f"\nResults saved to: {OUTPUT_FILE}")
print(df_results.tail(3))

# Parity plot
plt.figure(figsize=(10, 8))
plt.scatter(all_y_test, all_y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)

# Add perfect prediction line
min_val = min(all_y_test.min(), all_y_pred.min())
max_val = max(all_y_test.max(), all_y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

plt.xlabel('Experimental zT', fontsize=12)
plt.ylabel('Predicted zT', fontsize=12)
plt.title('Parity Plot: All Features (RFE) + MODNet', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

parity_file = FIGURE_DIR / "parity_plot_all_features_rfe.png"
plt.savefig(parity_file, dpi=300)
print(f"Parity plot saved to: {parity_file}")
plt.close()

print("\n" + "=" * 80)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 80)
print(f"\nFinal Statistics:")
print(f"  Total samples: {len(y)}")
print(f"  Selected features: {len(selected_features)}")
print(f"  Mean CV MAE: {np.mean(mae_values):.4f}")
print(f"  Mean CV RMSE: {np.mean(rmse_values):.4f}")
print(f"  Mean CV R²: {np.mean(r2_values):.4f}")
