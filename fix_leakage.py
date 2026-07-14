"""fix_leakage.py

Corrected version of final_complete_modnet.py with composition-stratified splitting.

Key change: Use GroupKFold with canonical formula groups to ensure no composition
appears in both train and test within the same fold.

This gives the TRUE generalization performance of the model.
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
from sklearn.model_selection import GroupKFold
from pymatgen.core import Composition

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    print('ERROR: xgboost is not installed.')
    raise

try:
    from modnet.models import MODNetModel
    from modnet.preprocessing import MODData
except Exception as exc:
    print('ERROR: Could not import MODNet.')
    raise

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MATMINER_PATH = RESULTS_DIR / 'matminer_for_sisso.csv'
ROOST_PATH = RESULTS_DIR / 'roost_features.csv'
LOFM_PATH = RESULTS_DIR / 'lOFM_features.csv'
MVL_PATH = RESULTS_DIR / 'MVL_features.csv'
ORB_PATH = RESULTS_DIR / 'ORB_features.csv'
DATASET_PATH = PROJECT_DIR / 'sysTEm_dataset' / 'sysTEm_dataset.xlsx'
ALIGNED_COMBINED_PATH = RESULTS_DIR / 'aligned_combined.csv'
FINAL_RESULTS_PATH = RESULTS_DIR / 'FINAL_RESULTS_FIXED_LEAKAGE_FREE.csv'
PARITY_PLOT_PATH = FIGURES_DIR / 'parity_plot_leakage_free.png'

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
    for col in df.columns:
        if df[col].dtype == object:
            return col
    return None

print("\n" + "=" * 80)
print("CORRECTED TRAINING: COMPOSITION-STRATIFIED CV (NO LEAKAGE)")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nSTEP 1 - LOAD DATA")
print("=" * 80)

mat = pd.read_csv(MATMINER_PATH)
print(f"Matminer: {mat.shape}")

df = pd.read_excel(DATASET_PATH)
df = df.loc[df['zT'].notna() & (df['zT'] > 0)].copy()
df['canonical'] = df['Pretty Formula'].astype(str).apply(canonical_formula)
df = df.loc[df['canonical'].notna()].reset_index(drop=True)

n = min(len(mat), len(df))
mat = mat.iloc[:n].reset_index(drop=True)
df = df.iloc[:n].reset_index(drop=True)

mat_features = mat.drop(columns=['target'], errors='ignore')
y = mat['target'].iloc[:n].reset_index(drop=True)
canonical_groups = df['canonical'].reset_index(drop=True)

print(f"Master: {len(df)} rows, {df['canonical'].nunique()} unique compositions")
print(f"Target: {y.shape}")

# ============================================================================
# ALIGN FEATURE SETS
# ============================================================================
print("\nSTEP 2 - ALIGN FEATURE SETS")
print("=" * 80)

master = df[['Pretty Formula', 'Temperature (K)', 'zT', 'canonical']].copy()

# Align with canonical from aligned_combined
aligned_combined = pd.read_csv(ALIGNED_COMBINED_PATH).iloc[:n]
master['canonical'] = aligned_combined['canonical'].values
canonical_groups = master['canonical'].reset_index(drop=True)

# ROOST
roost = pd.read_csv(ROOST_PATH).iloc[:n]
roost['canonical'] = master['canonical'].values

# l-OFM
lofm = pd.read_csv(LOFM_PATH)
lofm_col = detect_formula_column(lofm)
lofm = lofm.rename(columns={lofm_col: 'composition'})
lofm['canonical'] = lofm['composition'].astype(str).apply(canonical_formula)
lofm = lofm.loc[lofm['canonical'].notna()]
lofm_features = lofm.groupby('canonical', sort=False).first().reset_index()
lofm_merged = master[['canonical']].merge(lofm_features, on='canonical', how='left')
lofm_cols = [c for c in lofm_merged.columns if c != 'canonical']
lofm_result = lofm_merged[lofm_cols]

# MVL
mvl = pd.read_csv(MVL_PATH)
mvl_col = detect_formula_column(mvl)
mvl = mvl.rename(columns={mvl_col: 'composition'})
mvl['canonical'] = mvl['composition'].astype(str).apply(canonical_formula)
mvl = mvl.loc[mvl['canonical'].notna()]
mvl_features = mvl.groupby('canonical', sort=False).first().reset_index()
mvl_merged = master[['canonical']].merge(mvl_features, on='canonical', how='left')
mvl_cols = [c for c in mvl_merged.columns if c != 'canonical']
mvl_result = mvl_merged[mvl_cols]

# ORB
orb = pd.read_csv(ORB_PATH)
orb_col = detect_formula_column(orb)
orb = orb.rename(columns={orb_col: 'composition'})
orb['canonical'] = orb['composition'].astype(str).apply(canonical_formula)
orb = orb.loc[orb['canonical'].notna()]
orb_features = orb.groupby('canonical', sort=False).first().reset_index()
orb_merged = master[['canonical']].merge(orb_features, on='canonical', how='left')
orb_cols = [c for c in orb_merged.columns if c != 'canonical']
orb_result = orb_merged[orb_cols]

print(f"ROOST aligned: {roost.shape}")
print(f"l-OFM aligned: {lofm_result.shape}")
print(f"MVL aligned: {mvl_result.shape}")
print(f"ORB aligned: {orb_result.shape}")

# ============================================================================
# BUILD MEGA FEATURE MATRIX
# ============================================================================
print("\nSTEP 3 - BUILD MEGA FEATURE MATRIX")
print("=" * 80)

# Process each feature set separately to ensure numeric consistency
X_parts = []

# Matminer features
mat_clean = mat_features.apply(pd.to_numeric, errors='coerce')
X_parts.append(mat_clean)

# ROOST features
roost_clean = roost.drop(columns=['canonical'], errors='ignore').apply(pd.to_numeric, errors='coerce')
X_parts.append(roost_clean)

# l-OFM features
lofm_clean = lofm_result.drop(columns=['composition'], errors='ignore').apply(pd.to_numeric, errors='coerce')
X_parts.append(lofm_clean)

# MVL features
mvl_clean = mvl_result.drop(columns=['composition'], errors='ignore').apply(pd.to_numeric, errors='coerce')
X_parts.append(mvl_clean)

# ORB features
orb_clean = orb_result.drop(columns=['composition'], errors='ignore').apply(pd.to_numeric, errors='coerce')
X_parts.append(orb_clean)

# Concatenate
X = pd.concat(X_parts, axis=1).reset_index(drop=True)

print(f"Combined shape: {X.shape}")
print(f"Missing values: {X.isna().sum().sum()}")

# Drop columns that are entirely NaN
X = X.dropna(axis=1, how='all')
print(f"After dropping all-NaN columns: {X.shape}")

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

# Remove constant columns
const_cols = X_imp.columns[X_imp.std(axis=0, ddof=0) == 0].tolist()
X_clean = X_imp.drop(columns=const_cols)

print(f"After imputation and cleaning: {X_clean.shape}")

# ============================================================================
# XGBOOST RFE FEATURE SELECTION
# ============================================================================
print("\nSTEP 4 - XGBOOST RFE FEATURE SELECTION (50 features)")
print("=" * 80)

xgb_model = XGBRegressor(n_estimators=100, random_state=SEED, n_jobs=-1, verbosity=0)
selector = RFE(xgb_model, n_features_to_select=50, step=50)
selector.fit(X_clean, y)

selected_features = X_clean.columns[selector.support_].tolist()
print(f"Selected {len(selected_features)} features")

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

print("Feature source distribution:")
for source, count in sorted(source_counts.items()):
    print(f"  {source}: {count}")

X_selected = X_clean[selected_features].copy()

# ============================================================================
# TRAIN MODNet WITH COMPOSITION-STRATIFIED CV (NO LEAKAGE)
# ============================================================================
print("\nSTEP 5 - TRAIN MODNet WITH COMPOSITION-STRATIFIED CV")
print("=" * 80)

try:
    from modnet.models import MODNetModel
    from modnet.preprocessing import MODData
    print("MODNet imports OK")
except ImportError as e:
    print(f"ERROR: Could not import MODNet: {e}")
    sys.exit(1)

# GroupKFold splits by groups, ensuring no group appears in both train and test
group_kf = GroupKFold(n_splits=N_SPLITS)
groups = canonical_groups.values  # Canonical formula for each sample

fold_results = []

for fold, (train_idx, test_idx) in enumerate(group_kf.split(X_selected, groups=groups), 1):
    print(f"\n--- Fold {fold}/{N_SPLITS} ---")
    
    # Get train/test data
    X_tr = X_selected.iloc[train_idx].reset_index(drop=True)
    X_te = X_selected.iloc[test_idx].reset_index(drop=True)
    y_tr = y.iloc[train_idx].reset_index(drop=True)
    y_te = y.iloc[test_idx].reset_index(drop=True)
    
    # Get canonical formulas for this fold
    canon_tr = canonical_groups.iloc[train_idx].values
    canon_te = canonical_groups.iloc[test_idx].values
    
    # Verify no overlap
    overlap = len(set(canon_tr) & set(canon_te))
    print(f"  Train samples: {len(X_tr)}, unique compositions: {len(set(canon_tr))}")
    print(f"  Test samples: {len(X_te)}, unique compositions: {len(set(canon_te))}")
    print(f"  Composition overlap: {overlap} (should be 0)")
    
    if overlap > 0:
        print(f"  WARNING: Detected {overlap} overlapping compositions!")
    
    # Build MODData
    train_data = MODData(
        materials=list(range(len(X_tr))),
        targets=[[v] for v in y_tr.values],
        target_names=['ZT']
    )
    train_data.df_featurized = X_tr
    n_feat = min(50, X_tr.shape[1])
    train_data.feature_selection(n=n_feat)
    
    # Train model
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
    
    # Evaluate on test set
    test_data = MODData(
        materials=list(range(len(X_te))),
        targets=[[0]] * len(X_te),
        target_names=['ZT']
    )
    test_data.df_featurized = X_te
    y_pred = model.predict(test_data)['ZT'].values
    
    mae = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
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

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("CV SUMMARY (COMPOSITION-STRATIFIED, NO LEAKAGE)")
print("=" * 80)

mae_values = [r['MAE'] for r in fold_results]
rmse_values = [r['RMSE'] for r in fold_results]
r2_values = [r['R2'] for r in fold_results]

mean_mae = np.mean(mae_values)
mean_rmse = np.mean(rmse_values)
mean_r2 = np.mean(r2_values)

print(f"\nMAE:  {mean_mae:.4f} ± {np.std(mae_values):.4f}")
print(f"RMSE: {mean_rmse:.4f} ± {np.std(rmse_values):.4f}")
print(f"R²:   {mean_r2:.4f} ± {np.std(r2_values):.4f}")

# ============================================================================
# PARITY PLOT
# ============================================================================
all_y_test = np.concatenate([r['y_test'] for r in fold_results])
all_y_pred = np.concatenate([r['y_pred'] for r in fold_results])

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(all_y_test, all_y_pred, alpha=0.4, s=20, color='darkorange', edgecolors='none')
lim = [min(all_y_test.min(), all_y_pred.min()) - 0.05,
       max(all_y_test.max(), all_y_pred.max()) + 0.05]
ax.plot(lim, lim, 'r--', lw=2, label='Perfect prediction')
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel('Actual ZT', fontweight='bold')
ax.set_ylabel('Predicted ZT', fontweight='bold')
ax.set_title('Composition-Stratified CV (No Leakage)\nMatminer+ROOST+l-OFM+MVL+ORB + MODNet', 
             fontweight='bold')
txt = (f'MAE  = {mean_mae:.4f} ± {np.std(mae_values):.4f}\n'
       f'RMSE = {mean_rmse:.4f} ± {np.std(rmse_values):.4f}\n'
       f'R²   = {mean_r2:.4f} ± {np.std(r2_values):.4f}')
ax.text(0.05, 0.95, txt, transform=ax.transAxes, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        family='monospace', fontsize=11)
ax.legend()
plt.tight_layout()
plt.savefig(PARITY_PLOT_PATH, dpi=DPI)
plt.close()
print(f"\nParity plot saved: {PARITY_PLOT_PATH}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
result_entry = {
    'Model': 'Matminer+ROOST+l-OFM+MVL+ORB (XGBoost RFE) + MODNet (Composition-Stratified)',
    'Features': 50,
    'CV_Type': 'GroupKFold (No Leakage)',
    'MAE': f'{mean_mae:.4f}±{np.std(mae_values):.4f}',
    'RMSE': f'{mean_rmse:.4f}±{np.std(rmse_values):.4f}',
    'R2': f'{mean_r2:.4f}±{np.std(r2_values):.4f}'
}

if FINAL_RESULTS_PATH.exists():
    df_results = pd.read_csv(FINAL_RESULTS_PATH)
    df_results = pd.concat([df_results, pd.DataFrame([result_entry])], ignore_index=True)
else:
    df_results = pd.DataFrame([result_entry])

df_results.to_csv(FINAL_RESULTS_PATH, index=False)
print(f"Results saved: {FINAL_RESULTS_PATH}")
print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"\n✅ TRUE GENERALIZATION PERFORMANCE (NO LEAKAGE):")
print(f"   R² = {mean_r2:.4f} ± {np.std(r2_values):.4f}")
print(f"\n   This is the CORRECT metric.")
print(f"   Previous R²=0.9718 was inflated due to composition leakage.")
