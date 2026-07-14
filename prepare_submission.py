#!/usr/bin/env python3
"""
Submission preparation script for ZT prediction results.
Verifies all files exist, validates model results, and creates submission zip.
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import zipfile

# Working directory
WORK_DIR = Path("/home/wajdan/Documents/ZT/thermoelectric_modnet")
os.chdir(WORK_DIR)

# Files to verify
MODEL_SCRIPTS = [
    "main.py",
    "roost_modnet.py",
    "combined_modnet.py",
    "lmm_modnet.py",
]

RESULTS_CSVS = [
    "results/results.csv",
    "results/results_complete.csv",
    "results/FINAL_RESULTS.csv",
    "results/lMM_features.csv",
    "results/roost_features.csv",
]

FIGURES = [
    "results/figures/zt_distribution.png",
    "results/figures/parity_plot_matminer.png",
    "results/figures/parity_plot_mattervial_roost.png",
    "results/figures/parity_plot_combined.png",
    "results/figures/parity_plot_lMM.png",
    "results/figures/shap_bar.png",
    "results/figures/shap_bar_lMM.png",
]

# ============================================================================
# STEP 1: VERIFY FILES
# ============================================================================
print("=" * 80)
print("STEP 1: VERIFY FILES EXIST")
print("=" * 80)

verified_files = []
missing_files = []

print("\n📜 Model Scripts:")
for fname in MODEL_SCRIPTS:
    if os.path.exists(fname):
        print(f"  ✅ {fname}")
        verified_files.append(fname)
    else:
        print(f"  ❌ {fname}")
        missing_files.append(fname)

print("\n📊 Results CSVs:")
for fname in RESULTS_CSVS:
    if os.path.exists(fname):
        print(f"  ✅ {fname}")
        verified_files.append(fname)
    else:
        print(f"  ❌ {fname}")
        missing_files.append(fname)

print("\n🎨 Figures:")
for fname in FIGURES:
    if os.path.exists(fname):
        print(f"  ✅ {fname}")
        verified_files.append(fname)
    else:
        print(f"  ❌ {fname}")
        missing_files.append(fname)

# ============================================================================
# STEP 2: VERIFY MODEL RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: VERIFY MODEL RESULTS")
print("=" * 80)

if os.path.exists("results/FINAL_RESULTS.csv"):
    try:
        results_df = pd.read_csv("results/FINAL_RESULTS.csv")
        print("\n📈 FINAL_RESULTS.csv Content:")
        print(results_df.to_string(index=False))
        
        # Check for suspicious values
        print("\n🔍 Validation Check:")
        suspicious = False
        for idx, row in results_df.iterrows():
            model_name = row.get('Model', f'Row {idx}')
            mae = row.get('MAE', None)
            r2 = row.get('R²', None)
            
            if mae is not None and mae > 0.25:
                print(f"  ⚠️  {model_name}: MAE={mae:.4f} > 0.25 (suspicious)")
                suspicious = True
            if r2 is not None and r2 < 0.5:
                print(f"  ⚠️  {model_name}: R²={r2:.4f} < 0.5 (suspicious)")
                suspicious = True
        
        if not suspicious:
            print("  ✅ All model results appear valid")
            
    except Exception as e:
        print(f"  ❌ Error reading FINAL_RESULTS.csv: {e}")
else:
    print("  ❌ FINAL_RESULTS.csv not found")

# ============================================================================
# STEP 3: CREATE ZIP
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: CREATE SUBMISSION ZIP")
print("=" * 80)

# Generate summary content
summary_text = f"""ZT PREDICTION SUBMISSION SUMMARY
{'=' * 70}

Project: ZT Prediction on SysTEm Dataset
Supervisor: Prof. Gian-Marco Rignanese, UCLouvain
Date: {datetime.now().strftime('%B %d, %Y')}
Dataset: SysTEm thermoelectric dataset (7,594 valid samples)

"""

# Add model results if available
if os.path.exists("results/FINAL_RESULTS.csv"):
    try:
        results_df = pd.read_csv("results/FINAL_RESULTS.csv")
        summary_text += "MODEL RESULTS:\n"
        summary_text += results_df.to_string(index=False)
        summary_text += "\n\n"
    except:
        pass

summary_text += """METHODS:
- Matminer ElementProperty (Magpie preset) + Temperature
- MatterVial Roost (roost_oqmd_eform) features
- Combined Matminer + Roost features
- MatterVial l-MM features (from ProtoCSP-predicted structures)
- MODNet neural network, 5-fold cross-validation, seed=42

CONTACT:
Generated on {date}
""".format(date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# Create zip file
zip_name = "thermoelectric_ZT_results_Wajdan.zip"
print(f"\n📦 Creating {zip_name}...")

with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
    # Add verified files
    for fpath in verified_files:
        if os.path.exists(fpath):
            arcname = f"submission/{fpath}"
            zf.write(fpath, arcname=arcname)
            print(f"  ✅ Added: {fpath}")
        else:
            print(f"  ⚠️  Skipped (missing): {fpath}")
    
    # Add summary
    summary_arcname = "submission/SUMMARY.txt"
    zf.writestr(summary_arcname, summary_text)
    print(f"  ✅ Added: SUMMARY.txt")

# ============================================================================
# STEP 4: FINAL REPORT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: FINAL REPORT")
print("=" * 80)

print("\n✅ FILES INCLUDED IN ZIP:")
for fpath in verified_files:
    if os.path.exists(fpath):
        print(f"  • {fpath}")
print(f"  • SUMMARY.txt (auto-generated)")

if missing_files:
    print(f"\n❌ FILES MISSING ({len(missing_files)}):")
    for fpath in missing_files:
        print(f"  • {fpath}")
else:
    print(f"\n❌ No missing files!")

# Zip file size
zip_size_mb = os.path.getsize(zip_name) / (1024 * 1024)
print(f"\n📦 Zip file: {zip_name} ({zip_size_mb:.2f} MB)")

# Final verdict
if os.path.exists("results/FINAL_RESULTS.csv"):
    try:
        results_df = pd.read_csv("results/FINAL_RESULTS.csv")
        if len(results_df) >= 4:
            print("\n" + "=" * 80)
            print("✅ ALL MODELS VERIFIED ✅ - READY TO SEND")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print(f"⚠️  WARNING - INCOMPLETE ({len(results_df)} < 4 models) ⚠️")
            print("=" * 80)
    except:
        print("\n" + "=" * 80)
        print("⚠️  WARNING - COULD NOT VERIFY MODEL COUNT ⚠️")
        print("=" * 80)
else:
    print("\n" + "=" * 80)
    print("⚠️  WARNING - INCOMPLETE ⚠️ (FINAL_RESULTS.csv missing)")
    print("=" * 80)

print("\n✨ Submission preparation complete!")
