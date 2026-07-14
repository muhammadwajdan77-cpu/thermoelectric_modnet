#!/usr/bin/env python3
"""
Create FINAL_RESULTS.csv combining all 4 model results.
- Reads results_complete.csv for 3 completed models (Matminer, Roost, Combined)
- Monitors for lmm_modnet.py output to add l-MM results
"""

import pandas as pd
import os
from pathlib import Path

WORK_DIR = Path("/home/wajdan/Documents/ZT/thermoelectric_modnet")
os.chdir(WORK_DIR)

# Read existing results
results_df = pd.read_csv("results/results_complete.csv")

# Extract only Mean±Std rows
summary_rows = results_df[results_df['Fold'] == 'Mean±Std'].copy()
summary_rows = summary_rows.reset_index(drop=True)

print("=" * 80)
print("CREATING FINAL_RESULTS.csv")
print("=" * 80)

print("\n✅ Existing Models (from results_complete.csv):")
for idx, row in summary_rows.iterrows():
    model = row['Model']
    mae = row['MAE']
    rmse = row['RMSE']
    r2 = row['R2']
    print(f"  • {model}")
    print(f"    MAE={mae}, RMSE={rmse}, R²={r2}")

# Add column for future l-MM results
final_data = []

for idx, row in summary_rows.iterrows():
    final_data.append({
        'Model': row['Model'],
        'MAE': row['MAE'],
        'RMSE': row['RMSE'],
        'R²': row['R2']
    })

# Create DataFrame
final_df = pd.DataFrame(final_data)

# Save to CSV
final_df.to_csv("results/FINAL_RESULTS.csv", index=False)
print(f"\n✅ Saved: results/FINAL_RESULTS.csv (3 models)")
print("\nCurrent FINAL_RESULTS.csv:")
print(final_df.to_string(index=False))

print("\n⏳ Waiting for l-MM model results...")
print("   Once lmm_modnet.py completes, will add l-MM row to FINAL_RESULTS.csv")
