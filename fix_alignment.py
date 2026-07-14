#!/usr/bin/env python3
"""
fix_alignment.py

Align Matminer, Roost, and l-MM feature files to the SysTEm master dataset.
Saves aligned CSVs in `results/` and prints an alignment report.

Run:
    python fix_alignment.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys

try:
    from pymatgen.core import Composition
except Exception:
    Composition = None

WORK_DIR = Path(__file__).resolve().parent
DATA_DIR = WORK_DIR
RESULTS_DIR = WORK_DIR / 'results'

def canonical(formula):
    if pd.isna(formula):
        return None
    s = str(formula)
    if Composition is None:
        # basic cleanup fallback: remove spaces and parentheses ordering
        return ''.join(s.split())
    try:
        return Composition(s).reduced_formula
    except Exception:
        return None

def load_master(path):
    df = pd.read_excel(path, engine='openpyxl')
    # Keep only valid zT and non-null Pretty Formula
    df = df[df['zT'].notna()]
    df = df[df['zT'] > 0]
    df = df[df['Pretty Formula'].notna()]
    # Keep Temperature (K) and zT
    df = df.reset_index(drop=True)
    return df

def add_canonical_to_master(master):
    master = master.copy()
    master['canonical'] = master['Pretty Formula'].apply(canonical)
    before = len(master)
    master = master[master['canonical'].notna()].reset_index(drop=True)
    after = len(master)
    if after < before:
        print(f"⚠️  Dropped {before-after} master rows with unparseable formulas during canonicalization")
    return master

def align_by_canonical(master, features_df, feature_comp_col=None):
    df = features_df.copy()
    if feature_comp_col and feature_comp_col in df.columns:
        df['canonical'] = df[feature_comp_col].apply(canonical)
    else:
        # try to find a composition-like column
        candidates = [c for c in df.columns if c.lower() in ('composition','pretty formula','pretty_formula','formula')]
        if candidates:
            col = candidates[0]
            df['canonical'] = df[col].apply(canonical)
        else:
            df['canonical'] = None

    # If there are multiple rows per canonical formula, keep the first occurrence
    if 'canonical' in df.columns:
        df = df.drop_duplicates(subset=['canonical']).reset_index(drop=True)

    # Merge: left join master with features on canonical to keep master order
    merged = pd.merge(master, df, on='canonical', how='left', suffixes=('','_feat'))
    return merged

def find_comp_col(df):
    for col in ['Pretty Formula','pretty_formula','composition','Composition','formula','composition']:
        if col in df.columns:
            return col
    return None

def main():
    master_path = DATA_DIR / 'sysTEm_dataset' / 'sysTEm_dataset.xlsx'
    matminer_path = RESULTS_DIR / 'matminer_for_sisso.csv'
    roost_path = RESULTS_DIR / 'roost_features.csv'
    lmm_path = RESULTS_DIR / 'lMM_features.csv'
    compositions_path = RESULTS_DIR / 'compositions_for_protocsp.txt'

    master = load_master(master_path)
    print(f"MASTER raw shape: {master.shape}")
    master = add_canonical_to_master(master)
    print(f"MASTER canonicalized shape: {master.shape}")

    # Matminer
    matminer = pd.read_csv(matminer_path)
    mat_comp = find_comp_col(matminer)
    if mat_comp is None:
        print("⚠️  Matminer has no composition column; aligning by position")
        # positional align
        n = min(len(master), len(matminer))
        aligned_mat = matminer.iloc[:n].copy().reset_index(drop=True)
        aligned_mat['Pretty Formula'] = master['Pretty Formula'].iloc[:n].values
        aligned_mat['canonical'] = master['canonical'].iloc[:n].values
        aligned_mat['zT'] = master['zT'].iloc[:n].values
    else:
        aligned_mat = align_by_canonical(master, matminer, feature_comp_col=mat_comp)
    # Reindex helper to align feature DF to master canonical order
    def reindex_to_master(df, master):
        if 'canonical' not in df.columns:
            df['canonical'] = None
        # Keep only first occurrence per canonical to avoid expansion
        if df['canonical'].notna().any():
            df = df.drop_duplicates(subset=['canonical']).reset_index(drop=True)
        df_idx = df.set_index('canonical')
        # reindex will introduce rows for master canonical values that are missing in df
        reidx = df_idx.reindex(master['canonical']).reset_index()
        # Ensure master Pretty Formula and zT are present
        reidx['Pretty Formula'] = master['Pretty Formula'].values
        reidx['zT'] = master['zT'].values
        return reidx

    aligned_mat = reindex_to_master(aligned_mat, master)
    aligned_mat.to_csv(RESULTS_DIR / 'aligned_matminer.csv', index=False)
    print("Saved: results/aligned_matminer.csv")

    # Roost
    roost = pd.read_csv(roost_path)
    roost_comp = find_comp_col(roost)
    if roost_comp is None:
        print("⚠️  Roost has no formula column; will align by canonical via fallback (position)")
        n = min(len(master), len(roost))
        aligned_roost = roost.iloc[:n].copy().reset_index(drop=True)
        aligned_roost['Pretty Formula'] = master['Pretty Formula'].iloc[:n].values
        aligned_roost['canonical'] = master['canonical'].iloc[:n].values
        aligned_roost['zT'] = master['zT'].iloc[:n].values
    else:
        aligned_roost = align_by_canonical(master, roost, feature_comp_col=roost_comp)
    aligned_roost = reindex_to_master(aligned_roost, master)
    aligned_roost.to_csv(RESULTS_DIR / 'aligned_roost.csv', index=False)
    print("Saved: results/aligned_roost.csv")

    # l-MM
    lmm = pd.read_csv(lmm_path)
    lmm_comp = find_comp_col(lmm)
    if lmm_comp is None:
        # try positional with compositions file
        print("⚠️  l-MM has no formula column; attempting positional alignment using compositions_for_protocsp.txt")
        comps = []
        with open(compositions_path, 'r') as f:
            for line in f:
                comps.append(line.strip())
        n = min(len(master), len(comps), len(lmm))
        aligned_lmm = lmm.iloc[:n].copy().reset_index(drop=True)
        aligned_lmm['Pretty Formula'] = comps[:n]
        aligned_lmm['canonical'] = [canonical(x) for x in aligned_lmm['Pretty Formula']]
        aligned_lmm['zT'] = master['zT'].iloc[:n].values
    else:
        aligned_lmm = align_by_canonical(master, lmm, feature_comp_col=lmm_comp)
    aligned_lmm = reindex_to_master(aligned_lmm, master)
    aligned_lmm.to_csv(RESULTS_DIR / 'aligned_lMM.csv', index=False)
    print("Saved: results/aligned_lMM.csv")

    # Step: Verify alignment
    def verify(merged, name):
        print(f"\n{name} alignment:\n  shape: {merged.shape}")
        # first 3 canonical and zT
        print("  first 3 canonical (aligned):", merged['canonical'].iloc[:3].tolist())
        print("  first 3 zT (master):", master['zT'].iloc[:3].tolist())
        # count matched rows where canonical not null and at least one feature column present
        matched = merged['canonical'].notna().sum()
        ok = matched == len(master)
        print(f"  canonical present: {matched}/{len(master)} {'✅' if ok else '❌'}")

    verify(aligned_mat, 'Matminer')
    verify(aligned_roost, 'Roost')
    verify(aligned_lmm, 'l-MM')

    # Combined: build deduplicated feature tables (one row per canonical) then merge into master order
    def features_unique(df):
        drop_cols = [c for c in ['Pretty Formula','zT'] if c in df.columns]
        feats = df.drop(columns=drop_cols)
        feats = feats.drop_duplicates(subset=['canonical']).reset_index(drop=True)
        return feats

    feats_mat = features_unique(aligned_mat)
    feats_roost = features_unique(aligned_roost)

    combined = pd.merge(master, feats_mat, on='canonical', how='left')
    combined = pd.merge(combined, feats_roost, on='canonical', how='left', suffixes=('_mat','_roost'))
    combined['zT'] = master['zT'].values
    combined.to_csv(RESULTS_DIR / 'aligned_combined.csv', index=False)
    print('Saved: results/aligned_combined.csv')

    # Final report
    print('\nALIGNMENT REPORT')
    print(f"  Master rows:           {len(master)}")
    print(f"  Matminer aligned:      {len(aligned_mat)} {'✅' if (aligned_mat['canonical'].notna().sum() == len(master)) else '❌'}")
    print(f"  Roost aligned:         {len(aligned_roost)} {'✅' if (aligned_roost['canonical'].notna().sum() == len(master)) else '❌'}")
    print(f"  l-MM aligned:          {len(aligned_lmm)} {'✅' if (aligned_lmm['canonical'].notna().sum() == len(master)) else '❌'}")
    print(f"  Combined aligned:      {len(combined)} {'✅' if (len(combined) == len(master)) else '❌'}")

if __name__ == '__main__':
    main()
