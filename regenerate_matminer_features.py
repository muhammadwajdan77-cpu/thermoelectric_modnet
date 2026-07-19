#!/usr/bin/env python3
"""Regenerate Matminer composition features from SysTEm dataset with provenance tracking."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "sysTEm_dataset" / "sysTEm_dataset.xlsx"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OLD_MATMINER_PATH = RESULTS_DIR / "matminer_for_sisso.csv"
NEW_MATMINER_PATH = RESULTS_DIR / "matminer_for_sisso_v2.csv"
OUTPUT_FOLDS_PATH = RESULTS_DIR / "MATMINER_GROUPKFOLD_FOLDS_V2.csv"


def canonical_formula(formula: object) -> str:
    if pd.isna(formula):
        return ""
    text = str(formula).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    try:
        return Composition(text).reduced_formula
    except Exception:
        return text.replace(" ", "")


def load_sys_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Could not find SysTEm dataset at {DATASET_PATH}")
    df = pd.read_excel(DATASET_PATH)
    required = ["Pretty Formula", "zT", "Temperature (K)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"SysTEm dataset missing required columns: {missing}")
    return df


def generate_matminer_features() -> pd.DataFrame:
    sys_df = load_sys_dataset()
    sys_df = sys_df[["Pretty Formula", "zT", "Temperature (K)"]].copy()
    sys_df = sys_df.dropna(subset=["zT"]).copy()
    sys_df = sys_df[sys_df["zT"] > 0].copy()
    sys_df = sys_df.reset_index(drop=True)
    sys_df = sys_df.reset_index().rename(columns={"index": "sys_df_original_index"})

    featurizer = ElementProperty.from_preset("magpie")
    rows: list[dict] = []
    failed = 0
    valid = 0

    for _, row in sys_df.iterrows():
        formula = str(row["Pretty Formula"] or "").strip()
        if not formula or formula.lower() in {"nan", "none"}:
            continue
        try:
            comp = Composition(formula)
            features = np.asarray(featurizer.featurize(comp), dtype=float)
            if np.isnan(features).any():
                failed += 1
                continue
            record = {
                "formula": formula,
                "canonical_formula": canonical_formula(formula),
                "sys_df_original_index": int(row["sys_df_original_index"]),
                "Temperature_K": float(row["Temperature (K)"] or 300.0),
                "target": float(row["zT"]),
            }
            record.update({name: float(val) for name, val in zip(featurizer.feature_labels(), features)})
            rows.append(record)
            valid += 1
        except Exception:
            failed += 1

    if not rows:
        raise RuntimeError("No Matminer features could be generated")

    feature_df = pd.DataFrame(rows)
    feature_df.to_csv(NEW_MATMINER_PATH, index=False)
    print(f"Generated {len(feature_df)} rows -> {NEW_MATMINER_PATH}")
    print(f"Source valid SysTEm rows after cleaning: {len(sys_df)}")
    print(f"Skipped invalid formulas / featurization failures: {failed}")
    return feature_df


def check_integrity(mat_df: pd.DataFrame) -> None:
    feature_cols = [
        c for c in mat_df.columns
        if c not in {"formula", "canonical_formula", "sys_df_original_index", "Temperature_K", "target"}
        and pd.api.types.is_numeric_dtype(mat_df[c])
    ]
    if not feature_cols:
        raise RuntimeError("No feature columns found in generated Matminer file")

    X = mat_df[feature_cols].copy()
    imputer = SimpleImputer(strategy="mean")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_imp = X_imp.loc[:, X_imp.nunique(dropna=True) > 1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)
    nearest_other_dist = distances[:, 1]
    nearest_other_idx = indices[:, 1]

    near_dup_mask = nearest_other_dist < 0.01
    near_dup_rows = np.where(near_dup_mask)[0]
    print(f"Near-duplicate partner count (dist < 0.01): {len(near_dup_rows)} / {len(mat_df)}")

    if len(near_dup_rows) == 0:
        print("Integrity check passed: no near-duplicate feature pairs found.")
        return

    flagged = []
    for i in near_dup_rows:
        j = int(nearest_other_idx[i])
        if j == i:
            continue
        dist = float(nearest_other_dist[i])
        if dist < 0.01:
            flagged.append((i, j, dist))

    print("First 20 flagged pairs:")
    for i, j, dist in flagged[:20]:
        print(f"  {i} <-> {j} dist={dist:.6f}")
        print(f"    formula i: {mat_df.iloc[i]['formula']}")
        print(f"    formula j: {mat_df.iloc[j]['formula']}")
        print(f"    temp i: {mat_df.iloc[i]['Temperature_K']}")
        print(f"    temp j: {mat_df.iloc[j]['Temperature_K']}")
        print(f"    target i: {mat_df.iloc[i]['target']}")
        print(f"    target j: {mat_df.iloc[j]['target']}")

    different_formula_pairs = []
    for i, j, dist in flagged:
        key_i = canonical_formula(mat_df.iloc[i]['formula'])
        key_j = canonical_formula(mat_df.iloc[j]['formula'])
        if key_i and key_j and key_i != key_j:
            different_formula_pairs.append((i, j, dist, str(key_i), str(key_j)))

    if different_formula_pairs:
        print("Integrity check FAILED: different-formula near-duplicates detected.")
        for item in different_formula_pairs[:10]:
            i, j, dist, f1, f2 = item
            print(f"  {i} <-> {j} dist={dist:.6f} formulas={f1!r} vs {f2!r}")
        sys.exit(1)

    print("Integrity check passed: flagged pairs are either same-formula duplicates or expected composition-only repeats.")


def compare_with_old_file(mat_df: pd.DataFrame) -> None:
    if not OLD_MATMINER_PATH.exists():
        print("Old Matminer file not found; skipping comparison.")
        return

    old_df = pd.read_csv(OLD_MATMINER_PATH)
    print("\nOld vs new Matminer file summary")
    print(f"  old rows: {len(old_df)}")
    print(f"  new rows: {len(mat_df)}")
    if "target" in old_df.columns and "target" in mat_df.columns:
        print(f"  old target min/max/mean: {old_df['target'].min():.4f}/{old_df['target'].max():.4f}/{old_df['target'].mean():.4f}")
        print(f"  new target min/max/mean: {mat_df['target'].min():.4f}/{mat_df['target'].max():.4f}/{mat_df['target'].mean():.4f}")


def main() -> None:
    feature_df = generate_matminer_features()
    check_integrity(feature_df)
    compare_with_old_file(feature_df)
    print("\nRegeneration completed successfully.")


if __name__ == "__main__":
    main()
