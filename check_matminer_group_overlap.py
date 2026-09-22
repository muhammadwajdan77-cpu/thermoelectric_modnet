#!/usr/bin/env python3
"""Compare old Matminer GroupKFold grouping with corrected V2 canonical_formula grouping."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
SYS_PATH = ROOT / "sysTEm_dataset" / "sysTEm_dataset.xlsx"
OLD_MAT_PATH = RESULTS_DIR / "matminer_for_sisso.csv"
NEW_MAT_PATH = RESULTS_DIR / "matminer_for_sisso_v2.csv"


def canonical_formula(formula: object) -> str:
    if pd.isna(formula):
        raise ValueError("Empty formula cannot be canonicalized")
    return Composition(str(formula)).reduced_formula


def make_group_labels(formulas: pd.Series) -> np.ndarray:
    labels = []
    for formula in formulas.astype(str):
        text = formula.strip()
        if not text or text.lower() in {"nan", "none"}:
            labels.append("nan")
            continue
        try:
            labels.append(canonical_formula(text))
        except Exception:
            labels.append(text)
    return np.array(labels, dtype=object)


def count_overlap(groups: np.ndarray, canonical: np.ndarray, n_splits: int = 5) -> tuple[list[tuple[int, int]], int]:
    splits = GroupKFold(n_splits=n_splits).split(np.zeros(len(canonical)), canonical, groups=groups)
    fold_counts = []
    total = 0
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_set = set(canonical[train_idx])
        test_set = set(canonical[test_idx])
        overlap = train_set & test_set
        fold_counts.append((fold, len(overlap)))
        total += len(overlap)
    return fold_counts, total


def inspect_v2():
    mat = pd.read_csv(NEW_MAT_PATH)
    if "canonical_formula" not in mat.columns:
        raise ValueError("New Matminer file lacks canonical_formula column")

    groups = mat["canonical_formula"].astype(str).values
    print("V2 grouping")
    print("  rows:", len(mat))
    print("  unique canonical_formula:", len(np.unique(groups)))
    fold_counts, total = count_overlap(groups, groups)
    for fold, count in fold_counts:
        print(f"  fold {fold}: overlap count {count}")
    print("  total overlap count:", total)
    print("  max group size:", Counter(groups).most_common(1)[0][1])
    print("  groups with size > 1:", sum(1 for c in Counter(groups).values() if c > 1))


def inspect_old():
    sys_df = pd.read_excel(SYS_PATH)
    if OLD_MAT_PATH.exists():
        mat = pd.read_csv(OLD_MAT_PATH)
        n_rows = len(mat)
        source = "legacy CSV"
    elif NEW_MAT_PATH.exists():
        new_mat = pd.read_csv(NEW_MAT_PATH)
        n_rows = len(new_mat)
        source = "V2 CSV fallback"
    else:
        raise FileNotFoundError(f"Neither old nor new Matminer source CSV was found.")

    old_groups = make_group_labels(sys_df.iloc[:n_rows]["Pretty Formula"])
    sequential_groups = old_groups

    new_mat = pd.read_csv(NEW_MAT_PATH)
    canonical = new_mat["canonical_formula"].astype(str).values

    by_index_groups = None
    if "sys_df_original_index" in new_mat.columns:
        idxs = new_mat["sys_df_original_index"].astype(int)
        if idxs.max() < len(sys_df):
            by_index_groups = make_group_labels(sys_df.loc[idxs, "Pretty Formula"])

    print("Old grouping")
    print(f"  source: {source}")
    print("  old-group rows used:", n_rows)
    print("  old group unique:", len(np.unique(old_groups)))
    print("  canonical_formula unique (V2):", len(np.unique(canonical)))
    mismatch = sum(1 for a, b in zip(old_groups, canonical) if a != b)
    print("  rows where old group label != V2 canonical_formula:", mismatch)

    print("  sequential old-group overlap")
    fold_counts, total = count_overlap(sequential_groups, canonical)
    for fold, count in fold_counts:
        print(f"    fold {fold}: overlap count {count}")
    print("    total overlap count:", total)

    if by_index_groups is not None:
        print("  by-index old-group overlap")
        fold_counts, total = count_overlap(by_index_groups, canonical)
        for fold, count in fold_counts:
            print(f"    fold {fold}: overlap count {count}")
        print("    total overlap count:", total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Matminer GroupKFold overlap for old and V2 pipelines")
    parser.add_argument("--v2", action="store_true", help="Only inspect the V2 grouping")
    parser.add_argument("--old", action="store_true", help="Only inspect the old grouping")
    args = parser.parse_args()

    if not args.v2 and not args.old:
        args.v2 = True
        args.old = True

    if args.v2:
        inspect_v2()
        print()
    if args.old:
        inspect_old()


if __name__ == "__main__":
    main()
