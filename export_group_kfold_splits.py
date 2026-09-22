#!/usr/bin/env python3
"""Export exact-formula and parent-family GroupKFold split membership for the final 965-formula benchmark dataset."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

import run_grouped_modnet
import task3_part2_n965 as task3

ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "results" / "GROUP_KFOLD_SPLITS_READABLE.csv"


def build_split_records(row_level: pd.DataFrame, task_name: str, grouping_name: str, grouping_values: pd.Series, splitter) -> list[dict]:
    records: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(row_level, row_level["target"], groups=grouping_values.to_numpy()), start=1):
        train_set = set(train_idx.tolist())
        for row_id in range(len(row_level)):
            split = "train" if row_id in train_set else "test"
            records.append(
                {
                    "task": task_name,
                    "grouping": grouping_name,
                    "fold": fold,
                    "row_id": int(row_id),
                    "value": str(row_level.iloc[row_id][grouping_name] if grouping_name == "canonical_formula" else row_level.iloc[row_id]["_parent_family"]),
                    "split": split,
                }
            )
    return records


def main() -> None:
    row_level, _, _, _, _ = task3.load_data()
    row_level = row_level.copy().reset_index(drop=True)
    row_level["canonical_formula"] = row_level["canonical_formula"].astype(str).str.strip()
    row_level["_parent_family"] = row_level["formula"].map(lambda value: run_grouped_modnet.parent_family(value, threshold=0.05))

    exact_splitter = GroupKFold(n_splits=5, shuffle=True, random_state=task3.SEED)
    exact_records = build_split_records(
        row_level,
        task_name="Task 3 exact-formula",
        grouping_name="canonical_formula",
        grouping_values=row_level["canonical_formula"],
        splitter=exact_splitter,
    )

    parent_splitter = GroupKFold(n_splits=5)
    parent_records = build_split_records(
        row_level,
        task_name="Task 3 parent-family",
        grouping_name="parent_family",
        grouping_values=row_level["_parent_family"],
        splitter=parent_splitter,
    )

    all_records = exact_records + parent_records
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "grouping", "fold", "row_id", "value", "split"])
        writer.writeheader()
        writer.writerows(all_records)

    print(f"Wrote {len(all_records)} split rows to {OUTPUT_PATH}")
    print(f"Exact-formula rows: {len(exact_records)}")
    print(f"Parent-family rows: {len(parent_records)}")


if __name__ == "__main__":
    main()
