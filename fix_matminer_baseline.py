#!/usr/bin/env python3
"""Reproduce the deduplicated Matminer + MODNet GroupKFold baseline."""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from modnet.models import MODNetModel
from modnet.preprocessing import MODData
from pymatgen.core import Composition
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
INPUT_PATH = RESULTS_DIR / "matminer_for_sisso_v2.csv"
OUTPUT_PATH = RESULTS_DIR / "MATMINER_GROUPKFOLD_FOLDS_V3.csv"
TARGET = "target"
N_SPLITS = 5


def canonical_formula(value: object) -> str:
    if pd.isna(value):
        raise ValueError("Empty formula cannot be canonicalized")
    return Composition(str(value)).reduced_formula


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    print(f"raw shape: {df.shape}")
    required = {"formula", TARGET}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["_canonical_formula"] = df["formula"].map(canonical_formula)
    raw_unique = df["formula"].nunique(dropna=True)
    canonical_unique = df["_canonical_formula"].nunique(dropna=True)
    before = len(df)
    deduped = df.drop_duplicates(subset=["_canonical_formula"], keep="first").reset_index(drop=True)
    print(
        f"dropped {before - len(deduped)} duplicate feature vectors "
        f"({before} -> {len(deduped)} rows)"
    )
    print(f"Unique raw formula strings: {raw_unique}")
    print(f"Unique canonical formulas: {canonical_unique}")

    feature_columns = deduped.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns.remove(TARGET)
    if not feature_columns:
        raise ValueError("No numeric Matminer feature columns found")

    X = deduped[feature_columns]
    y = deduped[TARGET].astype(float)
    groups = deduped["_canonical_formula"].to_numpy(dtype=object)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        GroupKFold(n_splits=N_SPLITS).split(X, y, groups=groups), start=1
    ):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise AssertionError(f"Fold {fold} has {len(overlap)} overlapping groups")
        print(f"Fold {fold}/{N_SPLITS}: leak-check passed (overlap=0)")

        imputer = SimpleImputer(strategy="mean")
        X_train = pd.DataFrame(
            imputer.fit_transform(X.iloc[train_idx]),
            columns=feature_columns,
        )
        X_test = pd.DataFrame(
            imputer.transform(X.iloc[test_idx]),
            columns=feature_columns,
        )
        constant_columns = X_train.columns[X_train.nunique(dropna=True) <= 1]
        X_train = X_train.drop(columns=constant_columns)
        X_test = X_test.drop(columns=constant_columns)
        n_features = min(50, X_train.shape[1])

        if fold == 1:
            sample_count = min(5, len(test_idx))
            sample_positions = np.random.RandomState(42).choice(
                len(test_idx), size=sample_count, replace=False
            )
            similarities = cosine_similarity(X_test.iloc[sample_positions], X_train)
            print("Fold 0 near-duplicate check (cosine similarity > 0.999):")
            for position, row_similarities in zip(sample_positions, similarities):
                matches = np.flatnonzero(row_similarities > 0.999)
                max_similarity = float(row_similarities.max())
                formula = deduped.iloc[test_idx[position]]["formula"]
                print(
                    f"  formula={formula} max_train_cosine={max_similarity:.12f} "
                    f"near_duplicate_count={len(matches)}"
                )

        train_data = MODData(
            materials=list(range(len(train_idx))),
            targets=[[float(value)] for value in y.iloc[train_idx]],
            target_names=[TARGET],
        )
        train_data.df_featurized = X_train
        train_data.feature_selection(n=n_features)
        model = MODNetModel(
            [[[TARGET]]],
            weights={TARGET: 1},
            n_feat=n_features,
        )
        model.fit(
            train_data,
            val_fraction=0.1,
            lr=0.001,
            batch_size=64,
            loss="mae",
            epochs=100,
            verbose=0,
        )

        test_data = MODData(
            materials=list(range(len(test_idx))),
            targets=[[0.0]] * len(test_idx),
            target_names=[TARGET],
        )
        test_data.df_featurized = X_test
        predictions = model.predict(test_data)[TARGET].values
        actual = y.iloc[test_idx].to_numpy()
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        r2 = r2_score(actual, predictions)
        print(f"Fold {fold}/{N_SPLITS}: MAE={mae:.6f} RMSE={rmse:.6f} R²={r2:.6f}")
        fold_results.append((fold, mae, rmse, r2, len(train_idx), len(test_idx)))

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fold", "mae", "rmse", "r2", "n_train", "n_test"])
        for fold, mae, rmse, r2, n_train, n_test in fold_results:
            writer.writerow(
                [fold, f"{mae:.6f}", f"{rmse:.6f}", f"{r2:.6f}", n_train, n_test]
            )

    maes = np.array([row[1] for row in fold_results])
    r2s = np.array([row[3] for row in fold_results])
    print("\nFinal summary (V3)")
    print(f"MAE={maes.mean():.10f} ± {maes.std(ddof=1):.10f}")
    print(f"R²={r2s.mean():.10f} ± {r2s.std(ddof=1):.10f}")
    print(f"Saved fold-level results to {OUTPUT_PATH}")
    print("\nComparison")
    print("Reported (unverified):      MAE=0.1347, R²=0.7002")
    print("Old repo CSV (V2, no dedup): MAE=0.2240, R²=0.3520")
    print(f"New reproducible run (V3):  MAE={maes.mean():.10f}, R²={r2s.mean():.10f}")
    print(f"V3 MAE gap to reported: {abs(maes.mean() - 0.1347):.10f}")
    print(f"V3 MAE gap to V2: {abs(maes.mean() - 0.2240):.10f}")
    print(f"V3 R² gap to reported: {abs(r2s.mean() - 0.7002):.10f}")
    print(f"V3 R² gap to V2: {abs(r2s.mean() - 0.3520):.10f}")


if __name__ == "__main__":
    main()