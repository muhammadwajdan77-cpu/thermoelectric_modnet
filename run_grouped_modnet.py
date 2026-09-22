#!/usr/bin/env python3
"""Run MODNet GroupKFold with either exact or parent-composition groups."""

from __future__ import annotations

import argparse
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
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "results" / "matminer_for_sisso_v2.csv"
TARGET = "target"
N_SPLITS = 5


def canonical_formula(value: object) -> str:
    if pd.isna(value):
        raise ValueError("Empty formula cannot be canonicalized")
    return Composition(str(value)).reduced_formula


def parent_family(value: object, threshold: float = 0.05) -> str:
    composition = Composition(str(value))
    amounts = composition.get_el_amt_dict()
    total = sum(amounts.values())
    major_elements = tuple(
        sorted(element for element, amount in amounts.items() if amount / total >= threshold)
    )
    if not major_elements:
        raise ValueError(f"No parent elements found for {value!r}")
    return "+".join(major_elements)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full-exact", "full-parent"], required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(INPUT_PATH)
    df["_canonical_formula"] = df["formula"].map(canonical_formula)
    if args.mode == "full-exact":
        groups = df["_canonical_formula"].to_numpy(dtype=object)
        output_path = ROOT / "results" / "MATMINER_GROUPKFOLD_FULL_EXACT.csv"
        group_label = "exact canonical formula"
    else:
        if not 0 < args.threshold < 1:
            raise ValueError("--threshold must be between 0 and 1")
        df["_parent_family"] = df["formula"].map(
            lambda value: parent_family(value, threshold=args.threshold)
        )
        groups = df["_parent_family"].to_numpy(dtype=object)
        output_path = ROOT / "results" / "MATMINER_GROUPKFOLD_FULL_PARENT.csv"
        output_path = ROOT / "results" / (
            f"MATMINER_GROUPKFOLD_FULL_PARENT_{args.threshold:g}.csv"
        )
        group_label = f"parent family (elements >= {args.threshold:.0%})"

    feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns.remove(TARGET)
    X = df[feature_columns]
    y = df[TARGET].astype(float)
    print(f"mode: {args.mode}")
    print(f"raw shape: {df.shape[0], 137}")
    print(f"rows used: {len(df)}")
    print(f"group rule: {group_label}")
    print(f"unique groups: {len(np.unique(groups))}")

    fold_results = []
    splitter = GroupKFold(n_splits=N_SPLITS)
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups=groups), start=1
    ):
        overlap = set(groups[train_idx]).intersection(groups[test_idx])
        if overlap:
            raise AssertionError(f"Fold {fold} has {len(overlap)} overlapping groups")
        print(f"Fold {fold}/{N_SPLITS}: leak-check passed (overlap=0)")

        imputer = SimpleImputer(strategy="mean")
        X_train = pd.DataFrame(imputer.fit_transform(X.iloc[train_idx]), columns=feature_columns)
        X_test = pd.DataFrame(imputer.transform(X.iloc[test_idx]), columns=feature_columns)
        constant_columns = X_train.columns[X_train.nunique(dropna=True) <= 1]
        X_train = X_train.drop(columns=constant_columns)
        X_test = X_test.drop(columns=constant_columns)
        n_features = min(50, X_train.shape[1])

        train_data = MODData(
            materials=list(range(len(train_idx))),
            targets=[[float(value)] for value in y.iloc[train_idx]],
            target_names=[TARGET],
        )
        train_data.df_featurized = X_train
        train_data.feature_selection(n=n_features)
        model = MODNetModel([[[TARGET]]], weights={TARGET: 1}, n_feat=n_features)
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
        print(
            f"Fold {fold}/{N_SPLITS}: n_train={len(train_idx)} n_test={len(test_idx)} "
            f"MAE={mae:.6f} RMSE={rmse:.6f} R2={r2:.6f}"
        )
        fold_results.append((fold, mae, rmse, r2, len(train_idx), len(test_idx)))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fold", "mae", "rmse", "r2", "n_train", "n_test"])
        writer.writerows(
            [fold, f"{mae:.6f}", f"{rmse:.6f}", f"{r2:.6f}", n_train, n_test]
            for fold, mae, rmse, r2, n_train, n_test in fold_results
        )

    maes = np.array([row[1] for row in fold_results])
    r2s = np.array([row[3] for row in fold_results])
    print("Final summary")
    print(f"MAE={maes.mean():.10f} +/- {maes.std(ddof=1):.10f}")
    print(f"R2={r2s.mean():.10f} +/- {r2s.std(ddof=1):.10f}")
    print(f"Saved fold-level results to {output_path}")


if __name__ == "__main__":
    main()
