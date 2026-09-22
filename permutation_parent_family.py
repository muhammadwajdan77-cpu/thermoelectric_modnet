#!/usr/bin/env python3
"""Permutation control for the full-data parent-family MODNet pipeline."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from modnet.models import MODNetModel
from modnet.preprocessing import MODData
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from run_grouped_modnet import TARGET, canonical_formula, parent_family

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "results" / "matminer_for_sisso_v2.csv"
N_SPLITS = 5
THRESHOLD = 0.05


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    df["_canonical_formula"] = df["formula"].map(canonical_formula)
    df["_parent_family"] = df["formula"].map(
        lambda value: parent_family(value, threshold=THRESHOLD)
    )
    feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns.remove(TARGET)
    X = df[feature_columns]
    y = df[TARGET].astype(float).to_numpy()
    shuffled_y = np.random.RandomState(42).permutation(y)
    groups = df["_parent_family"].to_numpy(dtype=object)

    print(f"raw shape: {df.shape[0], 137}")
    print(f"rows used: {len(df)}")
    print(f"parent families: {len(np.unique(groups))}")
    print(f"shuffled target std={shuffled_y.std(ddof=1):.10f}")

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(
        GroupKFold(n_splits=N_SPLITS).split(X, shuffled_y, groups=groups), start=1
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
            targets=[[float(value)] for value in shuffled_y[train_idx]],
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
        actual = shuffled_y[test_idx]
        mae = mean_absolute_error(actual, predictions)
        r2 = r2_score(actual, predictions)
        fold_results.append((mae, r2))
        print(
            f"Fold {fold}/{N_SPLITS}: n_train={len(train_idx)} n_test={len(test_idx)} "
            f"MAE={mae:.6f} R2={r2:.6f}"
        )

    maes = np.array([row[0] for row in fold_results])
    r2s = np.array([row[1] for row in fold_results])
    print("Permutation test summary (parent family)")
    print(f"MAE={maes.mean():.10f} +/- {maes.std(ddof=1):.10f}")
    print(f"R2={r2s.mean():.10f} +/- {r2s.std(ddof=1):.10f}")


if __name__ == "__main__":
    main()
