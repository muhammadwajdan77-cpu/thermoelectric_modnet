#!/usr/bin/env python3
"""Classify near-duplicate Matminer feature pairs and, if clean, train MODNet with canonical-formula GroupKFold splits."""

from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from modnet.models import MODNetModel
from modnet.preprocessing import MODData
from pymatgen.core import Composition
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
INPUT_PATH = RESULTS_DIR / "matminer_for_sisso_v2.csv"
OUTPUT_FOLDS_PATH = RESULTS_DIR / "MATMINER_GROUPKFOLD_FOLDS_V2.csv"

TARGET_COLUMN = "target"
N_SPLITS = 5
METADATA_COLUMNS = {"formula", "canonical_formula", "sys_df_original_index", "Temperature_K", TARGET_COLUMN}


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


def get_feature_columns(mat_df: pd.DataFrame) -> List[str]:
    feature_cols = [
        c for c in mat_df.columns
        if c not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(mat_df[c])
    ]
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found in the Matminer file")
    return feature_cols


def build_feature_matrix(mat_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    feature_cols = get_feature_columns(mat_df)
    X = mat_df[feature_cols].copy()
    imputer = SimpleImputer(strategy="mean")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_imp = X_imp.loc[:, X_imp.nunique(dropna=True) > 1]
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns)
    return X_scaled, list(X_scaled.columns)


def find_near_duplicates(mat_df: pd.DataFrame) -> List[Tuple[int, int, float]]:
    X_scaled, _ = build_feature_matrix(mat_df)
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)
    flagged: List[Tuple[int, int, float]] = []
    for i in range(len(mat_df)):
        j = int(indices[i, 1])
        if j == i:
            continue
        dist = float(distances[i, 1])
        if dist < 0.01:
            flagged.append((i, j, dist))
    return flagged


def categorize_pairs(mat_df: pd.DataFrame, flagged: List[Tuple[int, int, float]]) -> Tuple[dict, List[dict]]:
    counts = {"a": 0, "b": 0, "c": 0}
    details: List[dict] = []

    for i, j, dist in flagged:
        formula_i = str(mat_df.iloc[i]["formula"]).strip()
        formula_j = str(mat_df.iloc[j]["formula"]).strip()
        canonical_i = canonical_formula(formula_i)
        canonical_j = canonical_formula(formula_j)

        if canonical_i and canonical_j and canonical_i == canonical_j:
            counts["a"] += 1
            continue

        try:
            set_i = {el.symbol for el in Composition(formula_i).elements}
            set_j = {el.symbol for el in Composition(formula_j).elements}
        except Exception:
            set_i = set()
            set_j = set()

        if set_i and set_j and set_i == set_j:
            counts["b"] += 1
        else:
            counts["c"] += 1
            details.append({
                "i": i,
                "j": j,
                "dist": dist,
                "formula_i": formula_i,
                "formula_j": formula_j,
                "canonical_i": canonical_i,
                "canonical_j": canonical_j,
                "elements_i": sorted(set_i),
                "elements_j": sorted(set_j),
            })

    print("Near-duplicate category counts")
    print(f"  a) same canonical formula: {counts['a']}")
    print(f"  b) different formula, same element set: {counts['b']}")
    print(f"  c) different element sets: {counts['c']}")

    if counts["c"] == 0:
        print("Category (c) count is 0; the old unrelated-composition corruption bug does not appear in this file.")
    else:
        print("Category (c) pairs detected:")
        for item in details[:20]:
            print(
                f"  {item['i']} <-> {item['j']} dist={item['dist']:.6f} | "
                f"{item['formula_i']} vs {item['formula_j']} | "
                f"elements={item['elements_i']} vs {item['elements_j']}"
            )

    return counts, details


def train_groupkfold(mat_df: pd.DataFrame) -> List[dict]:
    X_scaled, _ = build_feature_matrix(mat_df)
    y = mat_df[TARGET_COLUMN].astype(float).reset_index(drop=True)
    groups = mat_df["canonical_formula"].fillna("").astype(str).reset_index(drop=True)

    group_kf = GroupKFold(n_splits=N_SPLITS)
    fold_results: List[dict] = []

    for fold, (train_idx, test_idx) in enumerate(group_kf.split(X_scaled, y, groups=groups), start=1):
        train_groups = set(str(g) for g in groups.iloc[train_idx])
        test_groups = set(str(g) for g in groups.iloc[test_idx])
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise ValueError(f"Fold {fold} has composition overlap between train and test groups: {sorted(overlap)[:5]}")

        X_train = X_scaled.iloc[train_idx].reset_index(drop=True)
        X_test = X_scaled.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        train_data = MODData(
            materials=list(range(len(train_idx))),
            targets=[[float(v)] for v in y_train.values],
            target_names=[TARGET_COLUMN],
        )
        train_data.df_featurized = X_train
        train_data.feature_selection(n=min(50, X_train.shape[1]))

        model = MODNetModel([[[TARGET_COLUMN]]], weights={TARGET_COLUMN: 1}, n_feat=min(50, X_train.shape[1]))
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
            target_names=[TARGET_COLUMN],
        )
        test_data.df_featurized = X_test
        y_pred = model.predict(test_data)[TARGET_COLUMN].values

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Fold {fold}/{N_SPLITS}: MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f}")
        fold_results.append({"fold": fold, "mae": float(mae), "rmse": float(rmse), "r2": float(r2)})

    return fold_results


def save_fold_results(fold_results: List[dict]) -> Path:
    with OUTPUT_FOLDS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fold", "mae", "rmse", "r2"])
        for row in fold_results:
            writer.writerow([row["fold"], f"{row['mae']:.6f}", f"{row['rmse']:.6f}", f"{row['r2']:.6f}"])
    print(f"Saved fold-level results to {OUTPUT_FOLDS_PATH}")
    return OUTPUT_FOLDS_PATH


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input Matminer file not found: {INPUT_PATH}")

    mat_df = pd.read_csv(INPUT_PATH)
    if TARGET_COLUMN not in mat_df.columns:
        raise ValueError(f"Input file must contain a '{TARGET_COLUMN}' column")
    if "canonical_formula" not in mat_df.columns:
        raise ValueError("Input file must contain a 'canonical_formula' column")

    flagged = find_near_duplicates(mat_df)
    counts, details = categorize_pairs(mat_df, flagged)

    if counts["c"] > 0:
        print("Stopping before model training because category (c) contains non-zero corruption cases.")
        sys.exit(1)

    fold_results = train_groupkfold(mat_df)
    save_fold_results(fold_results)


if __name__ == "__main__":
    main()
