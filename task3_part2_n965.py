#!/usr/bin/env python3
"""Matched composition-vs-structure GroupKFold comparison for N=965."""

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
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import SelectKBest, mutual_info_regression

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MATMINER_PATH = RESULTS / "matminer_for_sisso_v2.csv"
STRUCTURE_PATH = RESULTS / "matminer_structure_features_full.csv"
OUTPUT_PATH = RESULTS / "TASK3_PART2_N965_FOLDS.csv"
PERMUTATION_PATH = RESULTS / "TASK3_PART2_N965_PERMUTATION.csv"
TARGET = "target"
SEED = 42
N_SPLITS = 5
N_RUNS = 3
N_FEAT = 50
EPOCHS = 100


def canonical_formula(value: object) -> str:
    return Composition(str(value)).reduced_formula


def is_dilute_dopant(formula: str) -> bool:
    amounts = Composition(formula).get_el_amt_dict()
    total = sum(amounts.values())
    return any(amount / total < 0.05 for amount in amounts.values())


def make_unique_columns(columns) -> list[str]:
    names: list[str] = []
    counts: dict[str, int] = {}
    for column in columns:
        base = str(column)
        count = counts.get(base, 0)
        counts[base] = count + 1
        names.append(base if count == 0 else f"{base}_{count}")
    return names


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, np.ndarray]:
    mat = pd.read_csv(MATMINER_PATH).copy()
    struct = pd.read_csv(STRUCTURE_PATH).copy()

    mat["canonical_formula"] = mat["canonical_formula"].astype(str).str.strip()
    struct["canonical_formula"] = struct["canonical_formula"].astype(str).str.strip()

    matched_formulas = struct["canonical_formula"].drop_duplicates().tolist()
    filtered = mat[mat["canonical_formula"].isin(matched_formulas)].copy().reset_index(drop=True)
    if len(struct) != 965 or len(matched_formulas) != 965:
        raise ValueError(f"Expected 965 matched structure formulas, got {len(matched_formulas)}")

    if filtered["canonical_formula"].nunique() != len(matched_formulas):
        raise ValueError("Filtered temperature table lost or duplicated canonical formulas")

    structure_columns = [c for c in struct.columns if c not in {"composition", "canonical_formula"}]
    struct_for_join = struct[["canonical_formula", *structure_columns]].copy()
    row_level = filtered.merge(struct_for_join, on="canonical_formula", how="left", validate="many_to_one")

    if len(row_level) != len(filtered):
        raise ValueError(f"Row-level join changed row count: {len(row_level)} != {len(filtered)}")
    if row_level["canonical_formula"].nunique() != len(matched_formulas):
        raise ValueError("Row-level join dropped formula coverage")

    base_columns = [
        c
        for c in row_level.columns
        if c not in {"formula", "canonical_formula", "sys_df_original_index", TARGET}
    ]
    X_base = row_level[base_columns].select_dtypes(include=[np.number]).copy()

    X_structure = row_level[structure_columns].apply(pd.to_numeric, errors="coerce")
    bond_columns = [c for c in structure_columns if str(c).startswith("BondFractions|")]
    if bond_columns:
        X_structure[bond_columns] = X_structure[bond_columns].fillna(0.0)
    all_nan = [c for c in structure_columns if c in X_structure.columns and X_structure[c].isna().all()]
    X_structure = X_structure.drop(columns=all_nan)

    print(f"Matched formulas: {len(matched_formulas)}")
    print(f"Temperature rows after join: {len(row_level)}")
    print(f"Arm A columns: {X_base.shape[1]}")
    print(f"Arm B structure columns after cleanup: {X_structure.shape[1]}")
    print(f"BondFractions zero-filled: {len(bond_columns)}")
    print(f"Fully-NaN structure columns dropped: {len(all_nan)}")
    return row_level, X_base, X_structure, row_level[TARGET].astype(float), row_level["canonical_formula"].astype(str).to_numpy()


def fit_predict(X: pd.DataFrame, y: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    train = X.iloc[train_idx]
    test = X.iloc[test_idx]
    fold_empty = train.columns[train.isna().all()]
    train = train.drop(columns=fold_empty)
    test = test.drop(columns=fold_empty)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(train)
    imputed_columns = train.columns[~pd.isna(imputer.statistics_)]
    train_imp = pd.DataFrame(
        imputer.transform(train), columns=imputed_columns
    )
    test_imp = pd.DataFrame(
        imputer.transform(test), columns=imputed_columns
    )
    constant = train_imp.columns[train_imp.nunique(dropna=True) <= 1]
    train_imp = train_imp.drop(columns=constant)
    test_imp = test_imp.drop(columns=constant)

    if train_imp.shape[1] == 0:
        raise ValueError(f"No usable features remain for fold with {len(train_idx)} training rows")

    n_features = min(N_FEAT, train_imp.shape[1])
    selector = SelectKBest(
        score_func=lambda features, target: mutual_info_regression(
            features, target, random_state=SEED
        ),
        k=n_features,
    )
    train_selected = pd.DataFrame(
        selector.fit_transform(train_imp, y.iloc[train_idx]),
        columns=train_imp.columns[selector.get_support()],
    )
    # MODNet indexes features by column label; duplicate names from composition + structure
    # descriptors can silently expand the selected matrix back to more than `k` features.
    # Make the selected names unique to preserve the exact selected feature count.
    train_selected = train_selected.copy()
    train_selected.columns = make_unique_columns(train_selected.columns)
    test_selected = pd.DataFrame(
        selector.transform(test_imp), columns=train_selected.columns
    )

    actual_n_features = min(n_features, train_selected.shape[1])
    if actual_n_features <= 0:
        raise ValueError(f"Feature selection left no usable features for fold with {len(train_idx)} rows")

    train_data = MODData(
        materials=list(range(len(train_idx))),
        targets=y.iloc[train_idx].to_numpy(dtype=float),
        target_names=[TARGET],
    )
    train_data.df_featurized = train_selected
    train_data.optimal_features = list(train_selected.columns)
    train_data.optimal_features_by_target = {TARGET: list(train_selected.columns)}
    # The sklearn SelectKBest stage already performs the supervised reduction for this run.
    # MODNet only needs the selected columns recorded under `optimal_features` to fit the model,
    # so we avoid the brittle internal mutual-information pass that crashes on a 2D target array.
    optimal_feature_count = len(train_data.get_optimal_descriptors())
    if optimal_feature_count == 0:
        raise ValueError(f"MODNet found zero optimal descriptors for fold with {len(train_idx)} rows")
    model_n_feat = min(N_FEAT, optimal_feature_count)
    model = MODNetModel([[[TARGET]]], weights={TARGET: 1}, n_feat=model_n_feat)
    model.fit(
        train_data,
        val_fraction=0.1,
        lr=0.001,
        batch_size=64,
        loss="mae",
        epochs=EPOCHS,
        verbose=0,
    )

    test_data = MODData(
        materials=list(range(len(test_idx))),
        targets=[[0.0]] * len(test_idx),
        target_names=[TARGET],
    )
    test_data.df_featurized = test_selected
    return model.predict(test_data)[TARGET].to_numpy()


def run_arm(label: str, X: pd.DataFrame, y: pd.Series, groups: np.ndarray, splits, run: int, shuffled_y=None):
    values = y if shuffled_y is None else pd.Series(shuffled_y, index=y.index)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups & test_groups
        if overlap:
            raise AssertionError(f"LEAKAGE: {label} run={run} fold={fold} overlap={len(overlap)}")
        print(f"{label} run={run} fold={fold}: leak-check passed (overlap=0)", flush=True)
        prediction = fit_predict(X, values, train_idx, test_idx)
        actual = values.iloc[test_idx].to_numpy()
        rows.append(
            {
                "model": label,
                "run": run,
                "fold": fold,
                "mae": mean_absolute_error(actual, prediction),
                "rmse": np.sqrt(mean_squared_error(actual, prediction)),
                "r2": r2_score(actual, prediction),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "group_overlap": len(overlap),
            }
        )
        print(f"  MAE={rows[-1]['mae']:.6f} R2={rows[-1]['r2']:.6f}", flush=True)
    return rows


def paired_comparison(result: pd.DataFrame) -> tuple[float, float, float, float, float, float, pd.DataFrame]:
    arm_a = result[result.model.str.startswith("Arm A")].copy()
    arm_b = result[result.model.str.startswith("Arm B")].copy()
    paired = arm_a[["run", "fold", "mae", "r2"]].merge(
        arm_b[["run", "fold", "mae", "r2"]],
        on=["run", "fold"],
        suffixes=("_A", "_B"),
    )
    if paired.empty:
        raise ValueError("No paired Arm A/Arm B results available for comparison")

    mae_diff = paired["mae_A"] - paired["mae_B"]
    r2_diff = paired["r2_A"] - paired["r2_B"]

    mae_mean_diff = float(mae_diff.mean())
    r2_mean_diff = float(r2_diff.mean())
    mae_t_stat, mae_p = stats.ttest_rel(paired["mae_A"], paired["mae_B"])[:2]
    r2_t_stat, r2_p = stats.ttest_rel(paired["r2_A"], paired["r2_B"])[:2]
    mae_wilcoxon = stats.wilcoxon(mae_diff, zero_method="wilcox", alternative="two-sided")
    r2_wilcoxon = stats.wilcoxon(r2_diff, zero_method="wilcox", alternative="two-sided")

    return (
        mae_mean_diff,
        mae_p,
        float(mae_t_stat),
        r2_mean_diff,
        r2_p,
        float(r2_t_stat),
        paired,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=N_RUNS)
    args = parser.parse_args()
    df, X_base, X_structure, y, groups = load_data()
    X_both = pd.concat([X_base.reset_index(drop=True), X_structure.reset_index(drop=True)], axis=1)
    y = y.reset_index(drop=True)
    groups = groups.astype(str)
    all_rows = []

    for run in range(1, args.runs + 1):
        seed = SEED + run - 1
        splitter = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        splits = list(splitter.split(X_base, y, groups=groups))
        print(f"=== RUN {run}/{args.runs} (seed={seed}) ===", flush=True)
        all_rows.extend(run_arm("Arm A: composition-only", X_base, y, groups, splits, run))
        all_rows.extend(run_arm("Arm B: composition+structure", X_both, y, groups, splits, run))

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    result = pd.DataFrame(all_rows)
    print("\nObserved summary")
    for label, frame in result.groupby("model"):
        print(label, "MAE", frame.mae.mean(), "+/-", frame.mae.std(ddof=1), "R2", frame.r2.mean(), "+/-", frame.r2.std(ddof=1))

    mae_mean_diff, mae_p, mae_t_stat, r2_mean_diff, r2_p, r2_t_stat, paired = paired_comparison(result)
    print(
        "Paired run×fold MAE comparison A-B: "
        f"mean_diff={mae_mean_diff:.8f} t={mae_t_stat:.6f} p={mae_p:.6f}"
    )
    print(
        "Paired run×fold R2 comparison A-B: "
        f"mean_diff={r2_mean_diff:.8f} t={r2_t_stat:.6f} p={r2_p:.6f}"
    )

    shuffled = np.random.RandomState(SEED).permutation(y.to_numpy())
    permutation_rows = []
    print("=== PERMUTATION TEST (one fixed target shuffle) ===", flush=True)
    permutation_rows.extend(run_arm("Arm A permutation", X_base, y, groups, splits, 1, shuffled))
    permutation_rows.extend(run_arm("Arm B permutation", X_both, y, groups, splits, 1, shuffled))
    pd.DataFrame(permutation_rows).to_csv(PERMUTATION_PATH, index=False)
    print(f"Saved observed folds to {OUTPUT_PATH}")
    print(f"Saved permutation folds to {PERMUTATION_PATH}")


if __name__ == "__main__":
    main()