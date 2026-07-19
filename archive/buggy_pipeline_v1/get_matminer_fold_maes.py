#!/usr/bin/env python3
"""Re-run Matminer + MODNet with GroupKFold and save fold-level MAEs plus a corrected resampling t-test comparison."""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy.stats import t as t_dist
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from modnet.models import MODNetModel
from modnet.preprocessing import MODData

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "target"
N_SPLITS = 5
TEST_SIZE = 1519
TRAIN_SIZE = 6075


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


def load_matminer_data() -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    mat_path = RESULTS_DIR / "matminer_for_sisso.csv"
    if not mat_path.exists():
        raise FileNotFoundError(f"Anchor file not found: {mat_path}")
    mat = pd.read_csv(mat_path)
    if TARGET_COLUMN not in mat.columns:
        raise ValueError(f"Anchor file must contain a '{TARGET_COLUMN}' column")

    dataset_path = ROOT / "sysTEm_dataset" / "sysTEm_dataset.xlsx"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Could not find SysTEm dataset at {dataset_path}")
    sys_df = pd.read_excel(dataset_path)
    if "Pretty Formula" not in sys_df.columns:
        raise ValueError("SysTEm dataset must contain a Pretty Formula column")

    n_rows = len(mat)
    groups = make_group_labels(sys_df.iloc[:n_rows]["Pretty Formula"])
    return mat, mat[TARGET_COLUMN], groups


def train_matminer_groupkfold() -> List[dict]:
    mat, y, groups = load_matminer_data()
    X = mat.drop(columns=[TARGET_COLUMN]).copy()

    imputer = SimpleImputer(strategy="mean")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_imp = X_imp.loc[:, X_imp.nunique(dropna=True) > 1]

    group_kf = GroupKFold(n_splits=N_SPLITS)
    fold_results: List[dict] = []
    groups_arr = np.asarray(groups, dtype=object)

    for fold, (train_idx, test_idx) in enumerate(group_kf.split(X_imp, y, groups=groups_arr), start=1):
        train_groups = {str(label) for label in groups_arr[train_idx]}
        test_groups = {str(label) for label in groups_arr[test_idx]}
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise ValueError(f"Fold {fold} has composition overlap between train and test")

        X_train = X_imp.iloc[train_idx].reset_index(drop=True)
        X_test = X_imp.iloc[test_idx].reset_index(drop=True)
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
        print(f"Fold {fold}/{N_SPLITS}: MAE={mae:.4f} R²={r2:.4f}", flush=True)
        fold_results.append({"fold": fold, "mae": float(mae), "rmse": float(rmse), "r2": float(r2)})

    return fold_results


def save_fold_results(fold_results: List[dict]) -> Path:
    output_path = RESULTS_DIR / "MATMINER_GROUPKFOLD_FOLDS.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fold", "mae", "rmse", "r2"])
        for row in fold_results:
            writer.writerow([row["fold"], f"{row['mae']:.6f}", f"{row['rmse']:.6f}", f"{row['r2']:.6f}"])
    print(f"Saved fold-level results to {output_path}")
    return output_path


def load_fold_maes(path: Path) -> List[float]:
    df = pd.read_csv(path)
    df = df[df["fold"].astype(str).str.lower() != "overall"]
    return df["mae"].astype(float).tolist()


def corrected_resampling_t_test(maes_a: List[float], maes_b: List[float]) -> Tuple[float, float, float]:
    diffs = np.array(maes_a, dtype=float) - np.array(maes_b, dtype=float)
    d_bar = float(np.mean(diffs))
    var_d = float(np.var(diffs, ddof=1))
    var_corrected = (1 / N_SPLITS + TEST_SIZE / TRAIN_SIZE) * var_d
    if var_corrected <= 0:
        t_stat = 0.0
        p_value = 1.0
    else:
        t_stat = d_bar / np.sqrt(var_corrected)
        p_value = float(2 * t_dist.sf(abs(t_stat), df=N_SPLITS - 1))
    return t_stat, p_value, d_bar


def save_statistical_results(results: List[dict]) -> Path:
    output_path = RESULTS_DIR / "STATISTICAL_COMPARISON_FINAL.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["comparison", "t_statistic", "p_value", "mean_diff", "significant"])
        for row in results:
            writer.writerow([row["comparison"], f"{row['t_statistic']:.6f}", f"{row['p_value']:.6f}", f"{row['mean_diff']:.6f}", str(row["significant"]).lower()])
    print(f"Saved statistical results to {output_path}")
    return output_path


def run_statistical_comparison(mat_path: Path) -> List[dict]:
    mat_maes = load_fold_maes(mat_path)
    crabnet_path = RESULTS_DIR / "CRABNET_CONTINUOUS_RESULTS.csv"
    hybrid_path = RESULTS_DIR / "CRABNET_LATENT_MODNET_RESULTS.csv"
    crabnet_maes = load_fold_maes(crabnet_path)
    hybrid_maes = load_fold_maes(hybrid_path)

    comparisons = [
        ("Matminer vs CrabNet+Temp", mat_maes, crabnet_maes),
        ("Matminer vs Hybrid", mat_maes, hybrid_maes),
        ("CrabNet+Temp vs Hybrid", crabnet_maes, hybrid_maes),
    ]

    print("\nComparison                     | t-stat | p-value | Significant?")
    print("-" * 65)
    results = []
    for label, left_maes, right_maes in comparisons:
        t_stat, p_value, mean_diff = corrected_resampling_t_test(left_maes, right_maes)
        significant = p_value < 0.05
        print(f"{label:<30} | {t_stat:6.3f} | {p_value:7.4f} | {'Yes' if significant else 'No':>11}")
        results.append({
            "comparison": label,
            "t_statistic": t_stat,
            "p_value": p_value,
            "mean_diff": mean_diff,
            "significant": significant,
        })

    save_statistical_results(results)
    return results


def main() -> None:
    fold_results = train_matminer_groupkfold()
    fold_path = save_fold_results(fold_results)
    run_statistical_comparison(fold_path)


if __name__ == "__main__":
    main()
