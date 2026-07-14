#!/usr/bin/env python3
"""Corrected resampling t-test for comparing cross-validated MAE results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import t as t_dist


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

# Requested files
RESULT_FILES = {
    "matminer": RESULTS_DIR / "HONEST_FINAL_RESULTS.csv",
    "crabnet": RESULTS_DIR / "CRABNET_CONTINUOUS_RESULTS.csv",
    "hybrid": RESULTS_DIR / "CRABNET_LATENT_MODNET_RESULTS.csv",
}

# Fallback files in case the requested baseline file only contains aggregate metrics.
FALLBACK_FILES = {
    "matminer": [RESULTS_DIR / "results.csv", RESULTS_DIR / "results_complete.csv", RESULTS_DIR / "results_complete_with_lMM.csv"],
}

K_FOLDS = 5
TEST_SIZE = 1519
TRAIN_SIZE = 6075


def parse_float(value: object) -> float:
    if value is None:
        raise ValueError("Missing value")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        raise ValueError("Empty value")
    if text.lower() in {"nan", "none", "null"}:
        return float("nan")
    if "±" in text:
        text = text.split("±", 1)[0]
    text = text.replace(",", "")
    return float(text)


def _normalize_header(header: str) -> str:
    return header.strip().lower().replace(" ", "")


def read_fold_metrics(path: Path) -> Tuple[List[float], List[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")

        fieldmap = { _normalize_header(name): name for name in reader.fieldnames }
        fold_key = None
        mae_key = None
        r2_key = None

        for candidate in ("fold", "folds", "foldnumber", "foldid"):
            if candidate in fieldmap:
                fold_key = fieldmap[candidate]
                break
        for candidate in ("mae", "meanabsoluteerror"):
            if candidate in fieldmap:
                mae_key = fieldmap[candidate]
                break
        for candidate in ("r2", "r_squared", "r2score"):
            if candidate in fieldmap:
                r2_key = fieldmap[candidate]
                break

        if fold_key is None or mae_key is None or r2_key is None:
            raise ValueError(f"Could not identify fold/mae/r2 columns in {path}")

        fold_maes: List[float] = []
        fold_r2s: List[float] = []
        for row in reader:
            fold_value = str(row.get(fold_key, "")).strip().lower()
            if fold_value in {"", "overall", "mean", "mean±std", "meanstd", "avg", "average"}:
                continue
            try:
                mae = parse_float(row.get(mae_key))
                r2 = parse_float(row.get(r2_key))
            except (TypeError, ValueError):
                continue
            fold_maes.append(mae)
            fold_r2s.append(r2)

    if not fold_maes:
        raise ValueError(f"No fold-level MAE rows found in {path}")
    return fold_maes, fold_r2s


def load_model_metrics(model_key: str) -> Dict[str, object]:
    primary_path = RESULT_FILES[model_key]
    try:
        maes, r2s = read_fold_metrics(primary_path)
    except (FileNotFoundError, ValueError):
        fallback_paths = FALLBACK_FILES.get(model_key, [])
        for path in fallback_paths:
            try:
                maes, r2s = read_fold_metrics(path)
                break
            except (FileNotFoundError, ValueError):
                continue
        else:
            raise

    mean_mae = float(np.mean(maes))
    std_mae = float(np.std(maes))
    mean_r2 = float(np.mean(r2s))
    return {
        "name": {
            "matminer": "Matminer + MODNet",
            "crabnet": "CrabNet + continuous",
            "hybrid": "Matminer + CrabNet latent + MODNet",
        }[model_key],
        "maes": maes,
        "r2s": r2s,
        "mean_mae": mean_mae,
        "std_mae": std_mae,
        "mean_r2": mean_r2,
    }


def corrected_resampling_t_test(maes_a: List[float], maes_b: List[float]) -> Tuple[float, float, float]:
    diffs = np.array(maes_a, dtype=float) - np.array(maes_b, dtype=float)
    d_bar = float(np.mean(diffs))
    var_d = float(np.var(diffs, ddof=1))
    var_corrected = (1 / K_FOLDS + TEST_SIZE / TRAIN_SIZE) * var_d
    if var_corrected <= 0:
        t_stat = 0.0
        p_value = 1.0
    else:
        t_stat = d_bar / np.sqrt(var_corrected)
        p_value = float(2 * t_dist.sf(abs(t_stat), df=K_FOLDS - 1))
    return t_stat, p_value, d_bar


def print_summary_table(models: Dict[str, Dict[str, object]]) -> None:
    print("Per-fold MAE values:")
    print("=" * 70)
    for key in ("matminer", "crabnet", "hybrid"):
        model = models[key]
        print(f"{model['name']}: {model['maes']}")
    print()

    print("Statistical Comparison (Corrected Resampling t-test):")
    print("=" * 60)
    print(f"{'Comparison':<35} | {'t-stat':>8} | {'p-value':>8} | {'Significant?':>13}")
    print("-" * 60)

    comparisons = [
        ("Matminer vs CrabNet+Temp", "matminer", "crabnet"),
        ("Matminer vs Hybrid", "matminer", "hybrid"),
        ("CrabNet+Temp vs Hybrid", "crabnet", "hybrid"),
    ]

    rows = []
    for label, left_key, right_key in comparisons:
        t_stat, p_value, mean_diff = corrected_resampling_t_test(
            models[left_key]["maes"],
            models[right_key]["maes"],
        )
        significant = p_value < 0.05
        rows.append((label, t_stat, p_value, mean_diff, significant))
        print(f"{label:<35} | {t_stat:8.3f} | {p_value:8.4f} | {'Yes' if significant else 'No':>13}")

    print()
    print(f"{'Model':<24} | {'Mean MAE':>10} | {'Std MAE':>9} | {'Mean R²':>10}")
    print("-" * 60)
    for key in ("matminer", "crabnet", "hybrid"):
        model = models[key]
        print(
            f"{model['name']:<24} | {model['mean_mae']:>10.4f} | {model['std_mae']:>9.4f} | {model['mean_r2']:>10.4f}"
        )

    return rows


def save_results(rows: List[Tuple[str, float, float, float, bool]]) -> None:
    output_path = RESULTS_DIR / "STATISTICAL_COMPARISON.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["comparison", "t_statistic", "p_value", "mean_diff", "significant"])
        for comparison, t_stat, p_value, mean_diff, significant in rows:
            writer.writerow([comparison, f"{t_stat:.6f}", f"{p_value:.6f}", f"{mean_diff:.6f}", str(significant).lower()])
    print(f"\nSaved results to {output_path}")


def main() -> None:
    models = {key: load_model_metrics(key) for key in ("matminer", "crabnet", "hybrid")}
    rows = print_summary_table(models)
    save_results(rows)


if __name__ == "__main__":
    main()
