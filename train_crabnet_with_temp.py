#!/usr/bin/env python3
"""Train CrabNet on zT prediction with temperature encoded as a proxy rare element."""

import re
import traceback
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim.swa_utils as swa_utils
from pymatgen.core.composition import Composition
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.optim.optimizer import Optimizer

try:
    from crabnet.crabnet_ import CrabNet
except Exception as exc:  # pragma: no cover - environment-dependent
    raise RuntimeError(f"CrabNet import failed: {exc}") from exc

DATA_PATH = Path("sysTEm_dataset") / "sysTEm_dataset.xlsx"
RESULT_CSV = Path("results") / "CRABNET_CONTINUOUS_RESULTS.csv"
PREDICTIONS_CSV = Path("results") / "CRABNET_CONTINUOUS_PREDICTIONS.csv"
PARITY_PNG = Path("results") / "figures" / "parity_crabnet_continuous.png"

NO_TEMP_MAE = 0.2169
NO_TEMP_R2 = 0.3989
BASELINE_MAE = 0.1347
BASELINE_R2 = 0.7002


def temp_to_element(temperature):
    if pd.isna(temperature):
        return "Og0.0"
    temp = float(temperature)
    x = round(temp / 1000, 3)
    return f"Og{x}"


def canonical_formula(formula):
    if formula is None:
        return None
    s = str(formula).strip()
    if s == "":
        return None
    try:
        return Composition(s).reduced_formula
    except Exception:
        return s


def is_valid_formula(formula):
    s = str(formula or "").strip()
    if not s:
        return False
    if re.search(r"wt%|vol%|%|\+", s):
        return False
    return True


def apply_swa_patch():
    """Apply a compatibility patch for SWA-related issues on newer PyTorch versions."""
    try:
        original_swa = swa_utils.AveragedModel.__init__
        if getattr(original_swa, "_patched_by_crabnet_temp", False):
            return

        def patched_swa_init(self, model, device=None, avg_fn=None, multi_avg_fn=None, use_buffers=False):
            try:
                return original_swa(
                    self,
                    model,
                    device=device,
                    avg_fn=avg_fn,
                    multi_avg_fn=multi_avg_fn,
                    use_buffers=use_buffers,
                )
            except TypeError as exc:
                if "use_buffers" in str(exc):
                    return original_swa(self, model, device=device, avg_fn=avg_fn, multi_avg_fn=multi_avg_fn)
                raise

        patched_swa_init._patched_by_crabnet_temp = True
        swa_utils.AveragedModel.__init__ = patched_swa_init
        print("Applied SWA compatibility patch", flush=True)
    except Exception as exc:
        print(f"SWA patch unavailable: {exc}", flush=True)


def ensure_dirs():
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    PARITY_PNG.parent.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_excel(path, engine="openpyxl")

    formula_col = None
    target_col = None
    temp_col = None
    for candidate in df.columns:
        name = str(candidate).strip().lower()
        if name in {"pretty formula", "formula"}:
            formula_col = candidate
        elif name in {"zt", "z_t", "z t", "target"}:
            target_col = candidate
        elif name in {"temperature (k)", "temperature", "temperature_k", "temperature(k)"}:
            temp_col = candidate

    if formula_col is None or target_col is None or temp_col is None:
        raise ValueError(f"Expected columns not found in dataset. Columns: {list(df.columns)}")

    df = df[[formula_col, target_col, temp_col]].copy()
    df.columns = ["Pretty Formula", "zT", "Temperature_K"]

    df = df.loc[df["zT"].notna() & df["Pretty Formula"].notna()].copy()
    df = df.loc[df["zT"] > 0].copy()
    df["Pretty Formula"] = df["Pretty Formula"].astype(str).str.strip()

    valid_mask = df["Pretty Formula"].apply(is_valid_formula)
    df = df.loc[valid_mask].copy()
    df = df.loc[df["Pretty Formula"] != ""].reset_index(drop=True)

    df["canonical_formula"] = df["Pretty Formula"].apply(canonical_formula)
    df = df.loc[df["canonical_formula"].notna()].copy()
    df["canonical_formula"] = df["canonical_formula"].astype(str).str.strip()
    df = df.loc[df["canonical_formula"] != ""].copy()

    df["formula_temp"] = df["Pretty Formula"] + " " + df["Temperature_K"].apply(temp_to_element)
    df["formula"] = df["formula_temp"]
    df["target"] = df["zT"].astype(float)

    print(f"Loaded dataset: {len(df)} rows, {df['canonical_formula'].nunique()} unique compositions", flush=True)
    print(f"Total encoded formulas: {df['formula'].nunique()}", flush=True)

    return df[["formula", "target", "canonical_formula", "Pretty Formula", "Temperature_K"]].copy()


def get_model(mat_prop, train_df, val_df, epochs, batch_size, learningrate, verbose=False, model_name="crabnet_temp"):
    return CrabNet(
        model_name=model_name,
        mat_prop=mat_prop,
        verbose=verbose,
        save=False,
        losscurve=False,
        learningcurve=False,
        batch_size=batch_size,
        epochs=epochs,
        lr=learningrate,
    )


def plot_parity(predictions: pd.DataFrame, save_path: Path) -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(predictions["true_zT"], predictions["predicted_zT"], s=14, alpha=0.4)
    min_val = min(predictions["true_zT"].min(), predictions["predicted_zT"].min())
    max_val = max(predictions["true_zT"].max(), predictions["predicted_zT"].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="k")
    plt.xlabel("True zT")
    plt.ylabel("Predicted zT")
    plt.title("CrabNet + temperature encoding")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_cv(df: pd.DataFrame, n_splits: int = 5):
    results = []
    all_predictions = []
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["canonical_formula"].values

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups), start=1):
        train_full = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        train_groups = set(train_full["canonical_formula"].unique())
        test_groups = set(test_df["canonical_formula"].unique())
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise RuntimeError(f"Group overlap detected in fold {fold_idx}: {len(overlap)} shared formulas")

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        train_idx_sub, val_idx = next(splitter.split(train_full, groups=train_full["canonical_formula"]))
        train_df = train_full.iloc[train_idx_sub].reset_index(drop=True)
        val_df = train_full.iloc[val_idx].reset_index(drop=True)

        print(
            f"Fold {fold_idx}/{n_splits}: train={len(train_df)} rows, val={len(val_df)} rows, test={len(test_df)} rows",
            flush=True,
        )

        train_df_cb = train_df[["formula", "target"]].copy()
        val_df_cb = val_df[["formula", "target"]].copy()
        test_df_cb = test_df[["formula", "target"]].copy()

        try:
            model = get_model(
                mat_prop="ZT",
                train_df=train_df_cb,
                val_df=val_df_cb,
                epochs=300,
                batch_size=256,
                learningrate=1e-3,
                verbose=False,
                model_name=f"crabnet_temp_fold_{fold_idx}",
            )
            model.fit(train_df=train_df_cb, val_df=val_df_cb)
        except Exception as exc:
            print(f"Training failed for fold {fold_idx}: {exc}", flush=True)
            traceback.print_exc()
            results.append({"fold": fold_idx, "mae": None, "rmse": None, "r2": None})
            continue

        try:
            pred, true = model.predict(test_df_cb, return_true=True)
        except Exception as exc:
            print(f"Prediction failed for fold {fold_idx}: {exc}", flush=True)
            traceback.print_exc()
            results.append({"fold": fold_idx, "mae": None, "rmse": None, "r2": None})
            continue

        mae = mean_absolute_error(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        r2 = r2_score(true, pred)
        print(f"Fold {fold_idx}/{n_splits}: MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f}", flush=True)

        fold_pred = test_df.copy()
        fold_pred["predicted_zT"] = pred
        fold_pred["true_zT"] = true
        fold_pred["fold"] = fold_idx
        all_predictions.append(fold_pred)
        results.append({"fold": fold_idx, "mae": mae, "rmse": rmse, "r2": r2})

    all_preds_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    overall_mae = None
    overall_rmse = None
    overall_r2 = None
    if not all_preds_df.empty:
        overall_mae = mean_absolute_error(all_preds_df["true_zT"], all_preds_df["predicted_zT"])
        overall_rmse = np.sqrt(mean_squared_error(all_preds_df["true_zT"], all_preds_df["predicted_zT"]))
        overall_r2 = r2_score(all_preds_df["true_zT"], all_preds_df["predicted_zT"])

    return results, (overall_mae, overall_rmse, overall_r2), all_preds_df


def save_results(results, overall, all_preds_df):
    summary_df = pd.DataFrame(results)
    if overall[0] is not None:
        summary_df = pd.concat(
            [summary_df, pd.DataFrame([{"fold": "overall", "mae": overall[0], "rmse": overall[1], "r2": overall[2]}])],
            ignore_index=True,
        )
    summary_df.to_csv(RESULT_CSV, index=False)

    if not all_preds_df.empty:
        all_preds_df.to_csv(PREDICTIONS_CSV, index=False)
        plot_parity(all_preds_df[["true_zT", "predicted_zT"]].copy(), PARITY_PNG)

    overall_mae, overall_rmse, overall_r2 = overall
    print("\nFinal comparison:", flush=True)
    print(f"CrabNet + Temperature encoding:   MAE={overall_mae:.4f} R²={overall_r2:.4f}", flush=True)
    print(f"CrabNet (no temperature):         MAE={NO_TEMP_MAE:.4f} R²={NO_TEMP_R2:.4f}", flush=True)
    print(f"MODNet + Matminer (baseline):     MAE={BASELINE_MAE:.4f} R²={BASELINE_R2:.4f}", flush=True)
    print(f"Saved metrics to {RESULT_CSV}", flush=True)
    print(f"Saved predictions to {PREDICTIONS_CSV}", flush=True)
    print(f"Saved parity plot to {PARITY_PNG}", flush=True)


def main():
    warnings.filterwarnings("ignore")
    ensure_dirs()
    apply_swa_patch()
    df = load_dataset(DATA_PATH)
    results, overall, all_preds_df = run_cv(df, n_splits=5)
    save_results(results, overall, all_preds_df)


if __name__ == "__main__":
    main()
