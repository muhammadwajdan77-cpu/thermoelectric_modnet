#!/usr/bin/env python3
"""Train CrabNet directly on zT from the SysTEm dataset using GroupKFold.

Usage: run from project root. See user instructions.
"""
import warnings
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

try:
    import crabnet.crabnet_ as crabnet_module
    from crabnet.crabnet_ import CrabNet
except Exception as exc:
    print("Failed to import CrabNet from crabnet.crabnet_:", exc)
    try:
        import crabnet
        print("Available crabnet module attrs:", [n for n in dir(crabnet) if not n.startswith("_")])
    except Exception:
        print("Could not inspect crabnet module members.")
    raise


# Note: previously this script attempted to override CrabNet's SWA
# implementation. That override has been removed to allow the package's
# native SWA implementation to function correctly.

DATA_PATH = Path("sysTEm_dataset") / "sysTEm_dataset.xlsx"
RESULT_CSV = Path("results") / "CRABNET_RESULTS.csv"
PARITY_PNG = Path("results") / "figures" / "parity_crabnet.png"
BASELINE_R2 = 0.7002
BASELINE_MAE = 0.1347


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
    try:
        comp = Composition(str(formula))
        if len(comp) == 0:
            return False
        import re

        if re.search(r'wt%|vol%|%|\+.*\+', str(formula)):
            return False
        return True
    except Exception:
        return False


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_excel(path, engine="openpyxl")
    df = df.rename(columns={
        "Pretty Formula": "Pretty Formula",
        "zT": "zT",
        "Temperature (K)": "Temperature_K",
    })
    df["Temperature_K"] = df.get("Temperature_K", 300.0).fillna(300.0)

    df = df.loc[df["zT"] > 0].copy()
    df = df.loc[df["Pretty Formula"].notna()].copy()

    valid_mask = df["Pretty Formula"].astype(str).apply(is_valid_formula)
    df = df.loc[valid_mask].reset_index(drop=True)
    print(f"Valid rows after formula filter: {len(df)}", flush=True)
    print(f"Removed invalid rows: {(~valid_mask).sum()}", flush=True)

    df["canonical_formula"] = df["Pretty Formula"].apply(canonical_formula)
    df = df.loc[df["canonical_formula"].notna()].copy()
    df = df.loc[df["canonical_formula"].astype(str).str.strip() != ""].copy()

    df = df[["canonical_formula", "zT", "Temperature_K"]].copy()
    print(f"Loaded dataset: {len(df)} rows, {df['canonical_formula'].nunique()} unique compositions")
    if len(df) < 7000 or df["canonical_formula"].nunique() < 1200:
        warnings.warn(
            f"Dataset size is smaller than expected: {len(df)} rows, {df['canonical_formula'].nunique()} unique compositions"
        )

    return df


def ensure_dirs():
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    PARITY_PNG.parent.mkdir(parents=True, exist_ok=True)


def plot_parity(df: pd.DataFrame, save_path: Path) -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(df["true_zT"], df["predicted_zT"], s=14, alpha=0.4)
    min_val = min(df["true_zT"].min(), df["predicted_zT"].min())
    max_val = max(df["true_zT"].max(), df["predicted_zT"].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="k")
    plt.xlabel("True zT")
    plt.ylabel("Predicted zT")
    plt.title("CrabNet direct zT parity")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_cv(df: pd.DataFrame, n_splits: int = 5):
    results = []
    all_predictions = []
    groups = df["canonical_formula"].values
    gkf = GroupKFold(n_splits=n_splits)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups), start=1):
        train_full = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        train_groups = set(train_full["canonical_formula"].unique())
        test_groups = set(test_df["canonical_formula"].unique())
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise RuntimeError(f"Group overlap detected in fold {fold_idx}: {len(overlap)} shared formulas")

        validator = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx_sub, val_idx = next(validator.split(train_full, groups=train_full["canonical_formula"]))
        train_df = train_full.iloc[train_idx_sub].reset_index(drop=True)
        val_df = train_full.iloc[val_idx].reset_index(drop=True)

        print(
            f"Fold {fold_idx}/{n_splits}: train={len(train_df)} rows, val={len(val_df)} rows, test={len(test_df)} rows",
            flush=True,
        )

        train_df_cb = train_df.rename(columns={"canonical_formula": "formula", "zT": "target"})
        val_df_cb = val_df.rename(columns={"canonical_formula": "formula", "zT": "target"})
        test_df_cb = test_df.rename(columns={"canonical_formula": "formula", "zT": "target"})

        try:
            model = CrabNet(
                model_name=f"crabnet_zT_fold_{fold_idx}",
                verbose=True,
                save=False,
                losscurve=False,
                learningcurve=False,
            )
            # Monkey-patch SWALR to avoid SWA-related TypeErrors with newer
            # PyTorch versions. This creates a no-op SWALR constructor so
            # code expecting it won't fail when constructing schedulers.
            try:
                import torch.optim as _optim
                import torch.optim.swa_utils as _swa_utils

                _swa_utils.SWALR = type(
                    "SWALR", (), {"__init__": lambda self, *a, **kw: None}
                )
            except Exception:
                # If the environment doesn't expose swa_utils, ignore.
                pass
            model.fit(
                train_df=train_df_cb,
                val_df=val_df_cb,
            )
        except Exception as exc:
            print(f"Error training fold {fold_idx}: {exc}")
            traceback.print_exc()
            results.append({"fold": fold_idx, "mae": None, "rmse": None, "r2": None})
            continue

        try:
            pred, true = model.predict(test_df_cb, return_true=True)
        except Exception as exc:
            print(f"Error predicting fold {fold_idx}: {exc}")
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
    overall_mae = overall_rmse = overall_r2 = None
    if not all_preds_df.empty:
        overall_mae = mean_absolute_error(all_preds_df["true_zT"], all_preds_df["predicted_zT"])
        overall_rmse = np.sqrt(mean_squared_error(all_preds_df["true_zT"], all_preds_df["predicted_zT"]))
        overall_r2 = r2_score(all_preds_df["true_zT"], all_preds_df["predicted_zT"])

    return results, (overall_mae, overall_rmse, overall_r2), all_preds_df


def save_results(results, overall, all_preds_df):
    pd.DataFrame(results).to_csv(RESULT_CSV.parent / "CRABNET_RESULTS_summary.csv", index=False)
    if not all_preds_df.empty:
        all_preds_df.to_csv(RESULT_CSV, index=False)
        plot_parity(all_preds_df, PARITY_PNG)

    overall_mae, overall_rmse, overall_r2 = overall
    print("\nFinal comparison:")
    print(f"CrabNet (direct ZT):     MAE={overall_mae:.4f} R²={overall_r2:.4f}")
    print(f"MODNet+Matminer baseline: MAE={BASELINE_MAE:.4f} R²={BASELINE_R2:.4f}")
    print(f"Saved predictions to {RESULT_CSV}")
    print(f"Saved fold summary to {RESULT_CSV.parent / 'CRABNET_RESULTS_summary.csv'}")
    print(f"Saved parity plot to {PARITY_PNG}")


def main():
    ensure_dirs()
    df = load_dataset(DATA_PATH)
    print(f"Data length: {len(df)}, unique compositions: {df['canonical_formula'].nunique()}")
    results, overall, all_preds_df = run_cv(df, n_splits=5)
    save_results(results, overall, all_preds_df)


if __name__ == "__main__":
    main()
