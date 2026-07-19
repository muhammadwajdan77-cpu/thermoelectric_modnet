#!/usr/bin/env python3
"""CrabNet continuous (temperature-encoded) retrain using canonical groups from matminer_for_sisso_v2.csv"""

import traceback
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from crabnet.crabnet_ import CrabNet
except Exception as exc:
    raise RuntimeError(f"CrabNet import failed: {exc}") from exc

RESULTS_DIR = Path("results")
INPUT_MATMINER = RESULTS_DIR / "matminer_for_sisso_v2.csv"
OUTPUT_CSV = RESULTS_DIR / "CRABNET_CONTINUOUS_V2_RESULTS.csv"
PREDICTIONS_CSV = RESULTS_DIR / "CRABNET_CONTINUOUS_V2_PREDICTIONS.csv"

SEED = 42
N_SPLITS = 5


def temp_to_element(temperature):
    if pd.isna(temperature):
        return "Og0.0"
    temp = float(temperature)
    x = round(temp / 1000, 3)
    return f"Og{x}"


def load_matminer(mat_path: Path) -> pd.DataFrame:
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing matminer file: {mat_path}")
    mat = pd.read_csv(mat_path)
    required = ["formula", "canonical_formula", "Temperature_K", "target"]
    for c in required:
        if c not in mat.columns:
            raise ValueError(f"Expected column '{c}' in {mat_path}")
    # build temperature-encoded formula used by CrabNet
    mat = mat.copy().reset_index(drop=True)
    mat["formula_T"] = mat["formula"].astype(str) + " " + mat["Temperature_K"].apply(temp_to_element)
    mat["canonical"] = mat["canonical_formula"].astype(str)
    mat["zT"] = mat["target"].astype(float)
    return mat


def get_model(mat_prop, train_df, val_df, epochs, batch_size, learningrate, verbose=False, model_name="crabnet_v2"):
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


def run_cv(mat: pd.DataFrame):
    gkf = GroupKFold(n_splits=N_SPLITS)
    groups = mat["canonical"].values
    results = []
    all_preds = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(mat, groups=groups), start=1):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups & test_groups
        assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} compositions overlap in fold {fold_idx}"
        print(f"Fold {fold_idx}: verified zero overlap ✓", flush=True)

        train_full = mat.iloc[train_idx].reset_index(drop=True)
        test_df = mat.iloc[test_idx].reset_index(drop=True)

        # create validation split from train respecting groups
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
        tr_sub, val_idx = next(splitter.split(train_full, groups=train_full["canonical"]))
        train_df = train_full.iloc[tr_sub].reset_index(drop=True)
        val_df = train_full.iloc[val_idx].reset_index(drop=True)

        print(f"Fold {fold_idx}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}", flush=True)

        train_cb = train_df[["formula_T", "zT"]].rename(columns={"formula_T": "formula", "zT": "target"})
        val_cb = val_df[["formula_T", "zT"]].rename(columns={"formula_T": "formula", "zT": "target"})
        test_cb = test_df[["formula_T", "zT"]].rename(columns={"formula_T": "formula", "zT": "target"})

        try:
            model = get_model(mat_prop="ZT", train_df=train_cb, val_df=val_cb, epochs=200, batch_size=128, learningrate=1e-3, verbose=False, model_name=f"crabnet_v2_fold_{fold_idx}")
            model.fit(train_df=train_cb, val_df=val_cb)
            pred, true = model.predict(test_cb, return_true=True)
        except Exception as exc:
            print(f"Fold {fold_idx} training/prediction failed: {exc}", flush=True)
            traceback.print_exc()
            results.append({"fold": fold_idx, "mae": None, "rmse": None, "r2": None})
            continue

        mae = mean_absolute_error(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        r2 = r2_score(true, pred)
        print(f"Fold {fold_idx}: MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f}", flush=True)

        fold_df = test_df.copy()
        fold_df["predicted_zT"] = pred
        fold_df["true_zT"] = true
        fold_df["fold"] = fold_idx
        all_preds.append(fold_df)

        results.append({"fold": fold_idx, "mae": float(mae), "rmse": float(rmse), "r2": float(r2)})

    all_preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    overall = (None, None, None)
    if not all_preds_df.empty:
        overall = (
            mean_absolute_error(all_preds_df["true_zT"], all_preds_df["predicted_zT"]),
            np.sqrt(mean_squared_error(all_preds_df["true_zT"], all_preds_df["predicted_zT"])),
            r2_score(all_preds_df["true_zT"], all_preds_df["predicted_zT"]),
        )

    return results, overall, all_preds_df


def save_results(results, overall, all_preds_df):
    df = pd.DataFrame(results)
    if overall[0] is not None:
        df = pd.concat([df, pd.DataFrame([{"fold": "overall", "mae": overall[0], "rmse": overall[1], "r2": overall[2]}])], ignore_index=True)
    df.to_csv(OUTPUT_CSV, index=False)
    if not all_preds_df.empty:
        all_preds_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Saved metrics to {OUTPUT_CSV}", flush=True)


def main():
    mat = load_matminer(INPUT_MATMINER)
    results, overall, preds = run_cv(mat)
    save_results(results, overall, preds)


if __name__ == '__main__':
    main()
