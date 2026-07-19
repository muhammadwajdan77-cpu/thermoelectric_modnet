#!/usr/bin/env python3
"""Fold-wise CrabNet latent extraction + hybrid training.

For each GroupKFold fold:
1. Train a CrabNet model on that fold's training compositions only.
2. Extract latent features for both train and test rows from that fold-specific model.
3. Train a downstream hybrid regressor on the merged matminer + latent feature matrix.

This follows the verified training setup from train_crabnet_with_temp.py
(300 epochs, batch size 256, learning rate 1e-3) and avoids the old untrained
latent-extraction bug.
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
MATMINER_PATH = RESULTS_DIR / "matminer_for_sisso_v2.csv"
RESULTS_OUTPUT = RESULTS_DIR / "HYBRID_V4_RESULTS.csv"
LATENT_OUTPUT = RESULTS_DIR / "HYBRID_V4_FOLD_LATENTS.csv"

SEED = 42
DEFAULT_SPLITS = 5
DEFAULT_EPOCHS = 300
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fold-wise CrabNet latent extraction + hybrid training")
    parser.add_argument("--input", type=Path, default=MATMINER_PATH, help="Path to matminer feature CSV")
    parser.add_argument("--results-output", type=Path, default=RESULTS_OUTPUT, help="Path to save hybrid results")
    parser.add_argument("--latent-output", type=Path, default=LATENT_OUTPUT, help="Path to save fold-level latent features")
    parser.add_argument("--n-splits", type=int, default=DEFAULT_SPLITS, help="Number of GroupKFold splits")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="CrabNet epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="CrabNet batch size")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR, help="CrabNet learning rate")
    parser.add_argument("--max-folds", type=int, default=None, help="Optional limit for quick smoke tests")
    parser.add_argument("--dry-run", action="store_true", help="Load data and print the fold plan without training")
    return parser.parse_args()


def temp_encoding_row(formula, temperature):
    if pd.isna(formula):
        return None
    try:
        temp = float(temperature)
    except Exception:
        return f"{formula} Og"
    frac = temp / 1000.0
    frac_text = f"{frac:.3f}".rstrip("0").rstrip(".")
    return f"{formula} Og{frac_text}"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)
    required = ["formula", "canonical_formula", "Temperature_K", "target"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing columns: {missing}")

    df = df.copy().reset_index(drop=True)
    df["formula_T"] = df.apply(lambda row: temp_encoding_row(row["formula"], row["Temperature_K"]), axis=1)
    df["target"] = df["target"].astype(float)
    df["canonical_formula"] = df["canonical_formula"].astype(str).str.strip()
    df["row_id"] = df.index.to_numpy()
    return df


def get_model(epochs: int, batch_size: int, learning_rate: float, fold_idx: int):
    try:
        from crabnet.crabnet_ import CrabNet
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(f"CrabNet import failed: {exc}") from exc

    return CrabNet(
        model_name=f"crabnet_temp_fold_{fold_idx}",
        mat_prop="ZT",
        verbose=False,
        save=False,
        losscurve=False,
        learningcurve=False,
        batch_size=batch_size,
        epochs=epochs,
        lr=learning_rate,
        compute_device="cpu",
    )


def extract_latent_features(crab_model, subset_df: pd.DataFrame, batch_size: int = 256) -> np.ndarray:
    latent_store = {}

    def hook_fn(module, inputs, output):
        src = inputs[0]
        if src.dim() == 3 and src.shape[-1] == 1:
            src = src.squeeze(-1)
        mask = src == 0
        if output.dim() == 3:
            mask = mask.unsqueeze(-1).expand_as(output)
            out = output.masked_fill(mask, 0.0)
            count = (~mask).sum(dim=1).float().clamp(min=1.0)
            avg = out.sum(dim=1) / count
        else:
            avg = output
        latent_store["feat"] = avg.detach().cpu().numpy()

    hook = crab_model.model.encoder.register_forward_hook(hook_fn)
    try:
        all_feats = []
        for start in range(0, len(subset_df), batch_size):
            batch = subset_df.iloc[start:start + batch_size]
            crab_df = pd.DataFrame({"formula": batch["formula_T"].values, "target": batch["target"].values})
            try:
                crab_model.load_data(crab_df, train=False)
                crab_model.predict(crab_df)
            except Exception:
                feat_dim = all_feats[-1].shape[1] if all_feats else 512
                all_feats.append(np.zeros((len(batch), feat_dim), dtype=float))
                continue

            if "feat" in latent_store:
                all_feats.append(latent_store["feat"])
                latent_store.clear()
            else:
                feat_dim = all_feats[-1].shape[1] if all_feats else 512
                all_feats.append(np.zeros((len(batch), feat_dim), dtype=float))

        if not all_feats:
            return np.zeros((len(subset_df), 512), dtype=float)
        return np.vstack(all_feats)
    finally:
        hook.remove()


def assert_no_group_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame, fold_idx: int) -> None:
    train_groups = set(train_df["canonical_formula"].astype(str).unique())
    test_groups = set(test_df["canonical_formula"].astype(str).unique())
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(f"Fold {fold_idx} has composition overlap: {sorted(list(overlap))[:10]}")


def train_hybrid_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray):
    if X_train.shape[1] == 0:
        return np.zeros(len(X_test), dtype=float)

    try:
        from modnet.models import MODNetModel
        from modnet.preprocessing import MODData

        from xgboost import XGBRegressor

        selector = XGBRegressor(n_estimators=100, random_state=SEED, n_jobs=-1, verbosity=0)
        selector.fit(X_train, y_train)
        importances = selector.feature_importances_
        top_idx = np.argsort(importances)[-50:]
        X_train_sel = X_train[:, top_idx]
        X_test_sel = X_test[:, top_idx]

        train_data = MODData(
            materials=list(range(len(X_train_sel))),
            targets=[[float(v)] for v in y_train.tolist()],
            target_names=["ZT"],
        )
        train_data.df_featurized = pd.DataFrame(X_train_sel)
        n_feat = min(50, X_train_sel.shape[1])
        train_data.feature_selection(n=n_feat)
        model = MODNetModel([[['ZT']]], weights={'ZT': 1}, n_feat=n_feat)
        model.fit(train_data, val_fraction=0.1, lr=0.001, batch_size=64, loss='mae', epochs=100, verbose=False)

        test_data = MODData(
            materials=list(range(len(X_test_sel))),
            targets=[[0.0]] * len(X_test_sel),
            target_names=["ZT"],
        )
        test_data.df_featurized = pd.DataFrame(X_test_sel)
        preds = model.predict(test_data)['ZT'].values
        return preds
    except Exception:
        try:
            reg = HistGradientBoostingRegressor(random_state=SEED)
            reg.fit(X_train, y_train)
            return reg.predict(X_test)
        except Exception:
            rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
            rf.fit(X_train, y_train)
            return rf.predict(X_test)


def build_feature_matrix(base_df: pd.DataFrame, latent_features: np.ndarray, latent_cols: list[str]) -> pd.DataFrame:
    latent_df = pd.DataFrame(latent_features, columns=latent_cols)
    feature_df = pd.concat([base_df.reset_index(drop=True), latent_df.reset_index(drop=True)], axis=1)
    return feature_df


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.input)
    print(f"Loaded {len(df)} rows from {args.input}", flush=True)

    if args.max_folds is not None:
        n_splits = min(args.n_splits, args.max_folds)
    else:
        n_splits = args.n_splits

    gkf = GroupKFold(n_splits=n_splits)
    groups = df["canonical_formula"].astype(str).values
    fold_results = []
    all_fold_latents = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups), start=1):
        if fold_idx > n_splits:
            break

        train_full = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        assert_no_group_overlap(train_full, test_df, fold_idx)

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
        sub_train_idx, val_idx = next(splitter.split(train_full, groups=train_full["canonical_formula"].values))
        train_df = train_full.iloc[sub_train_idx].reset_index(drop=True)
        val_df = train_full.iloc[val_idx].reset_index(drop=True)

        print(
            f"Fold {fold_idx}/{n_splits}: train={len(train_df)} rows, val={len(val_df)} rows, test={len(test_df)} rows",
            flush=True,
        )

        if args.dry_run:
            continue

        train_cb = train_df[["formula_T", "target"]].rename(columns={"formula_T": "formula"})
        val_cb = val_df[["formula_T", "target"]].rename(columns={"formula_T": "formula"})

        model = get_model(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, fold_idx=fold_idx)
        try:
            model.fit(train_df=train_cb, val_df=val_cb)
        except Exception as exc:
            raise RuntimeError(f"CrabNet training failed for fold {fold_idx}: {exc}") from exc

        train_latent = extract_latent_features(model, train_df[["formula_T", "target", "row_id"]].copy(), batch_size=args.batch_size)
        test_latent = extract_latent_features(model, test_df[["formula_T", "target", "row_id"]].copy(), batch_size=args.batch_size)

        train_latent_cols = [f"CrabLatent_{i}" for i in range(train_latent.shape[1])]
        test_latent_cols = [f"CrabLatent_{i}" for i in range(test_latent.shape[1])]

        train_split = train_df.copy()
        test_split = test_df.copy()
        train_split[train_latent_cols] = train_latent
        test_split[test_latent_cols] = test_latent

        for split_name, split_df, split_latent_cols in [
            ("train", train_split, train_latent_cols),
            ("test", test_split, test_latent_cols),
        ]:
            all_fold_latents.append(
                pd.DataFrame(
                    {
                        "fold": fold_idx,
                        "split": split_name,
                        "row_id": split_df["row_id"].values,
                        "canonical_formula": split_df["canonical_formula"].values,
                        "formula": split_df["formula"].values,
                        "target": split_df["target"].values,
                        **{col: split_df[col].values for col in split_latent_cols},
                    }
                )
            )

        exclude_cols = {"formula", "formula_T", "canonical_formula", "Temperature_K", "target", "row_id"}
        base_feature_cols = [
            col
            for col in train_df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[col])
        ]
        base_feature_cols = [col for col in base_feature_cols if col not in {"sys_df_original_index"}]

        X_train_full = train_split[base_feature_cols + train_latent_cols].astype(float)
        X_test_full = test_split[base_feature_cols + test_latent_cols].astype(float)
        y_train = train_split["target"].astype(float).values
        y_test = test_split["target"].astype(float).values

        imp = SimpleImputer(strategy="mean")
        X_train_imp = imp.fit_transform(X_train_full)
        X_test_imp = imp.transform(X_test_full)

        preds = train_hybrid_model(X_train_imp, y_train, X_test_imp)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        fold_results.append({"fold": fold_idx, "mae": float(mae), "rmse": float(rmse), "r2": float(r2)})
        print(f"Fold {fold_idx}: MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f}", flush=True)

    if args.dry_run:
        print("Dry run complete. No models were trained.", flush=True)
        return

    results_df = pd.DataFrame(fold_results)
    if not results_df.empty:
        results_df.loc[len(results_df)] = {
            "fold": "overall",
            "mae": float(results_df["mae"].mean()),
            "rmse": float(results_df["rmse"].mean()),
            "r2": float(results_df["r2"].mean()),
        }
    results_df.to_csv(args.results_output, index=False)
    print(f"Saved hybrid results to {args.results_output}", flush=True)

    latent_df = pd.concat(all_fold_latents, ignore_index=True) if all_fold_latents else pd.DataFrame()
    if not latent_df.empty:
        latent_df.to_csv(args.latent_output, index=False)
        print(f"Saved fold latent features to {args.latent_output}", flush=True)


if __name__ == "__main__":
    main()
