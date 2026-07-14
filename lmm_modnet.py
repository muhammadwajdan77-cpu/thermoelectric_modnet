#!/usr/bin/env python3
"""lmm_modnet.py

Train MODNet on MatterVial l-MM features for zT prediction.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from modnet.models import MODNetModel
from modnet.preprocessing import MODData

SEED = 42
np.random.seed(SEED)
try:
    tf.random.set_seed(SEED)
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LMM_FEATURES_PATH = RESULTS_DIR / "lMM_features.csv"
DATASET_PATH = ROOT / "sysTEm_dataset" / "sysTEm_dataset.xlsx"
BASELINE_RESULTS_PATH = RESULTS_DIR / "results_complete.csv"
FINAL_RESULTS_PATH = RESULTS_DIR / "FINAL_RESULTS.csv"


def load_data():
    print("Loading data...")
    df_lmm = pd.read_csv(LMM_FEATURES_PATH)
    print(f"  l-MM features loaded: {df_lmm.shape[0]} rows, {df_lmm.shape[1]} columns")

    nan_columns = df_lmm.columns[df_lmm.isna().all()].tolist()
    if nan_columns:
        print(f"  Dropping {len(nan_columns)} all-NaN columns from l-MM features")
        df_lmm = df_lmm.drop(columns=nan_columns)

    if "zT" in df_lmm.columns:
        df_lmm = df_lmm.drop(columns=["zT"])

    for col in ["composition"]:
        if col in df_lmm.columns:
            df_lmm[col] = df_lmm[col].astype(str).str.strip()

    df_dataset = pd.read_excel(DATASET_PATH, engine="openpyxl")
    df_dataset["Pretty Formula"] = df_dataset["Pretty Formula"].astype(str).str.strip()
    df_dataset["zT"] = pd.to_numeric(df_dataset["zT"], errors="coerce")

    compositions_lmm = set(df_lmm["composition"].dropna().astype(str))
    compositions_data = set(df_dataset["Pretty Formula"].dropna().astype(str))

    missing_in_data = sorted(list(compositions_lmm - compositions_data))
    missing_in_lmm = sorted(list(compositions_data - compositions_lmm))

    if missing_in_data:
        print("WARNING: compositions in lMM_features.csv missing from sysTEm_dataset.xlsx:")
        print(missing_in_data[:50])

    if missing_in_lmm:
        print("WARNING: compositions in sysTEm_dataset.xlsx missing from lMM_features.csv:")
        print(missing_in_lmm[:50])

    merged = df_lmm.merge(
        df_dataset[["Pretty Formula", "zT"]],
        left_on="composition",
        right_on="Pretty Formula",
        how="inner",
        indicator=True,
    )

    if merged.empty:
        raise RuntimeError("Alignment failed: no rows remain after merging l-MM features with the dataset.")

    merged = merged[merged["_merge"] == "both"].drop(columns=["Pretty Formula", "_merge"])

    if "zT" not in merged.columns:
        raise RuntimeError("Alignment failed: merged data does not contain zT target.")

    merged = merged[merged["zT"].notna()].copy()
    merged = merged[merged["zT"] > 0].copy()

    if merged.empty:
        raise RuntimeError("No samples with zT > 0 found after filtering.")

    feature_cols = [c for c in merged.columns if c not in ["composition", "structure_id", "filename", "zT"]]
    X = merged[feature_cols].copy()
    y = merged["zT"].astype(float).copy()

    print("\nDATA SUMMARY")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  zT range: {y.min():.4f} — {y.max():.4f}")

    return X, y


def train_modnet(X: pd.DataFrame, y: pd.Series):
    print("\nTraining MODNet with 5-fold CV...")
    imp = SimpleImputer(strategy="mean")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_maes, fold_rmses, fold_r2s = [], [], []
    all_y_true, all_y_pred = [], []
    fold_models = []
    fold_selected_features = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_imp), start=1):
        print(f"\n  Fold {fold}/5")
        X_tr = X_imp.iloc[train_idx].reset_index(drop=True)
        X_te = X_imp.iloc[test_idx].reset_index(drop=True)
        y_tr = y.iloc[train_idx].reset_index(drop=True)
        y_te = y.iloc[test_idx].reset_index(drop=True)

        train_data = MODData(
            materials=list(range(len(X_tr))),
            targets=[[v] for v in y_tr.values],
            target_names=["ZT"],
        )
        train_data.df_featurized = X_tr
        n_feat = min(50, X_tr.shape[1])
        train_data.feature_selection(n=n_feat)

        model = MODNetModel([[[("ZT")]]], weights={"ZT": 1}, n_feat=n_feat)
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
            materials=list(range(len(X_te))),
            targets=[[0]] * len(X_te),
            target_names=["ZT"],
        )
        test_data.df_featurized = X_te
        y_pred = model.predict(test_data)["ZT"].values

        mae = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        r2 = r2_score(y_te, y_pred)
        print(f"    MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_r2s.append(r2)
        all_y_true.extend(y_te.values.tolist())
        all_y_pred.extend(y_pred.tolist())
        fold_models.append(model)
        fold_selected_features.append(train_data.optimal_features[:n_feat])

    metrics = {
        "MAE": (np.mean(fold_maes), np.std(fold_maes)),
        "RMSE": (np.mean(fold_rmses), np.std(fold_rmses)),
        "R2": (np.mean(fold_r2s), np.std(fold_r2s)),
    }

    print("\nCROSS-VALIDATION SUMMARY")
    print(f"  MAE:  {metrics['MAE'][0]:.4f} ± {metrics['MAE'][1]:.4f}")
    print(f"  RMSE: {metrics['RMSE'][0]:.4f} ± {metrics['RMSE'][1]:.4f}")
    print(f"  R²:   {metrics['R2'][0]:.4f} ± {metrics['R2'][1]:.4f}")

    return (
        fold_maes,
        fold_rmses,
        fold_r2s,
        np.array(all_y_true),
        np.array(all_y_pred),
        fold_models,
        fold_selected_features,
    )


def save_parity_plot(y_true: np.ndarray, y_pred: np.ndarray):
    print("\nSaving parity plot...")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.4, s=20, color="steelblue", edgecolors="none")
    lim = [min(y_true.min(), y_pred.min()) - 0.05, max(y_true.max(), y_pred.max()) + 0.05]
    ax.plot(lim, lim, "r--", lw=2, label="Perfect prediction")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual zT", fontweight="bold")
    ax.set_ylabel("Predicted zT", fontweight="bold")
    ax.set_title("MatterVial l-MM + MODNet — Parity Plot", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "parity_plot_lMM.png", dpi=300)
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'parity_plot_lMM.png'}")


def shap_analysis(model, selected_features, X_imp):
    print("\nRunning SHAP analysis on fold 1 model...")
    X_sel = X_imp[selected_features].values
    bg = X_sel[np.random.choice(len(X_sel), min(100, len(X_sel)), replace=False)]

    def modnet_predict(x_arr):
        tmp = MODData(
            materials=list(range(len(x_arr))),
            targets=[[0]] * len(x_arr),
            target_names=["ZT"],
        )
        tmp.df_featurized = pd.DataFrame(x_arr, columns=selected_features)
        result = model.predict(tmp)["ZT"].values
        return result

    explainer = shap.KernelExplainer(modnet_predict, bg)
    sample = X_sel[np.random.choice(len(X_sel), min(50, len(X_sel)), replace=False)]
    shap_vals = explainer.shap_values(sample)
    if isinstance(shap_vals, list) and len(shap_vals) == 1:
        shap_vals = shap_vals[0]

    importance = np.abs(shap_vals).mean(axis=0)
    imp_df = pd.DataFrame({"feature": selected_features, "importance": importance})
    imp_df = imp_df.sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="orchid")
    ax.set_xlabel("Mean |SHAP value|", fontweight="bold")
    ax.set_title("Top 15 l-MM features by SHAP importance", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_bar_lMM.png", dpi=300)
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'shap_bar_lMM.png'}")

    return imp_df


def save_final_comparison(fold_maes, fold_rmses, fold_r2s):
    baseline = pd.read_csv(BASELINE_RESULTS_PATH)
    baseline_mean = baseline[baseline["Fold"] == "Mean±Std"].copy()
    summary_rows = []

    def get_metric(model_name, column):
        row = baseline_mean[baseline_mean["Model"] == model_name]
        if row.empty:
            return "N/A"
        return row.iloc[0][column]

    summary_rows.append({
        "Model": "Matminer + MODNet",
        "MAE": get_metric("Matminer + MODNet (Baseline)", "MAE"),
        "RMSE": get_metric("Matminer + MODNet (Baseline)", "RMSE"),
        "R2": get_metric("Matminer + MODNet (Baseline)", "R2"),
    })
    summary_rows.append({
        "Model": "MatterVial Roost + MODNet",
        "MAE": get_metric("MatterVial (Roost) + MODNet", "MAE"),
        "RMSE": get_metric("MatterVial (Roost) + MODNet", "RMSE"),
        "R2": get_metric("MatterVial (Roost) + MODNet", "R2"),
    })
    summary_rows.append({
        "Model": "Combined (Matminer + Roost) + MODNet",
        "MAE": get_metric("Combined (Matminer + Roost) + MODNet", "MAE"),
        "RMSE": get_metric("Combined (Matminer + Roost) + MODNet", "RMSE"),
        "R2": get_metric("Combined (Matminer + Roost) + MODNet", "R2"),
    })
    current_mae = f"{np.mean(fold_maes):.4f}±{np.std(fold_maes):.4f}"
    current_rmse = f"{np.mean(fold_rmses):.4f}±{np.std(fold_rmses):.4f}"
    current_r2 = f"{np.mean(fold_r2s):.4f}±{np.std(fold_r2s):.4f}"
    summary_rows.append({
        "Model": "MatterVial l-MM + MODNet",
        "MAE": current_mae,
        "RMSE": current_rmse,
        "R2": current_r2,
    })

    final_df = pd.DataFrame(summary_rows)
    final_df.to_csv(FINAL_RESULTS_PATH, index=False)

    print("\nFINAL COMPARISON TABLE")
    print(final_df.to_string(index=False))
    print(f"\nSaved: {FINAL_RESULTS_PATH}")

    all_rows = baseline.copy()
    new_rows = []
    for fold, (mae, rmse, r2) in enumerate(zip(fold_maes, fold_rmses, fold_r2s), start=1):
        new_rows.append({
            "Fold": fold,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Model": "MatterVial l-MM + MODNet",
        })
    new_rows.append({
        "Fold": "Mean±Std",
        "MAE": current_mae,
        "RMSE": current_rmse,
        "R2": current_r2,
        "Model": "MatterVial l-MM + MODNet",
    })
    appended_df = pd.concat([all_rows, pd.DataFrame(new_rows)], ignore_index=True)
    appended_df.to_csv(RESULTS_DIR / "results_complete_with_lMM.csv", index=False)
    print(f"Saved appended results table as: {RESULTS_DIR / 'results_complete_with_lMM.csv'}")


def main():
    X, y = load_data()
    fold_maes, fold_rmses, fold_r2s, all_y_true, all_y_pred, fold_models, fold_selected_features = train_modnet(X, y)
    save_parity_plot(all_y_true, all_y_pred)
    shap_analysis(fold_models[0], fold_selected_features[0], pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(X), columns=X.columns))
    save_final_comparison(fold_maes, fold_rmses, fold_r2s)
    print("\nAll tasks completed successfully.")


if __name__ == "__main__":
    main()
