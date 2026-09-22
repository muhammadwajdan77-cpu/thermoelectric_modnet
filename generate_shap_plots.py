import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
})

MATMINER_CSV = RESULTS_DIR / "matminer_for_sisso_v2.csv"
HYBRID_CSV = RESULTS_DIR / "HYBRID_V4_FOLD_LATENTS.csv"

N_SAMPLES = 500
MAX_BACKGROUND_SAMPLES = 100

try:
    sys.path.insert(0, str(ROOT / "MatterVial"))
    from mattervial.interpreter.help_scripts.integrated_plots_mattervial import get_shap_and_feature_decomposition
    MATTERVIAL_FUNCTION_AVAILABLE = True
except Exception as exc:
    MATTERVIAL_FUNCTION_AVAILABLE = False
    print(f"MatterVial helper unavailable ({exc}); using direct SHAP fallback.")

try:
    import shap
    from sklearn.ensemble import GradientBoostingRegressor
    SHAP_AVAILABLE = True
except Exception as exc:
    raise RuntimeError(f"SHAP or sklearn not available: {exc}")


def save_shap_plots(shap_values, X, feature_names, output_prefix, summary_name, bar_name, title_prefix):
    output_prefix = Path(output_prefix)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )
    ax = plt.gca()
    ax.set_title(f"{title_prefix} SHAP summary", pad=10, fontweight="bold")
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_ylabel("Features")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / summary_name, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    ax = plt.gca()
    ax.set_title(f"{title_prefix} mean |SHAP|", pad=10, fontweight="bold")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Features")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / bar_name, dpi=300, bbox_inches="tight")
    plt.close()

    if hasattr(shap_values, "values"):
        importances = np.abs(np.asarray(shap_values.values)).mean(axis=0)
    else:
        importances = np.abs(np.asarray(shap_values)).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": importances,
    }).sort_values("mean_abs_shap", ascending=False)
    importance_df.head(10).to_csv(FIGURES_DIR / f"{output_prefix.name}_top10.csv", index=False)
    print(f"Top 10 features for {output_prefix.name}:")
    print(importance_df.head(10).to_string(index=False))


def build_matminer_shap():
    if not MATMINER_CSV.exists():
        raise FileNotFoundError(f"Missing matminer CSV: {MATMINER_CSV}")

    mat = pd.read_csv(MATMINER_CSV)
    if "target" not in mat.columns:
        raise ValueError("Expected 'target' column in matminer_for_sisso_v2.csv")

    feature_cols = [
        col for col in mat.columns
        if col != "target" and pd.api.types.is_numeric_dtype(mat[col]) and col != "sys_df_original_index"
    ]
    if not feature_cols:
        raise ValueError(f"No numeric feature columns found in {MATMINER_CSV}")

    y = mat["target"].astype(float).to_numpy()
    X = mat[feature_cols].copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

    model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    sample_idx = np.random.choice(len(X), size=min(N_SAMPLES, len(X)), replace=False)
    X_sample = X.iloc[sample_idx].copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    save_shap_plots(
        shap_values,
        X_sample,
        X_sample.columns.tolist(),
        "shap_matminer",
        "shap_summary_matminer.png",
        "shap_bar_matminer.png",
        "Matminer + MODNet",
    )
    print("Saved Matminer SHAP summary and bar plots to results/figures")


def build_hybrid_shap():
    if not MATMINER_CSV.exists():
        raise FileNotFoundError(f"Missing matminer CSV: {MATMINER_CSV}")
    if not HYBRID_CSV.exists():
        raise FileNotFoundError(f"Missing hybrid latent CSV: {HYBRID_CSV}")

    mat = pd.read_csv(MATMINER_CSV)
    if "target" not in mat.columns:
        raise ValueError("Expected 'target' column in matminer_for_sisso_v2.csv")

    mat = mat.reset_index(drop=True)
    mat["row_id"] = mat.index.to_numpy()

    latent_df = pd.read_csv(HYBRID_CSV)
    if "row_id" not in latent_df.columns:
        raise ValueError(f"Expected 'row_id' column in {HYBRID_CSV}")
    if "split" not in latent_df.columns:
        raise ValueError(f"Expected 'split' column in {HYBRID_CSV}")

    latent_cols = [c for c in latent_df.columns if c.startswith("CrabLatent_")]
    if not latent_cols:
        raise ValueError(f"No CrabLatent_ columns found in {HYBRID_CSV}")

    test_latents = latent_df[latent_df["split"] == "test"].copy()
    if len(test_latents) == 0:
        raise ValueError(f"No test split rows found in {HYBRID_CSV}")

    merged = mat.merge(
        test_latents[["row_id", *latent_cols]],
        on="row_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(test_latents):
        raise ValueError(
            f"Merge mismatch: latent test rows={len(test_latents)} but merged rows={len(merged)}"
        )

    feature_cols = [
        col for col in merged.columns
        if col not in {"target", "row_id", "formula", "canonical_formula", "Temperature_K"}
        and pd.api.types.is_numeric_dtype(merged[col])
        and col != "sys_df_original_index"
    ]
    if not feature_cols:
        raise ValueError(f"No numeric hybrid feature columns found after merge")

    y = merged["target"].astype(float).to_numpy()
    X = merged[feature_cols].copy()

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    sample_idx = np.random.choice(len(X), size=min(N_SAMPLES, len(X)), replace=False)
    X_sample = X.iloc[sample_idx].copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    save_shap_plots(
        shap_values,
        X_sample,
        X_sample.columns.tolist(),
        "shap_hybrid",
        "shap_summary_hybrid.png",
        "shap_bar_hybrid.png",
        "Matminer + CrabNet latent + MODNet",
    )
    print(f"Saved hybrid SHAP summary/bar plots using {len(merged)} merged test rows from {HYBRID_CSV}")


if __name__ == "__main__":
    print("Running Matminer SHAP analysis...")
    build_matminer_shap()
    print("Running hybrid SHAP analysis...")
    build_hybrid_shap()
    print("Done.")
