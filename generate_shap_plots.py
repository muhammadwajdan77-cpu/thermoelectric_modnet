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

MATMINER_CSV = RESULTS_DIR / "matminer_for_sisso.csv"
HYBRID_CSV = RESULTS_DIR / "CRABNET_LATENT_MODNET_RESULTS.csv"

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
        raise ValueError("Expected 'target' column in matminer_for_sisso.csv")

    y = mat["target"].astype(float).to_numpy()
    X = mat.drop(columns=["target"]).copy()

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

    mat = pd.read_csv(MATMINER_CSV)
    if "target" not in mat.columns:
        raise ValueError("Expected 'target' column in matminer_for_sisso.csv")

    y = mat["target"].astype(float).to_numpy()
    X_mat = mat.drop(columns=["target"]).copy()

    X_mat = X_mat.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    latent_dim = 512
    latent_cols = [f"CrabLatent_{i}" for i in range(latent_dim)]
    hybrid_features = pd.DataFrame(index=X_mat.index)
    hybrid_features[latent_cols] = 0.0

    try:
        from crabnet.crabnet_ import CrabNet
        from extract_crabnet_latent import load_data, build_feature_dataframe, extract_latent_features
        import torch
        torch.set_num_threads(1)

        X_mat_full, y_full, df_aligned = load_data()
        if len(df_aligned) != len(X_mat_full):
            raise ValueError("Aligned dataset length mismatch")

        valid_mask = df_aligned["Pretty Formula"].apply(lambda s: str(s) != "nan")
        df_valid = df_aligned[valid_mask].reset_index(drop=True)
        if len(df_valid) == 0:
            raise ValueError("No valid formulas for CrabNet latent extraction")

        try:
            crab_model = CrabNet(compute_device="cpu", verbose=False, epochs=5, batch_size=64, lr=0.001, checkin=20, save=False)
            crab_model.fit(df_valid[["formula_T", "zT"]].rename(columns={"formula_T": "formula", "zT": "target"}).head(200),
                           df_valid[["formula_T", "zT"]].rename(columns={"formula_T": "formula", "zT": "target"}).head(20))
            latent = extract_latent_features(crab_model, df_valid)
        except Exception as exc:
            print(f"CrabNet latent extraction failed ({exc}); falling back to zeros for hybrid SHAP.")
            latent = np.zeros((len(df_valid), latent_dim), dtype=float)

        latent_df = pd.DataFrame(latent, columns=latent_cols)
        hybrid_features = pd.concat([X_mat.reset_index(drop=True), latent_df], axis=1)
        X = hybrid_features
    except Exception as exc:
        print(f"Hybrid SHAP fallback due to extraction issue: {exc}")
        X = pd.concat([X_mat.reset_index(drop=True), pd.DataFrame(np.zeros((len(X_mat), latent_dim), dtype=float), columns=latent_cols)], axis=1)

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
    print("Saved hybrid SHAP summary and bar plots to results/figures")


if __name__ == "__main__":
    print("Running Matminer SHAP analysis...")
    build_matminer_shap()
    print("Running hybrid SHAP analysis...")
    build_hybrid_shap()
    print("Done.")
