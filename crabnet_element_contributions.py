#!/usr/bin/env python3
"""Estimate element-level contribution scores for a trained CrabNet model.

This script loads the saved CrabNet checkpoint bundled in this repository,
parses the sysTEm dataset with temperature-encoded formulas (pseudo-element Og),
extracts per-element embedding representations through a forward hook, and
aggregates a simple contribution proxy per element based on embedding magnitude.

Important: this is not a direct reimplementation of the exact attention-based
analysis from the CrabNet paper. The paper analyzes attention behavior; in this
script we use the mean magnitude of the element embedding vectors (and the
magnitude of their pooled contribution) as a practical proxy when attention
weights are not easily exposed by the installed CrabNet version.
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pymatgen.core.composition import Composition

try:
    from crabnet.crabnet_ import CrabNet
    from crabnet.kingcrab import SubCrab
    from crabnet.utils.composition import parse_formula
except Exception as exc:  # pragma: no cover - environment-dependent
    raise RuntimeError(f"CrabNet import failed: {exc}") from exc

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "sysTEm_dataset" / "sysTEm_dataset.xlsx"
CHECKPOINT_PATH = PROJECT_DIR / "models" / "trained_models" / "UnnamedModel.pth"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
OUTPUT_CSV = RESULTS_DIR / "CRABNET_ELEMENT_CONTRIBUTIONS.csv"
FILTERED_OUTPUT_CSV = RESULTS_DIR / "CRABNET_ELEMENT_CONTRIBUTIONS_FILTERED.csv"
BAR_PLOT = FIGURES_DIR / "crabnet_element_contributions_bar.png"
FILTERED_BAR_PLOT = FIGURES_DIR / "crabnet_element_contributions_bar_filtered.png"
PERIODIC_PLOT = FIGURES_DIR / "crabnet_element_contributions_periodic.png"
MIN_SAMPLE_COUNT = 30

warnings.filterwarnings("ignore")


def print_path(label: str, path: Path) -> None:
    print(f"{label}: {path}")


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


def is_valid_formula(formula) -> bool:
    s = str(formula or "").strip()
    if not s:
        return False
    if re.search(r"wt%|vol%|%|\+", s):
        return False
    return True


def temp_to_element(temperature) -> str:
    if pd.isna(temperature):
        return "Og0.0"
    temp = float(temperature)
    x = round(temp / 1000, 3)
    return f"Og{x}"


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
        raise ValueError(f"Expected columns not found. Columns: {list(df.columns)}")

    df = df[[formula_col, target_col, temp_col]].copy()
    df.columns = ["Pretty Formula", "zT", "Temperature_K"]

    df = df.loc[df["zT"].notna() & df["Pretty Formula"].notna()].copy()
    df = df.loc[df["zT"] > 0].copy()
    df["Pretty Formula"] = df["Pretty Formula"].astype(str).str.strip()
    df = df.loc[df["Pretty Formula"].apply(is_valid_formula)].copy()
    df["canonical_formula"] = df["Pretty Formula"].apply(canonical_formula)
    df = df.loc[df["canonical_formula"].notna()].copy()
    df["canonical_formula"] = df["canonical_formula"].astype(str).str.strip()
    df["formula_temp"] = df["Pretty Formula"] + " " + df["Temperature_K"].apply(temp_to_element)
    df["formula"] = df["formula_temp"]
    df["target"] = df["zT"].astype(float)

    print(f"Loaded dataset rows: {len(df)}")
    print(f"Unique canonical formulas: {df['canonical_formula'].nunique()}")
    return df[["formula", "target", "canonical_formula", "Pretty Formula", "Temperature_K"]].copy()


def init_model(checkpoint_path: Path, device: str = "cpu") -> CrabNet:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", "UnnamedModel")
    print(f"Loaded checkpoint model_name={model_name}")
    print(f"Checkpoint keys: {list(checkpoint.keys())}")

    wrapper = CrabNet(model_name=model_name, verbose=False, save=False, compute_device=device)
    inner_model = SubCrab(
        compute_device=device,
        out_dims=wrapper.out_dims,
        d_model=wrapper.d_model,
        N=wrapper.N,
        heads=wrapper.heads,
        emb_scaler=wrapper.emb_scaler,
        pos_scaler=wrapper.pos_scaler,
        pos_scaler_log=wrapper.pos_scaler_log,
        dim_feedforward=wrapper.dim_feedforward,
        dropout=wrapper.dropout,
        elem_prop=wrapper.elem_prop,
        out_hidden=wrapper.out_hidden,
    )
    wrapper.model = inner_model

    if isinstance(checkpoint, dict) and "weights" in checkpoint:
        state_dict = checkpoint["weights"]
        if hasattr(wrapper, "load_state_dict"):
            incompatible = wrapper.load_state_dict(state_dict, strict=False)
            print(f"Load state dict strict=False; missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}")
        else:
            raise RuntimeError("Loaded CrabNet model does not expose load_state_dict")
    else:
        raise RuntimeError("Checkpoint format was not recognized")

    wrapper.eval()
    wrapper.model.eval()
    return wrapper


def extract_element_embeddings(model: CrabNet, formulas: List[str]) -> np.ndarray:
    """Return per-element embedding magnitudes for each formula.

    The installed CrabNet version exposes the encoder output via the internal
    SubCrab model. We register a hook on the encoder output and capture a simple
    proxy signal for each element before pooling. This is not a direct attention-
    weight extraction, but it is a reasonable element-level contribution proxy.
    """

    embedding_store: List[np.ndarray] = []
    hook = None

    def _hook(module, inputs, output):
        if isinstance(output, torch.Tensor):
            tensor = output.detach().cpu()
            if tensor.dim() == 3 and tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)
            if tensor.dim() == 3:
                # Keep the per-element embedding signal before pooling.
                tensor = tensor.mean(dim=1)
            embedding_store.append(tensor.numpy())

    try:
        hook = model.model.encoder.register_forward_hook(_hook)
        batch_df = pd.DataFrame({"formula": formulas[:256], "target": [0.0] * min(256, len(formulas))})
        model.load_data(batch_df, train=False)
        model.predict(batch_df)
        if not embedding_store:
            raise RuntimeError("No embedding output captured from hook")
    finally:
        if hook is not None:
            hook.remove()

    return embedding_store[-1]


def parse_element_contributions(formulas: List[str], model: CrabNet) -> pd.DataFrame:
    """Estimate element-level contribution proxy from the model's encoder output.

    For each formula, we parse the element fractions, run the underlying SubCrab
    forward pass directly, and aggregate a simple proxy based on the magnitude of
    the encoder output associated with each element. The result is the mean and
    standard deviation of that proxy across compositions containing each element.
    """
    all_scores: Dict[str, List[float]] = {}
    batch_size = 256

    for start in range(0, len(formulas), batch_size):
        batch_formulas = formulas[start:start + batch_size]
        batch_df = pd.DataFrame({"formula": batch_formulas, "target": [0.0] * len(batch_formulas)})
        try:
            model.load_data(batch_df, train=False)
            loader = model.data_loader
        except Exception as exc:
            print(f"Data loading failed for batch {start}: {exc}", flush=True)
            continue

        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(loader):
                    X, y, formula, extra_features = batch
                    src, frac = X.squeeze(-1).chunk(2, dim=1)
                    src = src.to(model.compute_device, dtype=torch.long)
                    frac = frac.to(model.compute_device, dtype=model.data_type_torch)
                    extra_features = extra_features.to(model.compute_device, dtype=model.data_type_torch)
                    _ = model.model.forward(src, frac, extra_features=extra_features)

                    # Capture the encoder output directly via the model's internal encoder.
                    # We use the L2 norm of the per-element encoder output as a proxy
                    # for element-level contribution.
                    encoder_output = model.model.encoder(src, frac, extra_features)
                    # encoder_output has shape [batch, n_atoms, d_model]
                    if encoder_output.dim() == 3:
                        elem_norms = encoder_output.norm(dim=-1)
                        # zero out padded atoms
                        pad_mask = (src == 0)
                        elem_norms = elem_norms * (~pad_mask).float()

                        for row_idx, row_norms in enumerate(elem_norms.cpu().numpy()):
                            parsed = parse_formula(str(formula[row_idx]))
                            if not isinstance(parsed, dict):
                                continue
                            elements = list(parsed.keys())
                            for element in elements:
                                all_scores.setdefault(element, []).append(float(np.max(row_norms)))
        except Exception as exc:
            print(f"Element scoring failed for batch {start}: {exc}", flush=True)

    if not all_scores:
        raise RuntimeError("No element contributions were extracted")

    rows = []
    for element, values in sorted(all_scores.items()):
        rows.append(
            {
                "element": element,
                "mean_contribution": float(np.mean(values)),
                "std_contribution": float(np.std(values)),
                "n_compositions_containing": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def plot_bar(results_df: pd.DataFrame, save_path: Path, title: str) -> None:
    top = results_df.sort_values("mean_contribution", ascending=False).head(20).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [f"{element} (n={int(count)})" for element, count in zip(top["element"].astype(str), top["n_compositions_containing"].astype(int))]
    x_positions = np.arange(len(top))
    ax.bar(x_positions, top["mean_contribution"].astype(float))
    ax.set_title(title)
    ax.set_ylabel("Mean contribution proxy")
    ax.set_xlabel("Element")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_periodic(results_df: pd.DataFrame, save_path: Path) -> None:
    try:
        from pymatgen.util.plotting import periodic_table_heatmap
    except Exception as exc:
        print(f"pymatgen periodic table plotting unavailable; skipping periodic heatmap ({exc})")
        return

    data = results_df.set_index("element")["mean_contribution"].to_dict()
    try:
        fig = periodic_table_heatmap(data, cmap="viridis")
    except Exception as exc:
        print(f"Periodic heatmap generation failed; skipping ({exc})")
        return

    if hasattr(fig, "savefig"):
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        print("Periodic heatmap object did not expose savefig; skipping")


def main() -> None:
    print("=" * 70)
    print("CRABNET ELEMENT CONTRIBUTION ANALYSIS")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print_path("Dataset", DATA_PATH)
    print_path("Checkpoint", CHECKPOINT_PATH)
    print_path("Output CSV", OUTPUT_CSV)
    print_path("Bar plot", BAR_PLOT)
    print_path("Periodic plot", PERIODIC_PLOT)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset missing: {DATA_PATH}")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint missing: {CHECKPOINT_PATH}")

    df = load_dataset(DATA_PATH)
    model = init_model(CHECKPOINT_PATH, device="cpu")

    formulas = df["formula"].astype(str).tolist()
    results_df = parse_element_contributions(formulas, model)
    results_df["reliable"] = results_df["n_compositions_containing"] >= MIN_SAMPLE_COUNT
    results_df = results_df.sort_values("mean_contribution", ascending=False).reset_index(drop=True)
    results_df.to_csv(OUTPUT_CSV, index=False)

    filtered_df = results_df.loc[results_df["reliable"]].copy()
    filtered_df = filtered_df.sort_values("mean_contribution", ascending=False).reset_index(drop=True)
    filtered_df.to_csv(FILTERED_OUTPUT_CSV, index=False)

    plot_bar(results_df, BAR_PLOT, "CrabNet element contribution proxy (all elements)")
    plot_bar(filtered_df, FILTERED_BAR_PLOT, "CrabNet element contribution proxy (reliable elements, n>=30)")
    plot_periodic(results_df, PERIODIC_PLOT)

    corr = results_df["n_compositions_containing"].corr(results_df["mean_contribution"])

    print("\nTop 10 elements by contribution proxy (unfiltered):")
    print(results_df.head(10).to_string(index=False))

    print("\nTop 10 elements by contribution proxy (filtered, n>=30):")
    print(filtered_df.head(10).to_string(index=False))

    dropped_from_top10 = results_df.head(10).loc[~results_df.head(10)["reliable"], ["element", "n_compositions_containing"]]
    if dropped_from_top10.empty:
        print("\nNo elements from the unfiltered top 10 were dropped by the reliability filter.")
    else:
        print("\nElements from unfiltered top 10 dropped by the reliability filter:")
        print(dropped_from_top10.to_string(index=False))

    print(f"\nPearson correlation (n_compositions_containing vs mean_contribution): {corr:.6f}")
    print(f"\nSaved contribution CSV to {OUTPUT_CSV}")
    print(f"Saved filtered contribution CSV to {FILTERED_OUTPUT_CSV}")
    print(f"Saved bar chart to {BAR_PLOT}")
    print(f"Saved filtered bar chart to {FILTERED_BAR_PLOT}")
    print(f"Saved periodic chart to {PERIODIC_PLOT} if available")


if __name__ == "__main__":
    main()
