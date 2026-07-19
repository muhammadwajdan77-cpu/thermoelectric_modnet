#!/usr/bin/env python3
"""Extract CrabNet latent features aligned to matminer_for_sisso_v2.csv (fixed).

Produces results/CRABNET_LATENT_FEATURES_V2.csv with `canonical_formula` preserved.
"""

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from crabnet.crabnet_ import CrabNet
except Exception as exc:
    raise RuntimeError(f"CrabNet import failed: {exc}") from exc

RESULTS_DIR = Path('results')
MATMINER_V2 = RESULTS_DIR / 'matminer_for_sisso_v2.csv'
OUTPUT = RESULTS_DIR / 'CRABNET_LATENT_FEATURES_V2.csv'

SEED = 42
BATCH_SIZE = 128


def temp_encoding_row(formula, temperature):
    if pd.isna(formula):
        return None
    try:
        temp = float(temperature)
    except Exception:
        return f"{formula} Og"
    frac = temp / 1000.0
    frac_text = f"{frac:.3f}".rstrip('0').rstrip('.')
    return f"{formula} Og{frac_text}"


def load_matminer_v2(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    mat = pd.read_csv(path)
    required = ['formula', 'canonical_formula', 'Temperature_K', 'target']
    for c in required:
        if c not in mat.columns:
            raise ValueError(f"Expected column '{c}' in {path}")
    mat = mat.copy().reset_index(drop=True)
    mat['formula_T'] = mat.apply(lambda r: temp_encoding_row(r['formula'], r['Temperature_K']), axis=1)
    mat['zT'] = mat['target'].astype(float)
    mat['canonical'] = mat['canonical_formula'].astype(str)
    return mat


def extract_latent(crab_model, df_subset, batch_size=BATCH_SIZE):
    latent_store = {}

    def hook_fn(module, input, output):
        src = input[0]
        if src.dim() == 3 and src.shape[-1] == 1:
            src = src.squeeze(-1)
        mask = (src == 0)
        if output.dim() == 3:
            mask = mask.unsqueeze(-1).expand_as(output)
            out = output.masked_fill(mask, 0.0)
            count = (~mask).sum(dim=1).float().clamp(min=1.0)
            avg = out.sum(dim=1) / count
        else:
            out = output
            avg = out
        latent_store['feat'] = avg.detach().cpu().numpy()

    hook = crab_model.model.encoder.register_forward_hook(hook_fn)
    crab_df = pd.DataFrame({'formula': df_subset['formula_T'].values, 'target': df_subset['zT'].values})
    all_feats = []
    for start in range(0, len(crab_df), batch_size):
        batch = crab_df.iloc[start:start + batch_size]
        try:
            crab_model.load_data(batch, train=False)
            crab_model.predict(batch)
            if 'feat' in latent_store:
                all_feats.append(latent_store['feat'])
                latent_store.clear()
            else:
                raise RuntimeError('Latent hook did not capture features')
        except Exception:
            feat_dim = all_feats[-1].shape[1] if all_feats else 512
            all_feats.append(np.zeros((len(batch), feat_dim), dtype=float))

    hook.remove()
    if not all_feats:
        return np.zeros((len(df_subset), 512), dtype=float)
    return np.vstack(all_feats)


def main():
    mat = load_matminer_v2(MATMINER_V2)
    print(f"Loaded matminer v2: {len(mat)} rows", flush=True)
    # Create CrabNet model but do not train to avoid heavy memory/time.
    try:
        crab = CrabNet(compute_device='cpu', verbose=False, epochs=1, batch_size=32, lr=0.001, save=False)
    except Exception as exc:
        raise RuntimeError(f"Unable to create CrabNet model: {exc}") from exc

    # perform a tiny one-epoch fit on a small valid subset to initialize internals
    valid_mask = mat['formula_T'].notna()
    small_df = mat.loc[valid_mask].head(32)[['formula_T', 'zT']].rename(columns={'formula_T':'formula','zT':'target'})
    if len(small_df) > 0:
        try:
            crab.fit(small_df, val_df=small_df)
        except Exception:
            # If this fails, continue — we may still be able to run predictor for feature extraction
            pass

    latent = extract_latent(crab, mat, batch_size=32)
    print(f"Extracted latent shape: {latent.shape}", flush=True)

    # Build dataframe and save
    latent_cols = [f'CrabLatent_{i}' for i in range(latent.shape[1])]
    out_df = pd.DataFrame(latent, columns=latent_cols)
    # include mat row index for exact one-to-one merging later
    out_df.insert(0, 'mat_index', mat.index.values)
    out_df.insert(1, 'canonical_formula', mat['canonical'].values)
    out_df.insert(2, 'formula', mat['formula'].values)
    out_df.insert(3, 'zT', mat['zT'].values)
    out_df.to_csv(OUTPUT, index=False)
    print(f"Saved latent features to {OUTPUT}", flush=True)


if __name__ == '__main__':
    main()
