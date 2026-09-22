# Thermoelectric MODNet Benchmarking Repository

This repository contains the trimmed, verified experiment workflow and final reported artifacts for thermoelectric property benchmarking.

## Included deliverables

- `README.md`, `LICENSE`, `requirements.txt`, `.gitignore`, `.gitmodules`
- `regenerate_matminer_features.py`
- `train_crabnet_with_temp.py`
- `hybrid_model_v4.py`
- `extract_matminer_structure_features_v2.py`
- `structure_comparison_v2.py`
- `generate_shap_plots.py`
- `crabnet_element_contributions.py`

## Final result artifacts

- `results/final/TASK1_MATMINER_MODNET_BASELINE_V3.csv`
- `results/final/CRABNET_CONTINUOUS_RESULTS.csv`
- `results/final/CRABNET_CONTINUOUS_V2_RESULTS.csv`
- `results/final/TASK2_HYBRID_V4_RESULTS.csv`
- `results/final/TASK3_MATMINE_STRUCTURE_FEATURES_FULL.csv`
- `results/final/TASK3_PART2_N965_FOLDS.csv`
- `results/final/TASK3_PART2_N965_PERMUTATION.csv`
- `results/final/TASK3_PARENT_FAMILY_NATIVE_FOLDS.csv`
- `results/GROUP_KFOLD_SPLITS_READABLE.csv`

## Supported datasets and submodules

- `data/protocsp_generated_structures/`
- `ProtoCSP/`
- `MatterVial/`
- `sysTEm_dataset/`

## Archive

- `archive/buggy_pipeline_v1/get_matminer_fold_maes.py`
- `archive/buggy_pipeline_v1/README.md`

This archive preserves a single buggy script for transparency only.

## Review findings and fixes

Rogério's review covered four real issues. The current authoritative outputs are in `results/final/`, and the final benchmarking workflow is the 965-formula, row-level temperature dataset used by `task3_part2_n965.py`.

1. Matminer + MODNet baseline reproducibility
   - Reported MAE = 0.1347 did not match the repo CSV (0.2240) and no script in the repo reproduced that number.
   - Investigation showed the dedup-based fix attempt was itself invalid because it discarded legitimate temperature-distinct measurements from the same composition.
   - Honest baseline: parent-family GroupKFold on the final dataset gives MAE = 0.2395 and R² = 0.230, with the documented fold-variance caveat.
   - Authoritative result: `results/final/TASK1_MATMINER_MODNET_BASELINE_V3.csv`

2. CrabNet + continuous temperature
   - The minor discrepancy was attributable to rerun variance and does not represent a model issue requiring further action.
   - Authoritative results: `results/final/CRABNET_CONTINUOUS_RESULTS.csv` and `results/final/CRABNET_CONTINUOUS_V2_RESULTS.csv`

3. Hybrid model (Matminer + CrabNet-latent + MODNet)
   - `crabnet_.py` was the standard installed package, not a missing custom wrapper; it is now version-pinned to avoid this confusion.
   - Row-order preservation in `extract_latent_features()` was verified end-to-end, and the silent zero-fallback was removed in favor of fail-loud behavior (it never fired in the reported results).
   - SHAP analysis was rerun on the correctly trained model.
   - Authoritative aggregate result: `results/final/TASK2_HYBRID_V4_RESULTS.csv`
   - The per-fold latent export is intentionally excluded from git because it is a large regenerable intermediate; recreate it locally with `python hybrid_model_v4.py` if needed.

4. Structure comparison
   - The structure features were regenerated using MODNet's proper structure featurizer, yielding 1768 features rather than the stale 13-feature output.
   - The comparison was filtered to `rank_1` structures only with deterministic tie-breaking, leaving 965 usable formulas out of 1262 after timeouts/failures.
   - The final comparison was run on the full temperature-resolved data with both exact-formula and parent-family grouping; this matched Rogério's original finding of no robust structure benefit.
   - Authoritative results: `results/final/TASK3_MATMINE_STRUCTURE_FEATURES_FULL.csv`, `results/final/TASK3_PART2_N965_FOLDS.csv`, and `results/final/TASK3_PARENT_FAMILY_NATIVE_FOLDS.csv`

The current authoritative artifacts live under `results/final/`. Stale and superseded outputs were archived under `archive/superseded_2026_09_21/` so the active repository remains reproducible and traceable.

## How to use

Clone with submodules initialized:

```bash
git clone --recurse-submodules https://github.com/muhammadwajdan77-cpu/thermoelectric_modnet.git
# or, if already cloned without --recurse-submodules:
git submodule update --init --recursive
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

Run the corrected scripts directly:

```bash
python regenerate_matminer_features.py
python train_crabnet_with_temp.py
python hybrid_model_v4.py
python extract_matminer_structure_features_v2.py
python structure_comparison_v2.py
python generate_shap_plots.py
python crabnet_element_contributions.py
```

## License

Licensed under the MIT License. See `LICENSE` for details.
