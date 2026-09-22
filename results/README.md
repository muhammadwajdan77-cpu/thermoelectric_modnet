# Current result inventory

This repository is now consolidated around the verified, reproducible artifacts. Stale and superseded files were moved to `archive/superseded_2026_09_21/` so the working tree stays readable and traceable.

## Task 1: Matminer + MODNet baseline
- `results/final/TASK1_MATMINER_MODNET_BASELINE_V3.csv` — reproducible exact-formula benchmark output produced by `python run_grouped_modnet.py --mode full-exact`.
- Source script: `run_grouped_modnet.py`
- Reproduction command: `python run_grouped_modnet.py --mode full-exact`

## Task 2: hybrid model and latent features
- `results/final/TASK2_HYBRID_V4_RESULTS.csv` — current hybrid-v4 results file.
- `results/final/TASK2_HYBRID_V4_FOLD_LATENTS.csv` — current fold-level latent feature export. This file is intentionally not tracked in git because it is a large regenerable intermediate (~207 MB); regenerate locally with `python hybrid_model_v4.py` if raw per-row latent vectors are needed.
- `results/checkpoints/` — checkpoint and MODData artifacts for the training workflow.
- Source script: `hybrid_model_v4.py`
- Reproduction command: `python hybrid_model_v4.py`

## Task 3: exact-formula vs parent-family benchmark
- `results/final/TASK3_MATMINE_STRUCTURE_FEATURES_FULL.csv` — merged structure features for the 965-formula temperature task.
- `results/final/TASK3_PART2_N965_FOLDS.csv` — corrected exact-formula fold results.
- `results/final/TASK3_PART2_N965_PERMUTATION.csv` — permutation control output.
- `results/final/TASK3_PARENT_FAMILY_NATIVE_FOLDS.csv` — corrected parent-family native MODNet fold results.
- Source scripts: `task3_part2_n965.py`, `run_grouped_modnet.py`
- Reproduction commands:
  - `python task3_part2_n965.py --runs 3`
  - `python run_grouped_modnet.py --mode full-parent`

## Readable split exports
- `results/GROUP_KFOLD_SPLITS_READABLE.csv` — export of the exact-formula and parent-family GroupKFold splits in CSV form.

## Archived and superseded artifacts
- `archive/superseded_2026_09_21/` — stale results, debug logs, old pickles, and obsolete scripts retained for traceability and not used in the current pipeline.
