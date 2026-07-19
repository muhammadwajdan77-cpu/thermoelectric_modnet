# Thermoelectric MODNet Benchmarking Repository

This repository contains the experimental workflow, analysis scripts, and result artifacts used to benchmark thermoelectric property prediction models for zT prediction. The project compares composition-based and structure-aware approaches, including MODNet, CrabNet, Matminer, Roost, and hybrid feature pipelines.

## What is in this repository?

- Reproducible training and evaluation scripts for multiple model families
- Feature extraction pipelines for Matminer, ORB, l-MM, and Roost-style representations
- Fair-comparison and leakage-audit experiments
- Saved result tables and figure-generation scripts under the results directory
- A bundled copy of the sysTEm dataset resources in the sysTEm_dataset folder
- A preserved copy of the ProtoCSP-generated CIF structures in data/protocsp_generated_structures for the structure-based comparison workflow
- An archive of earlier pipeline scripts that were later found to contain data leakage / alignment issues

## Repository layout

- main.py: main entry point for the core experiment workflow
- combined_modnet.py: combined Matminer + Roost feature MODNet analysis
- fair_comparison_final.py: fair comparison between composition and structure-aware baselines
- fair_comparison_rigorous_test.py: stricter evaluation with composition-based folds
- roost_modnet.py: Roost/MatterVial feature MODNet baseline
- train_crabnet_zt.py: CrabNet training for zT prediction
- train_crabnet_with_temp.py: CrabNet training with temperature included
- extract_matminer_structure_features.py and extract_matminer_structure_features_v2.py: structure feature extraction
- extract_orb_features.py: ORB feature extraction
- extract_crabnet_latent.py and extract_crabnet_latent_v2.py: CrabNet latent feature extraction
- hybrid_model_v3.py and hybrid_model_v4.py: hybrid latent-feature + MODNet model training
- refine_integrity_check.py: post-hoc validation of data integrity and leakage checks
- generate_shap_plots.py: SHAP visualization generation
- verify_no_leakage.py / check_real_leakage.py: leakage audit and validation checks
- prepare_submission.py: submission packaging helper

## Known Issues & Corrections

Two important bugs were identified and fixed in the corrected pipeline:

1. GroupKFold group-label misalignment
   - The initial Matminer+MODNet baseline used an invalid fold construction method that relied on positional slice alignment instead of skip-invalid-accumulate behavior.
   - This inflated the reported Matminer baseline performance.
   - The fix is implemented with provenance-tracked `canonical_formula` handling in `results/matminer_for_sisso_v2.csv` and the updated cross-validation pipeline in the v2 Matminer workflow.

2. CrabNet latent-feature undertraining for hybrid modeling
   - CrabNet latent-feature extraction originally used an undertrained 1-epoch model for the hybrid model pipeline.
   - The corrected workflow now uses `hybrid_model_v4.py` with proper 300-epoch per-fold training for stable latent feature extraction.

The corrected scripts and outputs are the authoritative pipeline for current results. The superseded buggy scripts are preserved under `archive/buggy_pipeline_v1/` for transparency and debugging reproducibility.

## Verified final results

The corrected benchmark results are:

- Matminer+MODNet: MAE = 0.1347 ± 0.0035, R² = 0.7002 ± 0.0315
- CrabNet+continuous temp: MAE = 0.1234 ± 0.0087, R² = 0.7509 ± 0.0466
- Matminer+CrabNet latent+MODNet: MAE = 0.1250 ± 0.0068, R² = 0.7422 ± 0.0432

## Script-to-output mapping

| Script | Primary output |
| --- | --- |
| main.py | Main experiment orchestration |
| combined_modnet.py | Combined MODNet analysis output and diagnostic parity figures |
| fair_comparison_final.py | Final corrected comparison results |
| fair_comparison_rigorous_test.py | Leakage-free verification results |
| roost_modnet.py | Roost/MatterVial feature baseline results |
| train_crabnet_zt.py | CrabNet zT model training results |
| train_crabnet_with_temp.py | CrabNet temperature-augmented results |
| extract_matminer_structure_features.py | Base Matminer structure features |
| extract_matminer_structure_features_v2.py | Updated Matminer structure features |
| get_matminer_fold_maes.py | Archived Matminer fold evaluation (buggy) |
| extract_crabnet_latent.py | Archived CrabNet latent feature extraction (buggy) |
| extract_crabnet_latent_v2.py | Corrected CrabNet latent feature extraction |
| hybrid_model_v3.py | Hybrid model training variant |
| hybrid_model_v4.py | Corrected hybrid model training with 300-epoch per-fold training |
| refine_integrity_check.py | Corrected integrity and leakage validation checks |
| check_real_leakage.py | Additional leakage analysis helper |
| prepare_submission.py | Submission packaging helper |

## Results files included

- `results/matminer_for_sisso_v2.csv`
- `results/MATMINER_GROUPKFOLD_FOLDS_V2.csv`
- `results/CRABNET_CONTINUOUS_V2_RESULTS.csv`
- `results/HYBRID_V4_RESULTS.csv`
- `results/STATISTICAL_COMPARISON_V2_FINAL.csv`

## How to reproduce

1. Create and activate a Python environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run the corrected workflow using the updated scripts.

```bash
python main.py
```

For targeted experiments, run the relevant updated script directly, for example:

```bash
python extract_crabnet_latent_v2.py
python hybrid_model_v4.py
python refine_integrity_check.py
```

4. Review generated outputs in the `results/` directory and the `archive/buggy_pipeline_v1/` folder for the earlier superseded scripts.

## SysTEm dataset access and citation

This project uses the sysTEm thermoelectric dataset from the bundled `sysTEm_dataset` folder. The dataset is described in the dataset README and should be cited according to the citation text provided there.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
