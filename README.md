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

- `results/matminer_for_sisso_v2.csv`
- `results/MATMINER_GROUPKFOLD_FOLDS_V2.csv`
- `results/CRABNET_CONTINUOUS_V2_RESULTS.csv`
- `results/HYBRID_V4_RESULTS.csv`
- `results/STRUCTURE_COMPARISON_V2_FINAL.csv`
- `results/STATISTICAL_COMPARISON_V2_FINAL.csv`
- `results/matminer_structure_features.csv`
- `results/CRABNET_ELEMENT_CONTRIBUTIONS.csv`
- `results/CRABNET_ELEMENT_CONTRIBUTIONS_FILTERED.csv`
- `results/group_kfold_splits_v2.pkl`
- `results/figures/crabnet_element_contributions_bar.png`
- `results/figures/crabnet_element_contributions_bar_filtered.png`
- `results/figures/shap_bar_hybrid.png`
- `results/figures/shap_bar_matminer.png`
- `results/figures/shap_summary_hybrid.png`
- `results/figures/shap_summary_matminer.png`
- `results/figures/shap_hybrid_top10.csv`
- `results/figures/shap_matminer_top10.csv`

## Supported datasets and submodules

- `data/protocsp_generated_structures/`
- `ProtoCSP/`
- `MatterVial/`
- `sysTEm_dataset/`

## Archive

- `archive/buggy_pipeline_v1/get_matminer_fold_maes.py`
- `archive/buggy_pipeline_v1/README.md`

This archive preserves a single buggy script for transparency only.

## How to use

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
