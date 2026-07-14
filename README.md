# Thermoelectric MODNet Benchmarking Repository

This repository contains the experimental workflow, analysis scripts, and result artifacts used to benchmark thermoelectric property prediction models for zT prediction. The project compares composition-based and structure-aware approaches, including MODNet, CrabNet, Matminer, Roost, and hybrid feature pipelines.

## What is in this repository?

- Reproducible training and evaluation scripts for multiple model families
- Feature extraction pipelines for Matminer, ORB, l-MM, and Roost-style representations
- Fair-comparison and leakage-audit experiments
- Saved result tables and figure-generation scripts under the results directory
- A bundled copy of the sysTEm dataset resources in the sysTEm_dataset folder
- A preserved copy of the ProtoCSP-generated CIF structures in data/protocsp_generated_structures for the structure-based comparison workflow

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
- extract_crabnet_latent.py: latent feature extraction for CrabNet
- generate_shap_plots.py: SHAP visualization generation
- verify_no_leakage.py: leakage audit and validation checks
- prepare_submission.py: submission packaging helper

## Installation

This repository was developed with Python 3.10+ and uses a pinned environment in the root requirements file.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For sub-project-specific dependencies, see:

- MatterVial/requirements.txt
- ProtoCSP/requirements.txt
- sysTEm_dataset/requirements.txt

## Requirements

The root requirements file contains the core dependencies used by the main analysis workflow:

- numpy
- pandas
- scikit-learn
- pymatgen
- matplotlib
- seaborn
- tensorflow
- modnet
- tqdm
- requests
- openpyxl

## Scripts and outputs

| Script | Primary output |
| --- | --- |
| main.py | Main experiment orchestration |
| combined_modnet.py | results/results_complete.csv and parity plots under results/figures |
| fair_comparison_final.py | results/FAIR_COMPARISON_FINAL.csv |
| fair_comparison_rigorous_test.py | results/FINAL_RESULTS_FIXED_LEAKAGE_FREE.csv |
| roost_modnet.py | Roost/MatterVial feature baseline results |
| train_crabnet_zt.py | results/CRABNET_RESULTS.csv |
| train_crabnet_with_temp.py | results/CRABNET_TEMP_RESULTS.csv |
| extract_matminer_structure_features.py | results/matminer_structure_features.csv |
| extract_matminer_structure_features_v2.py | results/matminer_structure_features.csv (updated variant) |
| extract_orb_features.py | results/ORB_features.csv |
| extract_crabnet_latent.py | results/CRABNET_LATENT_MODNET_RESULTS.csv |
| generate_shap_plots.py | SHAP-related CSVs and figures |
| verify_no_leakage.py | leakage audit output and validation reports |
| prepare_submission.py | submission-ready packaging helper |

## Current reported results

The current saved benchmark tables indicate the following best-performing result in results/FINAL_RESULTS.csv:

- Best model: Matminer + MODNet
- MAE: 0.1130 ± 0.0068
- R²: 0.7877 ± 0.0196

A stricter leakage-free comparison in results/FINAL_RESULTS_FIXED_LEAKAGE_FREE.csv reports a different, more conservative result:

- Best leakage-free model: Matminer+ROOST+l-OFM+MVL+ORB (XGBoost RFE) + MODNet (Composition-Stratified)
- MAE: 0.1448 ± 0.0150
- R²: 0.6574 ± 0.0800

## How to reproduce

1. Create and activate a Python environment.
2. Install dependencies with pip install -r requirements.txt.
3. Run the main workflow:

```bash
python main.py
```

4. For targeted experiments, run the relevant script directly, for example:

```bash
python combined_modnet.py
python fair_comparison_final.py
python verify_no_leakage.py
```

5. Review generated outputs in the results directory and the figures directory.

## SysTEm dataset access and citation

This project uses the sysTEm thermoelectric dataset from the bundled sysTEm_dataset folder. The dataset is described in the dataset README and should be cited according to the citation text provided there. If you use the dataset in published work, follow the citation guidance in sysTEm_dataset/README.md.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
