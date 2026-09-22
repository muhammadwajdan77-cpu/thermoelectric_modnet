This script contains the GroupKFold group-label misalignment bug that 
inflated the Matminer+MODNet baseline. Kept for transparency only — use 
regenerate_matminer_features.py instead.

Note: The incorrect grouping produced the leaky `0.1347` Matminer+MODNet result. The corrected, leakage-free workflow uses `results/matminer_for_sisso_v2.csv` and `results/MATMINER_GROUPKFOLD_FOLDS_V2.csv`.
