# Combined MODNet Script - Documentation

## Overview
`combined_modnet.py` combines pre-computed Matminer and MatterVial Roost features to train a MODNet model for zT prediction.

## Features

### 1. Feature Loading & Combination
- **Matminer Features**: 132 numerical features (excluding target and temperature)
- **Roost Features**: 128 learned representations from OQMD
- **Combined**: ~261 features (after removing duplicates)
- **Samples Aligned**: 7,594 samples across both datasets
- **Valid zT Samples** (zT > 0): 7,082 samples

### 2. 5-Fold Cross-Validation
- **Seed**: 42 (reproducible, same splits as baseline models)
- **Framework**: MODNet with pre-computed features via `df_featurized`
- **Feature Selection**: Automatic (min(30, n_features) selected)
- **Training**:
  - Optimizer: Adam with lr=0.001
  - Loss: MAE (Mean Absolute Error)
  - Batch size: 64
  - Epochs: 30
  - Validation fraction: 10%

### 3. Outputs

#### Parity Plot
- **Location**: `results/figures/parity_plot_combined.png`
- **Format**: Scatter plot with perfect prediction line
- **Metrics**: MAE, RMSE, R² displayed on plot

#### Results Table
- **Location**: `results/results_complete.csv`
- **Format**: Fold-by-fold metrics + Mean±Std summary
- **Columns**: Fold, MAE, RMSE, R², Model
- **Models Compared**:
  1. Matminer + MODNet (Baseline)
  2. MatterVial (Roost) + MODNet
  3. Combined (Matminer + Roost) + MODNet

## Data Processing

### Alignment Strategy
1. Load Matminer features (7594 samples)
2. Load Roost features (8237 samples)
3. Align to minimum (7594 samples)
4. Concatenate horizontally (260 combined features)
5. Add back Temperature_K from original dataset
6. Filter for positive zT only (7082 valid samples)

### Feature Handling
- **Imputation**: Mean strategy for missing values
- **Normalization**: Not applied (MODNet handles internally)
- **Duplicates**: Automatically removed (keep first occurrence)

## Expected Results

Based on the individual baseline models:
- **Matminer + MODNet**: MAE ≈ 0.1130, R² ≈ 0.7877
- **Roost + MODNet**: MAE ≈ 0.1206, R² ≈ 0.7336

**Combined Model Expected**: ~0.11-0.12 MAE, ~0.75-0.79 R² (synergy expected from feature combination)

## Usage

```bash
# Run the combined MODNet analysis
python combined_modnet.py

# Monitor progress (in another terminal)
tail -f /dev/null  # Check when parity_plot_combined.png appears
```

## Requirements

- `modnet`: Neural network model architecture
- `pymatgen`: Structure handling and validation
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: Cross-validation and metrics
- `matplotlib`: Visualization

## Performance Notes

- **Feature Selection**: MODNet computes Mutual Information for all 261 features (~6-10 seconds per fold)
- **Model Training**: ~5-10 minutes per fold
- **Total Runtime**: ~30-60 minutes for full 5-fold CV
- **GPU**: Optional (MODNet uses TensorFlow backend)

## Reproducibility

All parameters set for reproducible results:
- Random seed: 42
- KFold splits: deterministic
- Feature selection: deterministic
- No data augmentation

## Future Improvements

1. **Hyperparameter Tuning**:
   - Learning rate optimization (currently 0.001)
   - Batch size variation (currently 64)
   - Epoch count tuning (currently 30)

2. **Feature Selection**:
   - Increase from 30 to 50-80 features with cross-validation
   - Analyze SHAP importance of combined features

3. **Ensemble Methods**:
   - Combine predictions from all three models (Matminer, Roost, Combined)
   - Weighted ensemble based on individual performance

4. **Physical Insights**:
   - Extract feature importance rankings
   - Correlate top features with material properties
   - Identify synergistic feature pairs

## Related Files

- `matminer_modnet.py`: Matminer-only baseline
- `roost_modnet.py`: Roost-only baseline  
- `results/matminer_for_sisso.csv`: Pre-computed Matminer features
- `results/roost_features.csv`: Pre-computed Roost features
- `results/results_complete.csv`: Final comparison table
- `results/figures/parity_plot_combined.png`: Visualization
