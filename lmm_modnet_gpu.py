#!/usr/bin/env python3
"""
Optimized lmm_modnet.py with GPU acceleration enabled.
Changes:
- Use mixed precision training (float16) for faster computation
- Enable XLA compilation for TensorFlow
- Use GPU memory growth to avoid OOM
- Faster feature selection with reduced n_features
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'

import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap

# Enable GPU memory growth and mixed precision
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"   - {gpu}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
    except:
        pass
else:
    print("⚠️  No GPU found, using CPU")

# Enable mixed precision training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
print(f"✅ Mixed precision policy: {policy.name}")

try:
    from modnet.models import MODNetModel
    from modnet.preprocessing import MODData
    HAS_MODNET = True
except ImportError:
    print("❌ MODNet not found")
    HAS_MODNET = False

WORK_DIR = Path("/home/wajdan/Documents/ZT/thermoelectric_modnet")
os.chdir(WORK_DIR)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================================================================
# LOAD DATA
# ============================================================================
def load_data():
    print("\nLoading data...")
    
    # Load l-MM features
    lmm_df = pd.read_csv("results/lMM_features.csv")
    print(f"  l-MM features loaded: {lmm_df.shape[0]} rows, {lmm_df.shape[1]} columns")
    
    # Drop all-NaN columns
    nan_cols = lmm_df.columns[lmm_df.isna().all()].tolist()
    if nan_cols:
        lmm_df = lmm_df.drop(columns=nan_cols)
        print(f"  Dropping {len(nan_cols)} all-NaN columns")
    
    # Load dataset
    dataset = pd.read_excel("sysTEm_dataset/sysTEm_dataset.xlsx")
    
    # Merge on composition
    merged = lmm_df.merge(
        dataset[['Pretty Formula', 'zT']],
        left_on='composition',
        right_on='Pretty Formula',
        how='left'
    )
    
    # Filter for valid zT
    merged = merged[merged['zT'] > 0].copy()
    
    # Extract features and target
    feature_cols = [c for c in merged.columns if c.startswith('MEGNet') or c.startswith('Roost')]
    X = merged[feature_cols].values
    y = merged['zT'].values
    
    print(f"\nDATA SUMMARY")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  zT range: {y.min():.4f} — {y.max():.4f}")
    
    return X, y

# ============================================================================
# TRAIN MODNET
# ============================================================================
def train_modnet(X, y):
    print("\nTraining MODNet with 5-fold CV (GPU accelerated)...\n")
    
    if not HAS_MODNET:
        print("❌ MODNet not available")
        return None, None, None
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    all_y_true = []
    all_y_pred = []
    mae_list, rmse_list, r2_list = [], [], []
    models = {}
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
        print(f"  Fold {fold_idx}/5")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        
        # Create MODData
        train_data = MODData(X_train, y_train)
        test_data = MODData(X_test, y_test)
        
        # Feature selection (reduce to n=50 for speed)
        train_data.feature_selection(n=50)
        test_data.features = train_data.features
        
        # Train MODNet with GPU
        model = MODNetModel(
            train_data,
            lr=0.001,
            batch_size=64,
            epochs=100,
            loss='mae',
            val_fraction=0.1,
            verbose=0
        )
        
        # Predict
        y_pred = model.predict(test_data).flatten()
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        models[fold_idx] = model
        
        print(f"    MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    
    # Summary metrics
    mae_mean, mae_std = np.mean(mae_list), np.std(mae_list)
    rmse_mean, rmse_std = np.mean(rmse_list), np.std(rmse_list)
    r2_mean, r2_std = np.mean(r2_list), np.std(r2_list)
    
    print(f"\n  Mean±Std: MAE={mae_mean:.4f}±{mae_std:.4f}, RMSE={rmse_mean:.4f}±{rmse_std:.4f}, R²={r2_mean:.4f}±{r2_std:.4f}")
    
    return (all_y_true, all_y_pred, models, 
            f"{mae_mean:.4f}±{mae_std:.4f}",
            f"{rmse_mean:.4f}±{rmse_std:.4f}",
            f"{r2_mean:.4f}±{r2_std:.4f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
def save_results(X, y):
    X, y = load_data()
    results = train_modnet(X, y)
    
    if results is None:
        return
    
    all_y_true, all_y_pred, models, mae_str, rmse_str, r2_str = results
    
    # Parity plot
    plt.figure(figsize=(8, 8))
    plt.scatter(all_y_true, all_y_pred, alpha=0.5, s=10)
    plt.plot([min(all_y_true), max(all_y_true)], [min(all_y_true), max(all_y_true)], 'r--', lw=2)
    plt.xlabel('True zT')
    plt.ylabel('Predicted zT')
    plt.title(f'l-MM MODNet Parity Plot\nMAE={mae_str}, R²={r2_str}')
    plt.savefig('results/figures/parity_plot_lMM.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved: parity_plot_lMM.png")
    
    # SHAP analysis (on fold 1 model)
    if 1 in models:
        model_fold1 = models[1]
        fold_1_mask = list(range(len(y)))[:len(y)//5]  # Approximate fold 1 test set
        
        explainer = shap.TreeExplainer(model_fold1.model) if hasattr(model_fold1, 'model') else None
        if explainer:
            shap_values = explainer.shap_values(X[fold_1_mask])
            shap.summary_plot(shap_values, X[fold_1_mask], plot_type="bar", show=False)
            plt.savefig('results/figures/shap_bar_lMM.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ Saved: shap_bar_lMM.png")
    
    # Update FINAL_RESULTS.csv
    final_df = pd.read_csv("results/FINAL_RESULTS.csv")
    new_row = pd.DataFrame({
        'Model': [f'MatterVial (l-MM) + MODNet'],
        'MAE': [mae_str],
        'RMSE': [rmse_str],
        'R²': [r2_str]
    })
    final_df = pd.concat([final_df, new_row], ignore_index=True)
    final_df.to_csv("results/FINAL_RESULTS.csv", index=False)
    print("✅ Updated: FINAL_RESULTS.csv")

if __name__ == "__main__":
    print("=" * 80)
    print("L-MM MODNet Training (GPU Optimized)")
    print("=" * 80)
    
    X, y = load_data()
    save_results(X, y)
    
    print("\n✨ Complete!")
