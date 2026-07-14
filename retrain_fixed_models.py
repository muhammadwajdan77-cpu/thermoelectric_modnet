"""Retrain fixed MODNet models on aligned feature sets and update final results."""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from modnet.models import MODNetModel
from modnet.preprocessing import MODData

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

SEED = 42
N_SPLITS = 5
LR = 0.001
BATCH_SIZE = 64
EPOCHS = 100
LOSS = "mae"
N_FEAT = 50
TARGET_NAME = "ZT"


def numeric_feature_matrix(df: pd.DataFrame, target_column: str = "zT") -> pd.DataFrame:
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    if target_column in df_numeric.columns:
        df_numeric = df_numeric.drop(columns=[target_column])
    return df_numeric


def build_combined_features(matminer_path: Path, roost_path: Path):
    matminer = pd.read_csv(matminer_path)
    roost = pd.read_csv(roost_path)

    if "canonical" in matminer.columns and "canonical" in roost.columns:
        if not matminer["canonical"].astype(str).reset_index(drop=True).equals(
            roost["canonical"].astype(str).reset_index(drop=True)
        ):
            raise ValueError(
                "Combined feature load mismatch: canonical order differs between aligned_matminer.csv and aligned_roost.csv"
            )

    x_matminer = numeric_feature_matrix(matminer.drop(columns=["target"], errors="ignore"))
    x_roost = numeric_feature_matrix(roost)

    if "zT" not in matminer.columns and "zT" not in roost.columns:
        raise ValueError("Cannot find target zT in aligned_matminer.csv or aligned_roost.csv")

    y = matminer["zT"] if "zT" in matminer.columns else roost["zT"]
    if "zT" in matminer.columns and "zT" in roost.columns:
        if not np.allclose(matminer["zT"].fillna(0), roost["zT"].fillna(0), equal_nan=True):
            raise ValueError("zT values differ between aligned_matminer.csv and aligned_roost.csv")

    x_combined = pd.concat([x_matminer, x_roost], axis=1)
    x_combined = x_combined.loc[:, ~x_combined.columns.duplicated(keep="first")]

    return x_combined, y


def build_lmm_features(lmm_path: Path):
    lmm = pd.read_csv(lmm_path)
    if "zT" not in lmm.columns:
        raise ValueError("Cannot find target zT in aligned_lMM.csv")

    y = lmm["zT"].astype(float)
    x_lmm = numeric_feature_matrix(lmm)
    return x_lmm, y


def train_cv_modnet(X: pd.DataFrame, y: pd.Series, model_label: str):
    mask = y > 0
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    print(f"\n{'='*70}")
    print(f"TRAINING {model_label}")
    print(f"{'='*70}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target samples: {len(y)}  zT range: {y.min():.4f} — {y.max():.4f}")

    imputer = SimpleImputer(strategy="mean")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_maes = []
    fold_rmses = []
    fold_r2s = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_imp)):
        print(f"\n  Fold {fold + 1}/{N_SPLITS}")
        X_train = X_imp.iloc[train_idx].reset_index(drop=True)
        X_test = X_imp.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        train_data = MODData(
            materials=list(range(len(train_idx))),
            targets=[[v] for v in y_train.values],
            target_names=[TARGET_NAME],
        )
        train_data.df_featurized = X_train
        train_data.feature_selection(n=N_FEAT)

        model = MODNetModel([[[TARGET_NAME]]], weights={TARGET_NAME: 1}, n_feat=N_FEAT)
        model.fit(
            train_data,
            val_fraction=0.1,
            lr=LR,
            batch_size=BATCH_SIZE,
            loss=LOSS,
            epochs=EPOCHS,
            verbose=0,
        )

        test_data = MODData(
            materials=list(range(len(test_idx))),
            targets=[[0]] * len(test_idx),
            target_names=[TARGET_NAME],
        )
        test_data.df_featurized = X_test
        y_pred = model.predict(test_data)[TARGET_NAME].values

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"    MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_r2s.append(r2)

    mean_mae = np.mean(fold_maes)
    std_mae = np.std(fold_maes)
    mean_rmse = np.mean(fold_rmses)
    std_rmse = np.std(fold_rmses)
    mean_r2 = np.mean(fold_r2s)
    std_r2 = np.std(fold_r2s)

    print(f"\n{'-'*70}")
    print(f"{model_label} RESULTS")
    print(f"{'-'*70}")
    print(f"MAE:  {mean_mae:.4f} ± {std_mae:.4f}")
    print(f"RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"R²:   {mean_r2:.4f} ± {std_r2:.4f}")

    return {
        "MAE": mean_mae,
        "MAE_std": std_mae,
        "RMSE": mean_rmse,
        "RMSE_std": std_rmse,
        "R2": mean_r2,
        "R2_std": std_r2,
    }


def update_final_results(results_path: Path, metrics: dict):
    df = pd.read_csv(results_path)
    for model_name, metric in metrics.items():
        row_mask = df["Model"] == model_name
        values = {
            "MAE": f"{metric['MAE']:.4f}±{metric['MAE_std']:.4f}",
            "RMSE": f"{metric['RMSE']:.4f}±{metric['RMSE_std']:.4f}",
            "R2": f"{metric['R2']:.4f}±{metric['R2_std']:.4f}",
        }
        if row_mask.any():
            df.loc[row_mask, ["MAE", "RMSE", "R2"]]
            df.loc[row_mask, ["MAE", "RMSE", "R2"]] = [values["MAE"], values["RMSE"], values["R2"]]
        else:
            new_row = {"Model": model_name, **values}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(results_path, index=False)
    print(f"\nUpdated: {results_path}")


if __name__ == "__main__":
    combined_features, combined_target = build_combined_features(
        RESULTS_DIR / "aligned_matminer.csv",
        RESULTS_DIR / "aligned_roost.csv",
    )
    lmm_features, lmm_target = build_lmm_features(RESULTS_DIR / "aligned_lMM.csv")

    combined_metrics = train_cv_modnet(
        combined_features, combined_target, "Combined (Matminer + Roost) + MODNet"
    )
    lmm_metrics = train_cv_modnet(
        lmm_features, lmm_target, "MatterVial l-MM + MODNet"
    )

    update_final_results(
        RESULTS_DIR / "FINAL_RESULTS.csv",
        {
            "Combined (Matminer + Roost) + MODNet": combined_metrics,
            "MatterVial l-MM + MODNet": lmm_metrics,
        },
    )
