import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Composition


def canonical_formula(formula):
    if pd.isna(formula):
        raise ValueError('empty')
    return Composition(str(formula)).reduced_formula


def make_group_labels_method1(formulas):
    labels = []
    for formula in formulas.astype(str):
        text = formula.strip()
        if not text or text.lower() in {'nan', 'none'}:
            labels.append('nan')
            continue
        try:
            labels.append(canonical_formula(text))
        except Exception:
            labels.append(text)
    return np.array(labels, dtype=object)


def make_group_labels_method2(formulas, target_len):
    labels = []
    for formula in formulas.tolist():
        try:
            labels.append(canonical_formula(formula))
        except Exception:
            continue
        if len(labels) >= target_len:
            break
    return np.array(labels[:target_len], dtype=object)


def evaluate_leakage(groups, X_imp, y, title):
    group_kf = GroupKFold(n_splits=5)
    print(f'=== {title} ===')
    for fold, (train_idx, test_idx) in enumerate(group_kf.split(X_imp, y, groups=groups), start=1):
        X_train = X_imp.iloc[train_idx].values
        X_test = X_imp.iloc[test_idx].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        sample_size = min(200, len(X_test_scaled))
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(X_test_scaled), sample_size, replace=False)

        min_distances = []
        for i in sample_idx:
            dists = np.linalg.norm(X_train_scaled - X_test_scaled[i], axis=1)
            min_distances.append(dists.min())

        min_distances = np.array(min_distances)
        near_zero_count = np.sum(min_distances < 0.01)
        very_close_count = np.sum(min_distances < 0.5)

        print(f'Fold {fold}: train={len(train_idx)}, test={len(test_idx)}')
        print(f'  Median nearest-train distance (sampled {sample_size} test rows): {np.median(min_distances):.4f}')
        print(f'  Min: {min_distances.min():.4f}, Max: {min_distances.max():.4f}')
        print(f'  Near-identical (dist<0.01): {near_zero_count}/{sample_size}')
        print(f'  Suspiciously close (dist<0.5): {very_close_count}/{sample_size}')
        print()


if __name__ == '__main__':
    mat = pd.read_csv('results/matminer_for_sisso.csv')
    sys_df = pd.read_excel('sysTEm_dataset/sysTEm_dataset.xlsx')
    n_rows = len(mat)

    X = mat.drop(columns=['target']).copy()
    y = mat['target']
    imputer = SimpleImputer(strategy='mean')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_imp = X_imp.loc[:, X_imp.nunique(dropna=True) > 1]

    groups_method1 = make_group_labels_method1(sys_df.iloc[:n_rows]['Pretty Formula'])
    groups_method2 = make_group_labels_method2(sys_df['Pretty Formula'], n_rows)

    evaluate_leakage(groups_method1, X_imp, y, 'Method 1 (positional slice)')
    evaluate_leakage(groups_method2, X_imp, y, 'Method 2 (skip-invalid accumulate)')
