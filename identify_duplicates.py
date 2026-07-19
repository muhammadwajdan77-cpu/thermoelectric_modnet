import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

mat = pd.read_csv('results/matminer_for_sisso.csv')
sys_df = pd.read_excel('sysTEm_dataset/sysTEm_dataset.xlsx')
n_rows = len(mat)

X = mat.drop(columns=['target']).copy()
y = mat['target']
imputer = SimpleImputer(strategy='mean')
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_imp = X_imp.loc[:, X_imp.nunique(dropna=True) > 1]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

nn = NearestNeighbors(n_neighbors=2)
nn.fit(X_scaled)
distances, indices = nn.kneighbors(X_scaled)
nearest_other_dist = distances[:, 1]
nearest_other_idx = indices[:, 1]

near_dup_mask = nearest_other_dist < 0.01
n_near_dups = near_dup_mask.sum()
print(f'Total rows: {len(X_scaled)}')
print(f'Rows with a near-identical (dist<0.01) partner elsewhere in dataset: {n_near_dups}')
print(f'Percentage: {100 * n_near_dups / len(X_scaled):.1f}%')

sys_formulas = sys_df.iloc[:n_rows]['Pretty Formula'].values

dup_indices = np.where(near_dup_mask)[0][:10]
print('\nTop 10 duplicate-like pairs:')
for i in dup_indices:
    j = nearest_other_idx[i]
    if j == i:
        continue
    dist = nearest_other_dist[i]
    row_i = mat.iloc[i]
    row_j = mat.iloc[j]
    print(f'\nPair {i} <-> {j} | dist={dist:.6f}')
    print(f'  Formula i: {sys_formulas[i]}')
    print(f'  Formula j: {sys_formulas[j]}')
    print(f'  Temperature_K i: {row_i.get("Temperature_K", "N/A")}')
    print(f'  Temperature_K j: {row_j.get("Temperature_K", "N/A")}')
    print(f'  target i: {row_i.get("target", "N/A")}')
    print(f'  target j: {row_j.get("target", "N/A")}')
    print(f'  Row i features sample: {row_i.drop(labels=["target", "Temperature_K"], errors="ignore").head(5).to_dict()}')
    print(f'  Row j features sample: {row_j.drop(labels=["target", "Temperature_K"], errors="ignore").head(5).to_dict()}')
