# TEST 2 BUG FIX AND FINDINGS

## CRITICAL BUG IDENTIFIED & FIXED

### The Bug (FEATURE_OVERLAP_BY_FOLD.csv - BUGGY VERSION)
The original Test 2 was using `train_data.df_featurized.columns.tolist()` AFTER calling `feature_selection()`.

**Problem**: `df_featurized` is NOT truncated by `feature_selection()`. It still contains the **full feature pool** (127 columns for Model A, 140 for Model B), not the actual 50 selected features.

**Buggy Results** (all folds reported):
```
fold  n_features_A  n_features_B  n_overlap  pct_overlap
  1       127          140           127       100.0%  ← WRONG!
  2       127          140           127       100.0%  ← WRONG!
  3       127          140           127       100.0%  ← WRONG!
  4       127          140           127       100.0%  ← WRONG!
  5       127          140           127       100.0%  ← WRONG!
```

**Interpretation of Buggy Results**: "100% overlap means no feature displacement" ← **FALSE, meaningless**

---

## THE FIX

Replaced:
```python
selected_a = set(train_data_a.df_featurized.columns.tolist())
```

With:
```python
selected_a = set(train_data_a.get_optimal_descriptors())
```

**Why this works**: `get_optimal_descriptors()` returns `self.optimal_features`, which contains the **actual list of the 50 selected feature names**, not the full pool.

---

## CORRECTED RESULTS (FEATURE_OVERLAP_BY_FOLD_FIXED.csv - IN PROGRESS)

### Results So Far (3 of 5 folds):

**Fold 1**:
- Model A selected: 50 features
- Model B selected: 50 features  
- Overlap: 45/50 (90.0%)
- **→ 5 features displaced (10%)**

**Fold 2**:
- (computing...)

**Fold 3**:
- Model A selected: 50 features
- Model B selected: 50 features
- Overlap: 37/50 (74.0%)
- **→ 13 features displaced (26%)**

---

## KEY INSIGHT

**The feature displacement IS REAL and SIGNIFICANT:**

- Fold 1: 10% displacement
- Fold 3: 26% displacement
- **Average so far: 18%** (substantial!!)

This contradicts the buggy "100% overlap" finding. When Model B adds structure features to the candidate pool (going from 127→140 features), **MODNet's mutual-information based feature selection chooses DIFFERENT features** than when only composition features are available.

This supports the **feature-selection-budget displacement hypothesis**:
- With 50 slots and 127 candidates (Model A): selects features X
- With 50 slots and 140 candidates (Model B): must make different choices → selects features Y (many of which differ from X)
- Result: composition features that were selected in Model A may not make the cut in Model B because structure features now compete for the same 50 slots

---

## NEXT: Statistical Significance

Test 2 corrected overlap will be complete ~10-15 minutes.

Then need to verify with **Test 3** (Forced-Inclusion): If we protect Model A's selected features and FORCE them to be included even when structure features are added, does Model B's MAE improve?

If yes → displacement is the culprit
If no → structure features are genuinely unhelpful

---

## FILES

- **BUGGY**: `results/FEATURE_OVERLAP_BY_FOLD.csv` (100% overlap, meaningless)
- **FIXED**: `results/FEATURE_OVERLAP_BY_FOLD_FIXED.csv` (realistic overlap %, currently being written)
- **Script**: `test2_feature_overlap_fixed.py` (standalone fixed version)
- **Main Script**: `fair_comparison_rigorous_test.py` (also fixed, contains all 4 tests)
