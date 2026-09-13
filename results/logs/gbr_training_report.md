# GradientBoostingRegressor (GBR) Training Report

**Date**: 2026-09-13 16:45:20
**Total Sentences**: 1,010,961 (7,028 documents)
**Train Split**: 863,463 sentences (5,974 docs, 85%)
**Validation Holdout**: 147,498 sentences (1,054 docs, 15%)

## Hyperparameters
```json
{
  "n_estimators": 100,
  "learning_rate": 0.1,
  "max_depth": 4,
  "min_samples_split": 100,
  "min_samples_leaf": 50,
  "subsample": 0.8,
  "loss": "squared_error",
  "random_state": 42
}
```

## Performance Summary

| Metric | Config C1 (Replica) | Config C2 (Redundancy Swap) | Config C3 (Proposed) |
|---|---|---|---|
| **Features** | 5 features | 5 features | 4 features |
| **Train R²** | **0.2400** | **0.2400** | **0.2259** |
| **Val R²** | **0.2474** | **0.2474** | **0.2269** |
| **Train RMSE** | 0.1835 | 0.1835 | 0.1852 |
| **Val RMSE** | 0.1808 | 0.1808 | 0.1833 |
| **Train MAE** | 0.1287 | 0.1287 | 0.1308 |
| **Val MAE** | 0.1284 | 0.1284 | 0.1308 |
| **Fit Time** | 184.8s | 168.4s | 127.1s |
| **Negative Train R² Flag** | **False** | **False** | **False** |

## Ranked Feature Importances

### Config C1 / C2:
- **1. wmd**: `0.7659` (76.59%)
- **2. cosine_w2v**: `0.0694` (6.94%)
- **3. tfidf**: `0.0672` (6.72%)
- **4. ner**: `0.0581` (5.81%)
- **5. position**: `0.0393` (3.93%)

### Config C3:
- **1. tfidf**: `0.6670` (66.70%)
- **2. cosine_sbert**: `0.2445` (24.45%)
- **3. position**: `0.0481` (4.81%)
- **4. ner**: `0.0404` (4.04%)
