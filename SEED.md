# Reproducibility & Random Seeds

## Dataset Split Seed
- **Seed value**: `42`
- **Method**: Deterministic sort of all 7,128 unique case IDs, seeded Python `random.Random(42).shuffle()`, allocating:
  - Held-out test set: 100 documents (`data/splits/test_ids.txt`)
  - Training set: 7,028 documents (`data/splits/train_ids.txt`)
- **Generated on**: 2026-09-11
- **Status**: **FROZEN**. Per `01_DATA_SPEC.md` and `03_EXPERIMENT_PLAN.md`, this split must NEVER be regenerated or altered across experiments C1, C2, and C3.
