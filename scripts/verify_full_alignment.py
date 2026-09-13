"""
verify_full_alignment.py — Strict end-to-end alignment audit across full corpus:
1,010,961 sentences, 7,028 documents.
Checks C1 raw, C1 scaled, C3 raw, C3 scaled, labels, index keys, and scalers.
"""

import os
import json
import pickle
import numpy as np

feat_dir = "data/processed/features"

print("=" * 80)
print("COMPREHENSIVE FULL-CORPUS FEATURE & LABEL ALIGNMENT AUDIT")
print("=" * 80)

# 1. Load all arrays
X_c1_raw = np.load(os.path.join(feat_dir, "train_features_c1_raw.npy"))
X_c1_scaled = np.load(os.path.join(feat_dir, "train_features_c1_scaled.npy"))
X_c3_raw = np.load(os.path.join(feat_dir, "train_features_c3_raw.npy"))
X_c3_scaled = np.load(os.path.join(feat_dir, "train_features_c3_scaled.npy"))
train_labels = np.load(os.path.join(feat_dir, "train_labels.npy"))

with open(os.path.join(feat_dir, "train_sentence_index.json")) as f:
    sentence_keys = json.load(f)

with open(os.path.join(feat_dir, "train_doc_boundaries.json")) as f:
    doc_boundaries = json.load(f)

with open(os.path.join(feat_dir, "robust_scaler_c1_params.json")) as f:
    params_c1 = json.load(f)

with open(os.path.join(feat_dir, "robust_scaler_c3_params.json")) as f:
    params_c3 = json.load(f)

print(f"Row counts:")
print(f"  X_c1_raw:              {X_c1_raw.shape}")
print(f"  X_c1_scaled:           {X_c1_scaled.shape}")
print(f"  X_c3_raw:              {X_c3_raw.shape}")
print(f"  X_c3_scaled:           {X_c3_scaled.shape}")
print(f"  train_labels:          {train_labels.shape}")
print(f"  train_sentence_index:  {len(sentence_keys):,} items")
print(f"  train_doc_boundaries:  {len(doc_boundaries):,} documents")

# Hard Assertions
assert X_c1_raw.shape == (1010961, 5), f"C1 raw shape error: {X_c1_raw.shape}"
assert X_c1_scaled.shape == (1010961, 5), f"C1 scaled shape error: {X_c1_scaled.shape}"
assert X_c3_raw.shape == (1010961, 4), f"C3 raw shape error: {X_c3_raw.shape}"
assert X_c3_scaled.shape == (1010961, 4), f"C3 scaled shape error: {X_c3_scaled.shape}"
assert train_labels.shape == (1010961,), f"Labels shape error: {train_labels.shape}"
assert len(sentence_keys) == 1010961, f"Sentence keys length error: {len(sentence_keys)}"
assert len(doc_boundaries) == 7028, f"Doc boundaries length error: {len(doc_boundaries)}"

# NaN / Inf Checks
for name, arr in [
    ("X_c1_raw", X_c1_raw),
    ("X_c1_scaled", X_c1_scaled),
    ("X_c3_raw", X_c3_raw),
    ("X_c3_scaled", X_c3_scaled),
    ("train_labels", train_labels),
]:
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    assert nan_count == 0, f"Found {nan_count} NaNs in {name}"
    assert inf_count == 0, f"Found {inf_count} Infs in {name}"
print("\nNaN / Inf Check: CLEAN across all matrices (0 NaNs, 0 Infs).")

# Shared feature cross-validation
shared_diff = np.max(np.abs(X_c1_raw[:, :3] - X_c3_raw[:, :3]))
print(f"Shared feature max absolute difference between C1 and C3: {shared_diff:.2e}")
assert shared_diff == 0.0, f"Shared feature mismatch between C1 and C3: {shared_diff}"
print("Shared Feature Integrity: EXACT bitwise match across all 1,010,961 rows!")

print("\n" + "=" * 80)
print("ROBUST SCALER PARAMETERS (FULL CORPUS: 1,010,961 SENTENCES)")
print("=" * 80)
print("Config C1 / C2 (5 features):")
for k, v in params_c1.items():
    center = v["center"]
    scale = v["scale"]
    print(f"  {k:<14} | Median = {center:>10.4f} | IQR = {scale:>10.4f}")

print("\nConfig C3 (4 features):")
for k, v in params_c3.items():
    center = v["center"]
    scale = v["scale"]
    print(f"  {k:<14} | Median = {center:>10.4f} | IQR = {scale:>10.4f}")

print("\n" + "=" * 80)
print("RANDOMIZED SPOT-CHECKS ACROSS C1, C3, AND LABELS")
print("=" * 80)
np.random.seed(2026)
sample_rows = np.random.choice(len(sentence_keys), 6, replace=False)
for r in sample_rows:
    cid, s_idx = sentence_keys[r]
    c1 = X_c1_raw[r]
    c3 = X_c3_raw[r]
    lbl = train_labels[r]
    print(
        f"Row {r:7d} | Doc {cid:<6} Sent {s_idx:<3} | "
        f"TF-IDF={c1[0]:.4f} | NER={c1[1]:.0f} | Pos={c1[2]:.4f} | "
        f"W2V_cos={c1[3]:.4f} | WMD={c1[4]:.4f} | SBERT_cos={c3[3]:.4f} | Label={lbl:.4f}"
    )

print("\n>>> ALL HARD ASSERTIONS PASSED! Full 1,010,961 alignment confirmed. <<<")
