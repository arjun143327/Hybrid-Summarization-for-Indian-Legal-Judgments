"""
build_c3_features.py — Constructs full training feature matrix for C3 (4 features),
fits RobustScaler fresh on the full C3 training matrix, and saves scaled arrays and scaler.

Features:
  x_ij = [TF-IDF, NER, Position, Cosine_sbert]

Reuses shared features [TF-IDF, NER, Position] from train_features_c1_raw.npy to guarantee
exact consistency and zero redundant computation.
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))

from src.data.loader import load_split_ids, load_document
from src.data.preprocessing import segment_sentences
from src.features.embeddings_sbert import load_sbert_model, compute_sbert_cosine_features
from src.features.build_features import FeatureScalerPipeline, CONFIG_FEATURE_NAMES


def compute_sbert_cosine_all_docs(train_ids, batch_size=128):
    print(f"Loading SBERT model (all-MiniLM-L6-v2) on CPU...")
    # Set PyTorch thread count for optimal multi-core throughput
    num_threads = min(14, os.cpu_count() or 4)
    torch.set_num_threads(num_threads)
    model = load_sbert_model(device="cpu")

    total_docs = len(train_ids)
    print(f"Extracting SBERT cosine similarity for {total_docs:,} documents using {num_threads} CPU threads...")

    t0 = time.time()
    sbert_cosine_blocks = []
    total_sentences = 0

    for i, cid in enumerate(train_ids, 1):
        doc = load_document(cid)
        sents = segment_sentences(doc["judgment"])

        if not sents:
            sbert_cosine_blocks.append(np.array([], dtype=np.float32))
            continue

        cos_scores = compute_sbert_cosine_features(sents, model, batch_size=batch_size)
        sbert_cosine_blocks.append(cos_scores)
        total_sentences += len(sents)

        if i % 250 == 0 or i == total_docs:
            elapsed = time.time() - t0
            speed = i / elapsed
            eta_sec = (total_docs - i) / speed if speed > 0 else 0
            print(
                f"[{i:5d}/{total_docs}] docs encoded | "
                f"Sentences: {total_sentences:9,d} | "
                f"Elapsed: {elapsed:6.1f}s | "
                f"Speed: {speed:4.1f} docs/sec ({total_sentences/elapsed:5.1f} sents/sec) | "
                f"ETA: {eta_sec/60:4.1f} min"
            )

    total_time = time.time() - t0
    print(f"\nSBERT cosine extraction complete in {total_time:.2f}s ({total_time/60:.2f} minutes)!")
    print(f"Total sentences encoded: {total_sentences:,}")

    all_sbert_cosine = np.concatenate(sbert_cosine_blocks).astype(np.float32)
    return all_sbert_cosine


def main():
    print("=" * 80)
    print("BUILDING FULL TRAINING FEATURE MATRIX FOR CONFIG C3 (4 FEATURES)")
    print("Features: ['tfidf', 'ner', 'position', 'cosine_sbert']")
    print("=" * 80)

    out_dir = os.path.join("data", "processed", "features")
    c1_raw_path = os.path.join(out_dir, "train_features_c1_raw.npy")

    if not os.path.exists(c1_raw_path):
        raise FileNotFoundError(
            f"{c1_raw_path} not found. Run build_c1_features.py first to extract shared features!"
        )

    print(f"Loading shared features from {c1_raw_path}...")
    X_train_c1_raw = np.load(c1_raw_path)
    print(f"Loaded X_train_c1_raw shape: {X_train_c1_raw.shape}")

    # Columns 0: tfidf, 1: ner, 2: position
    shared_features = X_train_c1_raw[:, :3]
    total_sentences = len(shared_features)

    train_ids = load_split_ids("train")

    # Compute SBERT cosine column
    sbert_cosine_col = compute_sbert_cosine_all_docs(train_ids)
    assert len(sbert_cosine_col) == total_sentences, (
        f"Mismatch in sentence count: shared has {total_sentences}, SBERT has {len(sbert_cosine_col)}"
    )

    # Assemble C3 raw feature matrix: [tfidf, ner, position, cosine_sbert]
    print("\nAssembling C3 raw feature matrix...")
    X_train_c3_raw = np.column_stack([shared_features, sbert_cosine_col]).astype(np.float32)
    print(f"X_train_c3_raw shape: {X_train_c3_raw.shape} | Memory: {X_train_c3_raw.nbytes / (1024*1024):.2f} MB")

    c3_raw_path = os.path.join(out_dir, "train_features_c3_raw.npy")
    np.save(c3_raw_path, X_train_c3_raw)
    print(f"Saved C3 raw feature matrix to: {c3_raw_path}")

    # -------------------------------------------------------------
    # Fit RobustScaler FRESH on full C3 training matrix
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FITTING ROBUSTSCALER FRESH ON FULL 7,028-DOC TRAINING MATRIX (C3)")
    print("=" * 80)

    scaler_c3 = FeatureScalerPipeline(config="C3", scaler_type="robust")
    X_train_c3_scaled = scaler_c3.fit_transform(X_train_c3_raw)

    params = scaler_c3.get_scaling_params()
    print("\nLEARNED FULL-CORPUS ROBUSTSCALER PARAMETERS (C3):")
    print("-" * 75)
    for feat_name, p in params.items():
        print(f"  Feature: {feat_name:<12} | Median (center) = {p['center']:>10.4f} | IQR (scale) = {p['scale']:>10.4f}")
    print("-" * 75)

    # Save scaler and scaled matrix
    scaler_path = os.path.join(out_dir, "robust_scaler_c3.pkl")
    scaled_path = os.path.join(out_dir, "train_features_c3_scaled.npy")
    params_path = os.path.join(out_dir, "robust_scaler_c3_params.json")

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_c3, f)
    np.save(scaled_path, X_train_c3_scaled)
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"\nSaved RobustScaler model to: {scaler_path}")
    print(f"Saved scaled feature matrix to: {scaled_path} ({os.path.getsize(scaled_path)/(1024*1024):.2f} MB)")
    print(f"Saved scaling parameters to: {params_path}")

    # -------------------------------------------------------------
    # Hard Assertions & Spot-Check Validation Across All Matrices
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STRICT END-TO-END ALIGNMENT VALIDATION (ALL 1,010,961 ROWS)")
    print("=" * 80)

    labels_path = os.path.join(out_dir, "train_labels.npy")
    index_path = os.path.join(out_dir, "train_sentence_index.json")

    train_labels = np.load(labels_path)
    with open(index_path, "r") as f:
        sentence_keys = json.load(f)

    print(f"  train_labels.npy length:       {len(train_labels):,}")
    print(f"  X_train_c1_raw shape:          {X_train_c1_raw.shape}")
    print(f"  X_train_c3_raw shape:          {X_train_c3_raw.shape}")
    print(f"  train_sentence_index keys:     {len(sentence_keys):,}")

    assert len(train_labels) == 1010961, f"Expected 1,010,961 labels, got {len(train_labels)}"
    assert len(X_train_c1_raw) == 1010961, f"Expected 1,010,961 C1 rows, got {len(X_train_c1_raw)}"
    assert len(X_train_c3_raw) == 1010961, f"Expected 1,010,961 C3 rows, got {len(X_train_c3_raw)}"
    assert len(sentence_keys) == 1010961, f"Expected 1,010,961 sentence keys, got {len(sentence_keys)}"

    assert not np.isnan(X_train_c1_raw).any(), "NaN detected in C1 raw matrix!"
    assert not np.isnan(X_train_c3_raw).any(), "NaN detected in C3 raw matrix!"
    assert not np.isnan(train_labels).any(), "NaN detected in labels!"

    print("\nSpot-check alignment for 5 random sentence indices:")
    np.random.seed(42)
    sample_indices = np.random.choice(len(sentence_keys), size=5, replace=False)
    for idx in sample_indices:
        cid, s_idx = sentence_keys[idx]
        c1_row = X_train_c1_raw[idx]
        c3_row = X_train_c3_raw[idx]
        lbl = train_labels[idx]
        assert np.allclose(c1_row[:3], c3_row[:3]), f"Shared features mismatch at index {idx}!"
        print(f"  Row {idx:7d} -> Doc {cid:<6} Sent {s_idx:<3} | TF-IDF={c1_row[0]:.4f} | NER={c1_row[1]:.0f} | Pos={c1_row[2]:.4f} | Label={lbl:.4f}")

    print("\nALL HARD ASSERTIONS PASSED! PERFECT 1,010,961-ROW ALIGNMENT CONFIRMED.")

    # -------------------------------------------------------------
    # Package into processed_features_1010961.zip
    # -------------------------------------------------------------
    import shutil
    zip_base = os.path.join("data", "processed", "features_1010961_bundle")
    zip_path = os.path.join("data", "processed", "features", "processed_features_1010961.zip")
    print(f"\nPackaging all feature artifacts into {zip_path}...")
    shutil.make_archive(zip_base, "zip", out_dir)
    if os.path.exists(zip_base + ".zip"):
        os.replace(zip_base + ".zip", zip_path)
    print(f"Created {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
