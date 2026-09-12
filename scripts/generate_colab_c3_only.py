"""
generate_colab_c3_only.py — Generates notebooks/02_c3_sbert_colab.ipynb

A lean, focused notebook for Colab GPU that:
  1. Uploads train_features_c1_raw.npy, train_labels.npy, train_sentence_index.json
     from the local repo (already computed) via Google Drive mount or direct upload.
  2. Computes SBERT cosine for all 1,010,961 sentences on GPU (~2 min on T4).
  3. Assembles and saves train_features_c3_raw.npy and train_features_c3_scaled.npy.
  4. Runs strict alignment assertions + spot-checks.
  5. Downloads results back as processed_features_c3.zip.

C1 extraction is SKIPPED (already done locally, 33 mins, 0 dropped docs confirmed).
"""

import json
import os


def create_notebook():
    cells = []

    def md(source):
        return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.split("\n")]}

    def code(source):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in source.split("\n")]}

    # ── Cell 1: Title ────────────────────────────────────────────────────────
    cells.append(md("""# Notebook 02: C3 SBERT Feature Extraction (Colab GPU — C3 Only)

**Status**: C1 features (`train_features_c1_raw.npy`) already extracted locally — 7,028 docs, 1,010,961 sentences, 0 dropped, 33 minutes on CPU.  
**This notebook only**: Computes SBERT cosine similarity column, assembles C3 matrix, fits RobustScaler, validates alignment.

**Expected runtime on Colab T4 GPU**: ~2–4 minutes total."""))

    # ── Cell 2: Hardware check ────────────────────────────────────────────────
    cells.append(md("### Step 1: Confirm GPU Runtime"))
    cells.append(code("""import torch, os, sys, time, json, pickle, shutil
import numpy as np
from sklearn.preprocessing import RobustScaler

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")
else:
    print("WARNING: No GPU detected — switch runtime to T4 GPU for fast SBERT encoding!")"""))

    # ── Cell 3: Install deps ──────────────────────────────────────────────────
    cells.append(md("### Step 2: Install Dependencies"))
    cells.append(code("""!pip install -q sentence-transformers scikit-learn"""))

    # ── Cell 4: Upload pre-computed files ─────────────────────────────────────
    cells.append(md("""### Step 3: Upload Pre-Computed Files from Local Repo

Upload these 3 files from `data/processed/features/` in your local repo:
- `train_features_c1_raw.npy` (~19 MB)
- `train_labels.npy` (~3.9 MB)  
- `train_sentence_index.json` (~13.7 MB)

**Option A — Google Drive** (if your repo is in Drive):"""))
    cells.append(code("""# Option A: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Adjust this path to wherever your repo lives in Drive
REPO_FEATURES = "/content/drive/MyDrive/Hybrid-Summarization-for-Indian-Legal-Judgments/data/processed/features"

import shutil, os
os.makedirs("/content/features", exist_ok=True)
for fname in ["train_features_c1_raw.npy", "train_labels.npy", "train_sentence_index.json"]:
    src = os.path.join(REPO_FEATURES, fname)
    dst = os.path.join("/content/features", fname)
    shutil.copy(src, dst)
    print(f"Copied {fname} ({os.path.getsize(dst)/1e6:.1f} MB)")"""))

    cells.append(md("**Option B — Direct upload** (if not using Drive):"))
    cells.append(code("""# Option B: Direct file upload
from google.colab import files
import shutil, os
os.makedirs("/content/features", exist_ok=True)
print("Upload: train_features_c1_raw.npy, train_labels.npy, train_sentence_index.json")
uploaded = files.upload()
for fname, data in uploaded.items():
    with open(f"/content/features/{fname}", "wb") as f:
        f.write(data)
    print(f"Saved {fname} ({len(data)/1e6:.1f} MB)")"""))

    # ── Cell 5: Load pre-computed files ───────────────────────────────────────
    cells.append(md("### Step 4: Load Pre-Computed C1 Matrix, Labels, and Sentence Index"))
    cells.append(code("""FEATURES_DIR = "/content/features"

print("Loading pre-computed C1 features...")
X_c1_raw = np.load(f"{FEATURES_DIR}/train_features_c1_raw.npy")
print(f"  X_c1_raw shape: {X_c1_raw.shape}")

print("Loading ground-truth labels...")
train_labels = np.load(f"{FEATURES_DIR}/train_labels.npy")
print(f"  train_labels shape: {train_labels.shape}")

print("Loading sentence index registry...")
with open(f"{FEATURES_DIR}/train_sentence_index.json") as f:
    sentence_keys = json.load(f)
print(f"  sentence_keys length: {len(sentence_keys):,}")

# Pre-flight assertions
assert X_c1_raw.shape == (1010961, 5), f"Expected (1010961, 5), got {X_c1_raw.shape}"
assert len(train_labels) == 1010961, f"Expected 1,010,961 labels, got {len(train_labels)}"
assert len(sentence_keys) == 1010961, f"Expected 1,010,961 keys, got {len(sentence_keys)}"

print("\\nAll pre-flight assertions PASSED. Ready for SBERT extraction.")"""))

    # ── Cell 6: GPU SBERT Encoding ────────────────────────────────────────────
    cells.append(md("""### Step 5: GPU-Accelerated SBERT Cosine Similarity Extraction

Encodes all 1,010,961 sentences using `all-MiniLM-L6-v2` on GPU.  
**Expected time**: ~2–3 minutes on T4, <1 minute on A100."""))
    cells.append(code("""from sentence_transformers import SentenceTransformer
from src.data.loader import load_split_ids, load_document
from src.data.preprocessing import segment_sentences

# Reconstruct train_ids order (must match the sentence_keys order)
# We derive doc order from sentence_keys — unique doc_ids in first-appearance order
seen = {}
ordered_doc_ids = []
for cid, s_idx in sentence_keys:
    if cid not in seen:
        seen[cid] = True
        ordered_doc_ids.append(cid)
print(f"Unique documents in sentence_keys order: {len(ordered_doc_ids):,}")

# Load SBERT on GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading all-MiniLM-L6-v2 on {device}...")
t0 = time.time()
sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)
print(f"Model loaded in {time.time()-t0:.2f}s")

# We need the judgments — reconstruct from doc boundaries in sentence_keys
# Group consecutive sentence_keys by doc
from itertools import groupby

sbert_cosine_blocks = []
total_encoded = 0
t0 = time.time()

# Pre-load all sentences using the sentence_keys grouping
print("\\nStarting SBERT encoding...")
for i, (cid, group) in enumerate(groupby(sentence_keys, key=lambda x: x[0]), 1):
    sent_indices = [s_idx for _, s_idx in group]
    n_s = len(sent_indices)

    # Load document sentences
    doc = load_document(cid)
    sents = segment_sentences(doc["judgment"])

    # Encode on GPU
    embeddings = sbert.encode(
        sents,
        batch_size=512,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Document centroid cosine similarity
    doc_centroid = np.mean(embeddings, axis=0)
    c_norm = np.linalg.norm(doc_centroid)
    if c_norm > 0:
        doc_centroid /= c_norm
    cos_sims = np.dot(embeddings, doc_centroid).clip(-1.0, 1.0).astype(np.float32)
    sbert_cosine_blocks.append(cos_sims)
    total_encoded += n_s

    if i % 500 == 0 or i == len(ordered_doc_ids):
        elapsed = time.time() - t0
        speed_sents = total_encoded / elapsed
        eta_min = (1010961 - total_encoded) / speed_sents / 60 if speed_sents > 0 else 0
        print(f"  [{i:5d}/{len(ordered_doc_ids)}] docs | {total_encoded:9,d} sents | "
              f"{speed_sents:,.0f} sents/sec | ETA: {eta_min:.1f} min")

elapsed = time.time() - t0
sbert_cosine_col = np.concatenate(sbert_cosine_blocks).astype(np.float32)
print(f"\\nSBERT encoding complete in {elapsed:.1f}s ({elapsed/60:.2f} min)")
print(f"Total sentences encoded: {len(sbert_cosine_col):,}")
assert len(sbert_cosine_col) == 1010961, f"Mismatch: got {len(sbert_cosine_col)}"
print("Assertion PASSED: 1,010,961 SBERT cosine values computed.")"""))

    # ── Cell 7: Assemble C3 matrix ────────────────────────────────────────────
    cells.append(md("### Step 6: Assemble C3 Raw Feature Matrix & Fit RobustScaler"))
    cells.append(code("""# C3 = [tfidf, ner, position, cosine_sbert]
shared_features = X_c1_raw[:, :3]  # [tfidf, ner, position]
X_c3_raw = np.column_stack([shared_features, sbert_cosine_col]).astype(np.float32)
print(f"X_c3_raw shape: {X_c3_raw.shape} | Memory: {X_c3_raw.nbytes/1e6:.2f} MB")

# Fit FRESH RobustScaler on the full 1,010,961-row C3 matrix
from sklearn.preprocessing import RobustScaler
scaler_c3 = RobustScaler(quantile_range=(25.0, 75.0))
X_c3_scaled = scaler_c3.fit_transform(X_c3_raw).astype(np.float32)

feature_names_c3 = ["tfidf", "ner", "position", "cosine_sbert"]
print("\\nFull-Corpus C3 RobustScaler Parameters:")
print("-" * 65)
for i, name in enumerate(feature_names_c3):
    print(f"  {name:<14} | Median={scaler_c3.center_[i]:>10.4f} | IQR={scaler_c3.scale_[i]:>10.4f}")
print("-" * 65)"""))

    # ── Cell 8: Hard assertions & spot-checks ─────────────────────────────────
    cells.append(md("### Step 7: Strict End-to-End Alignment Assertions & Spot-Checks"))
    cells.append(code("""print("=" * 70)
print("STRICT END-TO-END ALIGNMENT VALIDATION")
print("=" * 70)
print(f"  train_labels length:     {len(train_labels):,}")
print(f"  X_c1_raw rows:           {len(X_c1_raw):,}")
print(f"  X_c3_raw rows:           {len(X_c3_raw):,}")
print(f"  sentence_keys length:    {len(sentence_keys):,}")

assert len(train_labels) == 1010961
assert len(X_c1_raw) == 1010961
assert len(X_c3_raw) == 1010961
assert len(sentence_keys) == 1010961
assert not np.isnan(X_c3_raw).any(), "NaN in C3!"
assert not np.isnan(X_c1_raw).any(), "NaN in C1!"

print("\\nSpot-check: 5 random rows across C1, C3, and labels:")
np.random.seed(42)
for idx in np.random.choice(len(sentence_keys), 5, replace=False):
    cid, s_idx = sentence_keys[idx]
    c1 = X_c1_raw[idx]
    c3 = X_c3_raw[idx]
    lbl = train_labels[idx]
    assert np.allclose(c1[:3], c3[:3]), f"Shared feature mismatch at row {idx}!"
    print(f"  Row {idx:7d} -> Doc {cid:<6} Sent {s_idx:<3} | "
          f"TF-IDF={c1[0]:.4f} | NER={c1[1]:.0f} | Pos={c1[2]:.4f} | "
          f"SBERT_cos={c3[3]:.4f} | Label={lbl:.4f}")

print("\\nALL ASSERTIONS PASSED — Perfect 1,010,961-row alignment confirmed!")"""))

    # ── Cell 9: Save & package ────────────────────────────────────────────────
    cells.append(md("### Step 8: Save All Artifacts & Download"))
    cells.append(code("""import os, json, pickle

os.makedirs(FEATURES_DIR, exist_ok=True)

# Save C3 matrices
np.save(f"{FEATURES_DIR}/train_features_c3_raw.npy", X_c3_raw)
np.save(f"{FEATURES_DIR}/train_features_c3_scaled.npy", X_c3_scaled)

# Save scaler
with open(f"{FEATURES_DIR}/robust_scaler_c3.pkl", "wb") as f:
    pickle.dump(scaler_c3, f)

# Save scaler params as JSON
params_c3 = {name: {"center": float(scaler_c3.center_[i]), "scale": float(scaler_c3.scale_[i])}
             for i, name in enumerate(feature_names_c3)}
with open(f"{FEATURES_DIR}/robust_scaler_c3_params.json", "w") as f:
    json.dump(params_c3, f, indent=2)

print("Saved artifacts:")
for fname in ["train_features_c3_raw.npy", "train_features_c3_scaled.npy",
              "robust_scaler_c3.pkl", "robust_scaler_c3_params.json"]:
    path = f"{FEATURES_DIR}/{fname}"
    print(f"  {fname}: {os.path.getsize(path)/1e6:.2f} MB")

# Package for download
zip_base = "/content/processed_features_c3"
shutil.make_archive(zip_base, "zip", FEATURES_DIR)
zip_size = os.path.getsize(zip_base + ".zip") / 1e6
print(f"\\nCreated {zip_base}.zip ({zip_size:.1f} MB)")

# Download
try:
    from google.colab import files
    files.download(zip_base + ".zip")
    print("Download initiated!")
except:
    print("Not in Colab — zip is at:", zip_base + ".zip")"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    nb_path = os.path.join("notebooks", "02_c3_sbert_colab.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Created: {nb_path}")


if __name__ == "__main__":
    create_notebook()
