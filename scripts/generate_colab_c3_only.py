"""
generate_colab_c3_only.py — Generates notebooks/02_c3_sbert_colab.ipynb

A self-contained notebook for Colab GPU that:
  1. Takes 'c3_colab_payload.zip' (70 MB containing pre-segmented sentences,
     pre-computed C1 features, labels, and sentence index keys).
  2. Runs SBERT 'all-MiniLM-L6-v2' on T4 GPU (~2-3 minutes for 1,010,961 sentences).
  3. Assembles full C3 matrix (1,010,961 x 4) [tfidf, ner, position, cosine_sbert].
  4. Fits fresh RobustScaler on full C3 corpus.
  5. Runs strict alignment assertions + randomized spot-checks.
  6. Automatically downloads processed_features_c3.zip.
"""

import json
import os


def create_notebook():
    cells = []

    def md(source):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        }

    def code(source):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")]
        }

    # Cell 1: Header
    cells.append(md("""# Notebook 02: C3 SBERT Feature Extraction (Colab T4 GPU)

**Project**: Hybrid Extractive-Abstractive Summarization for Indian Legal Judgments  
**Task**: Compute SBERT sentence-document cosine similarity for all 1,010,961 sentences across 7,028 training documents, assemble C3 raw matrix, fit fresh `RobustScaler`, and validate strict alignment.

---

### Quick Start (2 Steps):
1. **Upload Payload**: Drag and drop `c3_colab_payload.zip` (70 MB from your local repo's `data/` folder) into the **Files** panel on the left sidebar.
2. **Run**: Click **Runtime → Run All** (or press `Ctrl + F9`).

Expected runtime on Colab T4 GPU: **~2–3 minutes total**."""))

    # Cell 2: Hardware Check
    cells.append(md("### Step 1: Hardware & GPU Diagnostics"))
    cells.append(code("""import torch, os, sys, time, json, pickle, shutil, zipfile, gzip
import numpy as np

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:   ", torch.cuda.get_device_name(0))
    print("Total VRAM:   ", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")
else:
    print("WARNING: No GPU detected! Go to Runtime -> Change runtime type -> T4 GPU")"""))

    # Cell 3: Install dependencies
    cells.append(md("### Step 2: Install sentence-transformers & scikit-learn"))
    cells.append(code("""!pip install -q sentence-transformers scikit-learn"""))

    # Cell 4: Unpack Payload
    cells.append(md("""### Step 3: Unpack `c3_colab_payload.zip`
Unpacks pre-segmented legal sentences, pre-computed C1 features, labels, and composite join keys."""))
    cells.append(code("""PAYLOAD_ZIP = "/content/c3_colab_payload.zip"
assert os.path.exists(PAYLOAD_ZIP), (
    f"\\n[ERROR] '{PAYLOAD_ZIP}' not found!\\n"
    "Please upload 'c3_colab_payload.zip' (70 MB) to /content by dragging it into "
    "the Files panel on the left sidebar, then re-run this cell."
)

PAYLOAD_DIR = "/content/payload"
os.makedirs(PAYLOAD_DIR, exist_ok=True)
print(f"Unpacking {PAYLOAD_ZIP}...")
t0 = time.time()
with zipfile.ZipFile(PAYLOAD_ZIP, "r") as z:
    z.extractall(PAYLOAD_DIR)
print(f"Unpacked in {time.time()-t0:.2f}s:")
for f in os.listdir(PAYLOAD_DIR):
    size_mb = os.path.getsize(os.path.join(PAYLOAD_DIR, f)) / 1e6
    print(f"  - {f} ({size_mb:.2f} MB)")"""))

    # Cell 5: Load data & pre-flight assertions
    cells.append(md("### Step 4: Load Data & Pre-flight Assertions"))
    cells.append(code("""print("Loading C1 raw feature matrix...")
X_c1_raw = np.load(f"{PAYLOAD_DIR}/train_features_c1_raw.npy")
print(f"  X_c1_raw shape: {X_c1_raw.shape}")

print("Loading ground-truth labels...")
train_labels = np.load(f"{PAYLOAD_DIR}/train_labels.npy")
print(f"  train_labels shape: {train_labels.shape}")

print("Loading composite sentence index keys...")
with open(f"{PAYLOAD_DIR}/train_sentence_index.json") as f:
    sentence_keys = json.load(f)
print(f"  sentence_keys length: {len(sentence_keys):,}")

print("Loading pre-segmented document sentences...")
t0 = time.time()
with gzip.open(f"{PAYLOAD_DIR}/train_doc_sentences.json.gz", "rt", encoding="utf-8") as f:
    doc_sentences = json.load(f)
print(f"  Loaded {len(doc_sentences):,} documents in {time.time()-t0:.2f}s")

# Hard Pre-flight Assertions
assert X_c1_raw.shape == (1010961, 5), f"Expected (1010961, 5), got {X_c1_raw.shape}"
assert len(train_labels) == 1010961, f"Expected 1,010,961 labels, got {len(train_labels)}"
assert len(sentence_keys) == 1010961, f"Expected 1,010,961 keys, got {len(sentence_keys)}"
assert len(doc_sentences) == 7028, f"Expected 7,028 docs, got {len(doc_sentences)}"

print("\\n>>> ALL PRE-FLIGHT ASSERTIONS PASSED! Ready for GPU SBERT extraction. <<<")"""))

    # Cell 6: SBERT GPU Extraction
    cells.append(md("""### Step 5: GPU-Accelerated SBERT Cosine Similarity Extraction
Encodes all 1,010,961 sentences using `all-MiniLM-L6-v2` with batch size 512 on GPU.  
Computes sentence-to-document centroid cosine similarity for each judgment."""))
    cells.append(code("""from sentence_transformers import SentenceTransformer

# Reconstruct ordered doc_ids from sentence_keys to guarantee exact row order
seen = {}
ordered_doc_ids = []
for cid, s_idx in sentence_keys:
    if cid not in seen:
        seen[cid] = True
        ordered_doc_ids.append(cid)
print(f"Unique documents in join-key order: {len(ordered_doc_ids):,}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading SBERT (all-MiniLM-L6-v2) onto {device}...")
t0 = time.time()
sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)
print(f"Model loaded in {time.time()-t0:.2f}s")

sbert_cosine_blocks = []
total_encoded = 0
t_start = time.time()

print("\\nStarting GPU SBERT encoding (progress logged every 500 documents)...")
for i, cid in enumerate(ordered_doc_ids, 1):
    sents = doc_sentences[cid]
    if not sents:
        continue

    # GPU batch encoding with normalized unit vectors
    embeddings = sbert.encode(
        sents,
        batch_size=512,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Document centroid embedding (mean of sentence unit vectors)
    doc_centroid = np.mean(embeddings, axis=0)
    c_norm = np.linalg.norm(doc_centroid)
    if c_norm > 0:
        doc_centroid /= c_norm

    # Dot product of normalized vectors = cosine similarity
    cos_sims = np.dot(embeddings, doc_centroid).clip(-1.0, 1.0).astype(np.float32)
    sbert_cosine_blocks.append(cos_sims)
    total_encoded += len(sents)

    if i % 500 == 0 or i == len(ordered_doc_ids):
        elapsed = time.time() - t_start
        speed = total_encoded / elapsed
        eta_min = (1010961 - total_encoded) / speed / 60 if speed > 0 else 0
        print(f"  [{i:5d}/{len(ordered_doc_ids)}] docs | {total_encoded:9,d} sents | "
              f"{speed:,.0f} sents/sec | ETA: {eta_min:.1f} min")

total_time = time.time() - t_start
sbert_cosine_col = np.concatenate(sbert_cosine_blocks).astype(np.float32)
print(f"\\n>>> SBERT encoding COMPLETE in {total_time:.1f}s ({total_time/60:.2f} min)! <<<")
print(f"Total sentences encoded: {len(sbert_cosine_col):,}")
assert len(sbert_cosine_col) == 1010961, f"Mismatch: got {len(sbert_cosine_col)}"
print("Assertion PASSED: Exactly 1,010,961 SBERT cosine values computed.")"""))

    # Cell 7: Assemble C3 & Fit RobustScaler
    cells.append(md("""### Step 6: Assemble Full C3 Matrix & Fit Fresh RobustScaler
Combines shared features `[TF-IDF, NER, Position]` from C1 with the fresh SBERT cosine column.  
Fits `RobustScaler` fresh on all 1,010,961 sentences."""))
    cells.append(code("""# Columns 0, 1, 2 from C1 are shared: [tfidf, ner, position]
shared_features = X_c1_raw[:, :3]
X_c3_raw = np.column_stack([shared_features, sbert_cosine_col]).astype(np.float32)
print(f"X_c3_raw shape: {X_c3_raw.shape} | Memory: {X_c3_raw.nbytes / 1e6:.2f} MB")

from sklearn.preprocessing import RobustScaler
scaler_c3 = RobustScaler(quantile_range=(25.0, 75.0))
X_c3_scaled = scaler_c3.fit_transform(X_c3_raw).astype(np.float32)

feature_names_c3 = ["tfidf", "ner", "position", "cosine_sbert"]
print("\\n" + "=" * 65)
print("FULL-CORPUS C3 ROBUST SCALER PARAMETERS (1,010,961 SENTENCES)")
print("=" * 65)
for i, name in enumerate(feature_names_c3):
    print(f"  {name:<14} | Median = {scaler_c3.center_[i]:>10.4f} | IQR = {scaler_c3.scale_[i]:>10.4f}")
print("=" * 65)"""))

    # Cell 8: Strict End-to-End Alignment Validation
    cells.append(md("### Step 7: Strict Alignment Assertions & Spot-Checks"))
    cells.append(code("""print("=" * 70)
print("STRICT END-TO-END ALIGNMENT VALIDATION")
print("=" * 70)
print(f"  train_labels length:     {len(train_labels):,}")
print(f"  X_c1_raw rows:           {len(X_c1_raw):,}")
print(f"  X_c3_raw rows:           {len(X_c3_raw):,}")
print(f"  sentence_keys length:    {len(sentence_keys):,}")

assert len(train_labels) == 1010961, "Label length mismatch!"
assert len(X_c1_raw) == 1010961, "C1 length mismatch!"
assert len(X_c3_raw) == 1010961, "C3 length mismatch!"
assert len(sentence_keys) == 1010961, "Sentence keys length mismatch!"
assert not np.isnan(X_c3_raw).any(), "NaN found in C3 raw matrix!"
assert not np.isnan(X_c3_scaled).any(), "NaN found in C3 scaled matrix!"

print("\\nRandomized spot-checks across 5 sample rows:")
np.random.seed(42)
sample_indices = np.random.choice(len(sentence_keys), 5, replace=False)
for idx in sample_indices:
    cid, s_idx = sentence_keys[idx]
    c1 = X_c1_raw[idx]
    c3 = X_c3_raw[idx]
    lbl = train_labels[idx]
    assert np.allclose(c1[:3], c3[:3]), f"Shared feature mismatch at row {idx}!"
    print(f"  Row {idx:7d} -> Doc {cid:<6} Sent {s_idx:<3} | "
          f"TF-IDF={c1[0]:.4f} | NER={c1[1]:.0f} | Pos={c1[2]:.4f} | "
          f"SBERT_cos={c3[3]:.4f} | Label={lbl:.4f}")

print("\\n>>> ALL ALIGNMENT ASSERTIONS PASSED! Perfect 1,010,961-row integrity verified. <<<")"""))

    # Cell 9: Save & Download
    cells.append(md("### Step 8: Save Artifacts & Download"))
    cells.append(code("""OUT_DIR = "/content/processed_features_c3"
os.makedirs(OUT_DIR, exist_ok=True)

# Save arrays
np.save(f"{OUT_DIR}/train_features_c3_raw.npy", X_c3_raw)
np.save(f"{OUT_DIR}/train_features_c3_scaled.npy", X_c3_scaled)

# Save scaler
with open(f"{OUT_DIR}/robust_scaler_c3.pkl", "wb") as f:
    pickle.dump(scaler_c3, f)

# Save parameters JSON
params_c3 = {
    name: {"center": float(scaler_c3.center_[i]), "scale": float(scaler_c3.scale_[i])}
    for i, name in enumerate(feature_names_c3)
}
with open(f"{OUT_DIR}/robust_scaler_c3_params.json", "w") as f:
    json.dump(params_c3, f, indent=2)

print("Saved artifacts:")
for fname in ["train_features_c3_raw.npy", "train_features_c3_scaled.npy",
              "robust_scaler_c3.pkl", "robust_scaler_c3_params.json"]:
    path = os.path.join(OUT_DIR, fname)
    print(f"  {fname} ({os.path.getsize(path)/1e6:.2f} MB)")

# Create zip archive
zip_base = "/content/processed_features_c3"
shutil.make_archive(zip_base, "zip", OUT_DIR)
zip_size = os.path.getsize(zip_base + ".zip") / 1e6
print(f"\\nCreated {zip_base}.zip ({zip_size:.2f} MB)")

# Trigger browser download
from google.colab import files
files.download(zip_base + ".zip")
print("\\n>>> DOWNLOAD INITIATED! <<<")
print("Once downloaded, place processed_features_c3.zip in data/processed/features/ and unzip it.")"""))

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    out_path = os.path.join("notebooks", "02_c3_sbert_colab.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Successfully generated {out_path} with {len(cells)} cells.")


if __name__ == "__main__":
    create_notebook()
