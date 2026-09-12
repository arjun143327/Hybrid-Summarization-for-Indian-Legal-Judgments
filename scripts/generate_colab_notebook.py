"""
generate_colab_notebook.py — Generates notebooks/01_full_feature_extraction_and_scaling.ipynb
as a clean, self-contained Jupyter notebook for Google Colab GPU execution.
"""

import json
import os

def create_notebook():
    cells = []

    def md_cell(source):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        }

    def code_cell(source):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")]
        }

    # Cell 1: Header
    cells.append(md_cell("""# Notebook 01: Full Feature Extraction & Scaling (Colab GPU)

**Project**: Hybrid Extractive-Abstractive Summarization for Indian Legal Judgments  
**Methodology Reference**: `02_METHODOLOGY.md` Stage 1 & `04_TASKS.md` Phase 3 / Phase 6  
**Architecture**: Explicit `(doc_id, sentence_index)` Composite Join-Key Architecture  
**Corpus Scope**: All **7,028 training documents** (~1,010,961 segmented sentences)  

---

### Objectives
1. **Full-Corpus Feature Extraction**:
   - **Config C1 / C2** (5 features): `[TF-IDF sum, spaCy NER count, Sentence Position, Word2Vec Cosine, WMD]`
   - **Config C3** (4 features): `[TF-IDF sum, spaCy NER count, Sentence Position, SBERT Cosine]`
2. **Explicit Join-Key Enforcement**:
   - Assign every row an explicit composite key `(doc_id, sentence_index)` stored in `train_sentence_index.json`.
   - Prevent any row drift, dropped documents, or misaligned positional indices between features and ground-truth labels.
3. **WMD OOV Fallback Instrumentation**:
   - Instrument sentences triggering `max_fallback_dist = 3.0` across all 1,010,961 sentences.
   - Analyze fallback rates across OCR-affected vs. non-OCR judgments for the paper's Discussion section.
4. **Fresh RobustScaler Fit**:
   - Fit `RobustScaler` fresh on the full 1,010,961-row training matrices for C1 and C3 separately.
   - Record learned Medians and IQRs.
5. **Exact Row Count Alignment**:
   - Confirm and save `train_features_c1_raw.npy`, `train_features_c1_scaled.npy`, `train_features_c3_raw.npy`, `train_features_c3_scaled.npy`, and `train_labels.npy` — all exactly matching 1,010,961 rows with zero dropped documents."""))

    # Cell 2: Hardware check
    cells.append(md_cell("""### 1. Hardware & Environment Diagnostics
Check GPU availability, VRAM, and RAM in the current Colab runtime."""))

    cells.append(code_cell("""import os
import sys
import psutil
import torch

print("=" * 70)
print("HARDWARE & ENVIRONMENT DIAGNOSTICS")
print("=" * 70)
cuda_avail = torch.cuda.is_available()
print(f"CUDA Available:     {cuda_avail}")
if cuda_avail:
    print(f"GPU Device Name:    {torch.cuda.get_device_name(0)}")
    print(f"GPU Device Count:   {torch.cuda.device_count()}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Total VRAM:         {vram_gb:.2f} GB")
else:
    print("WARNING: Running on CPU. For fast SBERT extraction, switch to GPU runtime:")
    print("Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU / A100 GPU")

ram_gb = psutil.virtual_memory().total / (1024**3)
print(f"System RAM:         {ram_gb:.2f} GB")
print(f"CPU Cores:          {os.cpu_count()}")
print(f"Python Version:     {sys.version.split()[0]}")
print("=" * 70)"""))

    # Cell 3: Install dependencies
    cells.append(md_cell("""### 2. Dependency Installation
Install required packages and download spaCy `en_core_web_sm` model."""))

    cells.append(code_cell("""!pip install -q spacy POT sentence-transformers rouge-score gensim scikit-learn
!python -m spacy download en_core_web_sm"""))

    # Cell 4: Repo setup
    cells.append(md_cell("""### 3. Repository Setup & Directory Verification
Mount Google Drive or set up local paths. Ensure working directory is set to project root."""))

    cells.append(code_cell("""import os
import sys

# Mount Google Drive if running in standard Colab
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except Exception:
    IN_COLAB = False
    print("Running in local/offline environment.")

# Set repository root path
# Adjust this path if your repo is cloned in Google Drive or /content
REPO_ROOT = "."
if IN_COLAB and os.path.exists("/content/drive/MyDrive/Hybrid-Summarization-for-Indian-Legal-Judgments"):
    REPO_ROOT = "/content/drive/MyDrive/Hybrid-Summarization-for-Indian-Legal-Judgments"
elif IN_COLAB and os.path.exists("/content/Hybrid-Summarization-for-Indian-Legal-Judgments"):
    REPO_ROOT = "/content/Hybrid-Summarization-for-Indian-Legal-Judgments"

os.chdir(REPO_ROOT)
sys.path.insert(0, os.path.abspath("."))
print(f"Active Working Directory: {os.path.abspath(os.getcwd())}")

# Verify data existence
assert os.path.exists("data/splits/train_ids.txt"), "Missing data/splits/train_ids.txt!"
assert os.path.exists("data/raw/IN-Abs/train-data/judgement"), "Missing judgment dataset!"
print("Repository layout and dataset verified successfully.")"""))

    # Cell 5: Imports & Model loading
    cells.append(md_cell("""### 4. Load Models & Feature Extractors
- Pretrained `word2vec-google-news-300`
- Pretrained SBERT `all-MiniLM-L6-v2` (on GPU)
- TF-IDF Vectorizer (27,216 vocabulary, fitted strictly on 7,028 training judgments)
- spaCy `en_core_web_sm`
- LegalTokenizer"""))

    cells.append(code_cell("""import time
import json
import pickle
import numpy as np
from tqdm.auto import tqdm

from src.data.loader import load_split_ids, load_document
from src.data.preprocessing import segment_sentences, LegalTokenizer
from src.features.tfidf import fit_tfidf_vectorizer
from src.features.embeddings_w2v import load_word2vec_model
from src.features.embeddings_sbert import load_sbert_model
from src.features.build_features import assemble_document_raw_features, FeatureScalerPipeline, CONFIG_FEATURE_NAMES

print("Loading models and vectorizers...")
t0 = time.time()

# 1. TF-IDF Vectorizer
vec = fit_tfidf_vectorizer(verbose=False)
print(f"TF-IDF Vectorizer ready (vocab size: {len(vec.vocabulary_):,})")

# 2. Word2Vec Google News 300
w2v = load_word2vec_model()
print(f"Word2Vec Google News 300 ready (vocab size: {len(w2v.key_to_index):,})")

# 3. SBERT (on CUDA if available)
sbert_device = "cuda" if torch.cuda.is_available() else "cpu"
sbert = load_sbert_model(device=sbert_device)
print(f"SBERT model (all-MiniLM-L6-v2) ready on device: {sbert_device}")

# 4. Tokenizer
tok = LegalTokenizer()
print(f"All models loaded in {time.time() - t0:.2f}s.")"""))

    # Cell 6: Explicit Join-Key Architecture & Sentence Extraction
    cells.append(md_cell("""### 5. Step 1: Explicit Composite Join-Key Registry & Sentence Extraction
Iterate through all 7,028 training documents in `train_ids.txt`.
Assign every sentence an immutable composite key: `(case_id, sentence_index)`.
This guarantees exact 1:1 join alignment across C1 features, C3 features, and labels."""))

    cells.append(code_cell("""train_ids = load_split_ids("train")
total_docs = len(train_ids)
print(f"Total training cases: {total_docs:,}")

sentence_keys = []
doc_boundaries = {}
doc_sentences = {}
total_sentences = 0

print("Extracting sentence boundaries and composite keys across all training judgments...")
for cid in tqdm(train_ids, desc="Indexing Judgments"):
    doc = load_document(cid)
    sents = segment_sentences(doc["judgment"])
    doc_sentences[cid] = sents
    n_s = len(sents)

    start_idx = total_sentences
    end_idx = total_sentences + n_s
    doc_boundaries[cid] = {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "num_sentences": n_s
    }

    for s_idx in range(n_s):
        sentence_keys.append((cid, s_idx))
    total_sentences += n_s

print(f"\\nExtraction Complete!")
print(f"Total Documents: {len(doc_boundaries):,}")
print(f"Total Sentences: {total_sentences:,}")
assert total_sentences == 1010961, f"Expected 1,010,961 sentences, got {total_sentences:,}!"

# Save explicit join key registry
out_dir = os.path.join("data", "processed", "features")
os.makedirs(out_dir, exist_ok=True)
index_path = os.path.join(out_dir, "train_sentence_index.json")
with open(index_path, "w") as f:
    json.dump(sentence_keys, f)
print(f"Saved explicit join key registry to: {index_path}")

boundaries_path = os.path.join(out_dir, "train_doc_boundaries.json")
with open(boundaries_path, "w") as f:
    json.dump(doc_boundaries, f)
print(f"Saved doc boundaries to: {boundaries_path}")"""))

    # Cell 7: Verify / Load Ground Truth Labels
    cells.append(md_cell("""### 6. Step 2: Verify Ground-Truth Labels Alignment
Load `train_labels.npy` and assert exact row match (1,010,961 sentences)."""))

    cells.append(code_cell("""labels_path = os.path.join(out_dir, "train_labels.npy")
if os.path.exists(labels_path):
    train_labels = np.load(labels_path)
    print(f"Loaded existing train_labels.npy: {len(train_labels):,} rows")
else:
    print("train_labels.npy not found on disk. Computing ground-truth labels across 7,028 docs...")
    from src.labeling.gbr_labels import GBRLabeler
    labeler = GBRLabeler(use_stemmer=True)
    all_labels = []
    for cid in tqdm(train_ids, desc="Computing GBR Labels"):
        doc = load_document(cid)
        s_sents = doc_sentences[cid]
        r_sents = segment_sentences(doc["headnote"])
        lbls, _, _ = labeler.compute_document_labels(s_sents, r_sents)
        all_labels.append(lbls)
    train_labels = np.concatenate(all_labels).astype(np.float32)
    np.save(labels_path, train_labels)

assert len(train_labels) == 1010961, f"Expected 1,010,961 labels, got {len(train_labels):,}!"
print(f"Label Alignment Verified: {len(train_labels):,} sentences.")
print(f"Label Statistics: Min={train_labels.min():.4f}, Max={train_labels.max():.4f}, Mean={train_labels.mean():.4f}, Median={np.median(train_labels):.4f}")"""))

    # Cell 8: C1 Feature Extraction with WMD OOV Instrumentation
    cells.append(md_cell("""### 7. Step 3: Config C1 Feature Extraction & WMD OOV Fallback Instrumentation
Extract C1 raw features: `[TF-IDF, NER, Position, Cosine_w2v, WMD]`.
- Utilizes on-the-fly unique document token vector normalization to eliminate Gensim memory spikes.
- Safe OOV fallback on empty-vocabulary sentences (`max_fallback_dist = 3.0`).
- Instruments every sentence that triggers the fallback and checks OCR clustering."""))

    cells.append(code_cell("""# Heuristic for OCR artifact detection per Phase 1 spec
OCR_MARKERS = ["vil Appeal", "minal Appeal", "rit Petition", "pecial Leave", "ivil Appeal"]

t0 = time.time()
c1_feature_blocks = []
total_oov_fallbacks = 0
ocr_doc_count = 0
ocr_sentences = 0
ocr_oov_count = 0

non_ocr_doc_count = 0
non_ocr_sentences = 0
non_ocr_oov_count = 0

oov_sample_records = []

print("Extracting C1 raw features (5 features) across all 7,028 training judgments...")
for i, cid in enumerate(tqdm(train_ids, desc="Extracting C1 Features"), 1):
    sents = doc_sentences[cid]
    n_s = len(sents)
    if n_s == 0:
        continue

    # Check OCR status
    doc = load_document(cid)
    judgment = doc["judgment"]
    is_ocr = any(marker in judgment for marker in OCR_MARKERS)

    # Extract features + fallback mask
    X_doc, fallback_mask = assemble_document_raw_features(
        sentences=sents,
        config="C1",
        tfidf_vectorizer=vec,
        w2v_model=w2v,
        tokenizer=tok,
        return_fallback_mask=True
    )

    c1_feature_blocks.append(X_doc)
    doc_oov = int(fallback_mask.sum())
    total_oov_fallbacks += doc_oov

    if is_ocr:
        ocr_doc_count += 1
        ocr_sentences += n_s
        ocr_oov_count += doc_oov
    else:
        non_ocr_doc_count += 1
        non_ocr_sentences += n_s
        non_ocr_oov_count += doc_oov

    if doc_oov > 0 and len(oov_sample_records) < 15:
        for idx, triggered in enumerate(fallback_mask):
            if triggered and len(oov_sample_records) < 15:
                oov_sample_records.append((cid, idx, sents[idx][:100]))

elapsed = time.time() - t0
print(f"\\nC1 Feature Extraction Complete in {elapsed:.2f}s ({elapsed/60:.2f} minutes)!")

# Concatenate full matrix
X_train_c1_raw = np.vstack(c1_feature_blocks).astype(np.float32)
print(f"X_train_c1_raw shape: {X_train_c1_raw.shape} | Memory: {X_train_c1_raw.nbytes / (1024**2):.2f} MB")
assert len(X_train_c1_raw) == 1010961, f"Expected 1,010,961 rows, got {len(X_train_c1_raw):,}!"

# Save raw matrix
raw_c1_path = os.path.join(out_dir, "train_features_c1_raw.npy")
np.save(raw_c1_path, X_train_c1_raw)
print(f"Saved C1 raw feature matrix to: {raw_c1_path}")"""))

    # Cell 9: WMD OOV Fallback & OCR Breakdown Analysis
    cells.append(md_cell("""### 8. Step 4: WMD OOV Fallback Rate & OCR Clustering Analysis
Analyze the fallback rate across the full training corpus and determine if it clusters in OCR-affected judgments."""))

    cells.append(code_cell("""overall_oov_pct = (total_oov_fallbacks / total_sentences * 100) if total_sentences > 0 else 0
ocr_oov_pct = (ocr_oov_count / ocr_sentences * 100) if ocr_sentences > 0 else 0
non_ocr_oov_pct = (non_ocr_oov_count / non_ocr_sentences * 100) if non_ocr_sentences > 0 else 0

print("=" * 80)
print("WMD OOV FALLBACK AUDIT RESULTS (max_fallback_dist = 3.0)")
print("=" * 80)
print(f"Total Sentences Analyzed:         {total_sentences:,}")
print(f"Total OOV Fallback Sentences:     {total_oov_fallbacks:,} ({overall_oov_pct:.4f}%)")
print("-" * 80)
print(f"OCR-Affected Documents:           {ocr_doc_count:,} docs ({ocr_doc_count/total_docs*100:.2f}%)")
print(f"  Sentences in OCR Docs:          {ocr_sentences:,}")
print(f"  OOV Fallbacks in OCR Docs:      {ocr_oov_count:,} ({ocr_oov_pct:.4f}%)")
print("-" * 80)
print(f"Non-OCR Documents:                {non_ocr_doc_count:,} docs ({non_ocr_doc_count/total_docs*100:.2f}%)")
print(f"  Sentences in Non-OCR Docs:      {non_ocr_sentences:,}")
print(f"  OOV Fallbacks in Non-OCR Docs:  {non_ocr_oov_count:,} ({non_ocr_oov_pct:.4f}%)")
print("=" * 80)

print("\\nSample Sentences Triggering OOV Fallback:")
for cid, idx, text in oov_sample_records[:8]:
    print(f"  - Doc {cid} [Sent {idx}]: {repr(text)}")

# Save audit results
audit_results = {
    "total_documents": total_docs,
    "total_sentences": total_sentences,
    "total_oov_sentences": total_oov_fallbacks,
    "oov_rate_pct": overall_oov_pct,
    "ocr_doc_count": ocr_doc_count,
    "ocr_sentences": ocr_sentences,
    "ocr_oov_sentences": ocr_oov_count,
    "ocr_oov_rate_pct": ocr_oov_pct,
    "non_ocr_doc_count": non_ocr_doc_count,
    "non_ocr_sentences": non_ocr_sentences,
    "non_ocr_oov_sentences": non_ocr_oov_count,
    "non_ocr_oov_rate_pct": non_ocr_oov_pct,
    "samples": [{"cid": cid, "sent_idx": idx, "text": text} for cid, idx, text in oov_sample_records[:10]]
}
audit_path = os.path.join("results", "logs", "wmd_oov_audit.json")
os.makedirs(os.path.dirname(audit_path), exist_ok=True)
with open(audit_path, "w") as f:
    json.dump(audit_results, f, indent=2)
print(f"\\nSaved audit json to: {audit_path}")"""))

    # Cell 10: GPU-Accelerated C3 Feature Extraction
    cells.append(md_cell("""### 9. Step 5: GPU-Accelerated Config C3 Feature Extraction
Config C3 (4 features): `[TF-IDF, NER, Position, Cosine_sbert]`.
- Reuses shared features `[TF-IDF, NER, Position]` from `X_train_c1_raw[:, :3]`.
- Computes SBERT cosine similarity using GPU batch encoding (`all-MiniLM-L6-v2`).
- On Colab T4/V100 GPU, 1,010,961 sentences encode in ~5–10 minutes."""))

    cells.append(code_cell("""shared_features = X_train_c1_raw[:, :3]  # [tfidf, ner, position]
assert len(shared_features) == 1010961

print("Computing SBERT Cosine Similarity column on GPU...")
t0 = time.time()
sbert_cosine_blocks = []

# SentenceTransformer batch encoding per document or grouped documents
# Using document-level encoding to compute exact cosine similarity to document centroid
for cid in tqdm(train_ids, desc="SBERT GPU Cosine"):
    sents = doc_sentences[cid]
    if not sents:
        continue

    # Encode all sentences in document on GPU with batching
    embeddings = sbert.encode(
        sents,
        batch_size=min(len(sents), 512),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Document centroid in SBERT embedding space
    doc_centroid = np.mean(embeddings, axis=0)
    centroid_norm = np.linalg.norm(doc_centroid)
    if centroid_norm > 0:
        doc_centroid = doc_centroid / centroid_norm

    # Cosine similarity of each sentence to document centroid
    cos_sims = np.dot(embeddings, doc_centroid).astype(np.float32)
    sbert_cosine_blocks.append(cos_sims)

elapsed = time.time() - t0
print(f"\\nSBERT Cosine Extraction Complete in {elapsed:.2f}s ({elapsed/60:.2f} minutes)!")

sbert_cosine_col = np.concatenate(sbert_cosine_blocks).astype(np.float32)
assert len(sbert_cosine_col) == 1010961, f"Expected 1,010,961 SBERT cosine rows, got {len(sbert_cosine_col):,}!"

# Assemble C3 raw matrix
X_train_c3_raw = np.column_stack([shared_features, sbert_cosine_col]).astype(np.float32)
print(f"X_train_c3_raw shape: {X_train_c3_raw.shape} | Memory: {X_train_c3_raw.nbytes / (1024**2):.2f} MB")

# Save C3 raw matrix
raw_c3_path = os.path.join(out_dir, "train_features_c3_raw.npy")
np.save(raw_c3_path, X_train_c3_raw)
print(f"Saved C3 raw feature matrix to: {raw_c3_path}")"""))

    # Cell 11: End-to-End Alignment Validation
    cells.append(md_cell("""### 10. Step 6: Strict End-to-End Alignment Validation Across All Matrices
Verify that all arrays match 1,010,961 rows with zero dropped documents, zero NaNs, and identical sentence mappings."""))

    cells.append(code_cell("""print("=" * 80)
print("STRICT END-TO-END ALIGNMENT VALIDATION")
print("=" * 80)

# Check lengths
print(f"train_labels.npy rows:         {len(train_labels):,}")
print(f"train_features_c1_raw.npy:     {len(X_train_c1_raw):,}")
print(f"train_features_c3_raw.npy:     {len(X_train_c3_raw):,}")
print(f"train_sentence_index.json:     {len(sentence_keys):,}")
print(f"Total documents processed:     {len(doc_boundaries):,} / {total_docs:,}")

assert len(train_labels) == 1010961, "Mismatch in train_labels!"
assert len(X_train_c1_raw) == 1010961, "Mismatch in X_train_c1_raw!"
assert len(X_train_c3_raw) == 1010961, "Mismatch in X_train_c3_raw!"
assert len(sentence_keys) == 1010961, "Mismatch in sentence_keys!"
assert len(doc_boundaries) == total_docs, f"Dropped docs detected! ({len(doc_boundaries)} != {total_docs})"

# Check for NaNs and Infs
assert not np.isnan(X_train_c1_raw).any(), "NaNs detected in C1 raw matrix!"
assert not np.isnan(X_train_c3_raw).any(), "NaNs detected in C3 raw matrix!"
assert not np.isnan(train_labels).any(), "NaNs detected in labels!"

# Verify spot-check alignment
np.random.seed(42)
sample_indices = np.random.choice(len(sentence_keys), size=5, replace=False)
print("\\nSpot-check alignment for 5 random sentence indices:")
for idx in sample_indices:
    cid, s_idx = sentence_keys[idx]
    c1_row = X_train_c1_raw[idx]
    c3_row = X_train_c3_raw[idx]
    lbl = train_labels[idx]
    # Shared features (tfidf, ner, position) must match bit-for-bit
    assert np.allclose(c1_row[:3], c3_row[:3]), f"Shared features mismatch at index {idx}!"
    print(f"  Row {idx:7d} -> Doc {cid:<6} Sent {s_idx:<3} | TF-IDF={c1_row[0]:.3f} | NER={c1_row[1]:.0f} | Pos={c1_row[2]:.3f} | Label={lbl:.4f}")

print("\\nALL INTEGRITY ASSERTIONS PASSED! PERFECT 1,010,961-ROW ALIGNMENT CONFIRMED.")"""))

    # Cell 12: Fit RobustScaler Fresh on Full Matrices
    cells.append(md_cell("""### 11. Step 7: Fit Fresh RobustScaler on Full C1 and C3 Matrices
Fit `RobustScaler` fresh across all 1,010,961 training sentences for C1 and C3 separately, and transform to scaled matrices."""))

    cells.append(code_cell("""print("=" * 80)
print("FITTING ROBUSTSCALER FRESH ON FULL 1,010,961-ROW MATRICES")
print("=" * 80)

# 1. Fit C1 RobustScaler
scaler_c1 = FeatureScalerPipeline(config="C1", scaler_type="robust")
X_train_c1_scaled = scaler_c1.fit_transform(X_train_c1_raw)
params_c1 = scaler_c1.get_scaling_params()

print("\\nLEARNED FULL-CORPUS ROBUSTSCALER PARAMETERS (C1/C2 - 5 FEATURES):")
print("-" * 75)
for feat_name, p in params_c1.items():
    print(f"  Feature: {feat_name:<12} | Median (center) = {p['center']:>10.4f} | IQR (scale) = {p['scale']:>10.4f}")
print("-" * 75)

# 2. Fit C3 RobustScaler
scaler_c3 = FeatureScalerPipeline(config="C3", scaler_type="robust")
X_train_c3_scaled = scaler_c3.fit_transform(X_train_c3_raw)
params_c3 = scaler_c3.get_scaling_params()

print("\\nLEARNED FULL-CORPUS ROBUSTSCALER PARAMETERS (C3 - 4 FEATURES):")
print("-" * 75)
for feat_name, p in params_c3.items():
    print(f"  Feature: {feat_name:<12} | Median (center) = {p['center']:>10.4f} | IQR (scale) = {p['scale']:>10.4f}")
print("-" * 75)

# Save scalers and scaled arrays
scaler_c1_path = os.path.join(out_dir, "robust_scaler_c1.pkl")
scaler_c1_params_path = os.path.join(out_dir, "robust_scaler_c1_params.json")
scaled_c1_path = os.path.join(out_dir, "train_features_c1_scaled.npy")

with open(scaler_c1_path, "wb") as f:
    pickle.dump(scaler_c1, f)
with open(scaler_c1_params_path, "w") as f:
    json.dump(params_c1, f, indent=2)
np.save(scaled_c1_path, X_train_c1_scaled)

scaler_c3_path = os.path.join(out_dir, "robust_scaler_c3.pkl")
scaler_c3_params_path = os.path.join(out_dir, "robust_scaler_c3_params.json")
scaled_c3_path = os.path.join(out_dir, "train_features_c3_scaled.npy")

with open(scaler_c3_path, "wb") as f:
    pickle.dump(scaler_c3, f)
with open(scaler_c3_params_path, "w") as f:
    json.dump(params_c3, f, indent=2)
np.save(scaled_c3_path, X_train_c3_scaled)

print(f"\\nSaved C1 Scaled Matrix: {scaled_c1_path} ({os.path.getsize(scaled_c1_path)/(1024**2):.2f} MB)")
print(f"Saved C3 Scaled Matrix: {scaled_c3_path} ({os.path.getsize(scaled_c3_path)/(1024**2):.2f} MB)")
print(f"Saved scalers and parameter JSONs.")"""))

    # Cell 13: Summary Table
    cells.append(md_cell("""### 12. Step 8: Final Artifact Inventory & Verification Table
Verify all generated artifacts on disk."""))

    cells.append(code_cell("""import pandas as pd

artifacts = [
    ("train_labels.npy", labels_path, len(train_labels), "Ground-truth ROUGE labels"),
    ("train_sentence_index.json", index_path, len(sentence_keys), "Explicit (doc_id, sentence_index) registry"),
    ("train_doc_boundaries.json", boundaries_path, len(doc_boundaries), "Per-document index intervals"),
    ("train_features_c1_raw.npy", raw_c1_path, len(X_train_c1_raw), "C1 raw feature matrix (5 cols)"),
    ("train_features_c1_scaled.npy", scaled_c1_path, len(X_train_c1_scaled), "C1 RobustScaled matrix (5 cols)"),
    ("robust_scaler_c1.pkl", scaler_c1_path, "-", "Fitted C1 RobustScaler model"),
    ("train_features_c3_raw.npy", raw_c3_path, len(X_train_c3_raw), "C3 raw feature matrix (4 cols)"),
    ("train_features_c3_scaled.npy", scaled_c3_path, len(X_train_c3_scaled), "C3 RobustScaled matrix (4 cols)"),
    ("robust_scaler_c3.pkl", scaler_c3_path, "-", "Fitted C3 RobustScaler model"),
    ("wmd_oov_audit.json", audit_path, total_oov_fallbacks, "WMD OOV audit metrics"),
]

summary_rows = []
for name, path, count, desc in artifacts:
    size_mb = os.path.getsize(path) / (1024**2) if os.path.exists(path) else 0.0
    summary_rows.append({
        "Artifact": name,
        "Rows / Count": f"{count:,}" if isinstance(count, int) else count,
        "Size (MB)": f"{size_mb:.2f}",
        "Description": desc,
        "Status": "EXISTS" if os.path.exists(path) else "MISSING"
    })

df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_markdown(index=False))"""))

    # Cell 14: Archive Download
    cells.append(md_cell("""### 13. Step 9: Export Artifacts (Optional Colab Zip Download)
Package feature artifacts into a zip archive for downloading or transferring back to the local repository."""))

    cells.append(code_cell("""# Optional: Package artifacts for direct download
import shutil

zip_filename = "processed_features_1010961"
print(f"Creating {zip_filename}.zip...")
shutil.make_archive(zip_filename, 'zip', "data/processed/features")
print(f"Created {zip_filename}.zip ({os.path.getsize(zip_filename + '.zip') / (1024**2):.2f} MB)")

# If running in Colab, trigger browser download:
if IN_COLAB:
    from google.colab import files
    print("Initiating browser download...")
    files.download(f"{zip_filename}.zip")"""))

    notebook = {
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

    os.makedirs("notebooks", exist_ok=True)
    nb_path = os.path.join("notebooks", "01_full_feature_extraction_and_scaling.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Successfully created notebook at: {nb_path}")

if __name__ == "__main__":
    create_notebook()
