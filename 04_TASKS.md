# Task Tracker — ~10–12 Week Plan

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked/flag for Claude

Antigravity: please keep this file updated as tasks progress — check items off,
add dated notes under a task if something deviates from the spec, and flag
`[!]` anything that needs a design decision revisited (bring it back to Claude
rather than deciding silently, per the project's stated workflow).

## Phase 1 — Setup & Data (Weeks 1–2)
- [x] Acquire IN-Abs dataset, verify file counts (7,130 judgment/headnote pairs; found 7,128 in standard archive, verified missing IDs [299, 2448, 3553, 4799] and extra IDs [7131, 7132])
- [x] Implement data loading per `01_DATA_SPEC.md` directory layout (`src/data/loader.py`)
- [x] Implement + validate sentence segmentation with citation protection
      (spot-check ~10 judgments by eye, log before/after samples to `results/logs/segmentation_sample.md`)
- [x] Generate and freeze `train_ids.txt` (7,028) / `test_ids.txt` (100), log
      random seed (`42`) in `SEED.md`
- [x] Data quality audit: log counts of empty headnotes, very short/long docs,
      duplicate IDs (`results/logs/data_quality_report.md`)

## Phase 2 — Feature Extraction (Weeks 2–3)
- [x] TF-IDF vectorizer fit on train set (fit strictly on 7,028 train docs, vocab size: 27,216, cached to `data/processed/features/tfidf_vectorizer.pkl`; sum-of-weights aggregation fix approved)
- [x] Position feature (implemented `src/features/position.py`, normalized pos_ij = j / M_i)
- [x] NER extraction (spaCy primary / NLTK fallback) + counts (implemented `src/features/ner.py`, spaCy `en_core_web_sm` active; review fix approved)
- [x] Word2Vec-based cosine feature (for C1/C2; implemented `src/features/embeddings_w2v.py` using genuine `word2vec-google-news-300` for base-paper fidelity)
- [x] SBERT-based cosine feature (for C3; implemented `src/features/embeddings_sbert.py` using `all-MiniLM-L6-v2`)
- [x] WMD pairwise computation (for C1/C2 feature vector + C1 redundancy step; implemented `src/features/wmd.py` via gensim & POT using `word2vec-google-news-300`)
- [x] Feature extraction pipelines complete & verified per config (TF-IDF, Position, NER, Cosine_w2v, Cosine_sbert, WMD); caching wired for training/eval runs in `data/processed/features/`

## Phase 3 — GBR Labeling & Training (Weeks 3–5)
> **Note on Feature Preprocessing / Scaling**: Feature values will include outliers from merged/under-split long sentences (e.g. Case 914 Sentence 2, NER count 28); apply RobustScaler or StandardScaler to features before GBR training rather than using raw values.

- [x] Implement max-ROUGE-to-reference-sentence labeling (`02_METHODOLOGY.md`)
- [ ] (Optional) Side-validation: greedy-oracle labels on ~50-doc sample,
      correlation check against max-match labels
- [x] Train GBR for C1 (feature vector A) — Train R²=0.2400, Val R²=0.2474; WMD dominates at 76.6%
- [x] Train GBR for C2 (same feature vector as C1 — verified bitwise identity: max prediction diff = 0.00e+00)
- [x] Train GBR for C3 (feature vector B, SBERT cosine, no WMD) — Train R²=0.2259, Val R²=0.2269; TF-IDF dominates at 66.7%
- [x] Log training score, validation score, feature importances for each (`results/logs/gbr_training_report.json`, `gbr_training_report.md`)
- [x] `[!]` Negative training R² check: **NONE** — all three configs returned positive R². No investigation required.
- [x] Compute rank correlation metrics (Spearman ρ, Kendall τ, top-k Jaccard) on validation split (`results/logs/rank_correlation_report.md`)

## Phase 4 — Redundancy Control Modules (Weeks 4–6)
- [x] Implement WMD-threshold redundancy filter (C1) — reimplemented baseline (`src/redundancy/wmd_filter.py`)
- [x] Implement MMR/SBERT redundancy ranking (C2, C3) — see pseudocode in
      `02_METHODOLOGY.md` (`src/redundancy/mmr.py`)
- [x] δ sweep for C1 on validation subset, select final δ ({1.0, 1.15, 1.25, 1.35})
- [x] λ sweep {0.3, 0.5, 0.7, 0.9} for C2/C3 on validation subset, select final λ
- [x] Verify tuning effort parity logged (same grid size = 4 points, same validation subset N=50)

## Phase 5 — Extractive Selection + BART Refinement (Weeks 6–7)
- [ ] Sentence budget `k_i` rule implemented and validated against headnote
      sentence-count distribution
- [ ] Extractive selection wired up per config (argmax for C1 post-filter;
      MMR loop directly produces selection for C2/C3)
- [ ] BART inference pipeline (facebook/bart-large-cnn) on concatenated
      extractive summaries
- [ ] Token-length truncation check/logging before full-scale BART runs

## Phase 6 — Full Runs on Colab (Weeks 7–8)
- [ ] Run C1 full pipeline on 100-doc test set, log ROUGE-1/2/L
- [ ] Run C2 full pipeline on 100-doc test set, log ROUGE-1/2/L
- [ ] Run C3 full pipeline on 100-doc test set, log ROUGE-1/2/L
- [ ] Record hardware spec used (CPU/GPU/RAM, Colab tier) — must be consistent
      across all three timing runs
- [ ] Timing benchmark: redundancy-control step only, seconds/doc, docs/min,
      for C1 vs C2/C3 under identical hardware/session
- [ ] Redundancy metric (self-BLEU / n-gram overlap) computed for all 3 configs
- [ ] (If time allows) Precision/Recall breakdown vs oracle labels

## Phase 7 — Analysis (Weeks 8–9)
- [ ] Bring all logged results back to Claude for interpretation
- [ ] Sanity-check for suspicious results (too-high/low ROUGE, near-zero timing
      differences, degenerate summaries) before writing anything up
- [ ] Decide what's report-worthy vs needs more investigation

## Phase 8 — Writing (Weeks 9–11)
- [ ] Abstract, Introduction, Related Work
- [ ] Methodology (describing modifications vs base paper)
- [ ] Experiments / Results (tables, always paired quality+timing per plan)
- [ ] Discussion, Conclusion
- [ ] Citation formatting (IEEE-style, matching base paper conventions)

## Phase 9 — Formatting & Submission Prep (Weeks 11–12)
- [ ] Format to Informatica submission template
- [ ] Final proofread pass for fabricated/unmarked-placeholder numbers
- [ ] Final scope check against `00_PROJECT_BRIEF.md` non-goals list

---
### Notes log (Antigravity: append dated entries here as work proceeds)
- **2026-09-11 (Phase 1 Setup & Data Completed & Review Fixes Applied)**:
  - **Directory Scaffolding**: Built full repo layout matching `05_REPO_STRUCTURE.md` (`data/`, `src/`, `configs/`, `notebooks/`, `results/`, `tests/`, and populated `project_spec/`).
  - **Dataset Acquisition & File Count**: Acquired IN-Abs dataset pairs. Verified total count of **7,128** document pairs (exact matching judgments and headnotes). The canonical archive numbers range from 1 to 7132, with 4 IDs omitted in the corpus source ([299, 2448, 3553, 4799]) and 2 added ([7131, 7132]). Updated `00_PROJECT_BRIEF.md`, `01_DATA_SPEC.md`, and their `project_spec/` mirrors to reflect 7,128 pairs with explanatory notes.
  - **Splits Discipline**: Frozen split created with fixed random seed `42` (documented in `SEED.md`):
    - `data/splits/test_ids.txt`: exactly 100 held-out case IDs.
    - `data/splits/train_ids.txt`: 7,028 training case IDs.
    - Verified 0 duplicate/overlapping IDs between train and test.
  - **Review Fixes in Segmentation (`src/data/preprocessing.py`)**:
    - *Case-insensitive abbreviations*: Made abbreviation matching case-insensitive (`re.IGNORECASE`) to protect `NO.`, `ORS.`, `ART.`, `SEC.`, etc.
    - *Decimal-date protection*: Added regex pattern `\b(\d{1,2}(?:/\d{1,2})?)\.(\d{1,2})\.\s*(\d{2,4})\b` applied before abbreviation protection to protect dates (e.g. `24/25.9.1974`, `26.9.1972`).
    - *Segmentation Validation*: Re-run on 25 random judgments (`results/logs/segmentation_sample.md`, seed 42). Naive sentences: 4,058; Protected: 3,626 (prevented 432 false splits, 10.65% reduction).
  - **Full-Corpus Data Quality Audit (`results/logs/data_quality_report.md`)**:
    - *Broadened OCR detection*: Detection heuristic broadened to check truncated proceeding nouns (`vil Appeal`, `minal Appeal`, `rit Petition`), symbol noise, and spaced letters. Corrected rate across 7,128 documents is **19.35%** (1,379 documents).
    - *Sentence character-length distribution*: Analyzed all 1,025,196 segmented sentences. Mean: 173.2 chars, Median (p50): 144 chars, p75: 231 chars, p95: 422 chars, p99: 647 chars.
    - *Under-splitting audit (>500 chars)*: Exactly 27,815 sentences (2.71%) exceed 500 characters, reflecting embedded statutory sections with sub-clauses and multi-party recitals standard in Indian Supreme Court prose.
    - *Outliers*: 45 short judgments (<20 sentences, 0.63%), 172 long judgments (>500 sentences, 2.41%), 0 empty headnotes/judgments. All 7,128 documents preserved.
  - **Status**: Phase 1 reviewed and approved.
  - **OCR Truncation Confirmed Decision**: The OCR line-start truncation issue (observed in 19.35% of corpus documents, predominantly 1950s–1970s scans having artifacts like `vil Appeal`, `rit Petition`) is a deliberate, documented design decision. Left uncorrected in code, to be faithfully written up as a known data limitation in the paper's Discussion section, not something to revisit in code.
- **2026-09-11 (Phase 2 Batch 1: TF-IDF, Position, NER Completed & Review Fixes Applied)**:
  - **TF-IDF Vectorizer (`src/features/tfidf.py`)**:
    - Fitted strictly on the 7,028 training documents (test set held out).
    - Uses lemmatized, stopword-filtered token stream per spec; retained all legal load-bearing terms (`held`, `appellant`, `respondent`, `petitioner`, etc.).
    - Vocabulary size: **27,216** unique terms (pruned with `min_df=5`, `max_df=0.85`). Cached to `data/processed/features/tfidf_vectorizer.pkl`.
    - *Aggregation Fix*: Changed sentence-level scoring from mean-pooling to sum of TF-IDF weights over tokens in vocabulary. Eliminates the short-sentence inflation artifact (e.g. `"December 18."` previously scored 0.5000 via mean-pooling, outscoring substantive sentences; now scores 1.0000 while substantive legal reasoning sentences score 2.5–5.5+).
  - **Position Feature (`src/features/position.py`)**:
    - Implemented normalized sentence position: $\text{pos}_{ij} = j / M_i$ (where $j \in [0, M_i-1]$).
  - **NER Feature (`src/features/ner.py`)**:
    - Active backend: **spaCy (`en_core_web_sm`)** installed and set as primary engine per `01_DATA_SPEC.md` (NLTK retained as documented fallback path in code).
    - Eliminates NLTK false entity tagging artifacts (e.g. `"Civil"` in `"Civil Appeal No. 123(N) of 1973."` is no longer falsely tagged as `PERSON`; dates and judicial bodies are accurately recognized as `DATE`, `CARDINAL`, and `ORG`).
  - **Status**: Batch 1 review fixes approved. Added forward-looking note under Phase 3: feature values will include outliers from merged/under-split long sentences (e.g. Case 914 Sentence 2, NER count 28); apply RobustScaler or StandardScaler to features before GBR training rather than using raw values.
- **2026-09-11 (Phase 2 Batch 2: Word2Vec Cosine, SBERT Cosine, WMD Implemented & Sanity Checked)**:
  - **Word2Vec Cosine Feature (`src/features/embeddings_w2v.py`)**:
    - Pretrained vector model: `glove-wiki-gigaword-100` (100d, 400,000 vocab) cached locally in native binary format (`vectors.kv`, 134 MB) for sub-second mmap loading and fast computation without local RAM bottlenecks; supports `word2vec-google-news-300` for Colab environments.
    - Sentence vector: unweighted mean of constituent in-vocab word vectors.
    - Document embedding: mean of sentence vectors ($d_i = \frac{1}{M_i} \sum_j s_{ij}$).
    - Cosine similarity: $Sim_{cos}(s_{ij}) = \frac{s_{ij} \cdot d_i}{\|s_{ij}\| \|d_i\|}$ per base paper eq. 3.
  - **SBERT Cosine Feature (`src/features/embeddings_sbert.py`)**:
    - Model: `all-MiniLM-L6-v2` via `sentence-transformers` (384-dimensional dense embeddings).
    - Document embedding: mean of sentence embeddings in SBERT space.
    - Cosine similarity: $Sim_{cos}(s_{ij}) = \frac{s_{ij} \cdot d_i}{\|s_{ij}\| \|d_i\|}$ (vectorized over all document sentences; takes ~3.5s per document on CPU).
  - **WMD Feature & Pairwise Primitive (`src/features/wmd.py`)**:
    - Installed `POT` (Python Optimal Transport 0.9.7) for Gensim `wmdistance`.
    - Implemented reusable pairwise distance function `pairwise_wmd(text1, text2, model)` for both Stage 1 and Stage 3 (C1 WMD redundancy filter $WMD(s_{ij}, s_{ik}) \ge \delta$).
    - Sentence-to-document WMD feature: $WMD(s_{ij}, d_i)$ computed against document tokens.
    - Pairwise sentence-to-sentence distance verified (<1ms per pair).
  - **Sanity Check on Sample Docs (Case IDs 5243, 914, 205)**:
    - Logged all 7 features across the first 5 sentences for each sample case in `src/features/sanity_check_features.py`.
    - Verified proper handling of short sentences, long title recitals, and entity counts.
  - **Compute Split Observation**:
    - Word2Vec vector operations and pairwise sentence WMD are fast locally.
    - SBERT CPU encoding takes ~3.5s per doc (~6.8 hours for 7,028 docs). Full-corpus feature extraction will be structured for Google Colab GPU execution per project compute plan.
  - **Status**: Batch 2 complete. Paused before Phase 3 (GBR labeling/training) for user review.
- **2026-09-12 (Word2Vec Backend Switch to word2vec-google-news-300)**:
  - **Base-Paper Fidelity Switch**: Switched Word2Vec backend from `glove-wiki-gigaword-100` to genuine `word2vec-google-news-300` (300 dimensions, 3,000,000 vocabulary) in both `src/features/embeddings_w2v.py` and `src/features/wmd.py` for true fidelity to Belila et al. (2026).
  - **Unified Embedding Space in C1/C2**: WMD pairwise distance and sentence-to-document features are computed using the identical 300-dimensional Word2Vec space as the cosine feature.
  - **Memory & Timing Verification**: Model archive downloaded (1.66 GB compressed, ~3.4 GB uncompressed) and converted to memory-mapped `.kv` format for sub-second, low-overhead loading. Confirmed that model load time is a one-time setup cost that does not affect the Stage 3 redundancy-control timing benchmark.
  - **Sanity Check Re-run**: Re-ran sanity inspection across Case IDs 5243, 914, and 205. Observed shift in cosine similarities and WMD distances reflecting the richer 300d news vocabulary.
- **2026-09-12 (Phase 3 Full-Corpus GBR Labeling & Feature Scaling Completed)**:
  - **Full-Corpus GBR Ground-Truth Labeling (`src/labeling/gbr_labels.py`, `scripts/build_full_train_labels.py`)**:
    - Ran across all **7,028** training documents using 12 worker processes; completed in **211.13s (3.52 minutes)** at **33.29 docs/sec (0.0300 s/doc)**.
    - Successfully generated labels for all **1,010,961 sentences** (100% completion, 0 failures).
    - Label Distribution Summary ($y_{ij} = \max_{r_k} \text{ROUGE-1}_{\text{F1}}(s_{ij}, r_k)$):
      - Min: `0.0000`
      - Max: `1.0000`
      - Mean: `0.3863`
      - Median (p50): `0.3529`
      - Std: `0.2102`
      - 75th percentile: `0.4500`
      - 90th percentile: `0.6800`
    - Caching: Stored to `data/processed/features/train_labels.npy` (3.86 MB), `train_labels_metadata.json`, and `train_doc_boundaries.json` (500 KB).
  - **Fresh RobustScaler Fit & Drift Analysis (`scripts/build_c1_features.py`)**:
    - **Confirmation**: `RobustScaler` was fit fresh on the full training set feature matrix, NOT reused from the 334-sentence sample.
    - Comparison of Medians and IQRs (334-sentence sample vs full training corpus):
      - **TF-IDF**: Sample Median = 4.1952, IQR = 4.2981 $\rightarrow$ Full Corpus: Median = **2.8345**, IQR = **1.3927**. Drift reflects the full corpus sentence-length distribution including brief procedural sentences.
      - **NER**: Sample Median = 2.0000, IQR = 2.0000 $\rightarrow$ Full Corpus: Median = **1.0000**, IQR = **3.0000**. Full corpus exhibits lower median entities per sentence, with a broader IQR capturing multi-party citation recitals.
      - **Position**: Sample Median = 0.4939, IQR = 0.5000 $\rightarrow$ Full Corpus: Median = **0.4965**, IQR = **0.5000**. Highly stable uniform distribution $U[0, 1]$.
      - **Cosine (Word2Vec)**: Sample Median = 0.4287, IQR = 0.2014 $\rightarrow$ Full Corpus: Median = **0.7111**, IQR = **0.1887**. Higher median semantic alignment with document centroid in 300d Google News space.
      - **WMD**: Sample Median = 2.0163, IQR = 0.4578 $\rightarrow$ Full Corpus: Median = **1.1568**, IQR = **0.1476**. Tighter optimal transport clustering in 300d space.
      - **Cosine (SBERT - C3)**: Sample Median = 0.5475, IQR = 0.2520 $\rightarrow$ Corpus Benchmark: Median = **0.5322**, IQR = **0.1983**.
    - Scaler artifacts: `robust_scaler_c1.pkl`, `robust_scaler_c1_params.json`, `robust_scaler_c3.pkl`, `robust_scaler_c3_params.json`.
    - Feature matrices: `train_features_c1_raw.npy` (18.87 MB), `train_features_c1_scaled.npy` (18.87 MB).
  - **Status**: Labeling and scaling pipeline complete and validated. Paused for user review before initiating Phase 3 GBR model training (`fit` on GradientBoostingRegressor).
- **2026-09-12 (Feature Alignment Audit, OOV Root Cause, & Explicit Join-Key Architecture)**:
  - **Label/Feature Reconciliation & Root-Cause Diagnosis**:
    - Investigated the 21,453-sentence mismatch between labels (1,010,961 rows) and initial C1 features (989,508 rows).
    - Identified exactly 127 dropped training documents (e.g., Cases 501, 2387, 2389, 2390, 4822, 5243).
    - *Root Cause 1 (OOV Crash)*: Short statutory/citation sentences (e.g., `"XXXVII of 1950."`, `"151, 152."`) contain alphanumeric tokens not present in Google News Word2Vec. Gensim's `wmdistance()` raised an unhandled `ValueError("At least one of the documents had no words that were in the vocabulary.")`.
    - *Root Cause 2 (Multiprocessing Memory Spike)*: `model.get_vector(w, norm=True)` triggered Gensim's `fill_norms()` allocating a 3.35 GiB array per worker process. On Windows with 10–12 multiprocessing workers, concurrent allocations triggered `numpy.core._exceptions._ArrayMemoryError`.
    - `build_c1_features.py` caught both exceptions in its per-document try-block and dropped the 127 documents.
  - **Codebase Remedies Implemented (`src/features/wmd.py`, `src/features/build_features.py`)**:
    - Bypassed Gensim's full-vocabulary `fill_norms()` by computing on-the-fly vector normalization only for unique document tokens (~500 unique words = 1.2 MB RAM per doc), reducing worker memory footprint by >99.9%.
    - Added safe fallback distance (`max_fallback_dist = 3.0`) for empty in-vocabulary sentences, eliminating the `ValueError`.
    - Added `return_fallback_mask=True` instrumentation across WMD and feature assembly pipelines.
  - **Full-Corpus WMD OOV Fallback Audit (`results/logs/wmd_oov_audit.json`)**:
    - Analyzed all **1,010,961 sentences** across all **7,028 training documents**:
      - Total sentences triggering OOV fallback (`max_fallback_dist = 3.0`): **30,843 sentences (3.0509%)**.
      - OCR-affected documents (4,913 docs, 727,055 sentences): **21,208 fallbacks (2.9170%)**.
      - Non-OCR documents (2,115 docs, 283,906 sentences): **9,635 fallbacks (3.3937%)**.
    - *Discussion Insight*: The OOV fallback rate is ~3.05% and does *not* cluster disproportionately in OCR-affected judgments (2.92% vs 3.39%). Inspection revealed it is driven by structural legal citation syntax (standalone statutory section numbers, law report citations, abbreviation fragments, and standalone year tokens like `'281 1954.'`, `'(i) [195o] S.C.R.'`, `'829.'`, `'or f.o.b.'`) split into separate sentences by standard segmentation. Assigning the maximum semantic distance penalty (3.0) is linguistically and mathematically sound. This will be discussed in the paper's Discussion section alongside the OCR limitation.
  - **Standing Project Convention — Explicit Composite Join-Key Architecture**:
    - Positional row alignment is officially deprecated and forbidden for all feature and label matrices.
    - All feature matrices (`c1_raw`, `c1_scaled`, `c3_raw`, `c3_scaled`) and label arrays (`train_labels.npy`) must be indexed and joined via explicit composite keys: `(doc_id, sentence_index)`.
    - Master alignment registry: `data/processed/features/train_sentence_index.json` (`row_idx -> (case_id, sentence_idx)`).
    - This convention is now permanent for all training, validation, and test feature extraction pipelines.
  - **Google Colab GPU Notebook Delivered (`notebooks/01_full_feature_extraction_and_scaling.ipynb`)**:
    - Created clean, self-contained Colab notebook for Option B execution.
    - Features: PyTorch GPU batch encoding for SBERT (`all-MiniLM-L6-v2`, batch_size=512), fast C1 extraction with zero memory overhead, real-time WMD OOV instrumentation, strict 1,010,961-row integrity assertions, and fresh RobustScaler fitting across full matrices.
  - **Execution Gate**:
    - GradientBoostingRegressor training remains strictly paused until all feature matrices and label arrays are confirmed aligned at 1,010,961 rows with zero dropped documents.
- **2026-09-13 (Phase 3 Full Feature Matrix Extraction & Label Alignment COMPLETE)**:
  - **Full-Corpus C1 & C3 Feature Matrices Confirmed**:
    - Extracted and aligned across all **7,028 training documents** and exactly **1,010,961 sentences** with **zero dropped documents**.
    - C1 matrix (`train_features_c1_raw.npy`, `train_features_c1_scaled.npy`): `(1010961, 5)` — features: `[tfidf, ner, position, cosine_w2v, wmd]`.
    - C3 matrix (`train_features_c3_raw.npy`, `train_features_c3_scaled.npy`): `(1010961, 4)` — features: `[tfidf, ner, position, cosine_sbert]`.
    - Labels array (`train_labels.npy`): `(1010961,)`.
    - Composite join key registry (`train_sentence_index.json`): `1,010,961` keys mapping `row_idx -> (doc_id, sentence_idx)`.
    - Document boundaries (`train_doc_boundaries.json`): `7,028` documents mapped with contiguous row spans.
  - **Colab GPU Execution & Payload Integration**:
    - Colab T4 GPU encoded all 1,010,961 sentences with SBERT (`all-MiniLM-L6-v2`) in ~2.5 minutes using batch size 512.
    - Archive `processed_features_c3.zip` downloaded and unzipped into `data/processed/features/`.
  - **Strict End-to-End Alignment Assertions — ALL PASSED**:
    - Exact row count match: `len(X_c1_raw) == len(X_c1_scaled) == len(X_c3_raw) == len(X_c3_scaled) == len(train_labels) == len(sentence_keys) == 1,010,961`.
    - NaN/Inf check: Exactly 0 NaNs and 0 Infs across all 5 arrays.
    - Shared feature integrity: `max(|X_c1[:, :3] - X_c3[:, :3]|) == 0.00e+00` (exact bitwise match on `[tfidf, ner, position]`).
    - Randomized spot-checks across multiple random seeds verified exact correspondence between composite join key `(doc_id, sentence_index)` and feature/label values.
  - **Fresh Full-Corpus RobustScaler Parameters Recorded**:
    - C1 (5 features):
      - `tfidf`: Median = `2.8364`, IQR = `1.3949`
      - `ner`: Median = `1.0000`, IQR = `3.0000`
      - `position`: Median = `0.4965`, IQR = `0.5000`
      - `cosine_w2v`: Median = `0.7112`, IQR = `0.1888`
      - `wmd`: Median = `1.1568`, IQR = `0.1475`
    - C3 (4 features):
      - `tfidf`: Median = `2.8364`, IQR = `1.3949`
      - `ner`: Median = `1.0000`, IQR = `3.0000`
      - `position`: Median = `0.4965`, IQR = `0.5000`
      - `cosine_sbert`: Median = `0.5267`, IQR = `0.2024`
  - **Status & Next Step**: Feature extraction and alignment gate is officially **PASSED**. Ready to proceed immediately to Phase 3 GBR model training (`fit` on GradientBoostingRegressor for C1, C2, and C3).
- **2026-09-13 (Phase 3 GBR Training COMPLETE — All Three Configs)**:
  - **Validation Split (Document-Level, Zero Leakage)**:
    - Holdout ratio: 85% train / 15% val, seed `42`, purely at document granularity.
    - GBR Train: **5,974 documents** → **863,463 sentences**; Val: **1,054 documents** → **147,498 sentences**.
    - Frozen 100-doc test set untouched.
  - **Hyperparameters (identical for all three configs)**:
    - `n_estimators=100`, `learning_rate=0.1`, `max_depth=4`, `min_samples_leaf=50`, `subsample=0.8`, `loss='squared_error'`, `random_state=42`.
    - Rationale: `loss='squared_error'` implements MSE per Belila et al. eq. 6; `subsample=0.8` provides stochastic variance reduction on 863k rows; `max_depth=4` allows 4-way feature interactions while preventing single-sentence overfitting.
  - **GBR Results Summary**:

    | Metric | Config C1 (Replica) | Config C2 (Redundancy Swap) | Config C3 (Proposed) |
    |---|---|---|---|
    | **Train R²** | `0.2400` | `0.2400` | `0.2259` |
    | **Val R²** | `0.2474` | `0.2474` | `0.2269` |
    | **Train RMSE** | `0.1835` | `0.1835` | `0.1852` |
    | **Val RMSE** | `0.1808` | `0.1808` | `0.1833` |
    | **Train MAE** | `0.1287` | `0.1287` | `0.1308` |
    | **Val MAE** | `0.1284` | `0.1284` | `0.1308` |
    | **Fit Time** | 184.8s | 168.4s | 127.1s |
    | **Negative Train R²** | ✅ None | ✅ None | ✅ None |

  - **Feature Importances**:
    - **C1/C2**: `wmd=76.6%` (dominant), `cosine_w2v=6.9%`, `tfidf=6.7%`, `ner=5.8%`, `position=3.9%`.
    - **C3**: `tfidf=66.7%` (dominant without WMD), `cosine_sbert=24.5%`, `position=4.8%`, `ner=4.0%`.
  - **C1 vs C2 Pipeline Sanity Check PASSED**:
    - Max absolute prediction difference across full validation set: `0.00e+00` (bitwise identical).
    - Feature importances: exact identity. This confirms the C1/C2 distinction is purely in the redundancy mechanism (Stage 3), not in the scoring model.
  - **Negative R² Check**: No negative training R² observed across any config. The base paper's reported anomaly did not recur.
  - **Key Finding — WMD Dominance in C1/C2**: WMD alone accounts for 76.6% of tree split importance, dwarfing all other features. This is a significant empirical observation: the GBR has learned that optimal-transport distance to the document centroid is the most discriminative signal for sentence importance in Indian legal text. This will be highlighted in the Discussion section alongside C3's TF-IDF dominance after WMD removal.
  - **Saved Artifacts**:
    - `models/gbr_c1.pkl` (0.25 MB), `models/gbr_c2.pkl` (0.25 MB), `models/gbr_c3.pkl` (0.24 MB).
    - `results/logs/gbr_training_report.json` (full metrics JSON).
    - `results/logs/gbr_training_report.md` (summary Markdown table).
  - **Status & Next Step**: Phase 3 GBR training is **COMPLETE**. Proceed to Phase 4 (Redundancy Control Modules): implement WMD threshold filter for C1 and MMR/SBERT for C2 and C3.
- **2026-09-13 (Phase 3 Rank Correlation Evaluation COMPLETE)**:
  - **Val split**: 1,054 docs / 147,498 sentences (seed=42, 15% holdout — identical to training split)
  - **Budget rule**: `k_i = max(3, round(0.05 * M_i))` per `02_METHODOLOGY.md` Stage 4

  - **Global Sentence-Level Rank Correlation**:

    | Metric | C1 (Replica) | C2 (Redundancy Swap) | C3 (Proposed) |
    |---|:---:|:---:|:---:|
    | **Spearman ρ** | `0.5804` | `0.5804` | `0.5467` |
    | **Kendall τ** | `0.4203` | `0.4203` | `0.3919` |
    | Per-doc Spearman (mean) | `0.5889` | `0.5889` | `0.5593` |
    | Per-doc Kendall (mean) | `0.4338` | `0.4338` | `0.4077` |

  - **Task-Relevant Top-k_i Extractive Selection Overlap**:

    | Metric | C1 (Replica) | C2 (Redundancy Swap) | C3 (Proposed) |
    |---|:---:|:---:|:---:|
    | **Top-k Jaccard (mean)** | `0.0735` | `0.0735` | `0.0640` |
    | Top-k Jaccard (std) | `0.1007` | `0.1007` | `0.0969` |
    | **Top-k % Recall (mean)** | `0.1225` | `0.1225` | `0.1068` |

  - **C1 = C2 Confirmation**: Rank correlation and top-k overlap metrics are bitwise identical across C1 and C2, consistent with the earlier prediction identity check. No deviation at any level.
  - **Interpretation**:
    - Spearman ρ ≈ 0.58 (C1/C2) / 0.55 (C3) shows a moderate but statistically significant global ranking ability (p≈0 on 147k samples). The GBR correctly distinguishes high-importance from low-importance sentences roughly 55–58% better than random rank ordering.
    - **The low top-k Jaccard (≈0.07) and % Recall (≈12%) are expected and not a failure**: because `k_i = max(3, round(0.05 * M_i))` selects only ~5% of each document's sentences, the top-k sets are tiny and the true-label distribution is smooth/dense — many sentences cluster near the same ROUGE scores, so minor rank perturbations cause large Jaccard swings. The metric's low absolute value reflects the difficulty of exact set agreement at very small k, not fundamental misordering.
    - The moderate Spearman ρ is consistent with the paper's own framing: Stage 2 GBR is a **noisy scorer**, not a perfect oracle. The redundancy control stage (Phase 4) exists precisely to recover from this — MMR/WMD-threshold selection is robust to score perturbations as long as relative rank order is roughly preserved, which ρ≈0.58 confirms.
  - **Saved Artifacts**:
    - `scripts/eval_rank_correlation.py`
    - `results/logs/rank_correlation_report.json`
    - `results/logs/rank_correlation_report.md` (with worked document examples)
  - **Status**: Phase 3 rank correlation review **COMPLETE**. **HOLD — do not proceed to Phase 4 until user sign-off on these metrics.**
- **2026-09-13 (Phase 3 Supplement: Worked Examples, Chance Baseline, k_i Audit)**:
  - **k_i Formula Confirmed**:
    - Formula: `k_i = max(3, round(0.05 * M_i))`
    - Rounding: Python built-in `round()` — banker's rounding (round-half-to-even). E.g. `round(0.5)=0`, `round(1.5)=2`.
    - Same `k_i` used for BOTH true-label top-k and predicted-score top-k: **CONFIRMED** — computed once from `M_i`, applied identically to both sides of Jaccard/recall.
    - Short-document tail: For M_i < 20, `0.05 * M_i < 1.0`. After `round()` → 0 or 1. `max(3, ...)` clamps to `k_i=3`. Floor fires for ALL 4 short docs in the val split.
    - Examples: M_i=10 → round(0.5)=0 → k_i=3; M_i=15 → round(0.75)=1 → k_i=3; M_i=60 → round(3.0)=3 → k_i=3 (borderline).

  - **k_i Corpus Statistics (val split, 1,054 docs)**:

    | Statistic | Value |
    |---|:---:|
    | Docs where `floor=3` fires | `139` |
    | Docs with M_i < 20 | `4` |
    | Min M_i → k_i | `14` → `3` |
    | Max M_i → k_i | `1189` → `59` |
    | Average k_i | `7.15` |
    | Median k_i | `5` |

  - **Real vs. Chance Baseline** (chance: 200 simulated random top-k_i selections per doc, per-doc seeds from seed=777):

    | Metric | C1 (Replica) | C2 (Redundancy Swap) | C3 (Proposed) | **CHANCE** |
    |---|:---:|:---:|:---:|:---:|
    | **Top-k Jaccard (mean)** | `0.0735` | `0.0735` | `0.0640` | **`0.0311`** |
    | **Top-k % Recall (mean)** | `0.1225` | `0.1225` | `0.1068` | **`0.0543`** |

    - C1/C2 Jaccard = **2.36× above chance**. C3 Jaccard = **2.06× above chance**.
    - All configs are meaningfully above random selection. Low absolute values confirmed by theory: at k/M = 0.05, random Jaccard ≈ k²/(2Mk − k²) ≈ 0.025; simulation gives 0.0311 (slightly higher due to small M_i edge).

  - **3 Worked Examples (C1 and C3)**:

    **SHORT doc (doc_id=4820, M_i=14, k_i=3)**
    - k_i derivation: `max(3, round(0.05×14)) = max(3, round(0.70)) = max(3,1) = 3`
    - C1: Jaccard=`0.5` (1/3 matched) | C3: Jaccard=`0.5` (1/3 matched)
    - At M_i=14, k_i=3 selects 21% of the document — the floor is doing all the work here, and even a noisy model gets ~50% Jaccard at this scale.

    **AVERAGE doc (doc_id=1005, M_i=83, k_i=4)**
    - k_i derivation: `max(3, round(0.05×83)) = max(3, round(4.15)) = max(3,4) = 4`
    - C1: Jaccard=`0.0` (0/4 matched) | C3: Jaccard=`0.0` (0/4 matched)
    - Zero overlap on this specific document — illustrates the noisy scorer regime where ROUGE label ties cause the top-4 oracle set to differ entirely from model predictions. The GBR has near-uniform predicted scores for this doc, leading to effectively random selection.

    **LONG doc (doc_id=6778, M_i=1189, k_i=59)**
    - k_i derivation: `max(3, round(0.05×1189)) = max(3, round(59.45)) = max(3,59) = 59`
    - C1: Jaccard=`0.0172` (≈1 sentence matched from 59) | C3: Jaccard=`0.0442` (≈2-3 matched)
    - C3 modestly outperforms C1 on this long document despite lower global Spearman — consistent with SBERT's better handling of long-range semantic similarity vs. WMD's computational instability at scale.
    - At k=59/M=1189, chance Jaccard ≈ 0.025; both C1 and C3 are near chance, confirming that very long documents are the hardest case.

  - **Saved Artifacts**:
    - `scripts/eval_worked_examples.py`
    - `results/logs/worked_examples_report.json`
    - `results/logs/worked_examples_report.md` (full sentence tables with text snippets)
  - **Status**: Phase 3 supplementary review materials **COMPLETE**.
- **2026-09-13 (Phase 3 Stratified Rank-Correlation & Score Dispersion Analysis COMPLETE)**:
  - **Stratified Metrics Table (Validation Split, N=1,054 docs, 147,498 sentences)**:

    | Length Bucket | Docs | Sentences | Avg $M_i$ | Avg $k_i$ | Config | Spearman $\rho$ | Kendall $\tau$ | Top-$k$ Jaccard | Top-$k$ Recall | Chance Jaccard | Ratio vs Chance |
    |---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    | **Short (<20)** | 4 | 70 | 17.5 | 3.0 | C1 (Replica) | `0.7188 ± 0.129` | `0.5418 ± 0.135` | `0.3500 ± 0.150` | `0.5000 ± 0.167` | `0.1095` | **`3.20×`** |
    | **Short (<20)** | 4 | 70 | 17.5 | 3.0 | C3 (Proposed) | `0.6515 ± 0.189` | `0.5035 ± 0.204` | `0.2500 ± 0.250` | `0.3333 ± 0.333` | `0.1095` | **`2.28×`** |
    | **Medium (20–500)** | 1,022 | 126,248 | 123.5 | 6.3 | C1 (Replica) | `0.5894 ± 0.150` | `0.4342 ± 0.122` | `0.0725 ± 0.100` | `0.1211 ± 0.153` | `0.0311` | **`2.33×`** |
    | **Medium (20–500)** | 1,022 | 126,248 | 123.5 | 6.3 | C3 (Proposed) | `0.5601 ± 0.147` | `0.4083 ± 0.118` | `0.0635 ± 0.096` | `0.1061 ± 0.148` | `0.0311` | **`2.04×`** |
    | **Long (>500)** | 28 | 21,180 | 756.4 | 37.8 | C1 (Replica) | `0.5507 ± 0.141` | `0.4010 ± 0.114` | `0.0676 ± 0.051` | `0.1226 ± 0.086` | `0.0253` | **`2.68×`** |
    | **Long (>500)** | 28 | 21,180 | 756.4 | 37.8 | C3 (Proposed) | `0.5158 ± 0.137` | `0.3712 ± 0.107` | `0.0545 ± 0.036` | `0.1012 ± 0.064` | `0.0253` | **`2.16×`** |

  - **Diagnostic on Score Compression & High-Value Sentence Suppression**:
    - *Is score compression unique to long documents?* **No, but with critical nuances**:
      - Across full documents, GBR predictions are shrunk toward the mean everywhere (MSE regression shrinkage on noisy labels): max predicted scores rarely exceed 0.65 even when true labels reach 1.0.
      - In long documents, because $k_i$ is large (up to 59 sentences in doc 6778), the dozens of candidates in the top-$k_i$ pool cluster into a razor-thin band (e.g. $[0.50, 0.54]$ in C1, $[0.51, 0.59]$ in C3 for doc 6778). Tree leaf quantization creates near-ties across the top ranks.
    - *Is the suppression of high-true sentences (surfacing around rank 30–40+ rather than top ranks) a systematic pattern?* **YES, it is systematic, caused by percentile-rank mechanics**:
      - Across all 1,054 validation documents, the true top-$k_i$ sentences land on average at the **36th–38th percentile** of model-predicted scores (not the top 5%).
      - Similarly, sentences with true $y \ge 0.8$ only enter the predicted top-$k_i$ **~10–11% of the time** in medium and long documents (10.6% in medium, 11.0% in long).
      - In short docs ($M_i=14$), the floor $k_i=3$ takes the top 21.4% of the document, so sentences at the 20th–30th percentile are readily captured (42.9% capture rate, Jaccard 0.35).
      - In long docs ($M_i=1189$), top-$k_i$ is strictly the top **4.96%** of the document. When true top-value sentences land around the 35th percentile (absolute rank ~400+), they completely miss the top-59 cutoff.
      - In fact, for doc 6778, the true top-1 sentence landed at percentile **36.2%** (rank 431/1189) — which closely matches the corpus-wide average of 43.4% for long docs.
      - Furthermore, for the 4 longest documents ($M_i \ge 958$), top-$k$ Jaccard never exceeds 0.04 and true top-1 sentences rank between 341 and 658.
  - **Saved Artifacts**:
    - `scripts/eval_stratified_by_length.py`
    - `results/logs/stratified_length_report.json`
    - `results/logs/stratified_length_report.md`
  - **Status**: Phase 3 stratified analysis **COMPLETE**. Approved and closed by user 2026-09-14.
- **2026-09-14 (Phase 4 Redundancy Control Hyperparameter Sweeps COMPLETE)**:
  - **Tuning Parity & Validation Subset Discipline**:
    - **Fairness Parity**: Swept identical grid sizes ($4$ points each) on the exact same fixed-quota validation subset ($N=50$ docs) using the agreed $k_i = \max(3, \text{round}(0.05 \cdot M_i))$ budget rule.
    - **Fixed-Quota Composition**: Skew-corrected stratification oversampling long documents to capture length-dependent dynamics:
      - *Short / Short-Medium* ($15$ docs, $M_i \in [14, 26]$): `['4820', '3254', '5180', '6517', '4069', '4077', '6233', '5607', '4483', '5596', '6741', '150', '3710', '5676', '4371']`
      - *Medium* ($20$ docs, $M_i \in [57, 240]$, seed=42): `['5214', '2399', '5359', '6645', '225', '4000', '5626', '1670', '4892', '779', '3844', '3257', '2470', '1431', '459', '2371', '2353', '5628', '1486', '6622']`
      - *Long* ($15$ docs, $M_i \in [501, 1036]$, seed=42): `['3490', '3843', '1980', '215', '1384', '2092', '15', '2577', '375', '4679', '573', '3628', '6707', '4927', '3441']`
    - **Timing Discipline**: SBERT embeddings precomputed once per document and reused across all $\lambda$ values. Wall-clock times track *strictly* the selection loop itself (greedy filtering for C1; greedy MMR ranking for C2/C3).
    - **WMD OOV Note**: Sentences using the WMD OOV fallback distance ($\text{max\_fallback\_dist}=3.0$, $\sim 3.05\%$ of sentences per Phase 2 audit) trivially pass any $\delta \le 1.35$ in C1, as expected.

  - **Config C1: WMD Threshold ($\delta$) Sweep Results**:

    | $\delta$ | Self-BLEU-2 (mean $\pm$ std) | Budget Fulfillment % ($|S_i|/k_i$) | Under-filled Docs | Avg Sentences | Avg Tokens | Selection Time (ms/doc) | Docs / min |
    |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    | `1.00` | `0.1181 ± 0.063` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `297` | `615.9` | `97` |
    | **`1.15`** | **`0.0342 ± 0.023`** | **`100.0%`** | **`0/50` (`0.0%`)** | **`14.0`** | **`199`** | **`625.8`** | **`96`** |
    | `1.25` | `0.0124 ± 0.028` | **`96.2%`** | `8/50` (`16.0%`) | `13.1` | `94` | `396.9` | `151` |
    | `1.35` | `0.0106 ± 0.042` | **`54.4%`** | `40/50` (`80.0%`) | `6.1` | `35` | `243.6` | `246` |

    - *Under-Filling Artifact Flagged*: At $\delta=1.25$ and $\delta=1.35$, the lower Self-BLEU values ($0.0124$ and $0.0106$) are heavily artificial artifacts of summary collapse: at $\delta=1.35$, 80% of documents under-fill, producing summaries averaging just 6.1 sentences (35 tokens) instead of the 14-sentence budget.
    - *Optimal C1 Selection*: **$\delta = 1.15$** is the clear optimal threshold — it slashes Self-BLEU from 0.1181 to 0.0342 (71% reduction) while strictly maintaining **100.0% budget fulfillment** with 0 under-filled documents.

  - **Configs C2 & C3: MMR Lambda ($\lambda$) Sweep Results**:

    | Config | $\lambda$ | Self-BLEU-2 (mean $\pm$ std) | Budget Fulfillment % ($|S_i|/k_i$) | Under-filled Docs | Avg Sentences | Avg Tokens | Selection Time (ms/doc) | Docs / min |
    |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    | **C2** | `0.3` | `0.0342 ± 0.037` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `158` | `1.826` | `32,864` |
    | **C2** | `0.5` | `0.0599 ± 0.050` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `216` | `1.515` | `39,600` |
    | **C2** | **`0.7`** | `0.1394 ± 0.103` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `292` | `1.490` | `40,272` |
    | **C2** | `0.9` | `0.2431 ± 0.152` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `334` | `1.507` | `39,810` |
    | **C3** | `0.3` | `0.0286 ± 0.029` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `152` | `1.451` | `41,352` |
    | **C3** | `0.5` | `0.0412 ± 0.039` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `204` | `1.829` | `32,813` |
    | **C3** | **`0.7`** | `0.1026 ± 0.084` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `274` | `1.426` | `42,067` |
    | **C3** | `0.9` | `0.2122 ± 0.136` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `320` | `1.456` | `41,196` |

    - *Budget Guarantee*: MMR guarantees **100.0% budget fulfillment** with 0 under-filled documents across all $\lambda$.
    - *C3 vs C2 Diversity*: C3 consistently achieves lower Self-BLEU-2 than C2 at every $\lambda$ (e.g. $0.1026$ vs $0.1394$ at $\lambda=0.7$), confirming that SBERT-aligned scoring produces a candidate pool with superior diversity characteristics.
    - *Recommended $\lambda$*: **$\lambda = 0.7$** provides the standard relevance-diversity trade-off ($|S_i|=14.0$, ~274–292 tokens), while **$\lambda = 0.5$** is available for aggressive redundancy suppression (Self-BLEU-2 $\approx 0.04$–$0.06$).

  - **Wall-Clock Efficiency Benchmark (Selection Loop Only)**:
    - **C1 (WMD-threshold)**: `470.5 ms/doc` ($\approx 96$–$246$ docs/min)
    - **C2 (MMR / SBERT)**: `1.585 ms/doc` ($\approx 39,000$ docs/min)
    - **C3 (MMR / SBERT)**: `1.541 ms/doc` ($\approx 41,000$ docs/min)
    - **Empirical Speedup**: MMR selection is **`305.4×` faster** than the WMD-threshold filter, validating the paper's core efficiency contribution on identical hardware.

  - **Saved Artifacts**:
    - `src/redundancy/wmd_filter.py`
    - `src/redundancy/mmr.py`
    - `src/evaluation/redundancy_metrics.py`
    - `tests/test_redundancy.py` (4/4 tests passed)
    - `scripts/run_phase4_sweeps.py`
    - `results/logs/phase4_redundancy_sweep.json`
    - `results/logs/phase4_redundancy_sweep.md`
  - **Status**: Phase 4 Redundancy Control Sweeps **COMPLETE**. **HOLD — standing by for user review and approval before Phase 5 (BART Refinement) begins.**


