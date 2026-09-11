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
- [x] Word2Vec-based cosine feature (for C1/C2; implemented `src/features/embeddings_w2v.py` using `glove-wiki-gigaword-100` with mean document embedding & Sim_cos)
- [x] SBERT-based cosine feature (for C3; implemented `src/features/embeddings_sbert.py` using `all-MiniLM-L6-v2`)
- [x] WMD pairwise computation (for C1/C2 feature vector + C1 redundancy step; implemented `src/features/wmd.py` via gensim & POT)
- [ ] Cache feature matrices per config to `data/processed/features/`

## Phase 3 — GBR Labeling & Training (Weeks 3–5)
> **Note on Feature Preprocessing / Scaling**: Feature values will include outliers from merged/under-split long sentences (e.g. Case 914 Sentence 2, NER count 28); apply RobustScaler or StandardScaler to features before GBR training rather than using raw values.

- [ ] Implement max-ROUGE-to-reference-sentence labeling (`02_METHODOLOGY.md`)
- [ ] (Optional) Side-validation: greedy-oracle labels on ~50-doc sample,
      correlation check against max-match labels
- [ ] Train GBR for C1 (feature vector A)
- [ ] Train GBR for C2 (same feature vector as C1 — verify C1/C2 GBR weights
      end up identical if training data/seed identical, since inputs are the same)
- [ ] Train GBR for C3 (feature vector B, SBERT cosine, no WMD)
- [ ] Log training score, validation score, feature importances for each
- [ ] `[!]` If any GBR shows a negative training score (as base paper reported),
      flag and investigate before proceeding — do not silently report it

## Phase 4 — Redundancy Control Modules (Weeks 4–6)
- [ ] Implement WMD-threshold redundancy filter (C1) — reimplemented baseline
- [ ] Implement MMR/SBERT redundancy ranking (C2, C3) — see pseudocode in
      `02_METHODOLOGY.md`
- [ ] δ sweep for C1 on validation subset, select final δ
- [ ] λ sweep {0.3, 0.5, 0.7, 0.9} for C2/C3 on validation subset, select final λ
- [ ] Verify tuning effort parity logged (same grid size, same validation subset)

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
