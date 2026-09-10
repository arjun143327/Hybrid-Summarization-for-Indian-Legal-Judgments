# Task Tracker — ~10–12 Week Plan

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked/flag for Claude

Antigravity: please keep this file updated as tasks progress — check items off,
add dated notes under a task if something deviates from the spec, and flag
`[!]` anything that needs a design decision revisited (bring it back to Claude
rather than deciding silently, per the project's stated workflow).

## Phase 1 — Setup & Data (Weeks 1–2)
- [ ] Acquire IN-Abs dataset, verify file counts (7,130 judgment/headnote pairs)
- [ ] Implement data loading per `01_DATA_SPEC.md` directory layout
- [ ] Implement + validate sentence segmentation with citation protection
      (spot-check ~10 judgments by eye, log before/after samples)
- [ ] Generate and freeze `train_ids.txt` (7,030) / `test_ids.txt` (100), log
      random seed used for the split
- [ ] Data quality audit: log counts of empty headnotes, very short/long docs,
      duplicate IDs (per `01_DATA_SPEC.md` "known risks" section)

## Phase 2 — Feature Extraction (Weeks 2–3)
- [ ] TF-IDF vectorizer fit on train set
- [ ] Position feature
- [ ] NER extraction (spaCy/NLTK) + counts
- [ ] Word2Vec-based cosine feature (for C1/C2)
- [ ] SBERT-based cosine feature (for C3)
- [ ] WMD pairwise computation (for C1/C2 feature vector + C1 redundancy step)
- [ ] Cache feature matrices per config to `data/processed/features/`

## Phase 3 — GBR Labeling & Training (Weeks 3–5)
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
- (none yet)
