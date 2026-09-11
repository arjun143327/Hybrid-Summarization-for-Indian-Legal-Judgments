# Methodology Specification — Pipeline Stages

This is the algorithmic spec Antigravity should implement against. Formulas are
numbered to match the base paper's equations where directly reused, and prefixed
`NEW-` where this project introduces a change.

---

## Stage 1 — Sentence representation (all configs)

For document `d_i = {s_i1, ..., s_iM}`:

- **TF-IDF**: standard sklearn `TfidfVectorizer`, fit on the train set vocabulary,
  applied per-document. Use the lemmatized/stopword-removed token stream.
- **Position**: normalized position `pos_ij = j / M_i` (0 = first sentence, 1 = last).
- **NER count**: named entity count per sentence via spaCy `en_core_web_sm` (or NLTK
  if spaCy unavailable in the environment) — raw count, optionally normalized by
  sentence length.
- **Cosine** (base paper eq. 3):
  `Sim_cos(s_ij) = (s_ij · d_i) / (||s_ij|| ||d_i||)`
  where `d_i` = mean of sentence embeddings in the document. Base paper uses
  pretrained Word2Vec for this; **configs C1/C2 in this project also use Word2Vec
  to stay faithful to the base paper**, while **config C3 uses SBERT embeddings**
  for this cosine feature (see NEW-3 below).
- **WMD**: gensim `WmdSimilarity` / `wmdistance`, pretrained Word2Vec embeddings.
  Present in feature vectors for **C1 and C2 only**. Absent in **C3**.

### Feature vector per config
```
C1 (base replica): x_ij = [TF-IDF, NER, Position, Cosine_w2v, WMD]
C2 (redundancy-only swap): x_ij = [TF-IDF, NER, Position, Cosine_w2v, WMD]   # same as C1
C3 (proposed): x_ij = [TF-IDF, NER, Position, Cosine_sbert]                  # WMD dropped
```
C1 and C2 use an *identical* feature vector — they differ only in the redundancy
stage (Stage 3). This is intentional: it is what makes C1→C2 a clean, single-variable
ablation.

---

## Stage 2 — Supervised sentence importance scoring (all configs)

### Ground-truth label y_ij (NEW — base paper doesn't fully specify this)
Base paper says only "ROUGE-based alignment," no algorithm given. We define it
explicitly for reproducibility:

```
y_ij = max over reference sentences r_k in headnote_i of ROUGE-1_F1(s_ij, r_k)
```

i.e., each source sentence is scored by its best single-sentence match in the
reference headnote, not by comparison to the whole reference blob. Rationale:
IN-Abs headnotes (~932 words) are too long to score a single sentence against as
one unit without diluting signal; full greedy-oracle search is too expensive at
IN-Abs document lengths (hundreds of sentences/doc × 7,030 train docs). This
max-match approach is a reasonable middle ground — cheap (single pass, no
combinatorial search) and discriminative.

Optionally log a small side-validation: for ~50 sampled training docs, also compute
full greedy-oracle labels and report the label correlation (e.g. Spearman) with the
max-match labels, to justify the approximation in the paper if reviewers ask.

### Model
`scikit-learn GradientBoostingRegressor`, trained separately per config (C1, C2,
C3 each get their own GBR fit, since their input feature vectors differ). Objective:

```
L_GBR = (1/|S|) * sum_{i,j} (y_ij - fθ(x_ij))^2      [base paper eq. 6, MSE]
```

Log training score, validation score (holdout from train set — do NOT touch the
100-doc test set for hyperparameter tuning), and feature importances. **Watch for
a negative training score** — the base paper reported one and flagged it as a
problem; if we see the same, investigate (likely candidates: label noise from
the labeling scheme, feature scaling, insufficient tree depth/estimators) rather
than silently reporting it as a footnote.

---

## Stage 3 — Redundancy control (differs by config — this is the core contribution)

### C1 — WMD threshold (base paper eq. 7, reimplemented baseline)
```
Add s_ij to summary candidate set only if:
    WMD(s_ij, s_ik) ≥ δ   for all previously selected s_ik
```
δ is a tunable threshold — sweep a small grid (see 03_EXPERIMENT_PLAN.md) and
report the value used; base paper does not state its chosen δ, so ours must be
justified via a validation sweep, not copied blindly.

### C2 & C3 — NEW: MMR over SBERT embeddings (the proposed replacement)
Standard Maximal Marginal Relevance (Carbonell & Goldstein, 1998 — cited by the
base paper's own related work [6]), computed greedily:

```
MMR(s_ij) = λ · Relevance(s_ij) − (1 − λ) · max_{s_ik ∈ Selected} CosineSim(s_ij, s_ik)
```
where:
- `Relevance(s_ij)` = the GBR-predicted importance score `ŷ_ij` from Stage 2
  (reuses the supervised scoring signal, rather than re-deriving relevance
  independently — keeps the pipeline internally consistent)
- `CosineSim(s_ij, s_ik)` = cosine similarity between SBERT embeddings of the two
  sentences (`sentence-transformers`, e.g. `all-MiniLM-L6-v2` as a reasonable
  default — confirm final model choice against speed/quality tradeoff in Stage 3
  timing benchmark)
- Selection proceeds **greedily**: at each step, compute MMR score for all
  remaining candidate sentences given the currently-selected set, pick the
  argmax, add it, repeat until budget `k_i` reached (see Stage 4).
- `λ` is a tunable hyperparameter (0 ≤ λ ≤ 1) balancing relevance vs diversity.
  Sweep λ ∈ {0.3, 0.5, 0.7, 0.9} (see 03_EXPERIMENT_PLAN.md ablation table) and
  report the sensitivity, not just one chosen value.

### Illustrative pseudocode (for Antigravity to implement properly, not copy verbatim)
```python
def mmr_select(candidates, relevance_scores, embeddings, k, lam=0.7):
    """
    candidates: list of sentence ids
    relevance_scores: dict sid -> predicted GBR importance (relevance term)
    embeddings: dict sid -> SBERT embedding vector
    k: sentence budget for this document
    lam: MMR lambda
    """
    selected = []
    remaining = set(candidates)

    while remaining and len(selected) < k:
        best_sid, best_score = None, float("-inf")
        for sid in remaining:
            relevance = relevance_scores[sid]
            if selected:
                max_sim = max(
                    cosine_similarity(embeddings[sid], embeddings[sel])
                    for sel in selected
                )
            else:
                max_sim = 0.0  # first pick: no penalty term
            mmr_score = lam * relevance - (1 - lam) * max_sim
            if mmr_score > best_score:
                best_sid, best_score = sid, mmr_score
        selected.append(best_sid)
        remaining.remove(best_sid)

    return selected
```
Note the important behavioral difference from C1's rule: WMD-threshold is a
**hard filter** (binary admit/reject against a fixed δ), while MMR is a
**greedy ranking** that always produces exactly `k` sentences (no risk of
under-filling the summary if δ is set too strict, which is a real failure mode
of the base paper's method worth mentioning in Discussion).

---

## Stage 4 — Extractive selection (all configs)
```
S*_i = argmax_{S_i} sum_{s_ij ∈ S_i} ŷ_ij   s.t. |S_i| ≤ k_i     [base paper eq. 8]
```
For C1, this budget-constrained argmax is applied *after* the WMD filter has
already pruned candidates. For C2/C3, MMR's greedy loop *is* the selection
mechanism (it directly produces the top-`k_i` set) — no separate argmax step
needed, since MMR already jointly optimizes relevance and diversity in one pass.

`k_i` (document-adaptive sentence budget): base paper doesn't specify the exact
rule. Recommend a length-proportional budget, e.g. `k_i = max(3, round(0.05 * M_i))`
or similar, tuned so extractive-summary length roughly matches typical IN-Abs
headnote length in sentences — validate against training-set headnote sentence
counts before locking this in.

---

## Stage 5 — Abstractive refinement (all configs, unchanged from base paper)
Selected sentences concatenated in original document order → `facebook/bart-large-cnn`.
```
L_BART = − sum_{t=1}^{T} log P_φ(y_it | y_i,<t, X_i)      [base paper eq. 10]
```
Kept identical across C1/C2/C3 so BART is not a confound — any ROUGE/redundancy/
speed differences between configs must come from Stages 1–3.

Note on IN-Abs input length: BART's typical max input is ~1024 tokens. Given
judgment sentence counts, the concatenated extractive summary should already be
well under this if `k_i` is reasonable, but validate token counts before running
full BART inference at scale, and log any truncation events per document.

---

## Stage 6 — Joint framing (reporting only, not a literal joint-trained loss here)
Base paper eq. 11 states `L_total = λ1·L_GBR + λ2·L_BART` as a conceptual framing,
but GBR and BART are trained/run as separate stages in practice (GBR is not
backprop-connected to BART). Keep this same framing in the writeup for
consistency with the base paper's terminology, but implement them as the two
independent training/inference stages they actually are — do not attempt true
joint end-to-end optimization; that would be new scope beyond this paper's goals.
