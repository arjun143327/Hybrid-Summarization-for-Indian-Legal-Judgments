# Project Brief — Legal Summarization: MMR/SBERT Redundancy Control

## Context
This is an undergraduate research paper extending:

> Belila et al., "Semantic-Aware Hybrid Text Summarization Using Supervised Sentence
> Scoring and Redundancy Control," Informatica 50 (2026), 189–196.

Target venue: **Informatica journal** (same as base paper). Author is a 3rd-year CSE
undergraduate. This is a research codebase, not a production system — prioritize
correctness, reproducibility, and clean experiment logging over engineering polish.

## What the base paper does
A 3-stage hybrid summarization pipeline:
1. **Feature extraction**: each sentence gets `x_ij = [TF-IDF, NER count, Position, Cosine, WMD]`
2. **Supervised scoring**: a Gradient Boosting Regressor (GBR) predicts an importance
   score per sentence, trained with MSE against ROUGE-aligned pseudo-labels
3. **Redundancy control**: a sentence is added to the extractive summary only if its
   Word Mover's Distance (WMD) to every already-selected sentence exceeds threshold δ
4. **Abstractive refinement**: selected sentences → `facebook/bart-large-cnn` for
   fluency/coherence, trained/run with standard NLL

Evaluated on 100 CNN/DailyMail-style news articles, ROUGE-1/2/L F1.

Base paper's own stated limitations (this is what we are fixing):
- Semantic redundancy is not fully eliminated by the WMD filter
- WMD + TF-IDF are computationally expensive → poor scalability / no real-time use
- (also noted, not our focus: negative GBR training score, precision-recall imbalance)

## Our two contributions
1. **New domain**: Indian Supreme Court legal judgments (IN-Abs dataset), never tested
   in the base paper. Legal text has higher factual-accuracy stakes than news.
2. **Redundancy control swap**: replace the WMD-threshold redundancy rule with
   **Maximal Marginal Relevance (MMR) computed over Sentence-BERT embeddings**
   (cosine similarity), claimed to be both faster (no optimal-transport computation)
   and more effective at reducing redundancy.

Everything else in the pipeline (TF-IDF + position + NER + GBR scoring, BART
refinement) is kept identical to the base paper so that any measured differences
are attributable specifically to the redundancy-control swap.

## Framing / what "success" looks like
This is an **efficiency + quality tradeoff paper**, not a quality-only paper.
Every results table/section must report **both** ROUGE quality AND wall-clock time
together — never one without the other. Target claim: "comparable or improved
summarization quality with significantly reduced computation time."

## Three experimental configurations (see 03_EXPERIMENT_PLAN.md for full detail)
To keep the "redundancy swap" comparison fair and cleanly attributable, we run
THREE pipeline configurations, not two:

| Config | Feature vector | Redundancy mechanism | Purpose |
|---|---|---|---|
| **C1 — Base replica** | `[TF-IDF, NER, Position, Cosine, WMD]` | WMD threshold (δ) | Reimplemented control group, matches base paper exactly, run on IN-Abs |
| **C2 — Redundancy-only swap** | `[TF-IDF, NER, Position, Cosine, WMD]` | MMR over SBERT | Isolates the effect of just swapping redundancy control |
| **C3 — Proposed system** | `[TF-IDF, NER, Position, Cosine_SBERT]` (WMD dropped) | MMR over SBERT | Full proposed system — WMD removed everywhere, unified on SBERT |

C3 is the paper's headline system. C1 is the fair baseline. C2 is the ablation that
proves the reported gains come from the redundancy mechanism specifically, not from
unrelated feature changes.

## Explicit non-goals (DO NOT DO, unless the user asks)
- No Hindi / MILDSum multilingual extension (dataset is listed as a possible *future*
  supplement only — do not build it now)
- No new scoring model beyond GBR — do not swap in a different regressor
- No changes to the BART refinement stage
- No fabricated results, numbers, or citations anywhere — all numbers in later
  writeup must come from actual logged experiment runs. If a number is a projection
  or placeholder, it must be clearly marked `[PLACEHOLDER]` in any draft text.

## Division of labor (for Antigravity's awareness)
- **Claude (separate chat/project)**: literature/method design, result interpretation,
  academic writing. Antigravity should treat any spec doc handed to it (this folder)
  as already-validated design — implement faithfully, don't redesign silently.
- **Antigravity (this environment)**: writing/debugging all Python code, git/version
  control, task tracking, authoring notebooks for Colab.
- **Google Colab**: heavy compute — GBR training on full 7,030-doc train set, BART
  inference/fine-tuning, timing benchmarks — if local hardware is insufficient.

## Files in this spec bundle
- `00_PROJECT_BRIEF.md` — this file
- `01_DATA_SPEC.md` — IN-Abs dataset structure, loading, splits, preprocessing
- `02_METHODOLOGY.md` — exact formulas/pseudocode for every pipeline stage
- `03_EXPERIMENT_PLAN.md` — configs, ablations, timing benchmark protocol, metrics
- `04_TASKS.md` — task/progress tracker against the ~10–12 week plan
- `05_REPO_STRUCTURE.md` — suggested code/repo layout and file responsibilities
