# Experiment Plan

## Core comparisons

| Run | Feature vector | Redundancy | Domain | Purpose |
|---|---|---|---|---|
| **Base paper (reported)** | as published | WMD | CNN/DailyMail | reference numbers from the paper itself — NOT rerun, just cited |
| **C1 — replica** | `[TF-IDF, NER, Pos, Cosine_w2v, WMD]` | WMD threshold | IN-Abs | our reimplemented control group, new domain |
| **C2 — redundancy-only swap** | `[TF-IDF, NER, Pos, Cosine_w2v, WMD]` | MMR/SBERT | IN-Abs | isolates redundancy-mechanism effect |
| **C3 — proposed** | `[TF-IDF, NER, Pos, Cosine_sbert]` | MMR/SBERT | IN-Abs | full proposed system |

C1 vs C2 = the clean single-variable ablation (redundancy mechanism only).
C2 vs C3 = effect of also unifying the embedding space / dropping WMD as a feature.
C1 vs C3 = the paper's headline comparison (full proposed system vs faithful baseline).

Do NOT report C3 vs base-paper-published-numbers as if it were an apples-to-apples
comparison — different domain (IN-Abs vs CNN/DailyMail), so frame that comparison
qualitatively ("prior work / different domain") not as a strict baseline delta.

## Hyperparameter sweeps (ablations)

### MMR λ sweep
λ ∈ {0.3, 0.5, 0.7, 0.9}, run on C2 and C3, evaluated on a **validation subset**
(carved from train, NOT the 100-doc test set) to pick a final λ, then report that
one chosen λ's results on the frozen test set as the headline number. Report the
full sweep as a supporting ablation figure/table — this is standard practice and
also directly demonstrates the "not fully eliminated" redundancy claim is being
addressed with evidence, not just asserted.

### WMD δ threshold sweep (for C1, the baseline)
Similarly sweep δ over a small grid (e.g. 3–5 values spanning the observed WMD
distance distribution on a validation sample) so C1 is a *fairly tuned* baseline,
not a strawman. This matters for the paper's credibility — reviewers will
discount results if the baseline looks under-tuned relative to the proposed method.

### Tuning effort parity (explicit fairness rule)
Both C1 (δ) and C2/C3 (λ) get the **same number of grid points swept** and the
**same validation subset** for selection. Log this explicitly in the experiment
log so the methodology section can state tuning effort was matched.

## Metrics to compute, every run

1. **ROUGE-1 / ROUGE-2 / ROUGE-L F1** — `rouge-score` or `rouge` package,
   against IN-Abs reference headnotes, on the frozen 100-doc test set.
2. **Redundancy metric** — self-BLEU or n-gram overlap between summary sentences
   within a single document's output summary (i.e., how repetitive the summary is
   internally). Report as an explicit table alongside ROUGE — this is the direct
   evidence for the "MMR reduces redundancy more than WMD" claim.
3. **Wall-clock time per document** — timed *only* for the redundancy-control step
   in isolation (not the whole pipeline, since Stages 1/2/5 are shared/unchanged —
   timing them again adds noise, not signal). Report:
   - seconds/document (mean ± std over the 100-doc test set)
   - documents/minute (derived, for intuitive reporting)
   - **fixed, documented hardware**: record CPU model, RAM, whether GPU was used
     for embedding computation, and whether Colab or local — this must be identical
     across C1/C2/C3 timing runs, run in the same session/environment, not
     compared across different Colab sessions with different allocated hardware.
4. **Precision/Recall breakdown** (secondary, time-permitting) — sentence-level
   precision/recall of the extractive selection against the oracle sentence set
   used for GBR labels, to directly address the base paper's noted
   precision-recall imbalance.

## Reporting discipline
- Every results table/figure that shows ROUGE must be paired with the timing
  table for the same run — never present quality numbers in isolation, and
  never present speed numbers in isolation. This is the paper's core framing
  ("comparable or improved quality with significantly reduced computation time")
  and must be visible in every relevant table, not just summarized once in the abstract.
- Any number not yet produced by an actual run must be marked `[PLACEHOLDER]`
  in draft text — never filled in with a plausible-looking invented number.
- If C3 underperforms C1 on any ROUGE metric, report it honestly — the paper's
  claim is "comparable OR improved quality," which permits a quality/speed
  tradeoff, not a requirement that C3 wins on every metric. Do not cherry-pick.

## What NOT to add to this experiment plan (scope control)
- No Hindi/MILDSum runs
- No additional summarization baselines beyond what's needed for the ablation
  table above (i.e., don't go recruit RNES/SWAP-NET/etc. reimplementations —
  the base paper's Table 1 numbers are cited as prior context, not rerun)
- No new redundancy metric beyond self-BLEU/n-gram overlap unless the first
  results look ambiguous and a second metric is genuinely needed to clarify
