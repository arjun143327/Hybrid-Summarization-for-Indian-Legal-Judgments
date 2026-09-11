# Suggested Repository Structure

This is a suggested layout, not a mandate — Antigravity should adapt as needed,
but keep the config-parallel structure (C1/C2/C3 sharing code paths wherever
the spec says they're identical) so the "fair comparison" property is enforced
by the code structure itself, not just by convention.

```
legal-summarization/
├── README.md
├── requirements.txt
├── project_spec/                  # this handed-off spec bundle (read-only reference)
│   ├── 00_PROJECT_BRIEF.md
│   ├── 01_DATA_SPEC.md
│   ├── 02_METHODOLOGY.md
│   ├── 03_EXPERIMENT_PLAN.md
│   └── 04_TASKS.md
│
├── data/
│   ├── raw/in_abs/{judgments,headnotes}/
│   ├── splits/{train_ids.txt, test_ids.txt}
│   └── processed/{sentences/, features/}
│
├── src/
│   ├── data/
│   │   ├── loader.py               # IN-Abs loading, split management
│   │   └── preprocessing.py        # segmentation w/ citation protection, tokenization
│   ├── features/
│   │   ├── tfidf.py
│   │   ├── position.py
│   │   ├── ner.py
│   │   ├── embeddings_w2v.py       # for C1/C2 cosine + WMD
│   │   ├── embeddings_sbert.py     # for C3 cosine + all MMR
│   │   └── build_features.py       # assembles x_ij per config
│   ├── labeling/
│   │   └── gbr_labels.py           # max-ROUGE-to-reference-sentence labeling
│   ├── scoring/
│   │   └── gbr_model.py            # GBR train/predict, per-config
│   ├── redundancy/
│   │   ├── wmd_filter.py           # C1 baseline
│   │   └── mmr_sbert.py            # C2/C3 proposed
│   ├── selection/
│   │   └── extractive_select.py    # budget k_i, argmax / MMR-driven selection
│   ├── refinement/
│   │   └── bart_refine.py          # facebook/bart-large-cnn wrapper, shared by all configs
│   ├── evaluation/
│   │   ├── rouge_eval.py
│   │   ├── redundancy_metric.py    # self-BLEU / n-gram overlap
│   │   └── timing_benchmark.py     # isolated redundancy-step timing, hardware logging
│   └── pipeline.py                 # orchestrates C1/C2/C3 end-to-end runs
│
├── configs/
│   ├── c1_base_replica.yaml
│   ├── c2_redundancy_swap.yaml
│   └── c3_proposed.yaml            # each config file: feature list, redundancy
│                                       method, hyperparams (δ or λ), paths
│
├── notebooks/
│   ├── colab_train_gbr.ipynb       # heavy compute: full 7,030-doc GBR training
│   ├── colab_bart_inference.ipynb  # BART refinement at scale
│   └── colab_timing_benchmark.ipynb
│
├── results/
│   ├── logs/                        # raw run logs, hardware specs, hyperparameter sweeps
│   ├── tables/                       # generated ROUGE/timing/redundancy tables (csv/md)
│   └── figures/                        # any plots (λ sweep, δ sweep, etc.)
│
└── tests/
    └── (unit tests for segmentation, MMR selection, labeling — at minimum,
         sanity-check MMR greedy loop produces exactly k sentences with no
         duplicates, and that C1/C2 feature vectors are byte-identical given
         the same input document)
```

## Key structural invariants Antigravity should enforce in code (not just docs)
1. `configs/c1_*.yaml` and `configs/c2_*.yaml` must reference the **same**
   `build_features.py` code path and produce identical feature matrices — only
   `redundancy/` module selection should differ between C1 and C2.
2. All three configs' pipeline runs must consume the **same** frozen
   `data/splits/test_ids.txt` — enforce this in `pipeline.py` (e.g. assert the
   loaded test ID list hash matches a recorded value) rather than trusting it
   by convention.
3. `timing_benchmark.py` should isolate and time only the redundancy-control
   call itself (not feature extraction, not BART) and log hardware info
   (CPU/GPU/RAM, environment) alongside every timing result automatically —
   don't rely on manually remembering to note this per run.
4. Every experiment run should write a results log entry (JSON or CSV row)
   containing: config name, git commit hash, hyperparameters used, ROUGE
   scores, redundancy metric, timing, and hardware info — so nothing reported
   in the paper traces back to an unlogged, unreproducible run.
