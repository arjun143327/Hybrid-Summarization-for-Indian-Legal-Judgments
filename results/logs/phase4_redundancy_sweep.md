# Phase 4 Redundancy Control Hyperparameter Sweeps Report

Date: 2026-09-14 11:21:54
Validation subset size: $N=50$ documents (fixed quota: 15 short/short-med, 20 medium, 15 long)

> **Fairness Rule**: Identical sweep grid size (4 points each), identical validation subset, and identical budget rule ($k_i = \max(3, \text{round}(0.05 \cdot M_i))$).
> **Timing Methodology**: SBERT embeddings precomputed once per document; wall-clock times track strictly the selection loop (greedy candidate filtering for C1; greedy MMR ranking for C2/C3).
> **WMD OOV Note**: Sentences using the WMD OOV fallback distance ($\text{max\_fallback\_dist}=3.0$, $\sim 3.05\%$ of sentences per Phase 2 audit) trivially pass any $\delta \le 1.35$ in C1, as expected.

## 1. Validation Subset Composition

- **Short / Short-Medium** ($N=15$): `['4820', '3254', '5180', '6517', '4069', '4077', '6233', '5607', '4483', '5596', '6741', '150', '3710', '5676', '4371']`
- **Medium** ($N=20$): `['5214', '2399', '5359', '6645', '225', '4000', '5626', '1670', '4892', '779', '3844', '3257', '2470', '1431', '459', '2371', '2353', '5628', '1486', '6622']`
- **Long** ($N=15$): `['3490', '3843', '1980', '215', '1384', '2092', '15', '2577', '375', '4679', '573', '3628', '6707', '4927', '3441']`

## 2. Config C1: WMD Threshold ($\delta$) Sweep

| $\delta$ | Self-BLEU-2 (mean $\pm$ std) | Budget Fulfillment % ($|S_i|/k_i$) | Under-filled Docs | Avg Sentences | Avg Tokens | Selection Time (ms/doc) | Docs / min |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `1.00` | `0.1181 ± 0.063` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `297` | `615.9` | `97` |
| **`1.15`** | **`0.0342 ± 0.023`** | **`100.0%`** | **`0/50` (`0.0%`)** | **`14.0`** | **`199`** | **`625.8`** | **`96`** |
| `1.25` | `0.0124 ± 0.028` | **`96.2%`** | `8/50` (`16.0%`) | `13.1` | `94` | `396.9` | `151` |
| `1.35` | `0.0106 ± 0.042` | **`54.4%`** | `40/50` (`80.0%`) | `6.1` | `35` | `243.6` | `246` |

*Note on C1 selection timing*: C1's per-document selection time decreases as $\delta$ increases (from 625.8 ms at $\delta=1.15$ down to 243.6 ms at $\delta=1.35$) because a stricter $\delta$ produces a smaller accepted set, meaning each remaining candidate is compared against fewer already-selected sentences, reducing the total number of pairwise WMD computations performed per document.

*Under-Filling Artifact at $\delta \ge 1.25$*: At $\delta=1.25$ and $\delta=1.35$, the lower Self-BLEU values are artificial artifacts of severe summary under-filling (at $\delta=1.35$, 80% of documents under-fill, producing summaries averaging just 6.1 sentences and 35 tokens). **$\delta = 1.15$** is the optimal operating point, achieving 71% redundancy reduction with 100.0% budget fulfillment.

## 3. Configs C2 & C3: MMR Lambda ($\lambda$) Sweep

| Config | $\lambda$ | Self-BLEU-2 (mean $\pm$ std) | Budget Fulfillment % ($|S_i|/k_i$) | Under-filled Docs | Avg Sentences | Avg Tokens | Selection Time (ms/doc) | Docs / min |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **C2 (Redundancy Swap)** | **`0.3`** | **`0.0342 ± 0.037`** | **`100.0%`** | **`0/50` (`0.0%`)** | **`14.0`** | **`158`** | **`1.826`** | **`32,864`** |
| **C2 (Redundancy Swap)** | `0.5` | `0.0599 ± 0.050` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `216` | `1.515` | `39,600` |
| **C2 (Redundancy Swap)** | `0.7` | `0.1394 ± 0.103` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `292` | `1.490` | `40,272` |
| **C2 (Redundancy Swap)** | `0.9` | `0.2431 ± 0.152` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `334` | `1.507` | `39,810` |
| **C3 (Proposed)** | **`0.3`** | **`0.0286 ± 0.029`** | **`100.0%`** | **`0/50` (`0.0%`)** | **`14.0`** | **`152`** | **`1.451`** | **`41,352`** |
| **C3 (Proposed)** | `0.5` | `0.0412 ± 0.039` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `204` | `1.829` | `32,813` |
| **C3 (Proposed)** | `0.7` | `0.1026 ± 0.084` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `274` | `1.426` | `42,067` |
| **C3 (Proposed)** | `0.9` | `0.2122 ± 0.136` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `320` | `1.456` | `41,196` |

## 4. Operating Point Selection & Iso-Redundancy Formulation

- **C1 Selected Point**: $\delta = 1.15$ (Self-BLEU-2 = `0.0342`, 100.0% fulfillment).
- **C2 Selected Primary Point**: $\lambda = 0.3$ (Self-BLEU-2 = `0.0342`, 100.0% fulfillment).
  - *Clean Iso-Redundancy Comparison*: At $\lambda=0.3$, C2 achieves an **exact match** in internal redundancy (Self-BLEU-2 = `0.0342`) to C1 at $\delta=1.15$. Holding redundancy constant allows a fair, unconfounded comparison of selection speed and downstream ROUGE quality.
- **C3 Selected Primary Point**: $\lambda = 0.3$ (Self-BLEU-2 = `0.0286`, 100.0% fulfillment).
- **Secondary Candidate Retained**: $\lambda = 0.5$ is retained as a secondary candidate to carry into Phase 5 and Phase 6 in case $\lambda=0.3$ proves too aggressive once downstream ROUGE quality against reference headnotes is measured.

## 5. Headline Efficiency Comparison (Selected Operating Points)

Speedup is evaluated strictly at the chosen operating points ($\delta=1.15$ vs $\lambda=0.3$) rather than a blended average across the full grid (which was artificially depressed by the degenerate $\delta=1.35$ setting where 80% of docs under-filled):

- **C1 ($\delta=1.15$) Selection Time**: **`625.8 ms/doc`** (96 docs/min)
- **C2 ($\lambda=0.3$) Selection Time**: **`1.826 ms/doc`** (32,864 docs/min)
  - **Speedup vs C1**: **`342.7×` faster**
- **C3 ($\lambda=0.3$) Selection Time**: **`1.451 ms/doc`** (41,352 docs/min)
  - **Speedup vs C1**: **`431.3×` faster**
