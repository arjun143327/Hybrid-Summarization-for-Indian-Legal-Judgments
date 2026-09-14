# Phase 4 Redundancy Control Hyperparameter Sweeps Report

Date: 2026-09-14 11:21:54
Validation subset size: $N=50$ documents (fixed quota: 15 short/short-med, 20 medium, 15 long)

> **Fairness Rule**: Identical sweep grid size (4 points each), identical validation subset, and identical budget rule ($k_i = \max(3, 	ext{round}(0.05 \cdot M_i))$).
> **Timing Methodology**: SBERT embeddings precomputed once per document; wall-clock times track strictly the selection loop (greedy candidate filtering for C1; greedy MMR ranking for C2/C3).
> **WMD OOV Note**: Sentences using the WMD OOV fallback distance ($	ext{max\_fallback\_dist}=3.0$, $\sim 3.05\%$ of sentences per Phase 2 audit) trivially pass any $\delta \le 1.35$ in C1, as expected.

## 1. Validation Subset Composition

- **Short / Short-Medium** ($N=15$): `['4820', '3254', '5180', '6517', '4069', '4077', '6233', '5607', '4483', '5596', '6741', '150', '3710', '5676', '4371']`
- **Medium** ($N=20$): `['5214', '2399', '5359', '6645', '225', '4000', '5626', '1670', '4892', '779', '3844', '3257', '2470', '1431', '459', '2371', '2353', '5628', '1486', '6622']`
- **Long** ($N=15$): `['3490', '3843', '1980', '215', '1384', '2092', '15', '2577', '375', '4679', '573', '3628', '6707', '4927', '3441']`

## 2. Config C1: WMD Threshold ($\delta$) Sweep

| $\delta$ | Self-BLEU-2 (mean $\pm$ std) | Budget Fulfillment % ($|S_i|/k_i$) | Under-filled Docs | Avg Sentences | Avg Tokens | Selection Time (ms/doc) | Docs / min |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `1.0` | `0.1181 ± 0.063` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `297` | `615.9` | `97` |
| `1.15` | `0.0342 ± 0.023` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `199` | `625.8` | `96` |
| `1.25` | `0.0124 ± 0.028` | **`96.2%`** | `8/50` (`16.0%`) | `13.1` | `94` | `396.9` | `151` |
| `1.35` | `0.0106 ± 0.042` | **`54.4%`** | `40/50` (`80.0%`) | `6.1` | `35` | `243.6` | `246` |

## 3. Configs C2 & C3: MMR Lambda ($\lambda$) Sweep

| Config | $\lambda$ | Self-BLEU-2 (mean $\pm$ std) | Budget Fulfillment % ($|S_i|/k_i$) | Under-filled Docs | Avg Sentences | Avg Tokens | Selection Time (ms/doc) | Docs / min |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **C2 (Redundancy Swap)** | `0.3` | `0.0342 ± 0.037` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `158` | `1.826` | `32,864` |
| **C2 (Redundancy Swap)** | `0.5` | `0.0599 ± 0.050` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `216` | `1.515` | `39,600` |
| **C2 (Redundancy Swap)** | `0.7` | `0.1394 ± 0.103` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `292` | `1.490` | `40,272` |
| **C2 (Redundancy Swap)** | `0.9` | `0.2431 ± 0.152` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `334` | `1.507` | `39,810` |
| **C3 (Proposed)** | `0.3` | `0.0286 ± 0.029` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `152` | `1.451` | `41,352` |
| **C3 (Proposed)** | `0.5` | `0.0412 ± 0.039` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `204` | `1.829` | `32,813` |
| **C3 (Proposed)** | `0.7` | `0.1026 ± 0.084` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `274` | `1.426` | `42,067` |
| **C3 (Proposed)** | `0.9` | `0.2122 ± 0.136` | **`100.0%`** | `0/50` (`0.0%`) | `14.0` | `320` | `1.456` | `41,196` |

## 4. Efficiency Comparison (Selection Loop Only)

- **C1 (WMD-threshold)** average selection time: **`470.5 ms/doc`**
- **C2 (MMR / SBERT)** average selection time: **`1.585 ms/doc`**
- **C3 (MMR / SBERT)** average selection time: **`1.541 ms/doc`**
- **Speedup Factor**: MMR selection loop is **`305.4×` faster** than the WMD threshold filter.
