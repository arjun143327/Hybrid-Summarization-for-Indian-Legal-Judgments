# GBR Rank Correlation Report — Validation Split

**Date**: 2026-09-13 17:34:10  
**Val split**: 1,054 docs / 147,498 sentences  
**Budget rule**: k_i = max(3, round(0.05 * M_i))  avg k_i ≈ 7.2

## Global Sentence-Level Rank Correlation

| Metric | C1 (Replica) | C2 (Redundancy Swap) | C3 (Proposed) |
|---|:---:|:---:|:---:|
| **Spearman ρ** | 0.5804 | 0.5804 | 0.5467 |
| Spearman p | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| **Kendall τ** | 0.4203 | 0.4203 | 0.3919 |
| Kendall p | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| Per-doc Spearman (mean) | 0.5889 | 0.5889 | 0.5593 |
| Per-doc Kendall (mean) | 0.4338 | 0.4338 | 0.4077 |

## Task-Relevant: Top-k_i Extractive Selection Overlap

| Metric | C1 (Replica) | C2 (Redundancy Swap) | C3 (Proposed) |
|---|:---:|:---:|:---:|
| **Top-k Jaccard (mean)** | 0.0735 | 0.0735 | 0.0640 |
| Top-k Jaccard (std) | 0.1007 | 0.1007 | 0.0969 |
| **Top-k % Recall (mean)** | 0.1225 | 0.1225 | 0.1068 |
| Avg k_i | 7.2 | 7.2 | 7.2 |

## Worked Examples — Config C1 = C2 (identical scorer)

> True top-k = oracle ranking by ground-truth ROUGE label y_ij.  
> Pred top-k = GBR ranking by predicted score ŷ_ij.  
> Sentences in **bold** appear in BOTH sets.

### Doc 1005 — M_i=83 sents, k_i=4
- Jaccard: **0.0** | % Recall: **0%** (0/4 matched)
- True top-4: [10, 59, 61, 66]
- Pred top-4: [68, 70, 79, 80]

| Sent | True y | Pred ŷ | In Oracle? | In Model? |
|:---:|:---:|:---:|:---:|:---:|
| 10 | 0.6667 | — | ✅ | ❌ |
| 59 | 0.8627 | — | ✅ | ❌ |
| 61 | 0.7945 | — | ✅ | ❌ |
| 66 | 0.7018 | — | ✅ | ❌ |
| 68 | — | 0.3689 | ❌ | ✅ |
| 70 | — | 0.3371 | ❌ | ✅ |
| 79 | — | 0.4308 | ❌ | ✅ |
| 80 | — | 0.4124 | ❌ | ✅ |

### Doc 1010 — M_i=141 sents, k_i=7
- Jaccard: **0.0769** | % Recall: **14%** (1/7 matched)
- True top-7: [14, 16, 22, 35, 64, 76, **124**]
- Pred top-7: [116, **124**, 128, 134, 135, 136, 137]

| Sent | True y | Pred ŷ | In Oracle? | In Model? |
|:---:|:---:|:---:|:---:|:---:|
| 14 | 0.9565 | — | ✅ | ❌ |
| 16 | 0.7273 | — | ✅ | ❌ |
| 22 | 0.9091 | — | ✅ | ❌ |
| 35 | 0.5846 | — | ✅ | ❌ |
| 64 | 0.8308 | — | ✅ | ❌ |
| 76 | 0.6222 | — | ✅ | ❌ |
| 116 | — | 0.2857 | ❌ | ✅ |
| 124 | 0.6176 | 0.6176 | ✅ | ✅ |
| 128 | — | 0.3784 | ❌ | ✅ |
| 134 | — | 0.3714 | ❌ | ✅ |
| 135 | — | 0.4337 | ❌ | ✅ |
| 136 | — | 0.3125 | ❌ | ✅ |
| 137 | — | 0.506 | ❌ | ✅ |

### Doc 1013 — M_i=138 sents, k_i=7
- Jaccard: **0.0769** | % Recall: **14%** (1/7 matched)
- True top-7: [11, 13, 91, 122, 123, 124, **135**]
- Pred top-7: [24, 127, 130, 132, 133, 134, **135**]

| Sent | True y | Pred ŷ | In Oracle? | In Model? |
|:---:|:---:|:---:|:---:|:---:|
| 11 | 0.6061 | — | ✅ | ❌ |
| 13 | 0.6019 | — | ✅ | ❌ |
| 24 | — | 0.5524 | ❌ | ✅ |
| 91 | 0.6923 | — | ✅ | ❌ |
| 122 | 0.6575 | — | ✅ | ❌ |
| 123 | 1.0 | — | ✅ | ❌ |
| 124 | 0.5941 | — | ✅ | ❌ |
| 127 | — | 0.3051 | ❌ | ✅ |
| 130 | — | 0.5238 | ❌ | ✅ |
| 132 | — | 0.4706 | ❌ | ✅ |
| 133 | — | 0.3898 | ❌ | ✅ |
| 134 | — | 0.5778 | ❌ | ✅ |
| 135 | 0.6923 | 0.6923 | ✅ | ✅ |

## Worked Examples — Config C3 (Proposed)

> True top-k = oracle ranking by ground-truth ROUGE label y_ij.  
> Pred top-k = GBR ranking by predicted score ŷ_ij.  
> Sentences in **bold** appear in BOTH sets.

### Doc 1005 — M_i=83 sents, k_i=4
- Jaccard: **0.0** | % Recall: **0%** (0/4 matched)
- True top-4: [10, 59, 61, 66]
- Pred top-4: [69, 70, 74, 79]

| Sent | True y | Pred ŷ | In Oracle? | In Model? |
|:---:|:---:|:---:|:---:|:---:|
| 10 | 0.6667 | — | ✅ | ❌ |
| 59 | 0.8627 | — | ✅ | ❌ |
| 61 | 0.7945 | — | ✅ | ❌ |
| 66 | 0.7018 | — | ✅ | ❌ |
| 69 | — | 0.3059 | ❌ | ✅ |
| 70 | — | 0.3371 | ❌ | ✅ |
| 74 | — | 0.3704 | ❌ | ✅ |
| 79 | — | 0.4308 | ❌ | ✅ |

### Doc 1010 — M_i=141 sents, k_i=7
- Jaccard: **0.1667** | % Recall: **29%** (2/7 matched)
- True top-7: [14, 16, 22, 35, **64**, 76, **124**]
- Pred top-7: [63, **64**, 83, 95, **124**, 135, 137]

| Sent | True y | Pred ŷ | In Oracle? | In Model? |
|:---:|:---:|:---:|:---:|:---:|
| 14 | 0.9565 | — | ✅ | ❌ |
| 16 | 0.7273 | — | ✅ | ❌ |
| 22 | 0.9091 | — | ✅ | ❌ |
| 35 | 0.5846 | — | ✅ | ❌ |
| 63 | — | 0.5301 | ❌ | ✅ |
| 64 | 0.8308 | 0.8308 | ✅ | ✅ |
| 76 | 0.6222 | — | ✅ | ❌ |
| 83 | — | 0.4615 | ❌ | ✅ |
| 95 | — | 0.2885 | ❌ | ✅ |
| 124 | 0.6176 | 0.6176 | ✅ | ✅ |
| 135 | — | 0.4337 | ❌ | ✅ |
| 137 | — | 0.506 | ❌ | ✅ |

### Doc 1013 — M_i=138 sents, k_i=7
- Jaccard: **0.0** | % Recall: **0%** (0/7 matched)
- True top-7: [11, 13, 91, 122, 123, 124, 135]
- Pred top-7: [103, 121, 125, 127, 130, 131, 134]

| Sent | True y | Pred ŷ | In Oracle? | In Model? |
|:---:|:---:|:---:|:---:|:---:|
| 11 | 0.6061 | — | ✅ | ❌ |
| 13 | 0.6019 | — | ✅ | ❌ |
| 91 | 0.6923 | — | ✅ | ❌ |
| 103 | — | 0.3947 | ❌ | ✅ |
| 121 | — | 0.4274 | ❌ | ✅ |
| 122 | 0.6575 | — | ✅ | ❌ |
| 123 | 1.0 | — | ✅ | ❌ |
| 124 | 0.5941 | — | ✅ | ❌ |
| 125 | — | 0.2947 | ❌ | ✅ |
| 127 | — | 0.3051 | ❌ | ✅ |
| 130 | — | 0.5238 | ❌ | ✅ |
| 131 | — | 0.3733 | ❌ | ✅ |
| 134 | — | 0.5778 | ❌ | ✅ |
| 135 | 0.6923 | — | ✅ | ❌ |

