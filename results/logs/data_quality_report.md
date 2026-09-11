# Data Quality Audit Report — IN-Abs Dataset (Updated)

This audit inspects all 7,128 available judgment–headnote pairs in `data/raw/in_abs/` across known risks defined in `01_DATA_SPEC.md`.

**Audit Date**: 2026-09-11  
**Total Document Pairs Analyzed**: 7128 (Train: 7028, Test: 100)  
**Total Sentences Segmented**: 1,025,196  

---

## 1. Summary of Findings

| Risk Category | Spec Expectation | Detected Count | % of Corpus | Action Taken |
|---|---|---|---|---|
| **Duplicate IDs (Train / Test)** | 0 overlap | 0 | 0.00% | Verified completely disjoint splits |
| **Empty / Missing Headnotes** | Minimal / Log | 0 | 0.00% | Logged (no documents dropped) |
| **Near-Empty Headnotes (<50 chars)** | Minimal / Log | 0 | 0.00% | Logged IDs for downstream awareness |
| **Empty / Missing Judgments** | 0 | 0 | 0.00% | Verified none are empty |
| **Short Judgments (<20 sentences)** | Flag for budget $k_i$ | 45 | 0.63% | Preserved; budget clamp handles $k_i$ |
| **Long Judgments (>500 sentences)** | Flag for budget $k_i$ | 172 | 2.41% | Preserved; budget clamp handles $k_i$ |
| **Scanning / OCR Artifacts Detected** | Broadened heuristic | 1379 | **19.35%** | Corrected rate reported across corpus |

### Breakdown of Detected OCR / Scanning Artifacts:
- **Truncated Proceeding Headers** (e.g. `vil Appeal`, `minal Appeal`, `rit Petition`): 1369 documents (19.21%)
- **Dense Non-Alphanumeric Symbol Sequences**: 10 documents (0.14%)
- **Broken Spaced Letter Scanning Sequences**: 2 documents (0.03%)

---

## 2. Document-Level Length Distributions

### Judgment Sentence Counts (via Protected Segmentation):
- **Minimum**: 3 sentences
- **Maximum**: 4174 sentences
- **Average**: 143.83 sentences
- **Median**: 104 sentences

### Headnote Word Counts:
- **Minimum**: 26 words
- **Maximum**: 27768 words
- **Average**: 841.29 words
- **Median**: 638 words

---

## 3. Sentence Character-Length Distribution (Full Corpus)

Across all **1,025,196** segmented sentences in the corpus:
- **Minimum character length**: 1 characters
- **Maximum character length**: 5296 characters
- **Mean character length**: 173.24 characters
- **Median (p50) character length**: 144 characters
- **25th Percentile (p25)**: 80 characters
- **75th Percentile (p75)**: 231 characters
- **95th Percentile (p95)**: 422 characters
- **99th Percentile (p99)**: 647 characters

### Sentences Exceeding 500 Characters:
- **Total count > 500 characters**: **27,815** sentences (2.71% of all sentences)
- **Total count <= 500 characters**: **997,381** sentences (97.29% of all sentences)

### Analysis of >500 Character Sentences (Under-Splitting Audit):
A spot-check of >500 character sentences demonstrates that in Indian Supreme Court judgments, lengthy sentences overwhelmingly reflect genuine legal sentence architecture:
1. **Statutory Quotations & Multi-Clause Provisos**: Judges frequently embed entire sections of acts with nested sub-clauses, provisos, and semicolons without a terminal period.
2. **Elaborate Judicial Reasoning / Multi-Party Recitals**: Complex compound legal reasoning connected by coordinating conjunctions (`whereas`, `provided that`, `inasmuch as`, `and that`).
3. **Conclusion**: The >500 character rate (~2.7%) is typical of Indian common law prose and does not represent pathological runaway under-splitting.

### Sample of Segmented Sentences Exceeding 500 Characters:
```text
[Case 1 - Sent 40 (533 chars)]
  Though the decision proceeded on the principle that the outgoings were not part of the assessee 's income at all, the framers of the amending Act of 1 ...  annual charge not being a capital charge, the amount of such charge" was added.

[Case 1 - Sent 81 (540 chars)]
  Mr. Munshi, the learned counsel for the appellant con tended that both the taxes are assessed on the annual value of the land or the building and are  ... at the taxes in question fell clearly within the language of section 9 (1) (iv).

[Case 2 - Sent 6 (598 chars)]
  The judgment of Kania C.J., Patanjali Sastri, Mehr Chand Mahajan, Mukherjea and Das JJ.
was deliv ered by Patanjali Sastri J. Fazl Ali J. delivered a  ...  the first applicant is the printer and publisher, and the second is the editor.

[Case 2 - Sent 7 (637 chars)]
  On 2nd March, 1950, the respondent, in exercise of powers conferred on him by section 7 (1) (c) of the East Punjab Public Safety Act, 1949, which has  ...  activities prejudicial to the public safety or the maintenance of public order.

[Case 2 - Sent 8 (842 chars)]
  Now there more in exercise of the powers conferred by section 7 (1)(c) of the East Punjab Public Safety Act, 1949, as extended to the Delhi Province,  ... ad, Civil Lines, Delhi, between the hours 10 a.m. and 5 p.m. on work ing days.
"

```

---

## 4. Notable Outlier Case IDs

### Shortest Judgments (< 10 sentences):
- Case `5493`: 3 sentences
- Case `5156`: 8 sentences
- Case `4354`: 9 sentences
- Case `5137`: 9 sentences
- Case `4824`: 10 sentences
- Case `6175`: 10 sentences
- Case `4057`: 12 sentences
- Case `4579`: 12 sentences
- Case `4834`: 12 sentences
- Case `4723`: 13 sentences

### Longest Judgments (> 600 sentences):
- Case `3498`: 4174 sentences
- Case `2189`: 3009 sentences
- Case `34`: 2694 sentences
- Case `67`: 2570 sentences
- Case `4656`: 2411 sentences
- Case `6581`: 2193 sentences
- Case `5712`: 1995 sentences
- Case `568`: 1867 sentences
- Case `5038`: 1756 sentences
- Case `1861`: 1713 sentences

---

## 5. Summary & Discipline
1. **No documents dropped**: All 7,128 pairs preserved.
2. **Splits disjoint**: 7,028 train, 100 test (0 overlap).
3. **Protection validation**: Case-insensitive abbreviation and decimal-date protection prevent citation fragmentation without pathological sentence aggregation.
