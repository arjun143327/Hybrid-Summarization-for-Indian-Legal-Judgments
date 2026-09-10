# Data Specification — IN-Abs Dataset

## Source
IN-Abs (Indian-Abstractive) dataset: Indian Supreme Court judgments from the Legal
Information Institute of India, each paired with an expert-written abstractive
"headnote" summary.

- Total: 7,130 judgment–headnote pairs
- Train: 7,030 pairs
- Test: 100 pairs (held out, used for final reported metrics — mirrors base paper's
  100-doc test protocol so ROUGE numbers are comparable in scale/setup)
- Average judgment length: ~4,783 words
- Average headnote length: ~932 words

Supplementary (NOT used unless explicitly requested later): MILDSum (3,122 judgments,
English + Hindi). Do not build multilingual handling now — out of scope.

## Directory layout expected by the pipeline
```
data/
  raw/
    in_abs/
      judgments/        # one .txt per case, raw judgment text
      headnotes/         # one .txt per case, raw headnote/summary text
  splits/
    train_ids.txt        # 7,030 case IDs, one per line
    test_ids.txt          # 100 case IDs, one per line
  processed/
    sentences/            # per-doc sentence-segmented judgment (see below)
    features/               # per-doc extracted feature matrices (cached, per config)
```
Case IDs should be a stable identifier (e.g. filename stem) used to join judgment,
headnote, and all downstream artifacts.

## Preprocessing pipeline (applies identically across all 3 configs — see
00_PROJECT_BRIEF.md for config definitions)

1. **Normalization**: lowercase is NOT applied globally — legal text relies on
   capitalization for entity/party names; keep NER extraction on original case,
   only lowercase the TF-IDF tokenization path.
2. **Sentence segmentation**: use NLTK `sent_tokenize` as the default, BUT legal
   judgments have citation patterns (e.g. "AIR 1978 SC 597", "(1973) 2 SCC 235")
   that can falsely trigger sentence boundaries at the period after "AIR" or
   abbreviations. Flag this explicitly to Antigravity: a naive NLTK split WILL
   fragment citations. Recommended handling — a pre-pass regex protecting common
   legal abbreviation/citation patterns before segmentation (see snippet below),
   or use `nltk.tokenize.PunktSentenceTokenizer` extended with a custom
   abbreviation list including: "v", "vs", "AIR", "SCC", "SCR", "Cr", "L.J",
   "All", "Mad", "Cal", "Bom", "Ors", "Anr", "No", "Art", "S", "Sec".
3. **Tokenization**: NLTK word_tokenize for TF-IDF vocabulary construction.
4. **Stopword removal**: NLTK English stopword list; DO NOT strip legal
   stopwords/boilerplate connectors (e.g. "held", "appellant", "respondent") —
   these are semantically load-bearing in legal text, unlike in news.
5. **Lemmatization**: WordNetLemmatizer, applied only to the TF-IDF path (not to
   raw sentences used for NER, embeddings, or BART input — those need original
   surface form).

## Sentence-boundary protection snippet (illustrative, for Antigravity)
```python
import re

LEGAL_ABBREVS = [
    "AIR", "SCC", "SCR", "Cr", "L.J", "All", "Mad", "Cal", "Bom",
    "Ors", "Anr", "No", "Art", "Sec", "v", "vs",
]

def protect_citations(text: str) -> str:
    """Replace periods inside known legal abbreviations with a placeholder
    before sentence segmentation, then restore after."""
    for abbr in LEGAL_ABBREVS:
        pattern = rf"\b{re.escape(abbr)}\."
        text = re.sub(pattern, f"{abbr}<PERIOD>", text)
    return text

def restore_periods(sentences: list[str]) -> list[str]:
    return [s.replace("<PERIOD>", ".") for s in sentences]
```
This is a starting point, not a finished solution — validate against a sample of
~10 judgments by eye before running on the full 7,030-doc train set, since legal
citation formats vary. Log a sample of before/after segmentation to a file for
manual spot-checking.

## Ground-truth importance labels for GBR (y_ij)
Per the discussion already had with the user: use **max-ROUGE-to-reference-sentence**
labeling, NOT whole-reference-blob scoring and NOT full greedy-oracle search
(too expensive at IN-Abs document lengths; see 02_METHODOLOGY.md section on GBR
for the exact formula and rationale).

## Splits discipline
- Train/test split must be fixed and versioned (`train_ids.txt`, `test_ids.txt`)
  and reused identically across ALL THREE configs (C1, C2, C3) and the base-paper
  replica run. No re-splitting between experiments — this is required for the
  "same test set" fairness claim in 03_EXPERIMENT_PLAN.md.
- The 100-doc test set should be chosen once (e.g. random seed logged) and frozen.
  Do not cherry-pick or resample it after seeing results.

## Known data-quality risks to check for during loading (log counts, don't silently drop)
- Empty or near-empty headnotes
- Judgments with OCR artifacts (IN-Abs source documents can have scanning noise)
- Duplicate case IDs across train/test
- Extremely short judgments (<20 sentences) or extremely long ones (>500 sentences)
  that may need separate handling for the sentence budget `k_i` (see 02_METHODOLOGY.md)
