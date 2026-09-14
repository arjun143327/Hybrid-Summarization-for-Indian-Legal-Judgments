"""
redundancy_metrics.py — Internal summary redundancy and diversity metrics
per 03_EXPERIMENT_PLAN.md.

Computes:
1. Self-BLEU-2: Evaluates internal repetition across sentences in a summary.
   Treats each sentence as hypothesis against remaining sentences as references.
2. Unique N-gram Ratio (Distinct-1, Distinct-2): Ratio of unique to total n-grams.
3. Summary Length & Budget Fulfillment: |S_i|, k_i, and fulfillment percentage |S_i| / k_i.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from src.data.preprocessing import LegalTokenizer

_SMOOTH = SmoothingFunction().method1


def compute_self_bleu(
    sentences: List[str],
    n_gram: int = 2,
    tokenizer: Optional[LegalTokenizer] = None
) -> float:
    """
    Computes Self-BLEU across sentences in an extractive summary.

    Parameters
    ----------
    sentences : List[str]
        List of sentence strings in the summary.
    n_gram : int
        Max n-gram for BLEU (default 2 for Self-BLEU-2).
    tokenizer : Optional[LegalTokenizer]
        Tokenizer instance to convert sentences to word tokens.

    Returns
    -------
    float
        Mean Self-BLEU score in [0.0, 1.0]. Lower indicates less redundancy.
        Returns 0.0 if summary has <= 1 sentence.
    """
    if len(sentences) <= 1:
        return 0.0

    if tokenizer is None:
        tokenizer = LegalTokenizer()

    tokenized = [tokenizer.tokenize_for_tfidf(s) for s in sentences]
    # Filter out empty token lists
    tokenized = [toks for toks in tokenized if toks]
    if len(tokenized) <= 1:
        return 0.0

    if n_gram == 2:
        weights = (0.5, 0.5)
    elif n_gram == 3:
        weights = (1/3, 1/3, 1/3)
    elif n_gram == 4:
        weights = (0.25, 0.25, 0.25, 0.25)
    else:
        weights = tuple([1.0 / n_gram] * n_gram)

    scores = []
    for i, hyp in enumerate(tokenized):
        refs = [ref for j, ref in enumerate(tokenized) if j != i]
        bleu = sentence_bleu(
            refs,
            hyp,
            weights=weights,
            smoothing_function=_SMOOTH
        )
        scores.append(bleu)

    return float(np.mean(scores)) if scores else 0.0


def compute_distinct_ngrams(
    sentences: List[str],
    n: int = 2,
    tokenizer: Optional[LegalTokenizer] = None
) -> float:
    """
    Computes ratio of distinct n-grams to total n-grams in the summary.

    Parameters
    ----------
    sentences : List[str]
        List of sentence strings in the summary.
    n : int
        N-gram order (default 2 for distinct bigrams).
    tokenizer : Optional[LegalTokenizer]
        Tokenizer instance.

    Returns
    -------
    float
        Distinct-n ratio in [0.0, 1.0]. Higher indicates more diversity.
    """
    if not sentences:
        return 0.0

    if tokenizer is None:
        tokenizer = LegalTokenizer()

    tokens = []
    for s in sentences:
        tokens.extend(tokenizer.tokenize_for_tfidf(s))

    if len(tokens) < n:
        return 1.0 if tokens else 0.0

    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def evaluate_summary_redundancy(
    sentences: List[str],
    k_i: int,
    tokenizer: Optional[LegalTokenizer] = None
) -> Dict[str, Any]:
    """
    Evaluates complete redundancy and length stats for a selected summary.

    Parameters
    ----------
    sentences : List[str]
        Selected summary sentences in original document order.
    k_i : int
        Target sentence budget.
    tokenizer : Optional[LegalTokenizer]
        Tokenizer instance.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - "selected_count": int, |S_i|
        - "k_i": int
        - "budget_fulfillment": float, |S_i| / k_i
        - "under_filled": bool, |S_i| < k_i
        - "self_bleu_2": float
        - "distinct_2": float
        - "token_count": int
    """
    if tokenizer is None:
        tokenizer = LegalTokenizer()

    token_count = sum(len(tokenizer.tokenize_for_tfidf(s)) for s in sentences)
    count = len(sentences)
    fulfillment = count / k_i if k_i > 0 else 1.0

    return {
        "selected_count": count,
        "k_i": k_i,
        "budget_fulfillment": float(fulfillment),
        "under_filled": bool(count < k_i),
        "self_bleu_2": compute_self_bleu(sentences, n_gram=2, tokenizer=tokenizer),
        "distinct_2": compute_distinct_ngrams(sentences, n=2, tokenizer=tokenizer),
        "token_count": token_count,
    }
