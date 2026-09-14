"""
wmd_filter.py — WMD-threshold redundancy filter for Config C1 per 02_METHODOLOGY.md.

Implements the hard threshold filter (base paper eq. 7):
  Add s_ij to summary candidate set only if:
      WMD(s_ij, s_ik) >= delta for all previously selected s_ik

Candidates are evaluated in descending order of GBR predicted importance scores.
Selection halts when the sentence budget k_i is reached or all candidates are exhausted.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
from gensim.models import KeyedVectors

from src.data.preprocessing import LegalTokenizer
from src.features.wmd import pairwise_wmd


def wmd_threshold_filter(
    sentences: List[str],
    scores: np.ndarray,
    k_i: int,
    delta: float,
    keyed_vectors: KeyedVectors,
    tokenizer: Optional[LegalTokenizer] = None,
    norm: bool = True,
    max_fallback_dist: float = 3.0
) -> Dict[str, Any]:
    """
    Applies the greedy WMD threshold filter on a document's sentences.

    Parameters
    ----------
    sentences : List[str]
        List of segmented sentence texts in original document order.
    scores : np.ndarray
        1D array of predicted GBR importance scores (length M_i).
    k_i : int
        Sentence budget for this document.
    delta : float
        WMD distance threshold. Candidate must have WMD >= delta against
        all already-selected sentences to be admitted.
    keyed_vectors : KeyedVectors
        Pretrained word vector model (e.g. Word2Vec Google News).
    tokenizer : Optional[LegalTokenizer]
        Tokenizer instance. If None, instantiates LegalTokenizer.
    norm : bool
        Whether to L2-normalize vectors for WMD (default True).
    max_fallback_dist : float
        Distance assigned to OOV/infinite WMD pairs (default 3.0).

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - "selected_indices": list of sentence indices in selection order
        - "ordered_indices": list of sentence indices sorted by original appearance
        - "selected_count": int, |S_i|
        - "k_i": int, target budget
        - "budget_fulfillment": float, |S_i| / k_i
        - "under_filled": bool, whether |S_i| < k_i
        - "num_evaluated": int, number of candidate sentences considered
        - "num_rejected": int, number of candidate sentences rejected by filter
    """
    if tokenizer is None:
        tokenizer = LegalTokenizer()

    m_i = len(sentences)
    k_i = min(k_i, m_i)
    if k_i <= 0 or m_i == 0:
        return {
            "selected_indices": [],
            "ordered_indices": [],
            "selected_count": 0,
            "k_i": k_i,
            "budget_fulfillment": 0.0 if k_i > 0 else 1.0,
            "under_filled": k_i > 0,
            "num_evaluated": 0,
            "num_rejected": 0,
        }

    # Pre-tokenize all candidate sentences once for speed
    tokenized_sentences = [tokenizer.tokenize_for_tfidf(s) for s in sentences]

    # Rank candidates by descending predicted importance
    candidate_order = np.argsort(scores)[::-1]

    selected_indices: List[int] = []
    num_evaluated = 0
    num_rejected = 0

    for cand_idx in candidate_order:
        num_evaluated += 1
        cand_tokens = tokenized_sentences[cand_idx]

        if not selected_indices:
            # First candidate accepted unconditionally (highest predicted score)
            selected_indices.append(int(cand_idx))
            if len(selected_indices) == k_i:
                break
            continue

        # Check WMD against all previously selected sentences
        passes_filter = True
        for sel_idx in selected_indices:
            sel_tokens = tokenized_sentences[sel_idx]
            dist = pairwise_wmd(
                cand_tokens,
                sel_tokens,
                model=keyed_vectors,
                tokenizer=None,
                norm=norm
            )
            # Handle OOV fallback
            if not np.isfinite(dist) or dist == float("inf"):
                dist = max_fallback_dist

            if dist < delta:
                passes_filter = False
                num_rejected += 1
                break

        if passes_filter:
            selected_indices.append(int(cand_idx))
            if len(selected_indices) == k_i:
                break

    ordered_indices = sorted(selected_indices)
    selected_count = len(selected_indices)
    fulfillment = selected_count / k_i if k_i > 0 else 1.0

    return {
        "selected_indices": selected_indices,
        "ordered_indices": ordered_indices,
        "selected_count": selected_count,
        "k_i": k_i,
        "budget_fulfillment": float(fulfillment),
        "under_filled": bool(selected_count < k_i),
        "num_evaluated": num_evaluated,
        "num_rejected": num_rejected,
    }
