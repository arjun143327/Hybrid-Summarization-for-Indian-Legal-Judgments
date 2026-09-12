"""
wmd.py — Word Mover's Distance (WMD) computation per 02_METHODOLOGY.md.
Used in configs C1 and C2 for:
1. Sentence-to-document WMD feature extraction in Stage 1: x_ij = [TF-IDF, NER, Position, Cosine_w2v, WMD].
2. Pairwise sentence-to-sentence distance for C1 redundancy control filter in Stage 3:
   Add s_ij to summary candidate set only if WMD(s_ij, s_ik) >= delta for all previously selected s_ik.

Built as a reusable pairwise-distance function and feature extraction pipeline
using Gensim's KeyedVectors and Python Optimal Transport (POT).
"""

from typing import List, Optional, Union, Tuple
import numpy as np
from gensim.models import KeyedVectors

from src.data.preprocessing import LegalTokenizer


def pairwise_wmd(
    text_or_tokens1: Union[str, List[str]],
    text_or_tokens2: Union[str, List[str]],
    model: KeyedVectors,
    tokenizer: Optional[LegalTokenizer] = None,
    norm: bool = True
) -> float:
    """
    Computes the Word Mover's Distance (WMD) between two texts or token lists.
    Reusable for both Stage 1 feature extraction and Stage 3 (C1 redundancy filter).

    Parameters
    ----------
    text_or_tokens1 : Union[str, List[str]]
        First sentence (either raw text string or pre-tokenized list of words).
    text_or_tokens2 : Union[str, List[str]]
        Second sentence or document (raw text or token list).
    model : KeyedVectors
        Loaded pretrained word vector model.
    tokenizer : Optional[LegalTokenizer]
        Tokenizer instance to convert string to tokens. Defaults to LegalTokenizer.
    norm : bool
        Whether to normalize word vectors to unit length (default True, recommended by gensim).

    Returns
    -------
    float
        Word Mover's Distance between the two inputs.
        Returns 0.0 if identical or both empty.
        Returns float('inf') if one input has no in-vocabulary words.
    """
    if tokenizer is None:
        tokenizer = LegalTokenizer()

    # Convert strings to tokens if necessary
    tokens1 = (
        tokenizer.tokenize_for_tfidf(text_or_tokens1)
        if isinstance(text_or_tokens1, str)
        else text_or_tokens1
    )
    tokens2 = (
        tokenizer.tokenize_for_tfidf(text_or_tokens2)
        if isinstance(text_or_tokens2, str)
        else text_or_tokens2
    )

    # Empty checks
    if not tokens1 and not tokens2:
        return 0.0
    if not tokens1 or not tokens2:
        return float("inf")

    # Fast equality check
    if tokens1 == tokens2:
        return 0.0

    return float(model.wmdistance(tokens1, tokens2, norm=norm))


def compute_document_wmd_features(
    sentences: List[str],
    model: KeyedVectors,
    tokenizer: Optional[LegalTokenizer] = None,
    norm: bool = True,
    max_fallback_dist: float = 3.0,
    return_fallback_mask: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Computes the sentence-to-document WMD feature for all sentences in a document:
    WMD(s_ij, d_i)

    Parameters
    ----------
    sentences : List[str]
        List of segmented sentences for a single document.
    model : KeyedVectors
        Loaded pretrained word vector model.
    tokenizer : Optional[LegalTokenizer]
        Tokenizer to extract tokens. Defaults to LegalTokenizer.
    norm : bool
        Whether to normalize word vectors.
    max_fallback_dist : float
        Replacement value if WMD is infinite (e.g. sentence has zero in-vocab tokens).
    return_fallback_mask : bool
        If True, also returns a boolean array indicating which sentences triggered fallback.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, np.ndarray]
        1D array of WMD distance values of shape (len(sentences),).
        If return_fallback_mask is True, returns (wmd_scores, fallback_mask).
    """
    if not sentences:
        empty = np.array([], dtype=np.float32)
        return (empty, np.array([], dtype=bool)) if return_fallback_mask else empty

    if tokenizer is None:
        tokenizer = LegalTokenizer()

    # Pre-tokenize all sentences
    tokenized_sentences = [tokenizer.tokenize_for_tfidf(s) for s in sentences]

    # Full document tokens
    doc_tokens = [tok for s_tokens in tokenized_sentences for tok in s_tokens]
    doc_in_vocab = [w for w in doc_tokens if w in model]

    if not doc_in_vocab:
        fallback_scores = np.full(len(sentences), max_fallback_dist, dtype=np.float32)
        fallback_mask = np.ones(len(sentences), dtype=bool)
        return (fallback_scores, fallback_mask) if return_fallback_mask else fallback_scores

    from collections import Counter
    from scipy.spatial.distance import cdist
    from ot import emd2

    def get_norm_vec(w):
        v = model[w].astype(np.float64)
        if norm:
            nv = np.linalg.norm(v)
            if nv > 0:
                v = v / nv
        return v

    doc_counts = Counter(doc_in_vocab)
    doc_unique = list(doc_counts.keys())
    doc_v = np.array([get_norm_vec(w) for w in doc_unique], dtype=np.float64)
    doc_len = len(doc_in_vocab)
    b = np.array([doc_counts[w] / doc_len for w in doc_unique], dtype=np.float64)

    # Compute sentence-to-document WMD using exact POT EMD
    wmd_scores = []
    fallback_flags = []
    for s_tokens in tokenized_sentences:
        s_in_vocab = [w for w in s_tokens if w in model]
        if not s_in_vocab:
            wmd_scores.append(max_fallback_dist)
            fallback_flags.append(True)
            continue

        s_counts = Counter(s_in_vocab)
        s_unique = list(s_counts.keys())
        s_v = np.array([get_norm_vec(w) for w in s_unique], dtype=np.float64)
        s_len = len(s_in_vocab)
        a = np.array([s_counts[w] / s_len for w in s_unique], dtype=np.float64)

        M = cdist(s_v, doc_v, metric="euclidean")
        cost = emd2(a, b, M)
        if np.isinf(cost) or np.isnan(cost):
            cost = max_fallback_dist
            fallback_flags.append(True)
        else:
            fallback_flags.append(False)
        wmd_scores.append(float(cost))

    scores_arr = np.array(wmd_scores, dtype=np.float32)
    mask_arr = np.array(fallback_flags, dtype=bool)
    if return_fallback_mask:
        return scores_arr, mask_arr
    return scores_arr


def compute_pairwise_wmd_matrix(
    sentences: List[str],
    model: KeyedVectors,
    tokenizer: Optional[LegalTokenizer] = None,
    norm: bool = True
) -> np.ndarray:
    """
    Computes an (M_i, M_i) pairwise symmetric WMD distance matrix between all sentences.
    Useful for redundancy analysis and Stage 3 candidate filtering.

    Parameters
    ----------
    sentences : List[str]
        List of candidate sentences.
    model : KeyedVectors
        Loaded word vector model.
    tokenizer : Optional[LegalTokenizer]
        Tokenizer instance.
    norm : bool
        Normalize vectors to unit length.

    Returns
    -------
    np.ndarray
        Symmetric 2D matrix of shape (M_i, M_i).
    """
    n = len(sentences)
    dist_matrix = np.zeros((n, n), dtype=np.float32)

    if tokenizer is None:
        tokenizer = LegalTokenizer()

    tokenized_sents = [tokenizer.tokenize_for_tfidf(s) for s in sentences]

    for i in range(n):
        for j in range(i + 1, n):
            d = pairwise_wmd(
                tokenized_sents[i], tokenized_sents[j], model, tokenizer=tokenizer, norm=norm
            )
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix
