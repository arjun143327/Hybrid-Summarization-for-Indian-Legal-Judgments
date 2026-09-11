"""
wmd.py — Word Mover's Distance (WMD) computation per 02_METHODOLOGY.md.
Used in configs C1 and C2 for:
1. Sentence-to-document WMD feature extraction in Stage 1: x_ij = [TF-IDF, NER, Position, Cosine_w2v, WMD].
2. Pairwise sentence-to-sentence distance for C1 redundancy control filter in Stage 3:
   Add s_ij to summary candidate set only if WMD(s_ij, s_ik) >= delta for all previously selected s_ik.

Built as a reusable pairwise-distance function and feature extraction pipeline
using Gensim's KeyedVectors and Python Optimal Transport (POT).
"""

from typing import List, Optional, Union
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
    max_fallback_dist: float = 3.0
) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        1D array of WMD distance values of shape (len(sentences),).
    """
    if not sentences:
        return np.array([], dtype=np.float32)

    if tokenizer is None:
        tokenizer = LegalTokenizer()

    # Pre-tokenize all sentences
    tokenized_sentences = [tokenizer.tokenize_for_tfidf(s) for s in sentences]

    # Full document tokens
    doc_tokens = [tok for s_tokens in tokenized_sentences for tok in s_tokens]

    # Compute sentence-to-document WMD
    wmd_scores = []
    for s_tokens in tokenized_sentences:
        dist = pairwise_wmd(s_tokens, doc_tokens, model, tokenizer=tokenizer, norm=norm)
        if np.isinf(dist):
            dist = max_fallback_dist
        wmd_scores.append(dist)

    return np.array(wmd_scores, dtype=np.float32)


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
