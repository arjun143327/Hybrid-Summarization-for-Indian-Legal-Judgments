"""
embeddings_sbert.py — SBERT sentence and document embeddings and cosine similarity
per 02_METHODOLOGY.md Stage 1 (used in config C3).

Computes:
1. SBERT sentence embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`).
2. SBERT document embedding: Mean of sentence embeddings: d_i = (1 / M_i) * sum_j(s_ij).
3. Cosine similarity feature:
   Sim_cos(s_ij) = (s_ij · d_i) / (||s_ij|| ||d_i||)

This module also provides reusable sentence embedding functions that will be
consumed in Stage 3 for the MMR/SBERT redundancy control (configs C2 and C3).
"""

import os
from typing import List, Optional, Union
import numpy as np

# Suppress TensorFlow verbose oneDNN warnings when imported via transformers
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from sentence_transformers import SentenceTransformer

DEFAULT_SBERT_MODEL = "all-MiniLM-L6-v2"


def load_sbert_model(
    model_name: str = DEFAULT_SBERT_MODEL,
    device: Optional[str] = None
) -> SentenceTransformer:
    """
    Loads a SentenceTransformer model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (default: 'all-MiniLM-L6-v2').
    device : Optional[str]
        Device to run model on ('cuda', 'cpu', or None for automatic selection).

    Returns
    -------
    SentenceTransformer
        Loaded model instance.
    """
    return SentenceTransformer(model_name, device=device)


def encode_sentences(
    sentences: List[str],
    model: SentenceTransformer,
    batch_size: int = 64,
    show_progress_bar: bool = False,
    normalize_embeddings: bool = False
) -> np.ndarray:
    """
    Encodes a list of sentences into dense SBERT embeddings.

    Parameters
    ----------
    sentences : List[str]
        List of raw sentence strings.
    model : SentenceTransformer
        Loaded SentenceTransformer model.
    batch_size : int
        Mini-batch size for encoding.
    show_progress_bar : bool
        Whether to show tqdm progress bar.
    normalize_embeddings : bool
        If True, returns L2-normalized unit vectors.

    Returns
    -------
    np.ndarray
        2D array of shape (len(sentences), embedding_dim), e.g. (M_i, 384).
    """
    if not sentences:
        return np.empty((0, 384), dtype=np.float32)

    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings
    )
    return embeddings.astype(np.float32)


def compute_sbert_document_embedding(sentence_embeddings: np.ndarray) -> np.ndarray:
    """
    Computes document embedding as the mean of sentence embeddings in SBERT space:
    d_i = (1 / M_i) * sum_{j=1}^{M_i} s_{ij}

    Parameters
    ----------
    sentence_embeddings : np.ndarray
        2D array of sentence embeddings of shape (M_i, embedding_dim).

    Returns
    -------
    np.ndarray
        1D document embedding vector of shape (embedding_dim,).
    """
    if sentence_embeddings.shape[0] == 0:
        return np.zeros(sentence_embeddings.shape[1] if sentence_embeddings.ndim > 1 else 384, dtype=np.float32)
    return np.mean(sentence_embeddings, axis=0).astype(np.float32)


def compute_sbert_cosine_features(
    sentences: List[str],
    model: SentenceTransformer,
    batch_size: int = 64
) -> np.ndarray:
    """
    Computes SBERT cosine similarity feature Sim_cos(s_ij, d_i) for all sentences in a document.

    Parameters
    ----------
    sentences : List[str]
        List of segmented sentences for a single document.
    model : SentenceTransformer
        Loaded SBERT model.
    batch_size : int
        Batch size for sentence encoding.

    Returns
    -------
    np.ndarray
        1D array of cosine similarity scores of shape (len(sentences),).
    """
    if not sentences:
        return np.array([], dtype=np.float32)

    # Step 1: Encode all sentences into SBERT space
    sent_embs = encode_sentences(sentences, model, batch_size=batch_size)

    # Step 2: Compute document embedding as mean of sentence embeddings
    doc_emb = compute_sbert_document_embedding(sent_embs)

    # Step 3: Compute vectorized cosine similarity
    doc_norm = np.linalg.norm(doc_emb)
    sent_norms = np.linalg.norm(sent_embs, axis=1)

    denom = sent_norms * doc_norm
    # Prevent division by zero
    zero_mask = (denom == 0.0)
    denom[zero_mask] = 1e-12

    dot_products = np.dot(sent_embs, doc_emb)
    cos_sims = dot_products / denom
    cos_sims[zero_mask] = 0.0

    return np.clip(cos_sims, -1.0, 1.0).astype(np.float32)
