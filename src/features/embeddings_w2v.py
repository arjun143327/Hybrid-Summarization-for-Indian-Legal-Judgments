"""
embeddings_w2v.py — Word2Vec-based sentence and document embeddings and cosine similarity
per 02_METHODOLOGY.md Stage 1 (used in configs C1 and C2).

Computes:
1. Sentence embedding: Mean of constituent word vectors for in-vocabulary tokens.
2. Document embedding: Mean of sentence embeddings: d_i = (1 / M_i) * sum_j(s_ij).
3. Cosine similarity feature (base paper eq. 3):
   Sim_cos(s_ij) = (s_ij · d_i) / (||s_ij|| ||d_i||)

Pretrained Vector Choice:
-------------------------
Default local model: `glove-wiki-gigaword-100` (100-dimensional vectors, 400,000 vocab).
Why:
- Download size: 134 MB compressed vs 1.74 GB compressed for `word2vec-google-news-300`.
- Memory footprint: Loads in <1 second via memory-mapping (`mmap='r'`), fitting comfortably
  within local resource constraints while maintaining high lexical coverage for common English
  and legal terms.
- For Colab / high-RAM environments, `word2vec-google-news-300` or any custom vector path
  can be specified via `model_name_or_path`.
"""

import os
from typing import List, Optional, Union
import numpy as np
from gensim.models import KeyedVectors
import gensim.downloader as api

from src.data.preprocessing import LegalTokenizer

# Default local cache paths for pretrained vectors
DEFAULT_LOCAL_KV_PATH = os.path.join(
    os.path.expanduser("~"), "gensim-data", "glove-wiki-gigaword-100", "vectors.kv"
)
DEFAULT_FALLBACK_MODEL = "glove-wiki-gigaword-100"


def load_word2vec_model(
    model_name_or_path: Optional[str] = None,
    mmap: str = "r"
) -> KeyedVectors:
    """
    Loads pretrained word vectors using Gensim KeyedVectors.

    Parameters
    ----------
    model_name_or_path : Optional[str]
        Path to a saved KeyedVectors file (`.kv`) or a gensim-data model name
        (e.g., 'glove-wiki-gigaword-100', 'word2vec-google-news-300').
        If None, attempts to load the local cached binary at `DEFAULT_LOCAL_KV_PATH`,
        falling back to gensim.downloader if not found.
    mmap : str
        Memory-map mode for KeyedVectors.load ('r' recommended for fast read-only).

    Returns
    -------
    KeyedVectors
        Loaded word vector model.
    """
    if model_name_or_path is None:
        if os.path.isfile(DEFAULT_LOCAL_KV_PATH):
            model_name_or_path = DEFAULT_LOCAL_KV_PATH
        else:
            model_name_or_path = DEFAULT_FALLBACK_MODEL

    # If it's a local file path that exists, load via KeyedVectors.load
    if os.path.isfile(model_name_or_path):
        return KeyedVectors.load(model_name_or_path, mmap=mmap)

    # Otherwise, download or load via gensim.downloader
    return api.load(model_name_or_path)


def compute_sentence_vector(
    tokens: List[str],
    model: KeyedVectors
) -> np.ndarray:
    """
    Computes sentence embedding as the unweighted mean of constituent word vectors.

    Parameters
    ----------
    tokens : List[str]
        Tokenized words of the sentence.
    model : KeyedVectors
        Pretrained word vector model.

    Returns
    -------
    np.ndarray
        1D embedding vector of shape (vector_size,). Returns all zeros if no tokens in vocab.
    """
    vector_dim = model.vector_size
    valid_vecs = [model[w] for w in tokens if w in model]

    if not valid_vecs:
        return np.zeros(vector_dim, dtype=np.float32)

    return np.mean(valid_vecs, axis=0).astype(np.float32)


def compute_document_embedding(sentence_vectors: List[np.ndarray]) -> np.ndarray:
    """
    Computes document embedding as the mean of its sentence vectors:
    d_i = (1 / M_i) * sum_{j=1}^{M_i} s_{ij}

    Parameters
    ----------
    sentence_vectors : List[np.ndarray]
        List of sentence embedding vectors.

    Returns
    -------
    np.ndarray
        1D document embedding vector.
    """
    if not sentence_vectors:
        return np.zeros(100, dtype=np.float32)
    return np.mean(sentence_vectors, axis=0).astype(np.float32)


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Computes cosine similarity between two vectors per base paper eq. 3:
    Sim_cos(s_ij) = (s_ij · d_i) / (||s_ij|| ||d_i||)

    Returns 0.0 if either vector has zero norm.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    dot_product = np.dot(vec_a, vec_b)
    sim = dot_product / (norm_a * norm_b)
    return float(np.clip(sim, -1.0, 1.0))


def compute_w2v_cosine_features(
    sentences: List[str],
    model: KeyedVectors,
    tokenizer: Optional[LegalTokenizer] = None
) -> np.ndarray:
    """
    Computes the Word2Vec cosine similarity feature Sim_cos(s_ij) for all sentences
    in a document.

    Parameters
    ----------
    sentences : List[str]
        List of segmented sentences for a single document.
    model : KeyedVectors
        Loaded pretrained word vector model.
    tokenizer : Optional[LegalTokenizer]
        Tokenizer to extract tokens. Defaults to LegalTokenizer.

    Returns
    -------
    np.ndarray
        1D array of cosine similarity scores of shape (len(sentences),).
    """
    if not sentences:
        return np.array([], dtype=np.float32)

    if tokenizer is None:
        tokenizer = LegalTokenizer()

    # Step 1: Compute sentence vectors
    sentence_vectors = []
    for s in sentences:
        tokens = tokenizer.tokenize_for_tfidf(s)
        s_vec = compute_sentence_vector(tokens, model)
        sentence_vectors.append(s_vec)

    # Step 2: Compute document embedding (mean of sentence vectors)
    doc_embedding = compute_document_embedding(sentence_vectors)

    # Step 3: Compute cosine similarity of each sentence to document embedding
    similarities = [
        compute_cosine_similarity(s_vec, doc_embedding)
        for s_vec in sentence_vectors
    ]

    return np.array(similarities, dtype=np.float32)
