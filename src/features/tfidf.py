"""
tfidf.py — TF-IDF vectorizer and sentence scoring per 02_METHODOLOGY.md & 01_DATA_SPEC.md.

Implementation rules:
- Fit strictly on the training set (never on test).
- Tokenizer: lowercased, standard NLTK English stopwords (legal terms like 'held',
  'appellant', 'respondent' are explicitly retained), WordNet lemmatization.
- Computes TF-IDF importance per sentence:
  Sum of TF-IDF weights of sentence terms normalized by sentence length (or mean weight).
"""

import os
import re
import pickle
from typing import List, Optional, Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.data.loader import load_split_ids, load_document


class LegalTFIDFTokenizer:
    """
    Tokenizer tailored for legal text TF-IDF computation:
    - Standard NLTK English stopwords filtered out.
    - Crucial legal procedural / load-bearing words ('held', 'appellant', 'respondent',
      'petitioner', 'order', 'court') are preserved.
    - WordNet lemmatization with memoization for fast processing.
    """

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self._memo: Dict[str, str] = {}
        self._word_pat = re.compile(r"\b[a-zA-Z]{2,}\b")

    def _lemmatize(self, word: str) -> str:
        if word in self._memo:
            return self._memo[word]
        res = self.lemmatizer.lemmatize(word)
        self._memo[word] = res
        return res

    def __call__(self, text: str) -> List[str]:
        if not text:
            return []
        words = self._word_pat.findall(text.lower())
        return [
            self._lemmatize(w)
            for w in words
            if w not in self.stop_words
        ]


def fit_tfidf_vectorizer(
    train_ids: Optional[List[str]] = None,
    max_features: int = 50000,
    min_df: int = 5,
    max_df: float = 0.85,
    cache_path: str = os.path.join("data", "processed", "features", "tfidf_vectorizer.pkl"),
    verbose: bool = True,
) -> TfidfVectorizer:
    """
    Fits a TfidfVectorizer strictly on documents in the training set.
    Caches the fitted vectorizer to cache_path.
    """
    if os.path.exists(cache_path):
        if verbose:
            print(f"Loading cached TF-IDF vectorizer from {cache_path}...")
        with open(cache_path, "rb") as f:
            vectorizer = pickle.load(f)
        return vectorizer

    if train_ids is None:
        train_ids = load_split_ids("train")

    if verbose:
        print(f"Fitting TF-IDF vectorizer on {len(train_ids)} train documents...")

    tokenizer = LegalTFIDFTokenizer()
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )

    def train_corpus_generator():
        for idx, cid in enumerate(train_ids, 1):
            if verbose and (idx % 1000 == 0 or idx == len(train_ids)):
                print(f"Reading train doc {idx}/{len(train_ids)}...")
            doc = load_document(cid)
            yield doc["judgment"]

    vectorizer.fit(train_corpus_generator())

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(vectorizer, f)

    if verbose:
        print(f"Fitted vocabulary size: {len(vectorizer.vocabulary_):,}")
        print(f"Cached vectorizer to {cache_path}")

    return vectorizer


def compute_sentence_tfidf_scores(
    sentences: List[str],
    vectorizer: TfidfVectorizer,
) -> List[float]:
    """
    Computes a scalar TF-IDF score for each sentence in a document.
    Per standard extractive feature extraction:
    Sum of TF-IDF weights for terms present in the sentence divided by sentence word count.
    """
    if not sentences:
        return []

    tfidf_matrix = vectorizer.transform(sentences)
    scores = []
    for i, sent in enumerate(sentences):
        words = [w for w in sent.split() if len(w) > 1]
        n_words = max(len(words), 1)
        row = tfidf_matrix.getrow(i)
        row_sum = float(row.sum())
        scores.append(row_sum / n_words)

    return scores


def get_top_tfidf_terms_for_document(
    document_text: str,
    vectorizer: TfidfVectorizer,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Helper function to inspect the top-K highest TF-IDF terms in a document.
    """
    tfidf_vec = vectorizer.transform([document_text])
    feature_names = vectorizer.get_feature_names_out()
    coo = tfidf_vec.tocoo()
    sorted_items = sorted(zip(coo.col, coo.data), key=lambda x: x[1], reverse=True)

    top_terms = [
        (feature_names[idx], round(float(score), 4))
        for idx, score in sorted_items[:top_k]
    ]
    return top_terms
