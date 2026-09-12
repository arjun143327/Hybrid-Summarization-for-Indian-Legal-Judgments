"""
build_features.py — Feature matrix assembly and RobustScaler preprocessing per config
per 02_METHODOLOGY.md Stage 1 and 04_TASKS.md Phase 3.

Configs:
- C1 (Base replica): x_ij = [TF-IDF, NER, Position, Cosine_w2v, WMD]
- C2 (Redundancy-only swap): x_ij = [TF-IDF, NER, Position, Cosine_w2v, WMD]   (identical to C1)
- C3 (Proposed system): x_ij = [TF-IDF, NER, Position, Cosine_sbert]          (WMD dropped, SBERT cosine)

Scaling:
Per 04_TASKS.md Phase 3 guidance, feature values include outliers from merged/under-split
long sentences (e.g. Case 914 Sentence 2, NER count 28). RobustScaler (center on median,
scale by IQR) is applied across feature columns to normalize scale and prevent extreme
outliers from distorting GradientBoostingRegressor training.
"""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.features.tfidf import compute_sentence_tfidf_scores
from src.features.position import compute_position_features
from src.features.ner import compute_sentence_ner_counts
from src.features.embeddings_w2v import compute_w2v_cosine_features
from src.features.embeddings_sbert import compute_sbert_cosine_features
from src.features.wmd import compute_document_wmd_features
from src.data.preprocessing import LegalTokenizer

CONFIG_FEATURE_NAMES = {
    "C1": ["tfidf", "ner", "position", "cosine_w2v", "wmd"],
    "C2": ["tfidf", "ner", "position", "cosine_w2v", "wmd"],
    "C3": ["tfidf", "ner", "position", "cosine_sbert"],
}


def assemble_document_raw_features(
    sentences: List[str],
    config: str,
    tfidf_vectorizer,
    w2v_model=None,
    sbert_model=None,
    tokenizer: Optional[LegalTokenizer] = None,
) -> np.ndarray:
    """
    Extracts and concatenates the raw feature matrix for a single document's sentences:
    X_i in R^{M_i x D}

    Parameters
    ----------
    sentences : List[str]
        List of segmented sentences for the document.
    config : str
        One of 'C1', 'C2', 'C3'.
    tfidf_vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    w2v_model : KeyedVectors, optional
        Loaded Word2Vec model (required for C1 and C2).
    sbert_model : SentenceTransformer, optional
        Loaded SBERT model (required for C3).
    tokenizer : LegalTokenizer, optional
        Tokenizer instance.

    Returns
    -------
    np.ndarray
        Raw feature matrix of shape (len(sentences), n_features).
    """
    if config not in CONFIG_FEATURE_NAMES:
        raise ValueError(f"Unknown config: {config}. Must be one of {list(CONFIG_FEATURE_NAMES.keys())}")

    if not sentences:
        n_feats = len(CONFIG_FEATURE_NAMES[config])
        return np.empty((0, n_feats), dtype=np.float32)

    if tokenizer is None:
        tokenizer = LegalTokenizer()

    # Core shared features: TF-IDF, NER, Position
    tfidf_vals = np.array(compute_sentence_tfidf_scores(sentences, tfidf_vectorizer), dtype=np.float32)
    ner_vals = np.array(compute_sentence_ner_counts(sentences), dtype=np.float32)
    pos_vals = np.array(compute_position_features(sentences), dtype=np.float32)

    cols = [tfidf_vals, ner_vals, pos_vals]

    if config in ("C1", "C2"):
        if w2v_model is None:
            raise ValueError(f"w2v_model is required for config {config}")
        cos_w2v = compute_w2v_cosine_features(sentences, w2v_model, tokenizer=tokenizer)
        wmd_vals = compute_document_wmd_features(sentences, w2v_model, tokenizer=tokenizer)
        cols.extend([cos_w2v, wmd_vals])

    elif config == "C3":
        if sbert_model is None:
            raise ValueError(f"sbert_model is required for config {config}")
        cos_sbert = compute_sbert_cosine_features(sentences, sbert_model)
        cols.append(cos_sbert)

    # Stack column-wise into (M_i, n_features) matrix
    X_raw = np.column_stack(cols).astype(np.float32)
    return X_raw


class FeatureScalerPipeline:
    """
    Manages feature scaling using RobustScaler per the Phase 3 design specification.
    Fits strictly on the training set feature matrix to prevent data leakage,
    and applies the learned medians and IQRs to validation and test feature matrices.
    """

    def __init__(
        self,
        config: str = "C1",
        scaler_type: str = "robust",
        quantile_range: Tuple[float, float] = (25.0, 75.0),
    ):
        self.config = config
        self.feature_names = CONFIG_FEATURE_NAMES.get(config, [])
        self.scaler_type = scaler_type
        self.quantile_range = quantile_range

        if scaler_type == "robust":
            self.scaler = RobustScaler(quantile_range=quantile_range, with_centering=True, with_scaling=True)
        elif scaler_type == "standard":
            self.scaler = StandardScaler(with_mean=True, with_std=True)
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")

        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """
        Fits the scaler on the feature matrix X.
        """
        self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies learned scaling to feature matrix X.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureScalerPipeline must be fitted on training data before calling transform().")
        return self.scaler.transform(X).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fits scaler on X and transforms in a single call.
        """
        self.fit(X)
        return self.transform(X)

    def get_scaling_params(self) -> Dict[str, Dict[str, float]]:
        """
        Returns the center (median/mean) and scale (IQR/std) learned for each feature column.
        """
        if not self.is_fitted:
            return {}

        params = {}
        for idx, feat_name in enumerate(self.feature_names):
            center = float(self.scaler.center_[idx]) if hasattr(self.scaler, "center_") else 0.0
            scale = float(self.scaler.scale_[idx]) if hasattr(self.scaler, "scale_") else 1.0
            params[feat_name] = {
                "center": center,
                "scale": scale,
                "metric_center": "median" if self.scaler_type == "robust" else "mean",
                "metric_scale": "IQR" if self.scaler_type == "robust" else "std",
            }
        return params
