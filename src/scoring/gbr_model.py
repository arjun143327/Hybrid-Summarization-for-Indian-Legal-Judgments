"""
gbr_model.py — GradientBoostingRegressor training, evaluation, and scoring
per 02_METHODOLOGY.md Stage 2.

Implements:
1. Deterministic document-level train/validation split (zero document leakage).
2. GradientBoostingRegressor training with MSE loss (base paper eq. 6).
3. R² and RMSE evaluation, negative R² detection, and feature importance logging.
4. Model serialization and prediction interface.
"""

import os
import json
import time
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Standard hyperparameter set for Stage 2 GBR
DEFAULT_GBR_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 4,
    "min_samples_split": 100,
    "min_samples_leaf": 50,
    "subsample": 0.8,
    "loss": "squared_error",  # MSE per Belila et al. (2026) eq. 6
    "random_state": 42,
}

CONFIG_FEATURE_NAMES: Dict[str, List[str]] = {
    "C1": ["tfidf", "ner", "position", "cosine_w2v", "wmd"],
    "C2": ["tfidf", "ner", "position", "cosine_w2v", "wmd"],
    "C3": ["tfidf", "ner", "position", "cosine_sbert"],
}


def create_document_validation_split(
    doc_boundaries: Dict[str, Dict[str, int]],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Creates a strict document-level train/val split so that all sentences from
    a single case remain exclusively in either train or val (zero case leakage).

    Parameters
    ----------
    doc_boundaries : Dict[str, Dict[str, int]]
        Mapping of doc_id -> {"start_idx": ..., "end_idx": ..., "num_sentences": ...}.
    val_ratio : float
        Fraction of documents to hold out for validation (default: 0.15 = 1,054 docs).
    seed : int
        Random seed for reproducible document shuffling (default: 42).

    Returns
    -------
    train_indices : np.ndarray
        Array of row indices for training sentences.
    val_indices : np.ndarray
        Array of row indices for validation sentences.
    train_doc_ids : List[str]
        List of doc IDs in the training partition.
    val_doc_ids : List[str]
        List of doc IDs in the validation partition.
    """
    doc_ids = sorted(list(doc_boundaries.keys()))
    rng = np.random.RandomState(seed)
    shuffled_docs = rng.permutation(doc_ids).tolist()

    n_val = int(len(shuffled_docs) * val_ratio)
    val_doc_set = set(shuffled_docs[:n_val])
    train_doc_set = set(shuffled_docs[n_val:])

    train_indices = []
    val_indices = []

    for cid in doc_ids:
        b = doc_boundaries[cid]
        idx_range = list(range(b["start_idx"], b["end_idx"]))
        if cid in val_doc_set:
            val_indices.extend(idx_range)
        else:
            train_indices.extend(idx_range)

    train_indices_arr = np.array(train_indices, dtype=np.int64)
    val_indices_arr = np.array(val_indices, dtype=np.int64)
    train_doc_ids = sorted(list(train_doc_set))
    val_doc_ids = sorted(list(val_doc_set))

    return train_indices_arr, val_indices_arr, train_doc_ids, val_doc_ids


def train_gbr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    params: Optional[Dict[str, Any]] = None,
    config_name: str = "C1",
) -> Tuple[GradientBoostingRegressor, Dict[str, Any]]:
    """
    Trains a GradientBoostingRegressor model and evaluates train/val metrics.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target labels.
    X_val : np.ndarray
        Validation feature matrix.
    y_val : np.ndarray
        Validation target labels.
    feature_names : List[str]
        List of feature names corresponding to columns of X.
    params : Optional[Dict[str, Any]]
        Hyperparameters for GradientBoostingRegressor.
    config_name : str
        Configuration name (e.g. 'C1', 'C2', 'C3').

    Returns
    -------
    model : GradientBoostingRegressor
        Fitted model.
    metrics : Dict[str, Any]
        Dictionary of evaluation scores, feature importances, and diagnostics.
    """
    if params is None:
        params = dict(DEFAULT_GBR_PARAMS)

    print(f"\n--- Training GBR for Config {config_name} ---")
    print(f"Features: {feature_names}")
    print(f"Hyperparameters: {params}")
    print(f"X_train shape: {X_train.shape} | X_val shape: {X_val.shape}")

    t0 = time.time()
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    fit_duration = time.time() - t0

    print(f"Fitting completed in {fit_duration:.2f}s ({fit_duration / 60:.2f} min)")

    # Predictions
    t_eval = time.time()
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    eval_duration = time.time() - t_eval

    train_r2 = float(r2_score(y_train, y_train_pred))
    val_r2 = float(r2_score(y_val, y_val_pred))
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
    train_mae = float(mean_absolute_error(y_train, y_train_pred))
    val_mae = float(mean_absolute_error(y_val, y_val_pred))

    # Feature importances
    raw_importances = model.feature_importances_
    feat_importances = {
        name: float(imp)
        for name, imp in sorted(
            zip(feature_names, raw_importances), key=lambda x: x[1], reverse=True
        )
    }

    # Flag negative training R²
    is_negative_train_r2 = train_r2 < 0.0
    if is_negative_train_r2:
        print(f"  [ALERT] Negative training R² detected: {train_r2:.4f}!")
    else:
        print(f"  Train R²: {train_r2:.4f} | Train RMSE: {train_rmse:.4f} | Train MAE: {train_mae:.4f}")
        print(f"  Val R²:   {val_r2:.4f} | Val RMSE:   {val_rmse:.4f} | Val MAE:   {val_mae:.4f}")

    print("  Feature Importances (Ranked):")
    for rank, (fname, imp) in enumerate(feat_importances.items(), 1):
        print(f"    {rank}. {fname:<14} : {imp:.4f} ({imp * 100:.1f}%)")

    metrics = {
        "config": config_name,
        "features": feature_names,
        "hyperparameters": params,
        "n_train_samples": int(len(y_train)),
        "n_val_samples": int(len(y_val)),
        "fit_time_seconds": round(fit_duration, 2),
        "eval_time_seconds": round(eval_duration, 2),
        "train_r2": train_r2,
        "val_r2": val_r2,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "negative_train_r2_flag": is_negative_train_r2,
        "feature_importances": feat_importances,
    }

    return model, metrics


def save_gbr_model(model: GradientBoostingRegressor, save_path: str) -> None:
    """Saves fitted GBR model to pickle file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {save_path} ({os.path.getsize(save_path) / 1e6:.2f} MB)")


def load_gbr_model(load_path: str) -> GradientBoostingRegressor:
    """Loads fitted GBR model from pickle file."""
    with open(load_path, "rb") as f:
        model = pickle.load(f)
    return model


def predict_sentence_importance(
    model: GradientBoostingRegressor,
    X_scaled: np.ndarray
) -> np.ndarray:
    """
    Computes predicted importance scores ŷ_ij = f_θ(x_ij) for sentences.

    Parameters
    ----------
    model : GradientBoostingRegressor
        Trained GBR model.
    X_scaled : np.ndarray
        Scaled feature matrix of shape (M_i, d).

    Returns
    -------
    scores : np.ndarray
        1D array of predicted sentence importance scores of shape (M_i,).
    """
    if len(X_scaled) == 0:
        return np.array([], dtype=np.float32)
    preds = model.predict(X_scaled)
    return preds.astype(np.float32)
