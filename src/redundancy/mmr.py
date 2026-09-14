"""
mmr.py — Maximal Marginal Relevance (MMR) ranking over SBERT embeddings
per 02_METHODOLOGY.md (used in configs C2 and C3).

Implements greedy MMR:
  MMR(s_ij) = lambda * Relevance(s_ij) - (1 - lambda) * max_{s_ik in Selected} CosineSim(s_ij, s_ik)

where:
  - Relevance(s_ij) is the GBR predicted importance score y_hat_ij
  - CosineSim(s_ij, s_ik) is computed over SBERT embeddings
  - Selection proceeds greedily until the sentence budget k_i is reached.
"""

from typing import List, Dict, Any, Optional
import numpy as np


def mmr_select(
    scores: np.ndarray,
    embeddings: np.ndarray,
    k_i: int,
    lam: float = 0.7
) -> Dict[str, Any]:
    """
    Greedy Maximal Marginal Relevance (MMR) sentence selection.

    Parameters
    ----------
    scores : np.ndarray
        1D array of predicted GBR importance scores (length M_i).
    embeddings : np.ndarray
        2D array of SBERT sentence embeddings (shape M_i, D).
    k_i : int
        Sentence budget for this document.
    lam : float
        MMR lambda hyperparameter in [0, 1].
        lam=1.0 is pure relevance ranking; lam=0.0 maximally penalizes similarity.

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
    """
    m_i = len(scores)
    k_i = min(k_i, m_i)
    if k_i <= 0 or m_i == 0:
        return {
            "selected_indices": [],
            "ordered_indices": [],
            "selected_count": 0,
            "k_i": k_i,
            "budget_fulfillment": 0.0 if k_i > 0 else 1.0,
            "under_filled": k_i > 0,
        }

    # Normalize embeddings to unit L2 norm for fast dot-product cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    norm_embeds = embeddings / norms

    # Precompute pairwise cosine similarity matrix (M_i x M_i)
    sim_matrix = np.dot(norm_embeds, norm_embeds.T)
    # Clip for numerical stability
    np.clip(sim_matrix, -1.0, 1.0, out=sim_matrix)

    selected_indices: List[int] = []
    remaining_mask = np.ones(m_i, dtype=bool)

    # Step 1: select sentence with highest predicted importance score
    first_idx = int(np.argmax(scores))
    selected_indices.append(first_idx)
    remaining_mask[first_idx] = False

    if k_i == 1:
        return {
            "selected_indices": selected_indices,
            "ordered_indices": selected_indices,
            "selected_count": 1,
            "k_i": k_i,
            "budget_fulfillment": 1.0,
            "under_filled": False,
        }

    # Vector tracking maximum similarity to any currently selected sentence
    max_sim_to_selected = sim_matrix[first_idx].copy()

    # Greedy iterations for steps 2 .. k_i
    for _ in range(1, k_i):
        # Candidates are the unselected sentences
        cand_indices = np.where(remaining_mask)[0]
        if len(cand_indices) == 0:
            break

        cand_scores = scores[cand_indices]
        cand_max_sim = max_sim_to_selected[cand_indices]

        # MMR score calculation
        mmr_scores = lam * cand_scores - (1.0 - lam) * cand_max_sim

        # Pick candidate with highest MMR score
        best_cand_pos = int(np.argmax(mmr_scores))
        best_idx = int(cand_indices[best_cand_pos])

        selected_indices.append(best_idx)
        remaining_mask[best_idx] = False

        # Update maximum similarity vector with newly selected sentence
        np.maximum(max_sim_to_selected, sim_matrix[best_idx], out=max_sim_to_selected)

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
    }
