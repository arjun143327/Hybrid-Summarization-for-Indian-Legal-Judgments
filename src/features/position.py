"""
position.py — Sentence position feature extraction per 02_METHODOLOGY.md.

Definition:
For a document d_i = {s_i1, ..., s_iM}:
Normalized position pos_ij = j / M_i
where j in {0, ..., M_i - 1} is the 0-indexed sentence rank.
- First sentence: pos_i0 = 0.0
- Last sentence: pos_i(M-1) = (M-1) / M_i (approaches 1.0)
"""

from typing import List


def compute_position_features(sentences: List[str]) -> List[float]:
    """
    Computes normalized sentence position feature pos_ij = j / M_i.
    
    Args:
        sentences: List of segmented sentences for document i.
        
    Returns:
        List of floats representing normalized position for each sentence.
        If document has 1 sentence, returns [0.0].
        If document is empty, returns [].
    """
    m = len(sentences)
    if m == 0:
        return []
    if m == 1:
        return [0.0]

    # pos_ij = j / M_i per spec
    return [round(j / float(m), 6) for j in range(m)]
