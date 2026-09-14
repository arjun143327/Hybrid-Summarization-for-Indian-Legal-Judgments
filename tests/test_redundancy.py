"""
test_redundancy.py — Unit tests for Phase 4 Redundancy Control modules:
- wmd_threshold_filter (C1)
- mmr_select (C2/C3)
- compute_self_bleu & redundancy metrics
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath("."))
from src.redundancy.mmr import mmr_select
from src.evaluation.redundancy_metrics import (
    compute_self_bleu,
    compute_distinct_ngrams,
    evaluate_summary_redundancy,
)


def test_mmr_select_pure_relevance():
    # 4 sentences, lam=1.0 should strictly pick in descending order of scores
    scores = np.array([0.2, 0.9, 0.1, 0.7])
    # Dummy embeddings (orthogonal)
    embeds = np.eye(4)
    k_i = 3
    res = mmr_select(scores, embeds, k_i, lam=1.0)
    assert res["selected_count"] == 3
    assert res["selected_indices"] == [1, 3, 0]
    assert res["ordered_indices"] == [0, 1, 3]
    assert res["budget_fulfillment"] == 1.0
    assert not res["under_filled"]


def test_mmr_select_diversity_penalty():
    # Sentence 0 and 1 have identical embeddings, sentence 2 is distinct
    scores = np.array([0.9, 0.85, 0.7])
    embeds = np.array([
        [1.0, 0.0],
        [1.0, 0.0],  # Duplicate of 0
        [0.0, 1.0],  # Orthogonal to 0
    ])
    # With lam=0.3 (heavy diversity penalty), sentence 2 should beat sentence 1 for second slot
    res = mmr_select(scores, embeds, k_i=2, lam=0.3)
    assert res["selected_indices"][0] == 0
    assert res["selected_indices"][1] == 2  # picks diverse sentence 2 over duplicate 1


def test_self_bleu_identical_vs_distinct():
    # Identical sentences -> high self-BLEU
    rep_sents = [
        "The appellant was convicted under section 302 of the IPC.",
        "The appellant was convicted under section 302 of the IPC.",
    ]
    sb_rep = compute_self_bleu(rep_sents, n_gram=2)
    assert sb_rep > 0.8

    # Distinct sentences -> low self-BLEU
    div_sents = [
        "The appellant was convicted under section 302 of the Indian Penal Code.",
        "Revenue taxes were properly assessed by the municipal commissioner.",
    ]
    sb_div = compute_self_bleu(div_sents, n_gram=2)
    assert sb_div < sb_rep


def test_evaluate_summary_redundancy():
    sents = [
        "High Court dismissed the appeal filed by the defendant.",
        "The judgment of the trial court was confirmed accordingly.",
        "Special leave petition was subsequently rejected.",
    ]
    stats = evaluate_summary_redundancy(sents, k_i=3)
    assert stats["selected_count"] == 3
    assert stats["k_i"] == 3
    assert stats["budget_fulfillment"] == 1.0
    assert not stats["under_filled"]
    assert 0.0 <= stats["self_bleu_2"] <= 1.0
    assert 0.0 <= stats["distinct_2"] <= 1.0
    assert stats["token_count"] > 0
