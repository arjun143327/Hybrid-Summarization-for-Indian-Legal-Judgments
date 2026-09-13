"""
eval_stratified_by_length.py
Stratify Phase 3 rank-correlation metrics by document-length bucket:
  - Short:  M_i < 20 sentences
  - Medium: 20 <= M_i <= 500 sentences
  - Long:   M_i > 500 sentences
Evaluates C1 (replica) and C3 (proposed SBERT) on the validation split (seed=42, 15%).
Investigates score compression and ranking behavior in long documents vs short/medium.

Outputs:
  results/logs/stratified_length_report.json
  results/logs/stratified_length_report.md
"""

import os
import sys
import json
import pickle
import time
import numpy as np
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, os.path.abspath("."))
from src.scoring.gbr_model import create_document_validation_split


def ki_budget(m: int) -> int:
    return max(3, round(0.05 * m))


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 1.0


def recall_pct(true_set: set, pred_set: set) -> float:
    if not true_set:
        return 1.0
    return len(true_set & pred_set) / len(true_set)


def topk_sets(y_true, y_pred, k):
    k = min(k, len(y_true))
    ts = set(int(x) for x in np.argsort(y_true)[::-1][:k])
    ps = set(int(x) for x in np.argsort(y_pred)[::-1][:k])
    return ts, ps


def simulate_chance(m, k, n_trials=100, seed=0):
    rng = np.random.RandomState(seed)
    oracle = set(range(k))
    jacs, recs = [], []
    for _ in range(n_trials):
        rand_pred = set(rng.choice(m, size=min(k, m), replace=False).tolist())
        u = oracle | rand_pred
        jacs.append(len(oracle & rand_pred) / len(u) if u else 1.0)
        recs.append(len(oracle & rand_pred) / len(oracle) if oracle else 1.0)
    return float(np.mean(jacs)), float(np.mean(recs))


def load_model(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    print("=" * 75)
    print("STRATIFIED RANK-CORRELATION & SCORE-COMPRESSION EVALUATION")
    print("=" * 75)

    feat_dir = os.path.join("data", "processed", "features")
    logs_dir = os.path.join("results", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # 1. Load data
    print("\n[1/5] Loading feature matrices and labels...")
    X_c1 = np.load(os.path.join(feat_dir, "train_features_c1_scaled.npy"))
    X_c3 = np.load(os.path.join(feat_dir, "train_features_c3_scaled.npy"))
    y_all = np.load(os.path.join(feat_dir, "train_labels.npy"))
    with open(os.path.join(feat_dir, "train_doc_boundaries.json")) as fh:
        doc_boundaries = json.load(fh)

    # 2. Recreate validation split (seed=42, 15%)
    print("\n[2/5] Recreating val split (seed=42, 15%)...")
    _, val_idx, _, val_doc_ids = create_document_validation_split(
        doc_boundaries, val_ratio=0.15, seed=42
    )

    doc_local = {}
    cursor = 0
    for did in sorted(val_doc_ids):
        b = doc_boundaries[did]
        n = b["end_idx"] - b["start_idx"]
        doc_local[did] = (cursor, cursor + n)
        cursor += n

    X_c1_val = X_c1[val_idx]
    X_c3_val = X_c3[val_idx]
    y_val = y_all[val_idx]
    print(f"  Val set: {len(val_doc_ids)} docs, {len(val_idx):,} sentences")

    # 3. Predict with models
    print("\n[3/5] Generating predictions for C1 and C3...")
    model_c1 = load_model(os.path.join("models", "gbr_c1.pkl"))
    model_c3 = load_model(os.path.join("models", "gbr_c3.pkl"))
    yp_c1 = model_c1.predict(X_c1_val)
    yp_c3 = model_c3.predict(X_c3_val)

    # 4. Stratify documents
    print("\n[4/5] Evaluating metrics per document and per bucket...")
    buckets = {
        "short": {"name": "Short (<20)", "filter": lambda m: m < 20, "docs": []},
        "medium": {"name": "Medium (20-500)", "filter": lambda m: 20 <= m <= 500, "docs": []},
        "long": {"name": "Long (>500)", "filter": lambda m: m > 500, "docs": []},
    }

    doc_results = []
    rng_seed = 1234

    for did in sorted(val_doc_ids):
        lo, hi = doc_local[did]
        m = hi - lo
        k = ki_budget(m)
        yd = y_val[lo:hi]
        yp1 = yp_c1[lo:hi]
        yp3 = yp_c3[lo:hi]

        # Overlaps
        ts1, ps1 = topk_sets(yd, yp1, k)
        ts3, ps3 = topk_sets(yd, yp3, k)
        jac1 = jaccard(ts1, ps1)
        rec1 = recall_pct(ts1, ps1)
        jac3 = jaccard(ts3, ps3)
        rec3 = recall_pct(ts3, ps3)

        # Chance baseline
        cj, cr = simulate_chance(m, k, n_trials=50, seed=(rng_seed + int(did)) % 2**31)

        # Rank correlations
        if m >= 2:
            sr1, _ = spearmanr(yd, yp1)
            kt1, _ = kendalltau(yd, yp1)
            sr3, _ = spearmanr(yd, yp3)
            kt3, _ = kendalltau(yd, yp3)
            sr1 = float(sr1) if np.isfinite(sr1) else 0.0
            kt1 = float(kt1) if np.isfinite(kt1) else 0.0
            sr3 = float(sr3) if np.isfinite(sr3) else 0.0
            kt3 = float(kt3) if np.isfinite(kt3) else 0.0
        else:
            sr1, kt1, sr3, kt3 = 0.0, 0.0, 0.0, 0.0

        # Score distributions
        true_range = float(np.ptp(yd))
        true_max = float(np.max(yd))
        true_min = float(np.min(yd))
        true_mean = float(np.mean(yd))
        true_std = float(np.std(yd))

        # C1 predicted distribution
        c1_range = float(np.ptp(yp1))
        c1_max = float(np.max(yp1))
        c1_min = float(np.min(yp1))
        c1_std = float(np.std(yp1))

        # C3 predicted distribution
        c3_range = float(np.ptp(yp3))
        c3_max = float(np.max(yp3))
        c3_min = float(np.min(yp3))
        c3_std = float(np.std(yp3))

        # Ranking diagnostics: where do high-true-value sentences land in predicted rank?
        # Predicted ranking order (0-indexed rank: 0 is highest predicted score)
        c1_pred_order = np.argsort(yp1)[::-1]
        c3_pred_order = np.argsort(yp3)[::-1]
        # Invert to get rank of each sentence index (0 = rank 1, etc.)
        c1_ranks = np.empty_like(c1_pred_order)
        c1_ranks[c1_pred_order] = np.arange(m) + 1  # 1-indexed rank
        c3_ranks = np.empty_like(c3_pred_order)
        c3_ranks[c3_pred_order] = np.arange(m) + 1  # 1-indexed rank

        # True top-1 index and rank in pred
        true_top1_idx = int(np.argmax(yd))
        c1_true_top1_pred_rank = int(c1_ranks[true_top1_idx])
        c3_true_top1_pred_rank = int(c3_ranks[true_top1_idx])
        c1_true_top1_pct_rank = c1_true_top1_pred_rank / m
        c3_true_top1_pct_rank = c3_true_top1_pred_rank / m

        # True top-k indices: average rank in pred
        true_topk_indices = list(ts1)  # ts1 is true topk
        c1_avg_rank_of_true_topk = float(np.mean([c1_ranks[idx] for idx in true_topk_indices]))
        c3_avg_rank_of_true_topk = float(np.mean([c3_ranks[idx] for idx in true_topk_indices]))
        c1_pct_rank_of_true_topk = c1_avg_rank_of_true_topk / m
        c3_pct_rank_of_true_topk = c3_avg_rank_of_true_topk / m

        # High-true sentences (y >= 0.8)
        high_true_mask = yd >= 0.8
        n_high_true = int(np.sum(high_true_mask))
        if n_high_true > 0:
            c1_high_true_ranks = [int(c1_ranks[i]) for i in np.where(high_true_mask)[0]]
            c3_high_true_ranks = [int(c3_ranks[i]) for i in np.where(high_true_mask)[0]]
            c1_mean_high_true_rank = float(np.mean(c1_high_true_ranks))
            c3_mean_high_true_rank = float(np.mean(c3_high_true_ranks))
            c1_pct_high_true_rank = c1_mean_high_true_rank / m
            c3_pct_high_true_rank = c3_mean_high_true_rank / m
            c1_high_in_topk = int(sum(r <= k for r in c1_high_true_ranks))
            c3_high_in_topk = int(sum(r <= k for r in c3_high_true_ranks))
        else:
            c1_mean_high_true_rank = None
            c3_mean_high_true_rank = None
            c1_pct_high_true_rank = None
            c3_pct_high_true_rank = None
            c1_high_in_topk = None
            c3_high_in_topk = None

        doc_info = {
            "doc_id": did,
            "M_i": m,
            "k_i": k,
            "chance_jaccard": cj,
            "chance_recall": cr,
            "true_min": true_min,
            "true_max": true_max,
            "true_range": true_range,
            "true_mean": true_mean,
            "true_std": true_std,
            "n_high_true": n_high_true,
            "c1": {
                "spearman": sr1,
                "kendall": kt1,
                "jaccard": jac1,
                "recall": rec1,
                "pred_min": c1_min,
                "pred_max": c1_max,
                "pred_range": c1_range,
                "pred_std": c1_std,
                "true_top1_pred_rank": c1_true_top1_pred_rank,
                "true_top1_pct_rank": c1_true_top1_pct_rank,
                "avg_rank_of_true_topk": c1_avg_rank_of_true_topk,
                "pct_rank_of_true_topk": c1_pct_rank_of_true_topk,
                "mean_high_true_rank": c1_mean_high_true_rank,
                "pct_high_true_rank": c1_pct_high_true_rank,
                "high_in_topk": c1_high_in_topk,
            },
            "c3": {
                "spearman": sr3,
                "kendall": kt3,
                "jaccard": jac3,
                "recall": rec3,
                "pred_min": c3_min,
                "pred_max": c3_max,
                "pred_range": c3_range,
                "pred_std": c3_std,
                "true_top1_pred_rank": c3_true_top1_pred_rank,
                "true_top1_pct_rank": c3_true_top1_pct_rank,
                "avg_rank_of_true_topk": c3_avg_rank_of_true_topk,
                "pct_rank_of_true_topk": c3_pct_rank_of_true_topk,
                "mean_high_true_rank": c3_mean_high_true_rank,
                "pct_high_true_rank": c3_pct_high_true_rank,
                "high_in_topk": c3_high_in_topk,
            }
        }
        doc_results.append(doc_info)

        # Categorize into bucket
        for bkey, bdef in buckets.items():
            if bdef["filter"](m):
                bdef["docs"].append(doc_info)
                break

    # 5. Aggregate statistics per bucket
    print("\n[5/5] Compiling bucket aggregations...")
    bucket_summaries = {}
    for bkey, bdef in buckets.items():
        docs = bdef["docs"]
        n_docs = len(docs)
        if n_docs == 0:
            continue
        total_sents = sum(d["M_i"] for d in docs)
        avg_m = np.mean([d["M_i"] for d in docs])
        avg_k = np.mean([d["k_i"] for d in docs])

        # Chance
        ch_jac = np.mean([d["chance_jaccard"] for d in docs])
        ch_rec = np.mean([d["chance_recall"] for d in docs])

        # Ranges
        avg_true_range = np.mean([d["true_range"] for d in docs])
        avg_c1_range = np.mean([d["c1"]["pred_range"] for d in docs])
        avg_c3_range = np.mean([d["c3"]["pred_range"] for d in docs])

        avg_c1_std = np.mean([d["c1"]["pred_std"] for d in docs])
        avg_c3_std = np.mean([d["c3"]["pred_std"] for d in docs])

        # C1 metrics
        c1_spear = np.mean([d["c1"]["spearman"] for d in docs])
        c1_spear_std = np.std([d["c1"]["spearman"] for d in docs])
        c1_kend = np.mean([d["c1"]["kendall"] for d in docs])
        c1_kend_std = np.std([d["c1"]["kendall"] for d in docs])
        c1_jac = np.mean([d["c1"]["jaccard"] for d in docs])
        c1_jac_std = np.std([d["c1"]["jaccard"] for d in docs])
        c1_rec = np.mean([d["c1"]["recall"] for d in docs])
        c1_rec_std = np.std([d["c1"]["recall"] for d in docs])
        c1_top1_pct = np.mean([d["c1"]["true_top1_pct_rank"] for d in docs])
        c1_topk_pct = np.mean([d["c1"]["pct_rank_of_true_topk"] for d in docs])

        # C3 metrics
        c3_spear = np.mean([d["c3"]["spearman"] for d in docs])
        c3_spear_std = np.std([d["c3"]["spearman"] for d in docs])
        c3_kend = np.mean([d["c3"]["kendall"] for d in docs])
        c3_kend_std = np.std([d["c3"]["kendall"] for d in docs])
        c3_jac = np.mean([d["c3"]["jaccard"] for d in docs])
        c3_jac_std = np.std([d["c3"]["jaccard"] for d in docs])
        c3_rec = np.mean([d["c3"]["recall"] for d in docs])
        c3_rec_std = np.std([d["c3"]["recall"] for d in docs])
        c3_top1_pct = np.mean([d["c3"]["true_top1_pct_rank"] for d in docs])
        c3_topk_pct = np.mean([d["c3"]["pct_rank_of_true_topk"] for d in docs])

        # High true (y >= 0.8) stats
        docs_with_high = [d for d in docs if d["n_high_true"] > 0]
        n_docs_with_high = len(docs_with_high)
        if n_docs_with_high > 0:
            c1_high_pct = np.mean([d["c1"]["pct_high_true_rank"] for d in docs_with_high])
            c3_high_pct = np.mean([d["c3"]["pct_high_true_rank"] for d in docs_with_high])
            c1_high_capture = np.sum([d["c1"]["high_in_topk"] for d in docs_with_high]) / np.sum([d["n_high_true"] for d in docs_with_high])
            c3_high_capture = np.sum([d["c3"]["high_in_topk"] for d in docs_with_high]) / np.sum([d["n_high_true"] for d in docs_with_high])
        else:
            c1_high_pct, c3_high_pct = None, None
            c1_high_capture, c3_high_capture = None, None

        bucket_summaries[bkey] = {
            "bucket_name": bdef["name"],
            "n_docs": n_docs,
            "total_sentences": total_sents,
            "avg_mi": round(float(avg_m), 2),
            "avg_ki": round(float(avg_k), 2),
            "avg_true_range": round(float(avg_true_range), 4),
            "chance": {
                "jaccard": round(float(ch_jac), 4),
                "recall": round(float(ch_rec), 4),
            },
            "c1": {
                "spearman_mean": round(float(c1_spear), 4),
                "spearman_std": round(float(c1_spear_std), 4),
                "kendall_mean": round(float(c1_kend), 4),
                "kendall_std": round(float(c1_kend_std), 4),
                "jaccard_mean": round(float(c1_jac), 4),
                "jaccard_std": round(float(c1_jac_std), 4),
                "recall_mean": round(float(c1_rec), 4),
                "recall_std": round(float(c1_rec_std), 4),
                "jaccard_vs_chance": round(float(c1_jac / ch_jac), 2) if ch_jac > 0 else None,
                "recall_vs_chance": round(float(c1_rec / ch_rec), 2) if ch_rec > 0 else None,
                "avg_pred_range": round(float(avg_c1_range), 4),
                "avg_pred_std": round(float(avg_c1_std), 4),
                "true_top1_avg_pct_rank": round(float(c1_top1_pct), 4),
                "true_topk_avg_pct_rank": round(float(c1_topk_pct), 4),
                "high_true_avg_pct_rank": round(float(c1_high_pct), 4) if c1_high_pct else None,
                "high_true_capture_rate": round(float(c1_high_capture), 4) if c1_high_capture is not None else None,
            },
            "c3": {
                "spearman_mean": round(float(c3_spear), 4),
                "spearman_std": round(float(c3_spear_std), 4),
                "kendall_mean": round(float(c3_kend), 4),
                "kendall_std": round(float(c3_kend_std), 4),
                "jaccard_mean": round(float(c3_jac), 4),
                "jaccard_std": round(float(c3_jac_std), 4),
                "recall_mean": round(float(c3_rec), 4),
                "recall_std": round(float(c3_rec_std), 4),
                "jaccard_vs_chance": round(float(c3_jac / ch_jac), 2) if ch_jac > 0 else None,
                "recall_vs_chance": round(float(c3_rec / ch_rec), 2) if ch_rec > 0 else None,
                "avg_pred_range": round(float(avg_c3_range), 4),
                "avg_pred_std": round(float(avg_c3_std), 4),
                "true_top1_avg_pct_rank": round(float(c3_top1_pct), 4),
                "true_topk_avg_pct_rank": round(float(c3_topk_pct), 4),
                "high_true_avg_pct_rank": round(float(c3_high_pct), 4) if c3_high_pct else None,
                "high_true_capture_rate": round(float(c3_high_capture), 4) if c3_high_capture is not None else None,
            },
            "n_docs_with_high_true": n_docs_with_high,
        }

    # Print summary table
    print("\n" + "=" * 90)
    print("STRATIFIED RANK-CORRELATION & OVERLAP METRICS")
    print("=" * 90)
    header = f"{'Bucket':<18} | {'Docs':<5} | {'Avg M':<6} | {'Cfg':<4} | {'Spearman':<15} | {'Kendall':<15} | {'Jaccard':<15} | {'Recall':<15} | {'J/Chance':<8}"
    print(header)
    print("-" * len(header))
    for bkey, bsum in bucket_summaries.items():
        bname = bsum["bucket_name"]
        nd = bsum["n_docs"]
        avg_m = bsum["avg_mi"]
        for cfg in ["c1", "c3"]:
            cdata = bsum[cfg]
            sp_str = f"{cdata['spearman_mean']:.4f}±{cdata['spearman_std']:.3f}"
            kd_str = f"{cdata['kendall_mean']:.4f}±{cdata['kendall_std']:.3f}"
            jc_str = f"{cdata['jaccard_mean']:.4f}±{cdata['jaccard_std']:.3f}"
            rc_str = f"{cdata['recall_mean']:.4f}±{cdata['recall_std']:.3f}"
            jc_ratio = f"{cdata['jaccard_vs_chance']:.2f}x"
            print(f"{bname:<18} | {nd:<5} | {avg_m:<6.1f} | {cfg.upper():<4} | {sp_str:<15} | {kd_str:<15} | {jc_str:<15} | {rc_str:<15} | {jc_ratio:<8}")
        print("-" * len(header))

    # Print score compression and ranking behavior
    print("\n" + "=" * 90)
    print("SCORE COMPRESSION & RANKING BEHAVIOR DIAGNOSTICS")
    print("=" * 90)
    diag_hdr = f"{'Bucket':<18} | {'Cfg':<4} | {'Avg Pred Range':<15} | {'Avg Pred Std':<13} | {'Top-1 Pct Rank':<15} | {'Top-k Pct Rank':<15} | {'High-y (>=0.8) Capt':<20}"
    print(diag_hdr)
    print("-" * len(diag_hdr))
    for bkey, bsum in bucket_summaries.items():
        bname = bsum["bucket_name"]
        for cfg in ["c1", "c3"]:
            cdata = bsum[cfg]
            pr_str = f"{cdata['avg_pred_range']:.4f}"
            ps_str = f"{cdata['avg_pred_std']:.4f}"
            t1_str = f"{cdata['true_top1_avg_pct_rank']*100:.1f}%"
            tk_str = f"{cdata['true_topk_avg_pct_rank']*100:.1f}%"
            cap_str = f"{cdata['high_true_capture_rate']*100:.1f}%" if cdata['high_true_capture_rate'] is not None else "N/A"
            print(f"{bname:<18} | {cfg.upper():<4} | {pr_str:<15} | {ps_str:<13} | {t1_str:<15} | {tk_str:<15} | {cap_str:<20}")
        print("-" * len(diag_hdr))

    # Long docs detailed examination
    long_docs = buckets["long"]["docs"]
    print(f"\n" + "=" * 90)
    print(f"LONG DOCUMENTS DEEP-DIVE ({len(long_docs)} docs with M_i > 500)")
    print("=" * 90)
    ld_hdr = f"{'Doc ID':<8} | {'M_i':<5} | {'k_i':<4} | {'C1 Spear':<9} | {'C1 Range':<9} | {'C1 Top-1 Rk':<12} | {'C1 Jac':<7} | {'C3 Spear':<9} | {'C3 Range':<9} | {'C3 Jac':<7}"
    print(ld_hdr)
    print("-" * len(ld_hdr))
    for ld in sorted(long_docs, key=lambda x: -x["M_i"]):
        did = ld["doc_id"]
        m = ld["M_i"]
        k = ld["k_i"]
        c1s = ld["c1"]["spearman"]
        c1r = ld["c1"]["pred_range"]
        c1t1 = f"{ld['c1']['true_top1_pred_rank']}/{m}"
        c1j = ld["c1"]["jaccard"]
        c3s = ld["c3"]["spearman"]
        c3r = ld["c3"]["pred_range"]
        c3j = ld["c3"]["jaccard"]
        print(f"{did:<8} | {m:<5} | {k:<4} | {c1s:<9.4f} | {c1r:<9.4f} | {c1t1:<12} | {c1j:<7.4f} | {c3s:<9.4f} | {c3r:<9.4f} | {c3j:<7.4f}")

    # Check doc 6778 specifically
    doc_6778 = next((d for d in long_docs if d["doc_id"] == 6778 or d["doc_id"] == "6778"), None)
    if doc_6778:
        print("\n--- Specimen doc_id=6778 Details ---")
        print(f"  M_i={doc_6778['M_i']}, k_i={doc_6778['k_i']}")
        print(f"  True range: [{doc_6778['true_min']:.4f}, {doc_6778['true_max']:.4f}], spread={doc_6778['true_range']:.4f}")
        print(f"  C1 pred: [{doc_6778['c1']['pred_min']:.4f}, {doc_6778['c1']['pred_max']:.4f}], spread={doc_6778['c1']['pred_range']:.4f}, std={doc_6778['c1']['pred_std']:.4f}")
        print(f"  C3 pred: [{doc_6778['c3']['pred_min']:.4f}, {doc_6778['c3']['pred_max']:.4f}], spread={doc_6778['c3']['pred_range']:.4f}, std={doc_6778['c3']['pred_std']:.4f}")
        print(f"  C1 true top-1 rank: {doc_6778['c1']['true_top1_pred_rank']}/{doc_6778['M_i']} ({doc_6778['c1']['true_top1_pct_rank']*100:.1f}%)")
        print(f"  C3 true top-1 rank: {doc_6778['c3']['true_top1_pred_rank']}/{doc_6778['M_i']} ({doc_6778['c3']['true_top1_pct_rank']*100:.1f}%)")
        print(f"  Sentences with y>=0.8: {doc_6778['n_high_true']}")
        print(f"  C1 mean rank of y>=0.8: {doc_6778['c1']['mean_high_true_rank']:.1f} ({doc_6778['c1']['pct_high_true_rank']*100:.1f}%), in top-k: {doc_6778['c1']['high_in_topk']}/{doc_6778['n_high_true']}")
        print(f"  C3 mean rank of y>=0.8: {doc_6778['c3']['mean_high_true_rank']:.1f} ({doc_6778['c3']['pct_high_true_rank']*100:.1f}%), in top-k: {doc_6778['c3']['high_in_topk']}/{doc_6778['n_high_true']}")

    # Save JSON report
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "val_docs_total": len(val_doc_ids),
        "val_sentences_total": len(val_idx),
        "bucket_summaries": bucket_summaries,
        "long_docs_detail": [
            {
                "doc_id": d["doc_id"],
                "M_i": d["M_i"],
                "k_i": d["k_i"],
                "chance_jaccard": d["chance_jaccard"],
                "true_range": d["true_range"],
                "n_high_true": d["n_high_true"],
                "c1": d["c1"],
                "c3": d["c3"],
            }
            for d in sorted(long_docs, key=lambda x: -x["M_i"])
        ]
    }
    json_path = os.path.join(logs_dir, "stratified_length_report.json")
    with open(json_path, "w") as fh:
        json.dump(report_data, fh, indent=2, cls=NumpyEncoder)
    print(f"\nSaved JSON report -> {json_path}")

    # Generate and save Markdown report
    md_path = os.path.join(logs_dir, "stratified_length_report.md")
    write_markdown_report(report_data, md_path)
    print(f"Saved MD report   -> {md_path}")


def write_markdown_report(report, out_path):
    lines = []
    a = lines.append

    a("# Phase 3 Stratified Rank-Correlation & Score Dispersion Report")
    a("")
    a(f"Date: {report['timestamp']}")
    a(f"Validation split: {report['val_docs_total']} documents, {report['val_sentences_total']:,} sentences (seed=42, 15% holdout)")
    a("")
    a("## 1. Stratified Metrics Table by Document Length Bucket")
    a("")
    a("Document length buckets:")
    a("- **Short**: $M_i < 20$ sentences")
    a("- **Medium**: $20 \\le M_i \\le 500$ sentences")
    a("- **Long**: $M_i > 500$ sentences")
    a("")
    a("| Length Bucket | Docs | Sentences | Avg $M_i$ | Avg $k_i$ | Config | Spearman $\\rho$ (mean $\\pm$ std) | Kendall $\\tau$ (mean $\\pm$ std) | Top-$k$ Jaccard (mean $\\pm$ std) | Top-$k$ Recall (mean $\\pm$ std) | Chance Jaccard | Ratio vs Chance |")
    a("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for bkey, bsum in report["bucket_summaries"].items():
        bname = bsum["bucket_name"]
        nd = bsum["n_docs"]
        ns = bsum["total_sentences"]
        avg_m = bsum["avg_mi"]
        avg_k = bsum["avg_ki"]
        ch_j = bsum["chance"]["jaccard"]

        for cfg in ["c1", "c3"]:
            cdata = bsum[cfg]
            sp = f"`{cdata['spearman_mean']:.4f} ± {cdata['spearman_std']:.3f}`"
            kd = f"`{cdata['kendall_mean']:.4f} ± {cdata['kendall_std']:.3f}`"
            jc = f"`{cdata['jaccard_mean']:.4f} ± {cdata['jaccard_std']:.3f}`"
            rc = f"`{cdata['recall_mean']:.4f} ± {cdata['recall_std']:.3f}`"
            ch = f"`{ch_j:.4f}`"
            rat = f"**`{cdata['jaccard_vs_chance']:.2f}×`**"
            cfg_label = "C1 (Replica)" if cfg == "c1" else "C3 (Proposed)"
            a(f"| **{bname}** | {nd} | {ns:,} | {avg_m:.1f} | {avg_k:.1f} | {cfg_label} | {sp} | {kd} | {jc} | {rc} | {ch} | {rat} |")

    a("")
    a("## 2. Score Compression & Dispersion Analysis")
    a("")
    a("This section directly investigates whether the score compression observed in `doc_id=6778`")
    a("($M_i=1189$, predicted scores in $[0.51, 0.59]$ vs true scores up to $1.0$) is a systematic")
    a("characteristic of long documents or an isolated artifact.")
    a("")
    a("| Length Bucket | Config | Avg True Range | Avg Pred Range | Avg Pred Std | Top-1 True Sent Avg Rank % | Top-$k$ True Sents Avg Rank % | High-y ($\ge 0.8$) in Top-$k$ % |")
    a("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for bkey, bsum in report["bucket_summaries"].items():
        bname = bsum["bucket_name"]
        tr = f"`{bsum['avg_true_range']:.4f}`"
        for cfg in ["c1", "c3"]:
            cdata = bsum[cfg]
            cfg_label = "C1" if cfg == "c1" else "C3"
            pr = f"`{cdata['avg_pred_range']:.4f}`"
            ps = f"`{cdata['avg_pred_std']:.4f}`"
            t1 = f"`{cdata['true_top1_avg_pct_rank']*100:.1f}%`"
            tk = f"`{cdata['true_topk_avg_pct_rank']*100:.1f}%`"
            cap = f"`{cdata['high_true_capture_rate']*100:.1f}%`" if cdata['high_true_capture_rate'] is not None else "N/A"
            a(f"| **{bname}** | {cfg_label} | {tr} | {pr} | {ps} | {t1} | {tk} | {cap} |")

    a("")
    a("## 3. Deep Dive on Long Documents ($M_i > 500$)")
    a("")
    a("| Doc ID | $M_i$ | $k_i$ | True Range | C1 Pred Range | C1 Spearman | C1 Jaccard | C1 True Top-1 Rank | C3 Pred Range | C3 Spearman | C3 Jaccard | C3 True Top-1 Rank |")
    a("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for ld in report["long_docs_detail"]:
        did = ld["doc_id"]
        m = ld["M_i"]
        k = ld["k_i"]
        tr = f"`{ld['true_range']:.3f}`"
        c1r = f"`{ld['c1']['pred_range']:.3f}`"
        c1s = f"`{ld['c1']['spearman']:.3f}`"
        c1j = f"`{ld['c1']['jaccard']:.4f}`"
        c1t1 = f"`{ld['c1']['true_top1_pred_rank']}/{m}` (`{ld['c1']['true_top1_pct_rank']*100:.1f}%`)"
        c3r = f"`{ld['c3']['pred_range']:.3f}`"
        c3s = f"`{ld['c3']['spearman']:.3f}`"
        c3j = f"`{ld['c3']['jaccard']:.4f}`"
        c3t1 = f"`{ld['c3']['true_top1_pred_rank']}/{m}` (`{ld['c3']['true_top1_pct_rank']*100:.1f}%`)"
        a(f"| `{did}` | {m} | {k} | {tr} | {c1r} | {c1s} | {c1j} | {c1t1} | {c3r} | {c3s} | {c3j} | {c3t1} |")

    a("")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
