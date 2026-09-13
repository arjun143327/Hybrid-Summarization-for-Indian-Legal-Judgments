"""
eval_rank_correlation.py - Rank correlation metrics for GBR models on validation split.
"""

import os
import sys
import json
import time
import pickle
import numpy as np
from scipy.stats import spearmanr, kendalltau
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath("."))
from src.scoring.gbr_model import create_document_validation_split


class NumpyEncoder(json.JSONEncoder):
    """Serialize numpy scalar types to native Python for JSON."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def ki_budget(m_i: int) -> int:
    return max(3, round(0.05 * m_i))


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 1.0


def percent_overlap(true_set: set, pred_set: set) -> float:
    if not true_set:
        return 1.0
    return len(true_set & pred_set) / len(true_set)


def topk_overlap(y_true, y_pred, k):
    k = min(k, len(y_true))
    true_topk = set(np.argsort(y_true)[::-1][:k])
    pred_topk = set(np.argsort(y_pred)[::-1][:k])
    return jaccard(true_topk, pred_topk), percent_overlap(true_topk, pred_topk), true_topk, pred_topk


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate_config(model, X_val, y_val, val_doc_ids, doc_boundaries, config_name, n_examples=3):
    print(f"\n  [{config_name}] Predicting {len(y_val):,} sentences...")
    t0 = time.time()
    y_pred_all = model.predict(X_val)

    spear_rho, spear_p = spearmanr(y_val, y_pred_all)
    kend_tau, kend_p = kendalltau(y_val, y_pred_all)
    print(f"    Global Spearman rho = {spear_rho:.4f}  (p={spear_p:.2e})")
    print(f"    Global Kendall tau  = {kend_tau:.4f}  (p={kend_p:.2e})")

    # Build local row ranges for each val doc
    doc_local_ranges = {}
    cursor = 0
    for did in sorted(val_doc_ids):
        b = doc_boundaries[did]
        n = b["end_idx"] - b["start_idx"]
        doc_local_ranges[did] = (cursor, cursor + n)
        cursor += n

    jacs, pcts, ks = [], [], []
    spear_doc, kend_doc = [], []
    examples = []
    example_pool = sorted(val_doc_ids)[:n_examples]

    for did in sorted(val_doc_ids):
        lo, hi = doc_local_ranges[did]
        yd = y_val[lo:hi]
        yp = y_pred_all[lo:hi]
        m = hi - lo
        k = ki_budget(m)
        ks.append(k)

        jac, pct, true_set, pred_set = topk_overlap(yd, yp, k)
        jacs.append(jac)
        pcts.append(pct)

        if m >= 2:
            sr, _ = spearmanr(yd, yp)
            kt, _ = kendalltau(yd, yp)
            spear_doc.append(float(sr) if np.isfinite(sr) else 0.0)
            kend_doc.append(float(kt) if np.isfinite(kt) else 0.0)

        if did in example_pool:
            examples.append({
                "doc_id": did,
                "M_i": int(m),
                "k_i": int(k),
                "true_topk": [int(x) for x in sorted(true_set)],
                "pred_topk": [int(x) for x in sorted(pred_set)],
                "overlap": [int(x) for x in sorted(true_set & pred_set)],
                "jaccard": round(jac, 4),
                "pct_recall": round(pct, 4),
                "true_scores_at_true": {str(int(s)): round(float(yd[s]), 4) for s in sorted(true_set)},
                "pred_scores_at_pred": {str(int(s)): round(float(yp[s]), 4) for s in sorted(pred_set)},
                "true_scores_at_pred": {str(int(s)): round(float(yd[s]), 4) for s in sorted(pred_set)},
            })

    r = {
        "config": config_name,
        "n_val_docs": len(val_doc_ids),
        "n_val_sentences": len(y_val),
        "global_spearman_rho": round(float(spear_rho), 6),
        "global_spearman_pvalue": float(spear_p),
        "global_kendall_tau": round(float(kend_tau), 6),
        "global_kendall_pvalue": float(kend_p),
        "per_doc_spearman_mean": round(float(np.nanmean(spear_doc)), 6),
        "per_doc_kendall_mean": round(float(np.nanmean(kend_doc)), 6),
        "topk_jaccard_mean": round(float(np.mean(jacs)), 6),
        "topk_jaccard_std": round(float(np.std(jacs)), 6),
        "topk_recall_mean": round(float(np.mean(pcts)), 6),
        "avg_ki": round(float(np.mean(ks)), 2),
        "examples": examples,
    }
    print(f"    Jaccard mean={r['topk_jaccard_mean']:.4f}  std={r['topk_jaccard_std']:.4f}  recall={r['topk_recall_mean']:.4f}")
    print(f"    Per-doc Spearman={r['per_doc_spearman_mean']:.4f}  Kendall={r['per_doc_kendall_mean']:.4f}")
    print(f"    Done in {time.time()-t0:.1f}s")
    return r


def main():
    print("="*80)
    print("RANK CORRELATION EVALUATION — GBR Models (C1, C2, C3)")
    print("="*80)

    feat_dir = os.path.join("data","processed","features")
    logs_dir = os.path.join("results","logs")
    os.makedirs(logs_dir, exist_ok=True)

    print("\n[1/4] Loading matrices...")
    X_c1 = np.load(os.path.join(feat_dir,"train_features_c1_scaled.npy"))
    X_c3 = np.load(os.path.join(feat_dir,"train_features_c3_scaled.npy"))
    y_all = np.load(os.path.join(feat_dir,"train_labels.npy"))
    with open(os.path.join(feat_dir,"train_doc_boundaries.json")) as f:
        doc_boundaries = json.load(f)
    print(f"  X_c1={X_c1.shape}, X_c3={X_c3.shape}, y={y_all.shape}")

    print("\n[2/4] Recreating val split (seed=42, 15%)...")
    _, val_idx, _, val_doc_ids = create_document_validation_split(doc_boundaries, 0.15, 42)
    print(f"  Val: {len(val_doc_ids):,} docs, {len(val_idx):,} sentences")

    X_c1_val = X_c1[val_idx]
    X_c3_val = X_c3[val_idx]
    y_val = y_all[val_idx]

    print("\n[3/4] Loading models...")
    m1 = load_model("models/gbr_c1.pkl")
    m2 = load_model("models/gbr_c2.pkl")
    m3 = load_model("models/gbr_c3.pkl")

    print("\n[4/4] Evaluating configs...")
    results = {}
    results["C1"] = evaluate_config(m1, X_c1_val, y_val, val_doc_ids, doc_boundaries, "C1")
    results["C2"] = evaluate_config(m2, X_c1_val, y_val, val_doc_ids, doc_boundaries, "C2")
    results["C3"] = evaluate_config(m3, X_c3_val, y_val, val_doc_ids, doc_boundaries, "C3")

    json_path = os.path.join(logs_dir,"rank_correlation_report.json")
    with open(json_path,"w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved JSON -> {json_path}")

    # Markdown
    md_path = os.path.join(logs_dir,"rank_correlation_report.md")
    write_md(results, md_path)
    print(f"Saved Markdown -> {md_path}")

    print("\n"+"="*80+"  SUMMARY  "+"="*80)
    keys = [
        ("Global Spearman rho","global_spearman_rho"),
        ("Global Kendall tau","global_kendall_tau"),
        ("Per-doc Spearman mean","per_doc_spearman_mean"),
        ("Per-doc Kendall mean","per_doc_kendall_mean"),
        ("Top-k Jaccard mean","topk_jaccard_mean"),
        ("Top-k Jaccard std","topk_jaccard_std"),
        ("Top-k % Recall mean","topk_recall_mean"),
    ]
    print(f"  {'Metric':<35} {'C1':>10} {'C2':>10} {'C3':>10}")
    print("  "+"-"*65)
    for label,k in keys:
        print(f"  {label:<35} {results['C1'][k]:>10.4f} {results['C2'][k]:>10.4f} {results['C3'][k]:>10.4f}")
    print("="*80)


def write_md(results, out_path):
    c1,c2,c3 = results["C1"],results["C2"],results["C3"]
    L=[]
    L.append("# GBR Rank Correlation Report — Validation Split\n")
    L.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    L.append(f"**Val split**: {c1['n_val_docs']:,} docs / {c1['n_val_sentences']:,} sentences  ")
    L.append(f"**Budget rule**: k_i = max(3, round(0.05 * M_i))  avg k_i ≈ {c1['avg_ki']:.1f}\n")

    L.append("## Global Sentence-Level Rank Correlation\n")
    L.append("| Metric | C1 (Replica) | C2 (Redundancy Swap) | C3 (Proposed) |")
    L.append("|---|:---:|:---:|:---:|")
    L.append(f"| **Spearman ρ** | {c1['global_spearman_rho']:.4f} | {c2['global_spearman_rho']:.4f} | {c3['global_spearman_rho']:.4f} |")
    L.append(f"| Spearman p | {c1['global_spearman_pvalue']:.2e} | {c2['global_spearman_pvalue']:.2e} | {c3['global_spearman_pvalue']:.2e} |")
    L.append(f"| **Kendall τ** | {c1['global_kendall_tau']:.4f} | {c2['global_kendall_tau']:.4f} | {c3['global_kendall_tau']:.4f} |")
    L.append(f"| Kendall p | {c1['global_kendall_pvalue']:.2e} | {c2['global_kendall_pvalue']:.2e} | {c3['global_kendall_pvalue']:.2e} |")
    L.append(f"| Per-doc Spearman (mean) | {c1['per_doc_spearman_mean']:.4f} | {c2['per_doc_spearman_mean']:.4f} | {c3['per_doc_spearman_mean']:.4f} |")
    L.append(f"| Per-doc Kendall (mean) | {c1['per_doc_kendall_mean']:.4f} | {c2['per_doc_kendall_mean']:.4f} | {c3['per_doc_kendall_mean']:.4f} |\n")

    L.append("## Task-Relevant: Top-k_i Extractive Selection Overlap\n")
    L.append("| Metric | C1 (Replica) | C2 (Redundancy Swap) | C3 (Proposed) |")
    L.append("|---|:---:|:---:|:---:|")
    L.append(f"| **Top-k Jaccard (mean)** | {c1['topk_jaccard_mean']:.4f} | {c2['topk_jaccard_mean']:.4f} | {c3['topk_jaccard_mean']:.4f} |")
    L.append(f"| Top-k Jaccard (std) | {c1['topk_jaccard_std']:.4f} | {c2['topk_jaccard_std']:.4f} | {c3['topk_jaccard_std']:.4f} |")
    L.append(f"| **Top-k % Recall (mean)** | {c1['topk_recall_mean']:.4f} | {c2['topk_recall_mean']:.4f} | {c3['topk_recall_mean']:.4f} |")
    L.append(f"| Avg k_i | {c1['avg_ki']:.1f} | {c2['avg_ki']:.1f} | {c3['avg_ki']:.1f} |\n")

    for cfg_label, cfg in [("C1 = C2 (identical scorer)", c1), ("C3 (Proposed)", c3)]:
        L.append(f"## Worked Examples — Config {cfg_label}\n")
        L.append("> True top-k = oracle ranking by ground-truth ROUGE label y_ij.  ")
        L.append("> Pred top-k = GBR ranking by predicted score ŷ_ij.  ")
        L.append("> Sentences in **bold** appear in BOTH sets.\n")
        for ex in cfg.get("examples",[]):
            ts = set(ex["true_topk"])
            ps = set(ex["pred_topk"])
            ov = ts & ps
            L.append(f"### Doc {ex['doc_id']} — M_i={ex['M_i']} sents, k_i={ex['k_i']}")
            L.append(f"- Jaccard: **{ex['jaccard']}** | % Recall: **{ex['pct_recall']:.0%}** ({len(ov)}/{ex['k_i']} matched)")
            def fmt(s):
                return ", ".join(f"**{x}**" if x in (ts&ps) else str(x) for x in sorted(s))
            L.append(f"- True top-{ex['k_i']}: [{fmt(ts)}]")
            L.append(f"- Pred top-{ex['k_i']}: [{fmt(ps)}]")
            L.append("")
            L.append(f"| Sent | True y | Pred ŷ | In Oracle? | In Model? |")
            L.append("|:---:|:---:|:---:|:---:|:---:|")
            for s in sorted(ts | ps):
                ty = ex["true_scores_at_true"].get(str(s), "—")
                yp = ex["true_scores_at_pred"].get(str(s), "—")
                it = "✅" if s in ts else "❌"
                ip = "✅" if s in ps else "❌"
                L.append(f"| {s} | {ty} | {yp} | {it} | {ip} |")
            L.append("")

    with open(out_path,"w",encoding="utf-8") as f:
        f.write("\n".join(L)+"\n")


if __name__ == "__main__":
    main()
