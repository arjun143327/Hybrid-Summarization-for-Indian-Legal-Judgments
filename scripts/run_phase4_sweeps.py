"""
run_phase4_sweeps.py — Phase 4 Redundancy Control Hyperparameter Sweeps.

Runs:
1. Config C1: WMD-threshold filter over delta in {1.0, 1.15, 1.25, 1.35}.
2. Config C2: MMR over SBERT embeddings (C1 GBR scores) over lambda in {0.3, 0.5, 0.7, 0.9}.
3. Config C3: MMR over SBERT embeddings (C3 GBR scores) over lambda in {0.3, 0.5, 0.7, 0.9}.

Methodology constraints:
- Budget rule: k_i = max(3, round(0.05 * M_i)).
- Fixed-quota validation subset: N=50 docs (15 short/short-medium, 20 medium, 15 long).
- Pure selection timing: SBERT embeddings are precomputed once per doc before timing loops.
- Paired metrics: Self-BLEU-2 paired with budget fulfillment % (|S_i|/k_i).

Outputs:
- results/logs/phase4_redundancy_sweep.json
- results/logs/phase4_redundancy_sweep.md
"""

import os
import sys
import json
import gzip
import time
import pickle
from typing import List, Dict, Any
import numpy as np
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.abspath("."))
from src.scoring.gbr_model import create_document_validation_split
from src.data.preprocessing import LegalTokenizer
from src.redundancy.wmd_filter import wmd_threshold_filter
from src.redundancy.mmr import mmr_select
from src.evaluation.redundancy_metrics import evaluate_summary_redundancy


def ki_budget(m: int) -> int:
    return max(3, round(0.05 * m))


def load_model(path: str):
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


def get_doc_sentences(doc_sentences_dict: dict, doc_id: str) -> List[str]:
    raw = doc_sentences_dict.get(doc_id, doc_sentences_dict.get(str(doc_id), []))
    if isinstance(raw, dict):
        raw = raw.get("sentences", list(raw.values()))
    texts = []
    for s in raw:
        if isinstance(s, dict):
            texts.append(s.get("text", s.get("sentence", str(s))))
        else:
            texts.append(str(s))
    return texts


def main():
    print("=" * 80)
    print("PHASE 4: REDUNDANCY CONTROL HYPERPARAMETER SWEEPS (C1, C2, C3)")
    print("=" * 80)

    feat_dir = os.path.join("data", "processed", "features")
    logs_dir = os.path.join("results", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # 1. Load data & recreate validation split (seed=42, 15%)
    print("\n[1/7] Loading validation split and GBR models...")
    X_c1 = np.load(os.path.join(feat_dir, "train_features_c1_scaled.npy"))
    X_c3 = np.load(os.path.join(feat_dir, "train_features_c3_scaled.npy"))
    with open(os.path.join(feat_dir, "train_doc_boundaries.json")) as fh:
        doc_boundaries = json.load(fh)

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

    model_c1 = load_model(os.path.join("models", "gbr_c1.pkl"))
    model_c3 = load_model(os.path.join("models", "gbr_c3.pkl"))
    yp_c1_all = model_c1.predict(X_c1_val)
    yp_c3_all = model_c3.predict(X_c3_val)
    print(f"  Loaded models. Val split: {len(val_doc_ids)} docs, {len(val_idx):,} sentences.")

    # 2. Fixed-quota validation subset selection (N=50: 15 short/short-med, 20 med, 15 long)
    print("\n[2/7] Selecting fixed-quota validation subset (N=50)...")
    val_docs_meta = []
    for did in sorted(val_doc_ids):
        lo, hi = doc_local[did]
        m = hi - lo
        val_docs_meta.append({"doc_id": did, "M_i": m, "k_i": ki_budget(m)})

    # Sort docs by length
    sorted_by_len = sorted(val_docs_meta, key=lambda x: x["M_i"])

    # 15 Short / Short-Medium (all 4 short docs with M_i < 20 + next 11 shortest)
    short_quota = [d for d in sorted_by_len if d["M_i"] < 20]  # 4 docs
    short_med_pool = [d for d in sorted_by_len if d["M_i"] >= 20 and d["M_i"] <= 45]
    short_selected = short_quota + short_med_pool[:11]
    assert len(short_selected) == 15, f"Expected 15 short docs, got {len(short_selected)}"

    # 15 Long (sampled from the 28 docs with M_i > 500, seed=42)
    long_pool = [d for d in sorted_by_len if d["M_i"] > 500]
    rng_long = np.random.RandomState(42)
    long_indices = rng_long.choice(len(long_pool), size=15, replace=False)
    long_selected = [long_pool[i] for i in sorted(long_indices)]
    assert len(long_selected) == 15, f"Expected 15 long docs, got {len(long_selected)}"

    # 20 Medium (sampled from docs with 50 <= M_i <= 350, seed=42)
    med_pool = [d for d in sorted_by_len if 50 <= d["M_i"] <= 350]
    rng_med = np.random.RandomState(42)
    med_indices = rng_med.choice(len(med_pool), size=20, replace=False)
    med_selected = [med_pool[i] for i in sorted(med_indices)]
    assert len(med_selected) == 20, f"Expected 20 med docs, got {len(med_selected)}"

    subset_docs = short_selected + med_selected + long_selected
    assert len(subset_docs) == 50
    print(f"  Fixed-quota sample established: {len(subset_docs)} docs total:")
    print(f"    - Short / Short-Med: {len(short_selected)} docs (M_i range: {short_selected[0]['M_i']}..{short_selected[-1]['M_i']})")
    print(f"    - Medium:            {len(med_selected)} docs (M_i range: {min(d['M_i'] for d in med_selected)}..{max(d['M_i'] for d in med_selected)})")
    print(f"    - Long:              {len(long_selected)} docs (M_i range: {min(d['M_i'] for d in long_selected)}..{max(d['M_i'] for d in long_selected)})")

    # 3. Load text and precompute SBERT embeddings
    print("\n[3/7] Loading sentence text and precomputing SBERT embeddings...")
    with gzip.open(os.path.join("data", "train_doc_sentences.json.gz"), "rt", encoding="utf-8") as fh:
        doc_sentences_raw = json.load(fh)

    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = LegalTokenizer()

    # Pre-tokenize sentences and pre-encode with SBERT
    doc_cache = {}
    for dinfo in subset_docs:
        did = dinfo["doc_id"]
        lo, hi = doc_local[did]
        sents = get_doc_sentences(doc_sentences_raw, did)
        assert len(sents) == dinfo["M_i"], f"Length mismatch for doc {did}: {len(sents)} vs {dinfo['M_i']}"
        s_c1 = yp_c1_all[lo:hi]
        s_c3 = yp_c3_all[lo:hi]

        # SBERT precomputation (excluded from timing)
        sbert_vecs = sbert_model.encode(sents, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

        doc_cache[did] = {
            "sentences": sents,
            "scores_c1": s_c1,
            "scores_c3": s_c3,
            "sbert_embeds": sbert_vecs,
            "M_i": dinfo["M_i"],
            "k_i": dinfo["k_i"],
            "bucket": "short" if dinfo in short_selected else ("medium" if dinfo in med_selected else "long")
        }
    print(f"  Precomputation complete for all {len(doc_cache)} validation documents.")

    # 4. Load Word2Vec model for C1
    print("\n[4/7] Loading Word2Vec Google News model for C1 WMD filter...")
    kv_path = os.path.join(os.path.expanduser("~"), "gensim-data", "word2vec-google-news-300", "vectors.kv")
    kv = KeyedVectors.load(kv_path, mmap="r")
    print("  Word2Vec loaded.")

    # 5. Execute Config C1: WMD-threshold sweep over delta in {1.0, 1.15, 1.25, 1.35}
    delta_grid = [1.0, 1.15, 1.25, 1.35]
    print(f"\n[5/7] Running Config C1 delta sweep over {delta_grid}...")
    c1_results = {}

    for delta in delta_grid:
        print(f"  --> Evaluating C1 at delta={delta}...")
        doc_evals = []
        timing_records = []

        for dinfo in subset_docs:
            did = dinfo["doc_id"]
            ddata = doc_cache[did]
            sents = ddata["sentences"]
            sc = ddata["scores_c1"]
            ki = ddata["k_i"]

            # Pure selection timing
            t0 = time.perf_counter()
            filt_res = wmd_threshold_filter(
                sentences=sents,
                scores=sc,
                k_i=ki,
                delta=delta,
                keyed_vectors=kv,
                tokenizer=tokenizer,
                norm=True
            )
            elapsed_sec = time.perf_counter() - t0
            timing_records.append(elapsed_sec)

            # Extract chosen sentences in original document order
            sel_sents = [sents[i] for i in filt_res["ordered_indices"]]
            stats = evaluate_summary_redundancy(sel_sents, k_i=ki, tokenizer=tokenizer)

            doc_evals.append({
                "doc_id": did,
                "bucket": ddata["bucket"],
                "M_i": ddata["M_i"],
                "k_i": ki,
                "selected_count": filt_res["selected_count"],
                "budget_fulfillment": filt_res["budget_fulfillment"],
                "under_filled": filt_res["under_filled"],
                "self_bleu_2": stats["self_bleu_2"],
                "distinct_2": stats["distinct_2"],
                "token_count": stats["token_count"],
                "time_sec": elapsed_sec,
            })

        avg_sb2 = np.mean([e["self_bleu_2"] for e in doc_evals])
        std_sb2 = np.std([e["self_bleu_2"] for e in doc_evals])
        avg_ful = np.mean([e["budget_fulfillment"] for e in doc_evals])
        under_count = sum(1 for e in doc_evals if e["under_filled"])
        avg_sents = np.mean([e["selected_count"] for e in doc_evals])
        avg_tokens = np.mean([e["token_count"] for e in doc_evals])
        avg_time_ms = np.mean(timing_records) * 1000.0
        docs_per_min = 60.0 / np.mean(timing_records) if np.mean(timing_records) > 0 else 0.0

        c1_results[str(delta)] = {
            "delta": delta,
            "self_bleu_2_mean": round(float(avg_sb2), 4),
            "self_bleu_2_std": round(float(std_sb2), 4),
            "budget_fulfillment_mean": round(float(avg_ful * 100.0), 2),
            "under_filled_count": under_count,
            "under_filled_pct": round(float(under_count / len(doc_evals) * 100.0), 1),
            "avg_summary_sentences": round(float(avg_sents), 2),
            "avg_summary_tokens": round(float(avg_tokens), 1),
            "selection_time_ms_per_doc": round(float(avg_time_ms), 2),
            "docs_per_minute": round(float(docs_per_min), 1),
            "per_doc_evals": doc_evals,
        }
        print(f"      Self-BLEU-2: {avg_sb2:.4f} | Fulfillment: {avg_ful*100:.1f}% | Under-filled docs: {under_count}/{len(doc_evals)} | Time: {avg_time_ms:.1f} ms/doc")

    # 6. Execute Configs C2 & C3: MMR lambda sweep over {0.3, 0.5, 0.7, 0.9}
    lam_grid = [0.3, 0.5, 0.7, 0.9]
    print(f"\n[6/7] Running Configs C2 and C3 lambda sweeps over {lam_grid}...")

    c2_results = {}
    c3_results = {}

    for cfg_name, score_key, res_dict in [("C2", "scores_c1", c2_results), ("C3", "scores_c3", c3_results)]:
        print(f"  ==> Config {cfg_name} MMR Sweep...")
        for lam in lam_grid:
            doc_evals = []
            timing_records = []

            for dinfo in subset_docs:
                did = dinfo["doc_id"]
                ddata = doc_cache[did]
                sents = ddata["sentences"]
                sc = ddata[score_key]
                embeds = ddata["sbert_embeds"]
                ki = ddata["k_i"]

                # Pure selection timing
                t0 = time.perf_counter()
                mmr_res = mmr_select(
                    scores=sc,
                    embeddings=embeds,
                    k_i=ki,
                    lam=lam
                )
                elapsed_sec = time.perf_counter() - t0
                timing_records.append(elapsed_sec)

                sel_sents = [sents[i] for i in mmr_res["ordered_indices"]]
                stats = evaluate_summary_redundancy(sel_sents, k_i=ki, tokenizer=tokenizer)

                doc_evals.append({
                    "doc_id": did,
                    "bucket": ddata["bucket"],
                    "M_i": ddata["M_i"],
                    "k_i": ki,
                    "selected_count": mmr_res["selected_count"],
                    "budget_fulfillment": mmr_res["budget_fulfillment"],
                    "under_filled": mmr_res["under_filled"],
                    "self_bleu_2": stats["self_bleu_2"],
                    "distinct_2": stats["distinct_2"],
                    "token_count": stats["token_count"],
                    "time_sec": elapsed_sec,
                })

            avg_sb2 = np.mean([e["self_bleu_2"] for e in doc_evals])
            std_sb2 = np.std([e["self_bleu_2"] for e in doc_evals])
            avg_ful = np.mean([e["budget_fulfillment"] for e in doc_evals])
            under_count = sum(1 for e in doc_evals if e["under_filled"])
            avg_sents = np.mean([e["selected_count"] for e in doc_evals])
            avg_tokens = np.mean([e["token_count"] for e in doc_evals])
            avg_time_ms = np.mean(timing_records) * 1000.0
            docs_per_min = 60.0 / np.mean(timing_records) if np.mean(timing_records) > 0 else 0.0

            res_dict[str(lam)] = {
                "lambda": lam,
                "self_bleu_2_mean": round(float(avg_sb2), 4),
                "self_bleu_2_std": round(float(std_sb2), 4),
                "budget_fulfillment_mean": round(float(avg_ful * 100.0), 2),
                "under_filled_count": under_count,
                "under_filled_pct": round(float(under_count / len(doc_evals) * 100.0), 1),
                "avg_summary_sentences": round(float(avg_sents), 2),
                "avg_summary_tokens": round(float(avg_tokens), 1),
                "selection_time_ms_per_doc": round(float(avg_time_ms), 3),
                "docs_per_minute": round(float(docs_per_min), 1),
                "per_doc_evals": doc_evals,
            }
            print(f"    [{cfg_name}] lambda={lam} | Self-BLEU-2: {avg_sb2:.4f} | Fulfillment: {avg_ful*100:.1f}% | Under-filled: {under_count}/{len(doc_evals)} | Time: {avg_time_ms:.3f} ms/doc")

    # 7. Generate output reports
    print("\n[7/7] Compiling reports and saving artifacts...")
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validation_subset_size": len(subset_docs),
        "subset_quotas": {
            "short_short_med_count": len(short_selected),
            "medium_count": len(med_selected),
            "long_count": len(long_selected),
        },
        "subset_doc_ids": {
            "short_short_med": [d["doc_id"] for d in short_selected],
            "medium": [d["doc_id"] for d in med_selected],
            "long": [d["doc_id"] for d in long_selected],
        },
        "c1_delta_sweep": c1_results,
        "c2_lambda_sweep": c2_results,
        "c3_lambda_sweep": c3_results,
        "timing_summary": {
            "c1_wmd_filter_mean_ms_per_doc": round(float(np.mean([c1_results[str(d)]["selection_time_ms_per_doc"] for d in delta_grid])), 2),
            "c2_mmr_mean_ms_per_doc": round(float(np.mean([c2_results[str(l)]["selection_time_ms_per_doc"] for l in lam_grid])), 3),
            "c3_mmr_mean_ms_per_doc": round(float(np.mean([c3_results[str(l)]["selection_time_ms_per_doc"] for l in lam_grid])), 3),
            "mmr_speedup_vs_wmd": round(float(np.mean([c1_results[str(d)]["selection_time_ms_per_doc"] for d in delta_grid]) / np.mean([c3_results[str(l)]["selection_time_ms_per_doc"] for l in lam_grid])), 1)
        }
    }

    json_path = os.path.join(logs_dir, "phase4_redundancy_sweep.json")
    with open(json_path, "w") as fh:
        json.dump(report_data, fh, indent=2, cls=NumpyEncoder)
    print(f"  Saved JSON report -> {json_path}")

    md_path = os.path.join(logs_dir, "phase4_redundancy_sweep.md")
    write_markdown_report(report_data, md_path)
    print(f"  Saved MD report   -> {md_path}")

    print("\n" + "=" * 80)
    print("PHASE 4 SWEEP COMPLETE")
    print("=" * 80)


def write_markdown_report(report: dict, out_path: str):
    lines = []
    a = lines.append

    a("# Phase 4 Redundancy Control Hyperparameter Sweeps Report")
    a("")
    a(f"Date: {report['timestamp']}")
    a(f"Validation subset size: $N={report['validation_subset_size']}$ documents (fixed quota: 15 short/short-med, 20 medium, 15 long)")
    a("")
    a("> **Fairness Rule**: Identical sweep grid size (4 points each), identical validation subset, and identical budget rule ($k_i = \max(3, \text{round}(0.05 \cdot M_i))$).")
    a("> **Timing Methodology**: SBERT embeddings precomputed once per document; wall-clock times track strictly the selection loop (greedy candidate filtering for C1; greedy MMR ranking for C2/C3).")
    a("> **WMD OOV Note**: Sentences using the WMD OOV fallback distance ($\text{max\_fallback\_dist}=3.0$, $\sim 3.05\%$ of sentences per Phase 2 audit) trivially pass any $\delta \le 1.35$ in C1, as expected.")
    a("")
    a("## 1. Validation Subset Composition")
    a("")
    a(f"- **Short / Short-Medium** ($N=15$): `{report['subset_doc_ids']['short_short_med']}`")
    a(f"- **Medium** ($N=20$): `{report['subset_doc_ids']['medium']}`")
    a(f"- **Long** ($N=15$): `{report['subset_doc_ids']['long']}`")
    a("")
    a("## 2. Config C1: WMD Threshold ($\delta$) Sweep")
    a("")
    a("| $\delta$ | Self-BLEU-2 (mean $\pm$ std) | Budget Fulfillment % ($|S_i|/k_i$) | Under-filled Docs | Avg Sentences | Avg Tokens | Selection Time (ms/doc) | Docs / min |")
    a("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for d_str, res in report["c1_delta_sweep"].items():
        sb = f"`{res['self_bleu_2_mean']:.4f} ± {res['self_bleu_2_std']:.3f}`"
        ful = f"**`{res['budget_fulfillment_mean']:.1f}%`**"
        und = f"`{res['under_filled_count']}/50` (`{res['under_filled_pct']:.1f}%`)"
        snt = f"`{res['avg_summary_sentences']:.1f}`"
        tok = f"`{res['avg_summary_tokens']:.0f}`"
        tms = f"`{res['selection_time_ms_per_doc']:.1f}`"
        dpm = f"`{res['docs_per_minute']:.0f}`"
        a(f"| `{d_str}` | {sb} | {ful} | {und} | {snt} | {tok} | {tms} | {dpm} |")

    a("")
    a("## 3. Configs C2 & C3: MMR Lambda ($\lambda$) Sweep")
    a("")
    a("| Config | $\lambda$ | Self-BLEU-2 (mean $\pm$ std) | Budget Fulfillment % ($|S_i|/k_i$) | Under-filled Docs | Avg Sentences | Avg Tokens | Selection Time (ms/doc) | Docs / min |")
    a("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for cfg_code, sweep_key in [("C2 (Redundancy Swap)", "c2_lambda_sweep"), ("C3 (Proposed)", "c3_lambda_sweep")]:
        for l_str, res in report[sweep_key].items():
            sb = f"`{res['self_bleu_2_mean']:.4f} ± {res['self_bleu_2_std']:.3f}`"
            ful = f"**`{res['budget_fulfillment_mean']:.1f}%`**"
            und = f"`{res['under_filled_count']}/50` (`{res['under_filled_pct']:.1f}%`)"
            snt = f"`{res['avg_summary_sentences']:.1f}`"
            tok = f"`{res['avg_summary_tokens']:.0f}`"
            tms = f"`{res['selection_time_ms_per_doc']:.3f}`"
            dpm = f"`{res['docs_per_minute']:,.0f}`"
            a(f"| **{cfg_code}** | `{l_str}` | {sb} | {ful} | {und} | {snt} | {tok} | {tms} | {dpm} |")

    a("")
    a("## 4. Efficiency Comparison (Selection Loop Only)")
    a("")
    ts = report["timing_summary"]
    a(f"- **C1 (WMD-threshold)** average selection time: **`{ts['c1_wmd_filter_mean_ms_per_doc']:.1f} ms/doc`**")
    a(f"- **C2 (MMR / SBERT)** average selection time: **`{ts['c2_mmr_mean_ms_per_doc']:.3f} ms/doc`**")
    a(f"- **C3 (MMR / SBERT)** average selection time: **`{ts['c3_mmr_mean_ms_per_doc']:.3f} ms/doc`**")
    a(f"- **Speedup Factor**: MMR selection loop is **`{ts['mmr_speedup_vs_wmd']:.1f}×` faster** than the WMD threshold filter.")
    a("")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
