"""
build_full_train_labels.py — Computes ground-truth GBR labels across all 7,028 training documents
per 02_METHODOLOGY.md Stage 2 and 04_TASKS.md Phase 3.

Label formula:
  y_ij = max_{r_k in headnote_i} ROUGE-1_F1(s_ij, r_k)
"""

import os
import sys
import time
import json
import numpy as np
import multiprocessing as mp

sys.path.insert(0, os.path.abspath("."))

from src.data.loader import load_split_ids, load_document
from src.data.preprocessing import segment_sentences
from src.labeling.gbr_labels import GBRLabeler, get_label_distribution_summary

_worker_labeler = None


def init_worker():
    global _worker_labeler
    _worker_labeler = GBRLabeler(use_stemmer=True)


def process_doc_labels(cid: str):
    try:
        doc = load_document(cid)
        s_sents = segment_sentences(doc["judgment"])
        r_sents = segment_sentences(doc["headnote"])
        labels, _, _ = _worker_labeler.compute_document_labels(s_sents, r_sents)
        return {
            "case_id": cid,
            "num_sentences": len(s_sents),
            "labels": labels,
            "success": True,
        }
    except Exception as e:
        return {
            "case_id": cid,
            "num_sentences": 0,
            "labels": np.array([], dtype=np.float32),
            "success": False,
            "error": str(e),
        }


def main():
    print("=" * 80)
    print("RUNNING FULL-SCALE GBR LABEL GENERATION ACROSS 7,028 TRAINING DOCUMENTS")
    print("=" * 80)

    train_ids = load_split_ids("train")
    total_docs = len(train_ids)
    print(f"Total training cases loaded: {total_docs:,}")

    num_workers = min(12, os.cpu_count() or 4)
    print(f"Executing in parallel with {num_workers} worker processes...")

    t0 = time.time()
    boundaries = {}
    all_labels_list = []
    total_sentences = 0
    failures = []

    with mp.Pool(processes=num_workers, initializer=init_worker) as pool:
        # Using imap with chunksize for streaming progress logging
        chunksize = 25
        for i, res in enumerate(pool.imap(process_doc_labels, train_ids, chunksize=chunksize), 1):
            if not res["success"]:
                failures.append((res["case_id"], res.get("error", "Unknown error")))
                continue

            cid = res["case_id"]
            n_s = res["num_sentences"]
            lbls = res["labels"]

            start_idx = total_sentences
            end_idx = total_sentences + n_s
            boundaries[cid] = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "num_sentences": n_s,
            }

            all_labels_list.append(lbls)
            total_sentences += n_s

            if i % 500 == 0 or i == total_docs:
                elapsed = time.time() - t0
                speed = i / elapsed
                eta_sec = (total_docs - i) / speed if speed > 0 else 0
                print(
                    f"[{i:5d}/{total_docs}] docs processed | "
                    f"Sentences: {total_sentences:9,d} | "
                    f"Elapsed: {elapsed:6.1f}s | "
                    f"Speed: {speed:5.1f} docs/sec ({elapsed/i:6.4f} s/doc) | "
                    f"ETA: {eta_sec/60:4.1f} min"
                )

    total_time = time.time() - t0
    print("\nLabel generation complete!")
    print(f"Total execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Overall throughput: {total_docs/total_time:.2f} docs/sec ({total_time/total_docs:.4f} s/doc)")
    print(f"Total documents successfully processed: {len(boundaries):,}")
    print(f"Total sentences labeled: {total_sentences:,}")

    if failures:
        print(f"WARNING: {len(failures)} documents failed!")
        for cid, err in failures[:5]:
            print(f"  Doc {cid}: {err}")

    # Concatenate all labels into a single contiguous array
    print("\nConcatenating labels array...")
    train_labels = np.concatenate(all_labels_list).astype(np.float32)
    assert len(train_labels) == total_sentences, f"Length mismatch: {len(train_labels)} vs {total_sentences}"

    # Compute full distribution statistics
    stats = get_label_distribution_summary(train_labels)
    stats["total_docs"] = total_docs
    stats["total_sentences"] = total_sentences
    stats["execution_time_sec"] = round(total_time, 2)
    stats["sec_per_doc"] = round(total_time / total_docs, 4)
    stats["docs_per_sec"] = round(total_docs / total_time, 2)

    print("\nFULL 7,028-DOCUMENT TRAINING SET LABEL DISTRIBUTION:")
    print("=" * 60)
    print(f"  Count:   {stats['count']:,} sentences")
    print(f"  Min:     {stats['min']:.4f}")
    print(f"  Max:     {stats['max']:.4f}")
    print(f"  Mean:    {stats['mean']:.4f}")
    print(f"  Median:  {stats['p50']:.4f}")
    print(f"  Std:     {stats['std']:.4f}")
    print(f"  75th %:  {stats['p75']:.4f}")
    print(f"  90th %:  {stats['p90']:.4f}")
    print("=" * 60)

    # Save to disk
    os.makedirs(os.path.join("data", "processed", "features"), exist_ok=True)
    labels_path = os.path.join("data", "processed", "features", "train_labels.npy")
    meta_path = os.path.join("data", "processed", "features", "train_labels_metadata.json")
    boundaries_path = os.path.join("data", "processed", "features", "train_doc_boundaries.json")

    np.save(labels_path, train_labels)
    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)
    with open(boundaries_path, "w") as f:
        json.dump(boundaries, f)

    print(f"\nSaved training labels to: {labels_path} ({os.path.getsize(labels_path)/(1024*1024):.2f} MB)")
    print(f"Saved metadata to: {meta_path}")
    print(f"Saved doc boundaries to: {boundaries_path}")


if __name__ == "__main__":
    main()
