"""
test_parallel_labeler.py — Measure multi-process labeling speed on 100 docs.
"""
import time
import os
import sys

sys.path.insert(0, os.path.abspath("."))

import multiprocessing as mp
from src.data.loader import load_split_ids, load_document
from src.data.preprocessing import segment_sentences
from src.labeling.gbr_labels import GBRLabeler

_worker_labeler = None

def init_worker():
    global _worker_labeler
    _worker_labeler = GBRLabeler(use_stemmer=True)

def process_single_doc(cid):
    doc = load_document(cid)
    s_sents = segment_sentences(doc["judgment"])
    r_sents = segment_sentences(doc["headnote"])
    labels, _, _ = _worker_labeler.compute_document_labels(s_sents, r_sents)
    return len(s_sents), labels

def main():
    train_ids = load_split_ids("train")[:100]
    n_workers = min(12, os.cpu_count() or 4)
    print(f"Testing 100 docs with {n_workers} workers...")
    t0 = time.time()
    with mp.Pool(processes=n_workers, initializer=init_worker) as pool:
        results = pool.map(process_single_doc, train_ids, chunksize=10)
    elapsed = time.time() - t0
    total_sents = sum(r[0] for r in results)
    print(f"Finished 100 docs in {elapsed:.2f}s ({total_sents:,} sentences)")
    print(f"Throughput: {len(train_ids)/elapsed:.2f} docs/sec ({elapsed/len(train_ids):.4f} s/doc)")
    extrap = (elapsed / len(train_ids) * 7028) / 60
    print(f"Extrapolated for 7,028 docs: {extrap:.2f} minutes")

if __name__ == "__main__":
    main()
