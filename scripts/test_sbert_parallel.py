"""
test_sbert_parallel.py — Test SBERT multi-process speed on 50 docs.
"""
import os
import sys
import time
import multiprocessing as mp

sys.path.insert(0, os.path.abspath("."))

from src.data.loader import load_split_ids, load_document
from src.data.preprocessing import segment_sentences
from src.features.embeddings_sbert import load_sbert_model, compute_sbert_cosine_features

_sbert = None

def init_sbert():
    global _sbert
    import torch
    torch.set_num_threads(2)
    _sbert = load_sbert_model(device="cpu")

def proc_sbert_doc(cid):
    doc = load_document(cid)
    sents = segment_sentences(doc["judgment"])
    cos = compute_sbert_cosine_features(sents, _sbert)
    return len(sents), cos

def main():
    cids = load_split_ids("train")[:50]
    n_workers = 4
    print(f"Benchmarking SBERT across {n_workers} workers on {len(cids)} docs...")
    t0 = time.time()
    with mp.Pool(processes=n_workers, initializer=init_sbert) as pool:
        res = pool.map(proc_sbert_doc, cids, chunksize=5)
    elapsed = time.time() - t0
    total_s = sum(r[0] for r in res)
    print(f"50 docs ({total_s} sents) took {elapsed:.2f}s -> {len(cids)/elapsed:.2f} docs/sec ({total_s/elapsed:.1f} sents/sec)")
    extrap = (elapsed / len(cids) * 7028) / 60
    print(f"Extrapolated for 7,028 docs: {extrap:.2f} minutes")

if __name__ == "__main__":
    main()
