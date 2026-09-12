"""
test_c1_features_speed.py — Test extraction speed of all 5 C1 features across 100 docs.
"""
import os
import sys
import time
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.data.loader import load_split_ids, load_document
from src.data.preprocessing import segment_sentences, LegalTokenizer
from src.features.tfidf import fit_tfidf_vectorizer
from src.features.embeddings_w2v import load_word2vec_model
from src.features.build_features import assemble_document_raw_features

_vec = None
_w2v = None
_tok = None

def init_worker():
    global _vec, _w2v, _tok
    _vec = fit_tfidf_vectorizer(verbose=False)
    _w2v = load_word2vec_model()
    _tok = LegalTokenizer()

def process_doc(cid):
    doc = load_document(cid)
    sents = segment_sentences(doc["judgment"])
    X = assemble_document_raw_features(sents, "C1", _vec, _w2v, tokenizer=_tok)
    return len(sents), X

def main():
    cids = load_split_ids("train")[:100]
    n_workers = min(12, os.cpu_count() or 4)
    print(f"Testing C1 features extraction across {n_workers} workers on 100 docs...")
    t0 = time.time()
    with mp.Pool(processes=n_workers, initializer=init_worker) as pool:
        res = pool.map(process_doc, cids, chunksize=10)
    elapsed = time.time() - t0
    total_s = sum(r[0] for r in res)
    print(f"100 docs ({total_s:,} sentences) took {elapsed:.2f}s -> {len(cids)/elapsed:.2f} docs/sec ({elapsed/len(cids):.4f} s/doc)")
    extrap = (elapsed / len(cids) * 7028) / 60
    print(f"Extrapolated for 7,028 docs: {extrap:.2f} minutes")

if __name__ == "__main__":
    main()
