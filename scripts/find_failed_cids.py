"""
find_failed_cids.py — Identify which CIDs failed in C1 feature extraction and why.
"""
import os
import sys
import multiprocessing as mp

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

def check_doc(cid):
    try:
        doc = load_document(cid)
        sents = segment_sentences(doc["judgment"])
        if not sents:
            return cid, "EMPTY_SENTS", ""
        X = assemble_document_raw_features(sents, "C1", _vec, _w2v, tokenizer=_tok)
        return cid, "OK", ""
    except Exception as e:
        return cid, "ERROR", f"{type(e).__name__}: {str(e)}"

def main():
    train_ids = load_split_ids("train")
    print(f"Checking {len(train_ids)} docs for errors...")
    failed = []
    with mp.Pool(processes=10, initializer=init_worker) as pool:
        for cid, status, err in pool.imap_unordered(check_doc, train_ids, chunksize=50):
            if status != "OK":
                failed.append((cid, status, err))
                print(f"FAILURE [{len(failed)}]: Doc {cid} -> {status}: {err}")
    print(f"\nTotal failed docs: {len(failed)}")
    with open("data/processed/features/failed_c1_docs.txt", "w") as f:
        for cid, status, err in failed:
            f.write(f"{cid}\t{status}\t{err}\n")

if __name__ == "__main__":
    main()
