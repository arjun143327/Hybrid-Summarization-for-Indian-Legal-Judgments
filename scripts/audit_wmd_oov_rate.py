"""
audit_wmd_oov_rate.py — Instruments and measures the exact OOV fallback rate
(sentences triggering max_fallback_dist = 3.0) across all 7,028 training documents
(1,010,961 sentences), and analyzes clustering in OCR-affected judgments.
"""

import os
import sys
import time
import json
import numpy as np
import multiprocessing as mp

sys.path.insert(0, os.path.abspath("."))

from src.data.loader import load_split_ids, load_document
from src.data.preprocessing import segment_sentences, LegalTokenizer
from src.features.embeddings_w2v import load_word2vec_model

_w2v_vocab = None
_tokenizer = None


def init_worker():
    global _w2v_vocab, _tokenizer
    model = load_word2vec_model()
    _w2v_vocab = set(model.key_to_index.keys())
    _tokenizer = LegalTokenizer()


def audit_doc(cid: str):
    doc = load_document(cid)
    judgment = doc["judgment"]
    sents = segment_sentences(judgment)

    # Heuristic for OCR artifact detection per Phase 1 spec
    # Check for truncated proceeding nouns at line starts, symbol noise, etc.
    is_ocr_affected = False
    ocr_markers = ["vil Appeal", "minal Appeal", "rit Petition", "pecial Leave", "ivil Appeal"]
    for marker in ocr_markers:
        if marker in judgment:
            is_ocr_affected = True
            break

    n_sents = len(sents)
    oov_count = 0
    oov_samples = []

    for idx, s in enumerate(sents):
        tokens = _tokenizer.tokenize_for_tfidf(s)
        in_vocab = [w for w in tokens if w in _w2v_vocab]
        if not in_vocab:
            oov_count += 1
            if len(oov_samples) < 3:
                oov_samples.append((idx, s[:80]))

    return {
        "cid": cid,
        "n_sents": n_sents,
        "oov_count": oov_count,
        "is_ocr_affected": is_ocr_affected,
        "oov_samples": oov_samples,
    }


def main():
    print("=" * 80)
    print("AUDITING WMD OOV FALLBACK RATE (max_fallback_dist=3.0) ACROSS 7,028 TRAIN DOCS")
    print("=" * 80)

    train_ids = load_split_ids("train")
    total_docs = len(train_ids)
    print(f"Total training documents: {total_docs:,}")

    num_workers = min(12, os.cpu_count() or 4)
    print(f"Auditing with {num_workers} parallel workers...")

    t0 = time.time()
    total_sents = 0
    total_oov_sents = 0
    ocr_doc_count = 0
    ocr_total_sents = 0
    ocr_oov_sents = 0

    non_ocr_doc_count = 0
    non_ocr_total_sents = 0
    non_ocr_oov_sents = 0

    all_oov_samples = []

    with mp.Pool(processes=num_workers, initializer=init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(audit_doc, train_ids, chunksize=50), 1):
            n_s = res["n_sents"]
            n_oov = res["oov_count"]
            is_ocr = res["is_ocr_affected"]

            total_sents += n_s
            total_oov_sents += n_oov

            if is_ocr:
                ocr_doc_count += 1
                ocr_total_sents += n_s
                ocr_oov_sents += n_oov
            else:
                non_ocr_doc_count += 1
                non_ocr_total_sents += n_s
                non_ocr_oov_sents += n_oov

            if res["oov_samples"] and len(all_oov_samples) < 15:
                all_oov_samples.extend(res["oov_samples"][:2])

            if i % 1000 == 0 or i == total_docs:
                elapsed = time.time() - t0
                pct = (total_oov_sents / total_sents * 100) if total_sents > 0 else 0
                print(f"[{i:5d}/{total_docs}] docs | Sents: {total_sents:9,d} | OOV Fallbacks: {total_oov_sents:6,d} ({pct:5.2f}%) | Time: {elapsed:5.1f}s")

    elapsed = time.time() - t0
    overall_oov_pct = (total_oov_sents / total_sents * 100) if total_sents > 0 else 0
    ocr_oov_pct = (ocr_oov_sents / ocr_total_sents * 100) if ocr_total_sents > 0 else 0
    non_ocr_oov_pct = (non_ocr_oov_sents / non_ocr_total_sents * 100) if non_ocr_total_sents > 0 else 0

    print("\n" + "=" * 80)
    print("AUDIT RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Sentences Analyzed:         {total_sents:,}")
    print(f"Total OOV Fallback Sentences:     {total_oov_sents:,} ({overall_oov_pct:.4f}%)")
    print(f"Audit Execution Time:             {elapsed:.2f}s ({total_docs/elapsed:.1f} docs/sec)")
    print("-" * 80)
    print(f"OCR-Affected Documents:           {ocr_doc_count:,} docs ({ocr_doc_count/total_docs*100:.2f}%)")
    print(f"  Sentences in OCR Docs:          {ocr_total_sents:,}")
    print(f"  OOV Fallbacks in OCR Docs:      {ocr_oov_sents:,} ({ocr_oov_pct:.4f}%)")
    print("-" * 80)
    print(f"Non-OCR Documents:                {non_ocr_doc_count:,} docs ({non_ocr_doc_count/total_docs*100:.2f}%)")
    print(f"  Sentences in Non-OCR Docs:      {non_ocr_total_sents:,}")
    print(f"  OOV Fallbacks in Non-OCR Docs:  {non_ocr_oov_sents:,} ({non_ocr_oov_pct:.4f}%)")
    print("-" * 80)
    print("\nSample Sentences Triggering OOV Fallback:")
    for idx, s in all_oov_samples[:8]:
        print(f"  - (Sent {idx}): {repr(s)}")

    results = {
        "total_documents": total_docs,
        "total_sentences": total_sents,
        "total_oov_sentences": total_oov_sents,
        "oov_rate_pct": overall_oov_pct,
        "ocr_doc_count": ocr_doc_count,
        "ocr_sentences": ocr_total_sents,
        "ocr_oov_sentences": ocr_oov_sents,
        "ocr_oov_rate_pct": ocr_oov_pct,
        "non_ocr_doc_count": non_ocr_doc_count,
        "non_ocr_sentences": non_ocr_total_sents,
        "non_ocr_oov_sentences": non_ocr_oov_sents,
        "non_ocr_oov_rate_pct": non_ocr_oov_pct,
        "audit_time_sec": elapsed,
        "samples": [{"sent_idx": idx, "text": s} for idx, s in all_oov_samples[:10]]
    }

    out_path = os.path.join("results", "logs", "wmd_oov_audit.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full audit results to: {out_path}")


if __name__ == "__main__":
    main()
