"""
benchmark_scale.py — Test multi-core throughput for GBR labeling and feature extraction.
"""

import time
import multiprocessing as mp
from src.data.loader import load_split_ids, load_document
from src.data.preprocessing import segment_sentences, LegalTokenizer
from src.labeling.gbr_labels import GBRLabeler
from src.features.tfidf import fit_tfidf_vectorizer, compute_sentence_tfidf_scores
from src.features.position import compute_position_features
from src.features.ner import compute_sentence_ner_counts
from src.features.embeddings_w2v import load_word2vec_model, compute_w2v_cosine_features
from src.features.embeddings_sbert import load_sbert_model, compute_sbert_cosine_features
from src.features.wmd import compute_document_wmd_features


def label_doc(cid):
    labeler = GBRLabeler(use_stemmer=True)
    doc = load_document(cid)
    res = labeler.compute_labels_for_doc_dict(doc)
    return len(res["source_sentences"]), res["labels"]


def run_benchmark():
    train_ids = load_split_ids("train")
    num_cores = mp.cpu_count()
    print(f"Total CPU Cores Available: {num_cores}")
    print(f"Total Train Documents: {len(train_ids):,}")

    # Benchmark GBR Labeling on 50 docs
    test_ids = train_ids[:50]
    print(f"\nBenchmarking GBR Labeling on {len(test_ids)} docs across {min(14, num_cores)} workers...")
    t0 = time.time()
    with mp.Pool(processes=min(14, num_cores)) as pool:
        results = pool.map(label_doc, test_ids)
    elapsed = time.time() - t0
    total_sents = sum(r[0] for r in results)
    sec_per_doc = elapsed / len(test_ids)
    extrapolated_min = (sec_per_doc * len(train_ids)) / 60.0

    print(f"Labeling {len(test_ids)} docs took: {elapsed:.2f}s ({total_sents:,} sentences)")
    print(f"Speed: {sec_per_doc:.4f}s / doc ({len(test_ids)/elapsed:.1f} docs/sec)")
    print(f"Extrapolated time for ALL 7,028 docs: {extrapolated_min:.1f} minutes!")


if __name__ == "__main__":
    run_benchmark()
