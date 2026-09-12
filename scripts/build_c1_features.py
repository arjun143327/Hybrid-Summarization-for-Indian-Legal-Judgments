"""
build_c1_features.py — Extracts C1/C2 feature matrix (5 features) across all 7,028 training documents,
fits RobustScaler fresh on the full training matrix, and saves scaled arrays and scaler.

Features:
  x_ij = [TF-IDF, NER, Position, Cosine_w2v, WMD]
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import multiprocessing as mp

sys.path.insert(0, os.path.abspath("."))

from src.data.loader import load_split_ids, load_document
from src.data.preprocessing import segment_sentences, LegalTokenizer
from src.features.tfidf import fit_tfidf_vectorizer
from src.features.embeddings_w2v import load_word2vec_model
from src.features.build_features import assemble_document_raw_features, FeatureScalerPipeline, CONFIG_FEATURE_NAMES

_vec = None
_w2v = None
_tok = None


def init_c1_worker():
    global _vec, _w2v, _tok
    _vec = fit_tfidf_vectorizer(verbose=False)
    _w2v = load_word2vec_model()
    _tok = LegalTokenizer()


def process_c1_doc(cid: str):
    try:
        doc = load_document(cid)
        sents = segment_sentences(doc["judgment"])
        if not sents:
            return cid, 0, np.empty((0, 5), dtype=np.float32), np.empty((0,), dtype=bool), True

        X, fallback_mask = assemble_document_raw_features(
            sentences=sents,
            config="C1",
            tfidf_vectorizer=_vec,
            w2v_model=_w2v,
            tokenizer=_tok,
            return_fallback_mask=True,
        )
        return cid, len(sents), X, fallback_mask, True
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed processing document {cid}: {e}")


def main():
    print("=" * 80)
    print("BUILDING FULL TRAINING FEATURE MATRIX FOR CONFIG C1/C2 (5 FEATURES)")
    print("Features: ['tfidf', 'ner', 'position', 'cosine_w2v', 'wmd']")
    print("=" * 80)

    train_ids = load_split_ids("train")
    total_docs = len(train_ids)
    print(f"Total training documents: {total_docs:,}")

    num_workers = min(10, os.cpu_count() or 4)
    print(f"Executing extraction with {num_workers} parallel workers...")

    t0 = time.time()
    feature_blocks = []
    sentence_keys = []
    doc_boundaries = {}
    total_sentences = 0
    total_oov_fallbacks = 0

    with mp.Pool(processes=num_workers, initializer=init_c1_worker) as pool:
        chunksize = 20
        for i, (cid, n_s, X_doc, fallback_mask, success) in enumerate(
            pool.imap(process_c1_doc, train_ids, chunksize=chunksize), 1
        ):
            start_idx = total_sentences
            end_idx = total_sentences + n_s
            doc_boundaries[cid] = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "num_sentences": n_s,
            }

            for s_idx in range(n_s):
                sentence_keys.append((cid, s_idx))

            if n_s > 0:
                feature_blocks.append(X_doc)
                total_sentences += n_s
                total_oov_fallbacks += int(fallback_mask.sum())

            if i % 500 == 0 or i == total_docs:
                elapsed = time.time() - t0
                speed = i / elapsed
                eta_sec = (total_docs - i) / speed if speed > 0 else 0
                oov_pct = (total_oov_fallbacks / total_sentences * 100) if total_sentences > 0 else 0
                print(
                    f"[{i:5d}/{total_docs}] docs extracted | "
                    f"Sentences: {total_sentences:9,d} | "
                    f"OOV Fallbacks: {total_oov_fallbacks:6,d} ({oov_pct:4.2f}%) | "
                    f"Elapsed: {elapsed:6.1f}s | "
                    f"Speed: {speed:5.1f} docs/sec | "
                    f"ETA: {eta_sec/60:4.1f} min"
                )

    total_time = time.time() - t0
    print("\nFeature extraction complete!")
    print(f"Total execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Extraction speed: {total_docs/total_time:.2f} docs/sec ({total_time/total_docs:.4f} s/doc)")
    print(f"Total documents successfully processed: {len(doc_boundaries):,} / {total_docs:,} (0 dropped)")
    print(f"Total sentences extracted: {total_sentences:,}")
    print(f"Total WMD OOV fallbacks: {total_oov_fallbacks:,} ({total_oov_fallbacks/total_sentences*100:.4f}%)")

    # Concatenate all blocks into a contiguous matrix
    print("\nConcatenating full C1 raw feature matrix...")
    X_train_c1_raw = np.vstack(feature_blocks).astype(np.float32)
    print(f"X_train_c1_raw shape: {X_train_c1_raw.shape} | Memory: {X_train_c1_raw.nbytes / (1024*1024):.2f} MB")

    # Strict integrity assertion against labels
    out_dir = os.path.join("data", "processed", "features")
    labels_path = os.path.join(out_dir, "train_labels.npy")
    if os.path.exists(labels_path):
        train_labels = np.load(labels_path)
        assert len(X_train_c1_raw) == len(train_labels), (
            f"Shape mismatch: C1 raw has {len(X_train_c1_raw)} rows but train_labels has {len(train_labels)} rows!"
        )
        print(f"Confirmed exact alignment with {labels_path}: exactly {len(train_labels):,} rows.")

    # Save explicit join key index
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "train_sentence_index.json")
    with open(index_path, "w") as f:
        json.dump(sentence_keys, f)
    print(f"Saved explicit join key registry to: {index_path} ({len(sentence_keys):,} sentence keys)")

    # Save raw feature matrix
    raw_path = os.path.join(out_dir, "train_features_c1_raw.npy")
    np.save(raw_path, X_train_c1_raw)
    print(f"Saved raw feature matrix to: {raw_path}")

    # -------------------------------------------------------------
    # Fit RobustScaler FRESH on full training matrix
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FITTING ROBUSTSCALER FRESH ON FULL 7,028-DOC TRAINING MATRIX (C1/C2)")
    print("=" * 80)

    scaler_c1 = FeatureScalerPipeline(config="C1", scaler_type="robust")
    X_train_c1_scaled = scaler_c1.fit_transform(X_train_c1_raw)

    params = scaler_c1.get_scaling_params()
    print("\nLEARNED FULL-CORPUS ROBUSTSCALER PARAMETERS (C1/C2):")
    print("-" * 75)
    for feat_name, p in params.items():
        print(f"  Feature: {feat_name:<12} | Median (center) = {p['center']:>10.4f} | IQR (scale) = {p['scale']:>10.4f}")
    print("-" * 75)

    # Save scaler and scaled matrix
    scaler_path = os.path.join(out_dir, "robust_scaler_c1.pkl")
    scaled_path = os.path.join(out_dir, "train_features_c1_scaled.npy")
    params_path = os.path.join(out_dir, "robust_scaler_c1_params.json")

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_c1, f)
    np.save(scaled_path, X_train_c1_scaled)
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"\nSaved RobustScaler model to: {scaler_path}")
    print(f"Saved scaled feature matrix to: {scaled_path} ({os.path.getsize(scaled_path)/(1024*1024):.2f} MB)")
    print(f"Saved scaling parameters to: {params_path}")


if __name__ == "__main__":
    main()
