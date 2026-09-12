"""
sanity_check_labels.py — Sanity verification of GBR ground-truth labeling and feature scaling
for Case IDs 5243, 914, and 205 per Phase 3 instructions.

Verifies:
1. GBR label generation: y_ij = max_{r_k} ROUGE-1_F1(s_ij, r_k)
2. Label distribution across sample judgments (min, max, mean, median, std)
3. Top-5 highest-scoring sentences per document with their matching reference sentence
4. Feature matrix assembly for C1/C2 vs C3 and RobustScaler verification
"""

import time
import numpy as np

from src.data.loader import load_document
from src.labeling.gbr_labels import GBRLabeler, get_label_distribution_summary
from src.features.tfidf import fit_tfidf_vectorizer
from src.features.embeddings_w2v import load_word2vec_model
from src.features.embeddings_sbert import load_sbert_model
from src.features.build_features import assemble_document_raw_features, FeatureScalerPipeline, CONFIG_FEATURE_NAMES


def run_label_sanity_check():
    print("=" * 85)
    print("PHASE 3 GBR GROUND-TRUTH LABELING & SCALING SANITY CHECK")
    print("=" * 85)

    labeler = GBRLabeler(use_stemmer=True)
    sample_ids = ["5243", "914", "205"]

    label_results = {}

    for cid in sample_ids:
        doc = load_document(cid)
        t0 = time.time()
        res = labeler.compute_labels_for_doc_dict(doc)
        elapsed = time.time() - t0
        label_results[cid] = res

        src_sents = res["source_sentences"]
        ref_sents = res["reference_sentences"]
        labels = res["labels"]
        match_idx = res["match_indices"]
        match_txt = res["match_texts"]

        dist = get_label_distribution_summary(labels)

        print(f"\nDOCUMENT CASE ID: {cid}")
        print(f"  Source Sentences (M_i): {len(src_sents):,} | Reference Headnote Sentences (K_i): {len(ref_sents):,}")
        print(f"  Labeling Time: {elapsed:.3f}s ({len(src_sents) * len(ref_sents):,} sentence comparisons)")
        print(f"  Label Distribution (y_ij):")
        print(f"    - Min:    {dist['min']:.4f}")
        print(f"    - Max:    {dist['max']:.4f}")
        print(f"    - Mean:   {dist['mean']:.4f}")
        print(f"    - Median: {dist['p50']:.4f}")
        print(f"    - Std:    {dist['std']:.4f}")
        print(f"    - 75th %: {dist['p75']:.4f}")
        print(f"    - 90th %: {dist['p90']:.4f}")

        # Find top-5 highest-scoring sentences
        top5_indices = np.argsort(labels)[::-1][:5]

        print("\n  Top-5 Highest-Labeled Source Sentences (y_ij):")
        print("  " + "-" * 81)
        for rank, s_idx in enumerate(top5_indices, 1):
            s_text = src_sents[s_idx]
            s_snippet = s_text if len(s_text) <= 120 else s_text[:120] + "..."
            r_k = match_idx[s_idx]
            r_text = match_txt[s_idx]
            r_snippet = r_text if len(r_text) <= 120 else r_text[:120] + "..."

            print(f"  [Rank {rank}] Source Sent #{s_idx} | y_ij (ROUGE-1 F1) = {labels[s_idx]:.4f}")
            print(f"    Source:    \"{s_snippet}\"")
            print(f"    Ref Match (#{r_k}): \"{r_snippet}\"")
            print()

        # Also inspect lowest 2 sentences to ensure boilerplate/headers score low
        bot2_indices = np.argsort(labels)[:2]
        print("  Lowest-Labeled Sentences (Boilerplate Check):")
        for s_idx in bot2_indices:
            s_text = src_sents[s_idx]
            s_snippet = s_text if len(s_text) <= 80 else s_text[:80] + "..."
            print(f"    Sent #{s_idx}: \"{s_snippet}\" -> y_ij = {labels[s_idx]:.4f}")

        print("=" * 85)

    # -------------------------------------------------------------
    # Feature Assembly & RobustScaler Pipeline Verification
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print("FEATURE ASSEMBLY & ROBUSTSCALER VERIFICATION")
    print("=" * 85)

    print("Loading models for feature assembly...")
    vec = fit_tfidf_vectorizer()
    w2v = load_word2vec_model()
    sbert = load_sbert_model()

    all_raw_c1 = []
    all_raw_c3 = []

    for cid in sample_ids:
        doc = load_document(cid)
        sents = label_results[cid]["source_sentences"]

        x_c1 = assemble_document_raw_features(sents, config="C1", tfidf_vectorizer=vec, w2v_model=w2v)
        x_c3 = assemble_document_raw_features(sents, config="C3", tfidf_vectorizer=vec, sbert_model=sbert)

        all_raw_c1.append(x_c1)
        all_raw_c3.append(x_c3)

    X_sample_c1 = np.vstack(all_raw_c1)
    X_sample_c3 = np.vstack(all_raw_c3)

    print(f"\nAssembled Sample Feature Matrices across 3 docs ({X_sample_c1.shape[0]} total sentences):")
    print(f"  Config C1 shape: {X_sample_c1.shape} (Features: {CONFIG_FEATURE_NAMES['C1']})")
    print(f"  Config C3 shape: {X_sample_c3.shape} (Features: {CONFIG_FEATURE_NAMES['C3']})")

    # Fit RobustScaler
    scaler_c1 = FeatureScalerPipeline(config="C1", scaler_type="robust")
    X_scaled_c1 = scaler_c1.fit_transform(X_sample_c1)

    print("\nRobustScaler Parameters Learned on Sample Sentences (C1):")
    params = scaler_c1.get_scaling_params()
    for feat_name, p in params.items():
        print(f"  - {feat_name:<12}: Median (center) = {p['center']:>8.4f} | IQR (scale) = {p['scale']:>8.4f}")

    # Inspect outlier handling: Case 914 Sentence 2 (NER count 28)
    # Find sentence in stacked matrix
    c5243_len = len(label_results["5243"]["source_sentences"])
    case914_sent2_idx = c5243_len + 2  # Sent 2 in Case 914

    raw_row = X_sample_c1[case914_sent2_idx]
    scaled_row = X_scaled_c1[case914_sent2_idx]

    print("\nOutlier Scaling Demonstration (Case 914 Sentence 2 — merged appeal titles):")
    for feat_name, raw_val, scaled_val in zip(CONFIG_FEATURE_NAMES["C1"], raw_row, scaled_row):
        print(f"  {feat_name:<12}: Raw = {raw_val:>8.4f}  -->  RobustScaled = {scaled_val:>8.4f}")

    print("\nPipeline check SUCCESSFUL!")


if __name__ == "__main__":
    run_label_sanity_check()
