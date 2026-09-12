"""
sanity_check_features.py — Comprehensive sanity inspection of all Phase 2 features:
1. TF-IDF (sum of weights per sentence)
2. Position (normalized j / M_i)
3. NER count (spaCy en_core_web_sm)
4. Word2Vec Cosine Sim (Sim_cos(s_ij, d_i) via word2vec-google-news-300 mean pooling)
5. SBERT Cosine Sim (Sim_cos(s_ij, d_i) via all-MiniLM-L6-v2)
6. WMD to Document (Word Mover's Distance s_ij to d_i)
7. Pairwise WMD (s_ij to previous sentence s_i(j-1))
"""

import time
import numpy as np

from src.data.loader import load_document
from src.data.preprocessing import segment_sentences, LegalTokenizer
from src.features.tfidf import fit_tfidf_vectorizer, compute_sentence_tfidf_scores, get_top_tfidf_terms_for_document
from src.features.position import compute_position_features
from src.features.ner import get_active_ner_backend, compute_sentence_ner_counts, extract_sample_entities
from src.features.embeddings_w2v import load_word2vec_model, compute_w2v_cosine_features
from src.features.embeddings_sbert import load_sbert_model, compute_sbert_cosine_features
from src.features.wmd import pairwise_wmd


def run_sanity_check():
    print("=" * 80)
    print("PHASE 2 EMBEDDINGS & WMD FEATURE SANITY CHECK")
    print("=" * 80)

    # 1. Load TF-IDF vectorizer
    t0 = time.time()
    vec = fit_tfidf_vectorizer()
    vocab_size = len(vec.vocabulary_)
    print(f"Loaded TF-IDF Vectorizer (vocab: {vocab_size:,}) in {time.time() - t0:.2f}s")

    # 2. Check NER Backend
    active_ner = get_active_ner_backend()
    print(f"Active NER Backend: {active_ner}")

    # 3. Load Word2Vec model
    t0 = time.time()
    w2v_model = load_word2vec_model()
    print(f"Loaded Word2Vec Model ({len(w2v_model):,} vocab, {w2v_model.vector_size}d) in {time.time() - t0:.2f}s")

    # 4. Load SBERT model
    t0 = time.time()
    sbert_model = load_sbert_model("all-MiniLM-L6-v2")
    print(f"Loaded SBERT Model (all-MiniLM-L6-v2) in {time.time() - t0:.2f}s")

    tokenizer = LegalTokenizer()
    sample_ids = ["5243", "914", "205"]

    print("\n" + "=" * 80)

    for cid in sample_ids:
        print(f"\nDOCUMENT CASE ID: {cid}")
        doc = load_document(cid)
        j_text = doc["judgment"]
        sents = segment_sentences(j_text)
        M_i = len(sents)

        print(f"Total Sentences (M_i): {M_i}")

        # Compute full-doc features
        t_w2v_start = time.time()
        w2v_cos_all = compute_w2v_cosine_features(sents, w2v_model, tokenizer=tokenizer)
        t_w2v = time.time() - t_w2v_start

        t_sbert_start = time.time()
        sbert_cos_all = compute_sbert_cosine_features(sents, sbert_model)
        t_sbert = time.time() - t_sbert_start

        pos_feats = compute_position_features(sents)
        tfidf_scores = compute_sentence_tfidf_scores(sents[:10], vec)
        ner_counts = compute_sentence_ner_counts(sents[:10])

        # Prepare doc tokens for WMD
        doc_tokens = [tok for s in sents for tok in tokenizer.tokenize_for_tfidf(s)]
        print(f"Total Word Tokens in Document: {len(doc_tokens):,} ({len(set(doc_tokens)):,} unique)")
        print(f"Feature Timing on full {M_i} sentences: W2V Cosine: {t_w2v:.3f}s | SBERT Cosine: {t_sbert:.3f}s")

        print("\nFirst 5 Sentences with Full Feature Vector (C1/C2 vs C3):")
        print(f"{'Idx':<4} | {'Pos':<6} | {'TF-IDF':<8} | {'NER':<4} | {'Cos_w2v':<8} | {'Cos_sbert':<9} | {'WMD(s,doc)':<10} | {'WMD(s,prev)':<11}")
        print("-" * 75)

        for i in range(min(5, M_i)):
            s_toks = tokenizer.tokenize_for_tfidf(sents[i])
            wmd_doc = pairwise_wmd(s_toks, doc_tokens, w2v_model, tokenizer=tokenizer)
            wmd_prev = pairwise_wmd(s_toks, tokenizer.tokenize_for_tfidf(sents[i - 1]), w2v_model, tokenizer=tokenizer) if i > 0 else 0.0

            print(
                f"{i:<4} | "
                f"{pos_feats[i]:<6.4f} | "
                f"{tfidf_scores[i]:<8.4f} | "
                f"{ner_counts[i]:<4} | "
                f"{w2v_cos_all[i]:<8.4f} | "
                f"{sbert_cos_all[i]:<9.4f} | "
                f"{wmd_doc:<10.4f} | "
                f"{wmd_prev:<11.4f}"
            )

        print("\nExcerpts & Observations for First 5 Sentences:")
        for i in range(min(5, M_i)):
            s_snippet = sents[i] if len(sents[i]) < 90 else sents[i][:90] + "..."
            ents = extract_sample_entities(sents[i])
            ent_strs = [f"{e[0]} ({e[1]})" for e in ents[:3]]
            print(f"  [{i}] \"{s_snippet}\"")
            if ent_strs:
                print(f"       NER entities: {ent_strs}")

        print("-" * 80)


if __name__ == "__main__":
    run_sanity_check()
