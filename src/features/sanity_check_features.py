"""
sanity_check_features.py — Runs sanity inspection on TF-IDF, Position, and NER features
for sample documents per Phase 2 instructions.
"""

from src.data.loader import load_document
from src.data.preprocessing import segment_sentences
from src.features.tfidf import fit_tfidf_vectorizer, compute_sentence_tfidf_scores, get_top_tfidf_terms_for_document
from src.features.position import compute_position_features
from src.features.ner import get_active_ner_backend, compute_sentence_ner_counts, extract_sample_entities

def run_sanity_check():
    vec = fit_tfidf_vectorizer()
    vocab_size = len(vec.vocabulary_)
    active_ner = get_active_ner_backend()

    print("=" * 80)
    print(f"TF-IDF Fitted Vocabulary Size: {vocab_size:,}")
    print(f"Active NER Backend: {active_ner}")
    print("=" * 80)

    sample_ids = ["5243", "914", "205"]

    for cid in sample_ids:
        doc = load_document(cid)
        j_text = doc["judgment"]
        sents = segment_sentences(j_text)

        top_terms = get_top_tfidf_terms_for_document(j_text, vec, top_k=10)
        pos_feats = compute_position_features(sents)
        ner_counts = compute_sentence_ner_counts(sents[:10])
        tfidf_scores = compute_sentence_tfidf_scores(sents[:10], vec)

        print(f"\nDocument Case ID: {cid}")
        print(f"Total Sentences in Document (M_i): {len(sents)}")
        print("Top 10 TF-IDF Terms in Document:")
        for term, score in top_terms:
            print(f"  - {term}: {score}")

        print("\nFirst 5 Sentences with Extracted Features:")
        for i in range(min(5, len(sents))):
            s_snippet = sents[i] if len(sents[i]) < 100 else sents[i][:100] + "..."
            ents = extract_sample_entities(sents[i])
            ent_strs = [f"{e[0]} ({e[1]})" for e in ents[:4]]

            print(f"  [Sentence {i}]")
            print(f"    Text: \"{s_snippet}\"")
            print(f"    Normalized Position (j / M_i): {pos_feats[i]:.4f} ({i} / {len(sents)})")
            print(f"    TF-IDF Score: {tfidf_scores[i]:.4f}")
            print(f"    NER Count: {ner_counts[i]}")
            if ent_strs:
                print(f"    Sample Entities: {ent_strs}")
        print("-" * 80)

if __name__ == "__main__":
    run_sanity_check()
