"""
ner.py — Named Entity Recognition (NER) count feature per 02_METHODOLOGY.md & 01_DATA_SPEC.md.

Extraction rules:
- Uses spaCy (en_core_web_sm) if available; otherwise falls back gracefully to NLTK.
- Applies to original casing (never lowercased, per 01_DATA_SPEC.md).
- Computes raw named entity count per sentence.
"""

from typing import List, Tuple, Optional
import nltk
from nltk import pos_tag, ne_chunk, word_tokenize
from nltk.tree import Tree

# Global flag to track which engine is active
SPACY_AVAILABLE = False
_SPACY_NLP = None

try:
    import spacy
    try:
        _SPACY_NLP = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
        SPACY_AVAILABLE = True
    except Exception:
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False


def get_active_ner_backend() -> str:
    """Returns 'spaCy (en_core_web_sm)' or 'NLTK (ne_chunk)'."""
    return "spaCy (en_core_web_sm)" if SPACY_AVAILABLE else "NLTK (ne_chunk)"


def count_named_entities_spacy(sentence: str) -> int:
    """Count named entities using spaCy pipeline."""
    if not sentence.strip():
        return 0
    doc = _SPACY_NLP(sentence)
    return len(doc.ents)


def count_named_entities_nltk(sentence: str) -> int:
    """Count named entities using NLTK pos_tag + ne_chunk."""
    if not sentence.strip():
        return 0
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    chunked = ne_chunk(tagged)
    count = 0
    for subtree in chunked:
        if isinstance(subtree, Tree):
            count += 1
    return count


def compute_sentence_ner_counts(sentences: List[str]) -> List[int]:
    """
    Computes named entity count for each sentence in sentences list.
    Preserves original surface case.
    """
    if not sentences:
        return []

    if SPACY_AVAILABLE:
        # Use spaCy pipe for efficient batching if spaCy is installed
        counts = []
        for doc in _SPACY_NLP.pipe(sentences, batch_size=64):
            counts.append(len(doc.ents))
        return counts
    else:
        # NLTK fallback
        return [count_named_entities_nltk(s) for s in sentences]


def extract_sample_entities(sentence: str) -> List[Tuple[str, str]]:
    """
    Extracts entities with labels for sanity inspection.
    """
    if not sentence.strip():
        return []

    if SPACY_AVAILABLE:
        doc = _SPACY_NLP(sentence)
        return [(ent.text, ent.label_) for ent in doc.ents]
    else:
        tokens = word_tokenize(sentence)
        tagged = pos_tag(tokens)
        chunked = ne_chunk(tagged)
        entities = []
        for subtree in chunked:
            if isinstance(subtree, Tree):
                ent_text = " ".join([token for token, tag in subtree.leaves()])
                entities.append((ent_text, subtree.label()))
        return entities
