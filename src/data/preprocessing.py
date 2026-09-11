"""
preprocessing.py — Text preprocessing, sentence segmentation with legal citation protection,
and tokenization per 01_DATA_SPEC.md.
"""

import re
from typing import List
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Comprehensive legal abbreviations and citation components observed in Indian Supreme Court judgments
LEGAL_ABBREVS = [
    # Reporters & Law Journals
    "AIR", "SCC", "SCR", "Cr", "Cr.L.J", "L.J", "All", "Mad", "Cal", "Bom", 
    "Del", "SC", "ILR", "Comp", "Cas", "SCALE", "JT", "Supp", "KLT", "CTC",
    # Procedural & Party Designations
    "v", "vs", "Ors", "Anr", "No", "Nos", "Art", "Arts", "Sec", "Secs", "Cl", 
    "para", "paras", "Ref", "App", "Appl", "Govt", "Ltd", "Pvt", "Corp",
    # Judicial Titles & Honors
    "Hon'ble", "Hon", "Mr", "Mrs", "Ms", "Dr", "Prof", "C.J", "C.J.I", "J", "JJ",
    # Latin / Common Legal Terms
    "ibid", "id", "viz", "e.g", "i.e", "al", "etc"
]

# Date pattern to protect internal periods in dates (e.g., 24/25.9.1974, 26.9.1972, 1.5.1979, with optional newline)
DATE_REGEX = re.compile(r"\b(\d{1,2}(?:/\d{1,2})?)\.(\d{1,2})\.\s*(\d{2,4})\b")

# Case citation patterns (e.g., [1995] 2 S.C.R. 450, AIR 1978 SC 597, (1973) 2 SCC 235)
CITATION_REGEXES = [
    # AIR citations: AIR 1978 SC 597, A.I.R. 1950 S.C. 27
    re.compile(r"\bA\.?I\.?R\.?\s*(\d{4})?\s*([A-Z][A-Za-z.]*)?\s*(\d+)?", re.IGNORECASE),
    # SCC citations: (1973) 2 SCC 235, (2000) 1 S.C.C. 100, 1995 Supp (3) SCC 450
    re.compile(r"\(\d{4}\)\s*\d*\s*S\.?C\.?C\.?(\s*\(Supp\))?\s*\d+", re.IGNORECASE),
    # SCR citations: [1978] 2 S.C.R. 597, (1973) 2 SCR 235
    re.compile(r"[\[\(]\d{4}[\]\)]\s*\d*\s*S\.?C\.?R\.?\s*\d+", re.IGNORECASE),
    # Section numbers like Sec. 302, S. 302, Art. 21, Arts. 14, 19
    re.compile(r"\b(Sec|Secs|Art|Arts|S|cl)\.\s*\d+", re.IGNORECASE),
    # Versus: v. or vs.
    re.compile(r"\b(v|vs)\.", re.IGNORECASE),
    # Number: No. 123 or NO. 123
    re.compile(r"\bNo\.\s*\d+", re.IGNORECASE),
    # Single uppercase initial followed by period (e.g. A. K. Gopalan)
    re.compile(r"\b([A-Z])\.(?=\s*[A-Z])"),
]

PERIOD_PLACEHOLDER = "<PERIOD>"


def protect_citations(text: str) -> str:
    """
    Replaces periods inside known legal abbreviations, decimal dates, and citation patterns
    with a placeholder before sentence segmentation.
    """
    if not text:
        return ""

    # 1. Protect decimal dates (e.g. 24/25.9.1974, 26.9.1972, 1.5.1979) applied first
    text = DATE_REGEX.sub(rf"\1{PERIOD_PLACEHOLDER}\2{PERIOD_PLACEHOLDER}\3", text)

    # 2. Protect specific citation patterns
    def replace_periods_in_match(match):
        return match.group(0).replace(".", PERIOD_PLACEHOLDER)

    for regex in CITATION_REGEXES:
        text = regex.sub(replace_periods_in_match, text)

    # 3. Protect abbreviation list (case-insensitive to cover NO., ORS., ART., etc.)
    for abbr in LEGAL_ABBREVS:
        pattern = rf"\b{re.escape(abbr)}\."
        text = re.sub(
            pattern,
            lambda m: m.group(0).replace(".", PERIOD_PLACEHOLDER),
            text,
            flags=re.IGNORECASE,
        )

    # 4. Protect decimal numbers (e.g., 3.14, Rs. 10.50)
    text = re.sub(r"(\d+)\.(\d+)", rf"\1{PERIOD_PLACEHOLDER}\2", text)

    return text


def restore_periods(sentences: List[str]) -> List[str]:
    """
    Restores protected period placeholders back to actual periods.
    """
    return [s.replace(PERIOD_PLACEHOLDER, ".") for s in sentences]


def segment_sentences(text: str) -> List[str]:
    """
    Segments document text into sentences using citation-protected NLTK sent_tokenize.
    Filters out empty or pure-whitespace strings.
    """
    if not text or not text.strip():
        return []

    # Clean extreme whitespace/newlines while preserving sentence flow
    cleaned_text = re.sub(r"\r\n", "\n", text)
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)

    # Apply citation protection
    protected_text = protect_citations(cleaned_text)

    # Segment sentences
    raw_sentences = sent_tokenize(protected_text)

    # Restore periods
    sentences = restore_periods(raw_sentences)

    # Strip and filter
    final_sentences = [s.strip() for s in sentences if s.strip()]
    return final_sentences


class LegalTokenizer:
    """
    Tokenizer for the TF-IDF feature pipeline per 01_DATA_SPEC.md:
    - Lowercases text for TF-IDF
    - Standard NLTK English stopwords (DOES NOT strip legal stopwords like 'held', 'appellant')
    - WordNet Lemmatization
    """

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def tokenize_for_tfidf(self, text: str) -> List[str]:
        """
        Tokenize, remove punctuation, filter stopwords, and lemmatize for TF-IDF path.
        """
        if not text:
            return []

        tokens = word_tokenize(text.lower())
        processed = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalnum() and token not in self.stop_words
        ]
        return processed
