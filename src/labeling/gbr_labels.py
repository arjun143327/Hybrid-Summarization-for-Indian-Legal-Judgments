"""
gbr_labels.py — Ground-truth sentence importance labeling for GBR training
per 02_METHODOLOGY.md Stage 2.

Formula:
  y_ij = max_{r_k in headnote_i} ROUGE-1_F1(s_ij, r_k)

Each source sentence is scored by its best single-sentence match in the reference headnote.
This exact max-match-to-reference-sentence approach is the locked-in methodology decision
(whole-blob comparison and full greedy-oracle search are intentionally not used).
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from rouge_score import rouge_scorer
from rouge_score.rouge_scorer import _score_ngrams, _create_ngrams

from src.data.preprocessing import segment_sentences


class GBRLabeler:
    """
    Computes ground-truth supervision labels y_ij for Gradient Boosting Regressor (GBR)
    training using max ROUGE-1 F1 match against reference headnote sentences.
    """

    def __init__(self, use_stemmer: bool = True):
        self.scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=use_stemmer)
        self.tokenizer = self.scorer._tokenizer

    def compute_sentence_label(
        self,
        source_sentence: str,
        reference_sentences: List[str]
    ) -> Tuple[float, int, str]:
        """
        Computes y_ij for a single source sentence against all reference sentences:
        y_ij = max_{r_k} ROUGE-1_F1(s_ij, r_k)

        Returns
        -------
        Tuple[float, int, str]
            (max_f1_score, best_match_ref_idx, best_match_ref_text)
        """
        if not source_sentence or not source_sentence.strip() or not reference_sentences:
            return 0.0, -1, ""

        s_tokens = self.tokenizer.tokenize(source_sentence)
        if not s_tokens:
            return 0.0, -1, ""

        s_ngrams = _create_ngrams(s_tokens, 1)

        best_score = 0.0
        best_idx = -1
        best_ref = ""

        for k, r_sent in enumerate(reference_sentences):
            if not r_sent or not r_sent.strip():
                continue
            r_tokens = self.tokenizer.tokenize(r_sent)
            if not r_tokens:
                continue
            r_ngrams = _create_ngrams(r_tokens, 1)
            score = _score_ngrams(r_ngrams, s_ngrams).fmeasure

            if score > best_score:
                best_score = score
                best_idx = k
                best_ref = r_sent

        return float(best_score), best_idx, best_ref

    def compute_document_labels(
        self,
        source_sentences: List[str],
        reference_sentences: List[str]
    ) -> Tuple[np.ndarray, List[int], List[str]]:
        """
        Computes y_ij labels for all source sentences in a document.
        Pre-tokenizes reference sentences for optimal performance.

        Parameters
        ----------
        source_sentences : List[str]
            List of segmented sentences from the judgment text.
        reference_sentences : List[str]
            List of segmented sentences from the reference headnote.

        Returns
        -------
        Tuple[np.ndarray, List[int], List[str]]
            - labels: 1D array of shape (len(source_sentences),) with float values in [0.0, 1.0].
            - match_indices: List of best-matching reference sentence index for each source sentence.
            - match_texts: List of best-matching reference sentence text for each source sentence.
        """
        n_src = len(source_sentences)
        if n_src == 0 or not reference_sentences:
            return np.zeros(n_src, dtype=np.float32), [-1] * n_src, [""] * n_src

        # Pre-tokenize reference sentences once
        ref_ngrams_list = []
        valid_refs = []
        for k, r_sent in enumerate(reference_sentences):
            if not r_sent or not r_sent.strip():
                ref_ngrams_list.append(None)
                valid_refs.append(False)
                continue
            r_tokens = self.tokenizer.tokenize(r_sent)
            if not r_tokens:
                ref_ngrams_list.append(None)
                valid_refs.append(False)
            else:
                ref_ngrams_list.append(_create_ngrams(r_tokens, 1))
                valid_refs.append(True)

        labels = np.zeros(n_src, dtype=np.float32)
        match_indices = [-1] * n_src
        match_texts = [""] * n_src

        for j, s_sent in enumerate(source_sentences):
            if not s_sent or not s_sent.strip():
                continue
            s_tokens = self.tokenizer.tokenize(s_sent)
            if not s_tokens:
                continue
            s_ngrams = _create_ngrams(s_tokens, 1)

            best_f1 = 0.0
            best_k = -1

            for k, (is_valid, r_ngrams) in enumerate(zip(valid_refs, ref_ngrams_list)):
                if not is_valid:
                    continue
                score = _score_ngrams(r_ngrams, s_ngrams).fmeasure
                if score > best_f1:
                    best_f1 = score
                    best_k = k

            labels[j] = best_f1
            match_indices[j] = best_k
            match_texts[j] = reference_sentences[best_k] if best_k >= 0 else ""

        return labels, match_indices, match_texts

    def compute_labels_for_doc_dict(
        self,
        doc_dict: Dict[str, str]
    ) -> Dict:
        """
        Segments judgment and headnote and computes sentence-level ground truth labels y_ij.

        Parameters
        ----------
        doc_dict : Dict[str, str]
            Dictionary containing 'case_id', 'judgment', and 'headnote'.

        Returns
        -------
        Dict
            {
                'case_id': str,
                'source_sentences': List[str],
                'reference_sentences': List[str],
                'labels': np.ndarray,
                'match_indices': List[int],
                'match_texts': List[str]
            }
        """
        src_sents = segment_sentences(doc_dict["judgment"])
        ref_sents = segment_sentences(doc_dict["headnote"])

        labels, match_indices, match_texts = self.compute_document_labels(
            src_sents, ref_sents
        )

        return {
            "case_id": doc_dict.get("case_id", ""),
            "source_sentences": src_sents,
            "reference_sentences": ref_sents,
            "labels": labels,
            "match_indices": match_indices,
            "match_texts": match_texts,
        }


def get_label_distribution_summary(labels: np.ndarray) -> Dict[str, float]:
    """
    Computes summary statistics for a label array.
    """
    if len(labels) == 0:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "p50": 0.0}

    return {
        "count": int(len(labels)),
        "min": float(np.min(labels)),
        "max": float(np.max(labels)),
        "mean": float(np.mean(labels)),
        "std": float(np.std(labels)),
        "p50": float(np.median(labels)),
        "p75": float(np.percentile(labels, 75)),
        "p90": float(np.percentile(labels, 90)),
    }
