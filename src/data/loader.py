"""
loader.py — Data loading and split management for IN-Abs dataset.

Directory structure expected:
data/
  raw/
    in_abs/
      judgments/        # one .txt per case, raw judgment text
      headnotes/        # one .txt per case, raw headnote/summary text
  splits/
    train_ids.txt       # 7,030 case IDs, one per line
    test_ids.txt        # 100 case IDs, one per line
"""

import os
import random
from typing import List, Tuple, Dict, Optional


def get_available_case_ids(
    judgments_dir: str = os.path.join("data", "raw", "in_abs", "judgments"),
    headnotes_dir: str = os.path.join("data", "raw", "in_abs", "headnotes"),
) -> List[str]:
    """
    Returns a sorted list of case IDs that have both a judgment and headnote file.
    A case ID is the filename stem (e.g. '100' from '100.txt').
    """
    if not os.path.isdir(judgments_dir):
        raise FileNotFoundError(f"Judgments directory not found: {judgments_dir}")
    if not os.path.isdir(headnotes_dir):
        raise FileNotFoundError(f"Headnotes directory not found: {headnotes_dir}")

    j_ids = {
        os.path.splitext(f)[0]
        for f in os.listdir(judgments_dir)
        if f.endswith(".txt")
    }
    h_ids = {
        os.path.splitext(f)[0]
        for f in os.listdir(headnotes_dir)
        if f.endswith(".txt")
    }

    common_ids = sorted(list(j_ids.intersection(h_ids)), key=lambda x: int(x) if x.isdigit() else x)
    return common_ids


def create_and_freeze_splits(
    case_ids: List[str],
    train_count: int = 7028,
    test_count: int = 100,
    seed: int = 42,
    splits_dir: str = os.path.join("data", "splits"),
) -> Tuple[List[str], List[str]]:
    """
    Splits case_ids into train and test sets using a fixed seed,
    and writes train_ids.txt and test_ids.txt to splits_dir.
    """
    os.makedirs(splits_dir, exist_ok=True)
    train_path = os.path.join(splits_dir, "train_ids.txt")
    test_path = os.path.join(splits_dir, "test_ids.txt")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Splits already exist at {splits_dir}. Loading existing splits...")
        return load_split_ids("train", splits_dir), load_split_ids("test", splits_dir)

    if len(case_ids) < (train_count + test_count):
        raise ValueError(
            f"Available case IDs ({len(case_ids)}) is less than requested train ({train_count}) + test ({test_count})"
        )

    # Sort deterministically before shuffling
    sorted_ids = sorted(case_ids, key=lambda x: int(x) if x.isdigit() else x)
    rng = random.Random(seed)
    shuffled = sorted_ids.copy()
    rng.shuffle(shuffled)

    # Sample test set first (100 docs)
    test_ids = sorted(shuffled[:test_count], key=lambda x: int(x) if x.isdigit() else x)
    # Remaining for train
    remaining_ids = shuffled[test_count:]
    train_ids = sorted(remaining_ids[:train_count], key=lambda x: int(x) if x.isdigit() else x)

    with open(train_path, "w", encoding="utf-8") as f:
        for cid in train_ids:
            f.write(f"{cid}\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for cid in test_ids:
            f.write(f"{cid}\n")

    print(f"Splits created: {len(train_ids)} train IDs, {len(test_ids)} test IDs (Seed: {seed})")
    return train_ids, test_ids


def load_split_ids(split_name: str, splits_dir: str = os.path.join("data", "splits")) -> List[str]:
    """
    Loads case IDs for a given split ('train' or 'test').
    """
    filename = f"{split_name}_ids.txt"
    filepath = os.path.join(splits_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Split file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids


def load_document(
    case_id: str,
    judgments_dir: str = os.path.join("data", "raw", "in_abs", "judgments"),
    headnotes_dir: str = os.path.join("data", "raw", "in_abs", "headnotes"),
) -> Dict[str, str]:
    """
    Loads raw judgment text and headnote text for a specific case ID.
    """
    j_path = os.path.join(judgments_dir, f"{case_id}.txt")
    h_path = os.path.join(headnotes_dir, f"{case_id}.txt")

    if not os.path.exists(j_path):
        raise FileNotFoundError(f"Judgment file not found: {j_path}")
    if not os.path.exists(h_path):
        raise FileNotFoundError(f"Headnote file not found: {h_path}")

    with open(j_path, "r", encoding="utf-8", errors="replace") as f:
        judgment_text = f.read()

    with open(h_path, "r", encoding="utf-8", errors="replace") as f:
        headnote_text = f.read()

    return {
        "case_id": case_id,
        "judgment": judgment_text,
        "headnote": headnote_text,
    }


def load_dataset_split(
    split_name: str,
    splits_dir: str = os.path.join("data", "splits"),
    judgments_dir: str = os.path.join("data", "raw", "in_abs", "judgments"),
    headnotes_dir: str = os.path.join("data", "raw", "in_abs", "headnotes"),
    limit: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Loads all documents for a split.
    """
    case_ids = load_split_ids(split_name, splits_dir=splits_dir)
    if limit is not None:
        case_ids = case_ids[:limit]

    docs = []
    for cid in case_ids:
        docs.append(load_document(cid, judgments_dir, headnotes_dir))
    return docs
