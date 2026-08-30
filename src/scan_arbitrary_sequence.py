"""The reusable tool: scan ANY DNA sequence a user provides, not just the
106 labeled training examples.

The trained classifier only ever saw 57-letter windows during training, so
to analyze a longer sequence we slide a 57-letter window across it,
score every window, and keep the highest-scoring ("most promoter-like")
regions. For each of those candidate regions we then run the same
saturation-mutagenesis interpretability method from mutate_scan.py to flag
exactly which bases the model is relying on. This is what makes the tool
useful on new/unlabeled sequences instead of just re-confirming a result on
data that was already labeled by hand.
"""

import pickle
from pathlib import Path

import numpy as np

from data_utils import SEQUENCE_LENGTH
from featurize import featurize_sequences
from mutate_scan import scan_sequence

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "model.pkl"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["k"]


def clean_sequence(raw: str) -> str:
    seq = "".join(raw.split()).lower()
    if not seq:
        raise ValueError("Empty sequence.")
    invalid = set(seq) - set("acgt")
    if invalid:
        raise ValueError(f"Sequence contains non-ACGT characters: {sorted(invalid)}")
    if len(seq) < SEQUENCE_LENGTH:
        raise ValueError(f"Sequence must be at least {SEQUENCE_LENGTH} bp long.")
    return seq


def sliding_window_scores(model, k, sequence: str, step: int = 1):
    """Score every 57-bp window in the sequence. Returns a list of
    (window_start_index, promoter_probability)."""
    starts = list(range(0, len(sequence) - SEQUENCE_LENGTH + 1, step))
    windows = [sequence[s : s + SEQUENCE_LENGTH] for s in starts]
    X = featurize_sequences(windows, k=k)
    probs = model.predict_proba(X)[:, 1]
    return list(zip(starts, probs))


def top_candidate_windows(model, k, sequence: str, top_n: int = 3, step: int = 1):
    """Return up to top_n non-overlapping windows with the highest
    promoter probability, sorted best-first."""
    scored = sliding_window_scores(model, k, sequence, step=step)
    scored.sort(key=lambda pair: pair[1], reverse=True)

    chosen = []
    for start, prob in scored:
        if all(abs(start - c_start) >= SEQUENCE_LENGTH for c_start, _ in chosen):
            chosen.append((start, prob))
        if len(chosen) >= top_n:
            break
    return chosen


def analyze_sequence(raw_sequence: str, top_n: int = 3, step: int = 1) -> list[dict]:
    """Full pipeline for a user-submitted sequence:
    1. Clean/validate input
    2. Find the most promoter-like region(s)
    3. Run saturation mutagenesis on each to get a per-base importance score

    Returns a list of dicts, one per candidate region, best first.
    """
    model, k = load_model()
    sequence = clean_sequence(raw_sequence)
    candidates = top_candidate_windows(model, k, sequence, top_n=top_n, step=step)

    results = []
    for start, prob in candidates:
        window_seq = sequence[start : start + SEQUENCE_LENGTH]
        importance = scan_sequence(model, k, window_seq)
        results.append(
            {
                "start": start,
                "end": start + SEQUENCE_LENGTH,
                "window_sequence": window_seq,
                "promoter_probability": float(prob),
                "importance_profile": importance,
            }
        )
    return results


if __name__ == "__main__":
    # A made-up longer stretch of DNA containing a real promoter (DEOP1)
    # embedded in the middle, to sanity-check that the scanner finds it.
    filler = "acgtacgtacgtacgtacgtacgtacgtacgtacgtacgt"
    known_promoter = (
        "cagaaacgttttattcgaacatcgatctcgtcttgtgttagaattctaacatacggt"
    )
    demo_sequence = filler + known_promoter + filler

    print(f"Scanning a {len(demo_sequence)}-bp sequence...\n")
    for i, result in enumerate(analyze_sequence(demo_sequence, top_n=2), start=1):
        print(f"Region {i}: positions {result['start']}-{result['end']}")
        print(f"  probability = {result['promoter_probability']:.3f}")
        print(f"  sequence    = {result['window_sequence']}")
        print()
