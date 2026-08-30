"""TATA-box mode of the reusable scanner (see scan_arbitrary_sequence.py for
the E. coli/bacterial version). Uses the model trained on synthetic data
in train_tata_model.py, and 60-letter windows instead of 57.

Same idea: slide the window across a user-submitted sequence, keep the
highest-scoring region(s), and run saturation mutagenesis on them to flag
the exact bases driving the prediction.
"""

from pathlib import Path

from featurize import featurize_sequences
from mutate_scan import load_model, scan_sequence
from synthetic_tata_data import SEQUENCE_LENGTH

TATA_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "tata_model.pkl"


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
    starts = list(range(0, len(sequence) - SEQUENCE_LENGTH + 1, step))
    windows = [sequence[s : s + SEQUENCE_LENGTH] for s in starts]
    X = featurize_sequences(windows, k=k)
    probs = model.predict_proba(X)[:, 1]
    return list(zip(starts, probs))


def top_candidate_windows(model, k, sequence: str, top_n: int = 3, step: int = 1):
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
    model, k = load_model(TATA_MODEL_PATH)
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
    from synthetic_tata_data import generate_dataset

    demo_record = next(r for r in generate_dataset(n_per_class=1, seed=7) if r.label == 1)
    filler = "acgtacgtacgtacgtacgtacgtacgtacgtacgtacgt"
    demo_sequence = filler + demo_record.sequence + filler

    print(f"Ground truth: TATA box planted at index {demo_record.tata_start} within the inner sequence\n")
    for i, result in enumerate(analyze_sequence(demo_sequence, top_n=2), start=1):
        print(f"Region {i}: positions {result['start']}-{result['end']}")
        print(f"  probability = {result['promoter_probability']:.3f}")
        print(f"  sequence    = {result['window_sequence']}")
