"""Day 3-4: "Virtual CRISPR" saturation mutagenesis.

For each true-positive promoter sequence, mutate one position at a time to
each of the other three bases, re-score the trained model, and record how
much the promoter-probability drops. Averaging this across many promoters
gives a per-position "importance profile": positions where mutations cause
big confidence drops are positions the model has learned are functionally
important, without ever being told where the -10/-35 boxes are.
"""

import pickle
from pathlib import Path

import numpy as np

from data_utils import TSS_INDEX, load_records
from featurize import featurize_sequences

BASES = "acgt"
MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "model.pkl"
PROFILE_PATH = Path(__file__).resolve().parent.parent / "data" / "importance_profile.npy"


def load_model(path: Path = MODEL_PATH):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["k"]


def promoter_probability(model, k, sequence: str) -> float:
    X = featurize_sequences([sequence], k=k)
    return model.predict_proba(X)[0, 1]


def scan_sequence(model, k, sequence: str) -> np.ndarray:
    """Return an array of length SEQUENCE_LENGTH: the max confidence drop
    observed at each position when mutating it to any other base."""
    baseline = promoter_probability(model, k, sequence)
    drops = np.zeros(len(sequence))
    for pos in range(len(sequence)):
        original_base = sequence[pos]
        worst_drop = 0.0
        for base in BASES:
            if base == original_base:
                continue
            mutated = sequence[:pos] + base + sequence[pos + 1 :]
            prob = promoter_probability(model, k, mutated)
            drop = baseline - prob
            worst_drop = max(worst_drop, drop)
        drops[pos] = worst_drop
    return drops


def average_importance_profile(model, k, sequences: list[str]) -> np.ndarray:
    profiles = [scan_sequence(model, k, seq) for seq in sequences]
    return np.mean(profiles, axis=0)


def top_mutations(model, k, sequence: str, top_n: int = 3) -> list[dict]:
    """For a single sequence, find the individual single-letter mutations
    that cause the biggest confidence drop, across all positions and all
    alternate bases. This is the "before/after" demo: pick one sequence and
    show exactly which single-letter changes matter most."""
    baseline = promoter_probability(model, k, sequence)
    mutations = []
    for pos in range(len(sequence)):
        original_base = sequence[pos]
        for base in BASES:
            if base == original_base:
                continue
            mutated_seq = sequence[:pos] + base + sequence[pos + 1 :]
            prob = promoter_probability(model, k, mutated_seq)
            mutations.append(
                {
                    "position": pos,
                    "original_base": original_base,
                    "mutated_base": base,
                    "baseline_probability": baseline,
                    "mutated_probability": prob,
                    "drop": baseline - prob,
                }
            )
    mutations.sort(key=lambda m: m["drop"], reverse=True)
    return mutations[:top_n]


def main():
    model, k = load_model()
    records = load_records()
    promoters = [r.sequence for r in records if r.label == 1]

    print(f"Running saturation mutagenesis on {len(promoters)} promoter sequences...")
    profile = average_importance_profile(model, k, promoters)

    np.save(PROFILE_PATH, profile)
    print(f"Saved importance profile to {PROFILE_PATH}")

    # Positions relative to the transcription start site (index TSS_INDEX == TSS).
    # -10 box is typically found around relative position -10 to -7,
    # -35 box around relative position -35 to -30.
    for i, importance in enumerate(profile):
        relative_pos = i - TSS_INDEX
        print(f"position {relative_pos:+4d}  importance {importance:.4f}")

    top_positions = np.argsort(profile)[::-1][:10]
    print("\nTop 10 most important positions (index, relative-to-TSS, importance):")
    for idx in top_positions:
        relative_pos = idx - TSS_INDEX
        print(f"  index {idx:2d}  pos {relative_pos:+4d}  importance {profile[idx]:.4f}")


if __name__ == "__main__":
    main()
