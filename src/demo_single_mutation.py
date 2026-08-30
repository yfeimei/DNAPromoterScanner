"""Live demo: pick one sequence, show its baseline promoter-probability,
then show the single-letter mutations that cause the biggest confidence
collapse. This is the most compelling thing to show someone in person --
"change one letter, watch the AI's confidence fall apart."

Usage:
    python demo_single_mutation.py --mode bacterial
    python demo_single_mutation.py --mode tata
"""

import argparse
from pathlib import Path

from data_utils import load_records
from mutate_scan import load_model, promoter_probability, top_mutations
from synthetic_tata_data import generate_dataset

BACTERIAL_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "model.pkl"
TATA_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "tata_model.pkl"


def pick_strongest_example(model, k, candidates: list[tuple[str, str]]) -> tuple[str, str]:
    """Pick the candidate with the highest baseline promoter-probability.

    All candidates here are true positives (real or planted promoters), so
    the one the model is most confident about makes the clearest live demo
    -- this is choosing a representative illustrative example, not
    cherry-picking a false result. Every candidate is a genuine promoter;
    we're only choosing which one to show."""
    best_name, best_seq, best_prob = None, None, -1.0
    for name, seq in candidates:
        prob = promoter_probability(model, k, seq)
        if prob > best_prob:
            best_name, best_seq, best_prob = name, seq, prob
    return best_name, best_seq


def pick_bacterial_example(model, k) -> tuple[str, str]:
    records = load_records()
    candidates = [(r.name, r.sequence) for r in records if r.label == 1]
    return pick_strongest_example(model, k, candidates)


def pick_tata_example(model, k, n_candidates: int = 20) -> tuple[str, str]:
    records = generate_dataset(n_per_class=n_candidates, seed=3)
    candidates = [(r.name, r.sequence) for r in records if r.label == 1]
    return pick_strongest_example(model, k, candidates)


def main():
    parser = argparse.ArgumentParser(description="Single-sequence mutation demo")
    parser.add_argument("--mode", choices=["bacterial", "tata"], default="tata")
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "bacterial":
        model, k = load_model(BACTERIAL_MODEL_PATH)
        name, sequence = pick_bacterial_example(model, k)
    else:
        model, k = load_model(TATA_MODEL_PATH)
        name, sequence = pick_tata_example(model, k)

    baseline = promoter_probability(model, k, sequence)
    print(f"Mode: {args.mode}")
    print(f"Sequence: {name}")
    print(f"  {sequence}")
    print(f"Baseline promoter-probability: {baseline:.1%}\n")

    mutations = top_mutations(model, k, sequence, top_n=args.top_n)
    print(f"Top {args.top_n} single-letter mutations that most collapse confidence:\n")
    for m in mutations:
        print(
            f"  position {m['position']:2d}: "
            f"{m['original_base']} -> {m['mutated_base']}   "
            f"{m['baseline_probability']:.1%} -> {m['mutated_probability']:.1%}  "
            f"(drop {m['drop']:.1%})"
        )


if __name__ == "__main__":
    main()
