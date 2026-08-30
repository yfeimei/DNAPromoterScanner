"""Validate the saturation-mutagenesis method on the synthetic TATA dataset.

This is the key control experiment: because we planted the TATA-box motif
ourselves in synthetic_tata_data.py, we know EXACTLY where it is in every
positive sequence. The classifier was never told this position during
training -- it only saw whole 60-letter sequences labeled 0/1. If the
mutation-scan importance profile lights up at the same position we planted
the motif, that's a clean, checkable proof that the method correctly
locates the functional element, with a known ground truth to compare
against (unlike the E. coli case, where we only have literature values).
"""

from pathlib import Path

import numpy as np

from mutate_scan import average_importance_profile, load_model
from synthetic_tata_data import (
    TATA_LENGTH,
    TATA_OFFSET_FROM_TSS,
    TSS_INDEX,
    generate_dataset,
)

TATA_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "tata_model.pkl"
TATA_PROFILE_PATH = Path(__file__).resolve().parent.parent / "data" / "tata_importance_profile.npy"

N_VALIDATION_SEQUENCES = 100


def main():
    model, k = load_model(TATA_MODEL_PATH)

    records = generate_dataset(n_per_class=N_VALIDATION_SEQUENCES, seed=999)
    positives = [r for r in records if r.label == 1]

    print(f"Running saturation mutagenesis on {len(positives)} synthetic promoters...")
    profile = average_importance_profile(model, k, [r.sequence for r in positives])
    np.save(TATA_PROFILE_PATH, profile)
    print(f"Saved importance profile to {TATA_PROFILE_PATH}\n")

    # Ground-truth planted region (nominal position, ignoring per-sequence jitter)
    planted_start = TSS_INDEX + TATA_OFFSET_FROM_TSS
    planted_end = planted_start + TATA_LENGTH
    planted_region_importance = float(np.mean(profile[planted_start:planted_end]))
    overall_mean = float(np.mean(profile))
    fold = planted_region_importance / overall_mean if overall_mean else float("nan")

    print(f"Overall mean importance across all 60 positions: {overall_mean:.4f}")
    print(
        f"Mean importance in the planted TATA region "
        f"(indices {planted_start}-{planted_end}): {planted_region_importance:.4f} "
        f"({fold:.2f}x background)"
    )

    top_positions = np.argsort(profile)[::-1][:10]
    print("\nTop 10 most important positions found by the model:")
    for idx in top_positions:
        in_planted_region = planted_start <= idx < planted_end
        flag = "  <-- inside planted TATA region" if in_planted_region else ""
        print(f"  index {idx:2d}  importance {profile[idx]:.4f}{flag}")

    print(
        "\nIf the top positions cluster inside the planted region and the "
        "fold-enrichment is well above 1x, the method correctly re-discovers "
        "a functional motif using only the classifier's learned behavior -- "
        "with a fully known ground truth, unlike the E. coli case where we "
        "only have literature values to compare against."
    )


if __name__ == "__main__":
    main()
