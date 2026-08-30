"""Plot the synthetic TATA-box importance profile (produced by
tata_validate.py) with the ground-truth planted region shaded. Saves a PNG
for the poster/write-up. Unlike the bacterial version, here the shaded
region is not from the literature -- it's exactly where WE planted the
motif during data generation, so this plot is a direct visual check of the
method against a fully known answer.
"""

from pathlib import Path

import matplotlib

# Non-interactive backend: this script only saves a PNG, and requiring a GUI
# backend (Tkinter) breaks on Windows installs that skipped the tcl/tk
# option and on headless servers. Must precede the pyplot import.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from synthetic_tata_data import TATA_LENGTH, TATA_OFFSET_FROM_TSS, TSS_INDEX  # noqa: E402

PROFILE_PATH = Path(__file__).resolve().parent.parent / "data" / "tata_importance_profile.npy"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "tata_importance_profile.png"


def main():
    profile = np.load(PROFILE_PATH)
    positions = np.arange(len(profile)) - TSS_INDEX

    planted_start = TATA_OFFSET_FROM_TSS
    planted_end = TATA_OFFSET_FROM_TSS + TATA_LENGTH

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(positions, profile, color="mediumseagreen", width=0.8)
    ax.axvspan(
        planted_start - 0.5,
        planted_end - 0.5,
        color="salmon",
        alpha=0.4,
        label="Planted TATA box (ground truth, nominal position)",
    )

    ax.set_xlabel("Position relative to transcription start site (TSS)")
    ax.set_ylabel("Mutation-sensitivity (importance)")
    ax.set_title("Where the AI's confidence drops the most (synthetic TATA-box demo)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
