"""Plot the bacterial importance profile (produced by mutate_scan.py) with
the known -35 and -10 regulatory regions shaded, so the finding is visible
at a glance instead of buried in a table of 57 numbers. Saves a PNG for use
in the research poster/write-up.
"""

from pathlib import Path

import matplotlib

# This script only writes a PNG, so force the non-interactive "Agg" backend.
# Without it matplotlib tries to open a GUI window, which needs Tkinter --
# not always installed with Python on Windows, and unavailable on headless
# servers. Must be set before pyplot is imported.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from compare_to_known_motifs import KNOWN_REGIONS  # noqa: E402
from data_utils import TSS_INDEX  # noqa: E402

PROFILE_PATH = Path(__file__).resolve().parent.parent / "data" / "importance_profile.npy"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "importance_profile.png"

REGION_COLORS = ["salmon", "khaki"]


def main():
    profile = np.load(PROFILE_PATH)
    positions = np.arange(len(profile)) - TSS_INDEX

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(positions, profile, color="steelblue", width=0.8)

    for (name, (start, end)), color in zip(KNOWN_REGIONS.items(), REGION_COLORS):
        ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.4, label=name)

    ax.set_xlabel("Position relative to transcription start site (TSS)")
    ax.set_ylabel("Mutation-sensitivity (importance)")
    ax.set_title("Where the AI's confidence drops the most (E. coli promoters)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
