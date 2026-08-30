"""Day 4: Compare the AI-discovered importance profile against the known
biology of E. coli promoters.

Textbook consensus elements (Hawley & McClure 1983 and standard molecular
biology references):
  -35 element: consensus "TTGACA", centered ~ -35 to -30 relative to TSS
  -10 element ("Pribnow box"): consensus "TATAAT", centered ~ -10 to -7

This script does NOT claim to discover new biology -- it checks whether a
simple k-mer classifier, trained with no positional information at all,
independently rediscovers these two known regions as important. That
agreement (or disagreement) is the actual research result.
"""

from pathlib import Path

import numpy as np

from data_utils import TSS_INDEX

PROFILE_PATH = Path(__file__).resolve().parent.parent / "data" / "importance_profile.npy"

# Known regions, expressed as (start, end) relative to the TSS (inclusive).
KNOWN_REGIONS = {
    "-35 element": (-35, -30),
    "-10 element (Pribnow box)": (-10, -7),
}


def region_average_importance(profile: np.ndarray, start: int, end: int) -> float:
    indices = [TSS_INDEX + rel for rel in range(start, end + 1)]
    indices = [i for i in indices if 0 <= i < len(profile)]
    return float(np.mean(profile[indices])) if indices else float("nan")


def main():
    profile = np.load(PROFILE_PATH)
    overall_mean = float(np.mean(profile))

    print(f"Overall mean importance across all 57 positions: {overall_mean:.4f}\n")
    print("Known regulatory region vs. AI-flagged importance:")
    for name, (start, end) in KNOWN_REGIONS.items():
        avg = region_average_importance(profile, start, end)
        fold = avg / overall_mean if overall_mean else float("nan")
        print(f"  {name:30s} avg importance {avg:.4f}  ({fold:.2f}x background)")

    print(
        "\nIf both regions show importance well above 1x background, the model "
        "has independently rediscovered the known promoter architecture using "
        "only sequence statistics -- a concrete, checkable validation of the "
        "method, not just a black-box accuracy number."
    )


if __name__ == "__main__":
    main()
