"""Rule-based motif labeling.

Deliberately NOT an AI/LLM step -- plain pattern matching, so the website
can label results instantly, for free, with no API calls and no per-user
cost. Given a candidate region and its per-position importance profile
(from mutate_scan.scan_sequence), this looks at the DNA directly under the
most mutation-sensitive stretch and checks whether it resembles a known,
well-established consensus motif within a small number of mismatches.

This is what turns a raw importance profile into something a non-expert
user can actually understand ("this looks like a known -10 box") instead
of just a wall of numbers.
"""

import numpy as np

KNOWN_MOTIFS = {
    "-10 element (Pribnow box)": "tataat",
    "-35 element": "ttgaca",
}

# Used in TATA-box demo mode (tata_scan_arbitrary_sequence.py). Kept
# separate from KNOWN_MOTIFS so bacterial-mode results are never labeled
# against a eukaryotic motif by mistake.
TATA_MOTIFS = {
    "TATA box (Goldberg-Hogness box)": "tataaaa",
}

MAX_MISMATCHES = 2


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def most_important_span(importance: np.ndarray, span_length: int) -> tuple[int, int]:
    """Find the span_length-wide window of consecutive positions with the
    highest total importance score."""
    best_start, best_score = 0, -1.0
    for start in range(len(importance) - span_length + 1):
        score = float(np.sum(importance[start : start + span_length]))
        if score > best_score:
            best_start, best_score = start, score
    return best_start, best_start + span_length


def label_window(
    window_sequence: str, importance: np.ndarray, motifs: dict[str, str] = KNOWN_MOTIFS
) -> list[dict]:
    """Return a list of matches (possibly empty) describing any known motif
    that approximately lines up with the model's most mutation-sensitive
    region of this window. Pass motifs=TATA_MOTIFS for TATA-box demo mode."""
    labels = []
    for name, consensus in motifs.items():
        span_length = len(consensus)
        start, end = most_important_span(importance, span_length)
        candidate = window_sequence[start:end]
        mismatches = hamming(candidate, consensus)
        if mismatches <= MAX_MISMATCHES:
            labels.append(
                {
                    "name": name,
                    "consensus": consensus,
                    "matched_text": candidate,
                    "position_in_window": (start, end),
                    "mismatches": mismatches,
                }
            )
    return labels


if __name__ == "__main__":
    # sanity example: a window whose most important 6 letters are "tataat"
    fake_window = "acgtac" + "tataat" + "acgtacgtacgtacgtacgtacgtacgtacgtacgtacgtacgtac"
    fake_importance = np.zeros(len(fake_window))
    fake_importance[6:12] = 1.0  # mark the "tataat" span as most important
    print(label_window(fake_window, fake_importance))
