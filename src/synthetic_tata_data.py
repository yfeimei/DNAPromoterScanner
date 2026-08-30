"""Synthetic TATA-box dataset generator.

This does NOT use any downloaded or real genomic data. It generates
sequences programmatically:
  - "promoter" (positive) sequences: random background DNA with a TATA-box
    motif inserted at a biologically realistic position and distance from
    the transcription start site (TSS)
  - "non-promoter" (negative) sequences: pure random background DNA, no
    motif inserted

The real TATA box consensus is usually written TATAWAW (W = A or T), found
roughly 25-30 bp upstream of the TSS in many eukaryotic core promoters
(the "Goldberg-Hogness box"). We resolve each W randomly to A or T, allow a
small position jitter (+/- 3 bp) and an occasional single point mutation,
so the task is not a trivial exact-string-match and mimics natural
sequence variability.

Because the "ground truth" motif location is something WE planted, this
dataset lets us do a clean control experiment: does the saturation
mutagenesis method (mutate_scan.py) correctly re-discover exactly where we
put it, using only the trained classifier -- with no positional
information ever given to the model during training?
"""

import random
from dataclasses import dataclass

BASES = "acgt"
SEQUENCE_LENGTH = 60
TSS_INDEX = 50  # index within the sequence corresponding to the TSS

# TATAWAW resolved with lowercase w as the wildcard marker
TATA_TEMPLATE = "tatawaw"
TATA_LENGTH = len(TATA_TEMPLATE)

# The 3' end of the TATA box sits ~25-30 bp upstream of the TSS in real
# eukaryotic promoters; this offset places the motif's start there.
TATA_OFFSET_FROM_TSS = -28
POSITION_JITTER = 3  # +/- bp of natural variability in exact placement
POINT_MUTATION_RATE = 0.15  # fraction of positives with one extra mismatch


@dataclass
class SyntheticRecord:
    label: int  # 1 = has planted TATA box, 0 = pure random background
    name: str
    sequence: str
    tata_start: int | None  # ground-truth index of the planted motif, or None


def _random_background(rng: random.Random, length: int = SEQUENCE_LENGTH) -> str:
    return "".join(rng.choice(BASES) for _ in range(length))


def _resolve_template(rng: random.Random, template: str = TATA_TEMPLATE) -> str:
    return "".join(rng.choice("at") if c == "w" else c for c in template)


def generate_positive(rng: random.Random, name: str) -> SyntheticRecord:
    background = list(_random_background(rng))
    motif = list(_resolve_template(rng))

    if rng.random() < POINT_MUTATION_RATE:
        pos = rng.randrange(len(motif))
        motif[pos] = rng.choice(BASES)

    jitter = rng.randint(-POSITION_JITTER, POSITION_JITTER)
    start = TSS_INDEX + TATA_OFFSET_FROM_TSS + jitter
    start = max(0, min(start, SEQUENCE_LENGTH - TATA_LENGTH))

    background[start : start + TATA_LENGTH] = motif
    sequence = "".join(background)
    return SyntheticRecord(label=1, name=name, sequence=sequence, tata_start=start)


def generate_negative(rng: random.Random, name: str) -> SyntheticRecord:
    sequence = _random_background(rng)
    return SyntheticRecord(label=0, name=name, sequence=sequence, tata_start=None)


def generate_dataset(
    n_per_class: int = 200, seed: int = 42
) -> list[SyntheticRecord]:
    rng = random.Random(seed)
    records = []
    for i in range(n_per_class):
        records.append(generate_positive(rng, name=f"synthetic_promoter_{i}"))
    for i in range(n_per_class):
        records.append(generate_negative(rng, name=f"synthetic_background_{i}"))
    rng.shuffle(records)
    return records


if __name__ == "__main__":
    records = generate_dataset(n_per_class=5, seed=1)
    for r in records[:6]:
        marker = f"(TATA planted at index {r.tata_start})" if r.tata_start is not None else "(no motif)"
        print(f"[{r.label}] {r.name}: {r.sequence} {marker}")
