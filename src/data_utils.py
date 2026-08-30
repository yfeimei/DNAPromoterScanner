"""Day 1: Load and parse the UCI E. coli promoter dataset.

File format of data/promoters.data, one record per line:
    <class>,<name>, <57-letter DNA sequence>
class is "+" (promoter) or "-" (non-promoter).
The 57-letter window covers positions -50 to +7 relative to the
transcription start site (TSS). Position -50 is index 0 in the sequence;
the TSS (+1) is at index 50.
"""

from dataclasses import dataclass
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "promoters.data"

SEQUENCE_LENGTH = 57
TSS_INDEX = 50  # index within the 57-length sequence corresponding to +1


@dataclass
class PromoterRecord:
    label: int  # 1 = promoter, 0 = non-promoter
    name: str
    sequence: str  # lowercase, length 57, alphabet {a, c, g, t}


def load_records(path: Path = DATA_PATH) -> list[PromoterRecord]:
    records = []
    # encoding is pinned so the file parses identically on Windows (whose
    # default is the ANSI code page, not UTF-8) as on macOS/Linux
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls, name, seq = line.split(",")
            label = 1 if cls.strip() == "+" else 0
            name = name.strip()
            seq = seq.strip().lower()
            if len(seq) != SEQUENCE_LENGTH:
                raise ValueError(f"Unexpected sequence length for {name}: {len(seq)}")
            records.append(PromoterRecord(label=label, name=name, sequence=seq))
    return records


def train_test_split_records(
    records: list[PromoterRecord], test_fraction: float = 0.2, seed: int = 42
):
    """Simple stratified split so both classes are represented in the test set."""
    import random

    rng = random.Random(seed)
    positives = [r for r in records if r.label == 1]
    negatives = [r for r in records if r.label == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    def split(group):
        n_test = max(1, int(len(group) * test_fraction))
        return group[n_test:], group[:n_test]

    pos_train, pos_test = split(positives)
    neg_train, neg_test = split(negatives)

    train = pos_train + neg_train
    test = pos_test + neg_test
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


if __name__ == "__main__":
    records = load_records()
    n_pos = sum(r.label == 1 for r in records)
    n_neg = sum(r.label == 0 for r in records)
    print(f"Loaded {len(records)} records: {n_pos} promoters, {n_neg} non-promoters")
    print("Example record:", records[0])
