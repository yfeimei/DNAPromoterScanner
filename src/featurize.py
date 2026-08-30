"""Day 1: Turn DNA sequences into k-mer frequency feature vectors.

Each sequence is broken into overlapping substrings of length k (k-mers),
e.g. for k=4 the sequence "acgt" contributes the k-mer "acgt" itself, while
a longer sequence contributes many overlapping k-mers. The feature vector
for a sequence is the normalized count of each possible k-mer.
"""

from itertools import product

import numpy as np

BASES = "acgt"


def all_kmers(k: int) -> list[str]:
    return ["".join(p) for p in product(BASES, repeat=k)]


def kmer_index(k: int) -> dict[str, int]:
    return {kmer: i for i, kmer in enumerate(all_kmers(k))}


def sequence_to_kmer_vector(sequence: str, k: int, index: dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(index), dtype=np.float64)
    n_windows = len(sequence) - k + 1
    for i in range(n_windows):
        kmer = sequence[i : i + k]
        if kmer in index:
            vec[index[kmer]] += 1
    if n_windows > 0:
        vec /= n_windows  # normalize to frequencies so sequence length doesn't matter
    return vec


def featurize_sequences(sequences: list[str], k: int = 4) -> np.ndarray:
    index = kmer_index(k)
    return np.vstack([sequence_to_kmer_vector(seq, k, index) for seq in sequences])


if __name__ == "__main__":
    from data_utils import load_records

    records = load_records()
    X = featurize_sequences([r.sequence for r in records], k=4)
    print(f"Feature matrix shape: {X.shape}  (n_sequences x n_possible_4-mers)")
