"""Train a classifier to detect the planted synthetic TATA-box motif.

Same method as train_model.py (k-mer features + logistic regression), but
trained on the synthetic dataset from synthetic_tata_data.py instead of
real E. coli data. No data is downloaded; everything is generated locally.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from data_utils import train_test_split_records
from featurize import featurize_sequences
from synthetic_tata_data import generate_dataset

K = 4  # k-mer length, same default as the bacterial model
# Default C=1.0 similarly over-regularizes here, keeping every probability
# within ~0.45-0.63. C=10 was chosen the same way as the bacterial model
# (comparing 5-fold CV ROC-AUC and probability spread): similar AUC to
# C=1.0 (0.959 vs 0.954) but a much wider, more useful probability range
# (0.23-0.96) without the near-total saturation seen at C=100.
C = 10.0
N_PER_CLASS = 300  # synthetic data is free to generate, so use a larger set
MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "tata_model.pkl"


def main():
    records = generate_dataset(n_per_class=N_PER_CLASS, seed=42)
    train_records, test_records = train_test_split_records(records)

    X_train = featurize_sequences([r.sequence for r in train_records], k=K)
    y_train = np.array([r.label for r in train_records])
    X_test = featurize_sequences([r.sequence for r in test_records], k=K)
    y_test = np.array([r.label for r in test_records])

    model = LogisticRegression(max_iter=2000, C=C)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"k = {K}, C = {C}, synthetic examples per class = {N_PER_CLASS}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Test ROC-AUC:  {roc_auc_score(y_test, y_prob):.3f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["no TATA box", "has TATA box"]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "k": K}, f)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
