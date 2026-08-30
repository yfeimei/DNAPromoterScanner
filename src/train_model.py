"""Day 2: Train a classifier to distinguish promoters from non-promoters.

Uses logistic regression on k-mer frequency features. Logistic regression is
chosen deliberately over a deep model: the dataset is small (106 sequences),
so a simple model avoids overfitting and is fast enough to iterate on in an
afternoon while still giving a per-feature weight we can reason about later.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from data_utils import load_records, train_test_split_records
from featurize import featurize_sequences

K = 5  # k-mer length
# scikit-learn's default C=1.0 over-regularizes on this small dataset (only
# ~85 training sequences vs hundreds of k-mer features), which squashes
# every predicted probability to within a few percent of 0.5 and makes the
# mutation-scan step nearly undetectable. C=100 was chosen by comparing
# 5-fold cross-validated ROC-AUC and probability spread across k in {3,4,5}
# and C in {1, 10, 100, 1000}: k=5/C=100 gave the best, most stable AUC
# (0.996 +/- 0.007) with a healthy probability spread (~0.15-0.89) without
# pushing to the extreme near-0/near-1 saturation seen at C=1000.
C = 100.0
MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "model.pkl"


def main():
    records = load_records()
    train_records, test_records = train_test_split_records(records)

    X_train = featurize_sequences([r.sequence for r in train_records], k=K)
    y_train = np.array([r.label for r in train_records])
    X_test = featurize_sequences([r.sequence for r in test_records], k=K)
    y_test = np.array([r.label for r in test_records])

    model = LogisticRegression(max_iter=2000, C=C)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"k = {K}, C = {C}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Test ROC-AUC:  {roc_auc_score(y_test, y_prob):.3f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["non-promoter", "promoter"]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "k": K}, f)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
