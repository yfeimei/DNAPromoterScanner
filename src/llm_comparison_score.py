"""Score the comparison once results_template.csv has been filled in with
the chatbot's answers (see llm_comparison_prepare.py).

Reports accuracy for her own classifier and for the chatbot on the exact
same sequences, plus where they agreed/disagreed with the true label and
with each other -- the actual evidence for "does Gen AI get this right?"
instead of an assumption either way.
"""

import csv
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent.parent / "comparison" / "results_template.csv"


def main():
    if not CSV_PATH.exists():
        raise SystemExit(
            f"{CSV_PATH} not found. Run llm_comparison_prepare.py first, "
            "then fill in the llm_answer column."
        )

    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    missing = [r["id"] for r in rows if not r["llm_answer"].strip()]
    if missing:
        print(
            f"Warning: {len(missing)} row(s) still have an empty llm_answer "
            f"column (ids: {', '.join(missing)}). Scoring only the filled-in rows.\n"
        )
    rows = [r for r in rows if r["llm_answer"].strip()]
    if not rows:
        raise SystemExit("No filled-in rows to score yet.")

    def normalize(value: str) -> str:
        value = value.strip().lower()
        if value.startswith("prom") or value.startswith("yes"):
            return "promoter"
        if value.startswith("non") or value.startswith("no"):
            return "non-promoter"
        return value

    n = len(rows)
    classifier_correct = sum(
        1 for r in rows if normalize(r["classifier_prediction"]) == r["true_label"]
    )
    llm_correct = sum(1 for r in rows if normalize(r["llm_answer"]) == r["true_label"])
    agreement = sum(
        1
        for r in rows
        if normalize(r["classifier_prediction"]) == normalize(r["llm_answer"])
    )

    print(f"Scored {n} sequences\n")
    print(f"Her classifier accuracy: {classifier_correct}/{n} ({classifier_correct/n:.0%})")
    print(f"Chatbot accuracy:        {llm_correct}/{n} ({llm_correct/n:.0%})")
    print(f"Classifier/chatbot agreement rate: {agreement}/{n} ({agreement/n:.0%})")

    print("\nPer-sequence breakdown:")
    print(f"{'id':>3}  {'true':<12} {'classifier':<12} {'chatbot':<12} notes")
    for r in rows:
        c_pred = normalize(r["classifier_prediction"])
        l_pred = normalize(r["llm_answer"])
        notes = []
        if c_pred != r["true_label"]:
            notes.append("classifier WRONG")
        if l_pred != r["true_label"]:
            notes.append("chatbot WRONG")
        print(
            f"{r['id']:>3}  {r['true_label']:<12} {c_pred:<12} {l_pred:<12} "
            f"{'; '.join(notes)}"
        )

    print(
        "\nInterpretation guide: if the chatbot accuracy is close to 50%, "
        "it's essentially guessing on sequences with no obvious keyword to "
        "latch onto -- meaningfully worse than a small model trained "
        "specifically for this task. If it does well, check whether its "
        "written reasoning (llm_reasoning column) is actually sound biology "
        "or just a confident-sounding guess -- accuracy alone doesn't prove "
        "correct reasoning."
    )


if __name__ == "__main__":
    main()
