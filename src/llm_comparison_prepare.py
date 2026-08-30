"""Prepare a fair, blind comparison: her own classifier vs. a general-purpose
Gen AI chatbot (ChatGPT/Claude/Gemini), tested on the exact same sequences.

This produces two files in comparison/:
  - prompts.txt          One prompt per sequence, to paste into a free
                          chatbot one at a time. No labels included -- the
                          chatbot sees only the raw DNA, same as her model.
  - results_template.csv  A scoring sheet, pre-filled with the true label
                          and her own classifier's prediction. She fills in
                          the "llm_answer" and "llm_reasoning" columns after
                          asking the chatbot, then runs
                          llm_comparison_score.py to see the final comparison.

No API calls -- this is the same offline, human-in-the-loop workflow used
in generate_ai_explanations.py, just applied to the detection task itself
instead of the write-up.
"""

import csv
import random
from pathlib import Path

from data_utils import load_records
from featurize import featurize_sequences
from mutate_scan import load_model

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "comparison"
MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "model.pkl"

N_PROMOTERS = 8
N_NON_PROMOTERS = 8
SEED = 7


def select_sample():
    records = load_records()
    promoters = [r for r in records if r.label == 1]
    non_promoters = [r for r in records if r.label == 0]

    rng = random.Random(SEED)
    sample = rng.sample(promoters, N_PROMOTERS) + rng.sample(non_promoters, N_NON_PROMOTERS)
    rng.shuffle(sample)  # so order gives no hint about the true label
    return sample


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = select_sample()

    model, k = load_model(MODEL_PATH)
    X = featurize_sequences([r.sequence for r in sample], k=k)
    classifier_probs = model.predict_proba(X)[:, 1]

    prompts_path = OUTPUT_DIR / "prompts.txt"
    csv_path = OUTPUT_DIR / "results_template.csv"

    with open(prompts_path, "w", encoding="utf-8") as f:
        f.write(
            "Paste EACH numbered prompt below into a free chatbot (ChatGPT, "
            "Claude, or Gemini) ONE AT A TIME, in a NEW/fresh chat each time "
            "(so earlier answers can't bias later ones). Record its answer "
            "in results_template.csv.\n\n"
        )
        for i, record in enumerate(sample, start=1):
            f.write(
                f"--- Prompt {i} ---\n"
                f"Here is a 57-letter DNA sequence: {record.sequence}\n"
                f"Is this likely to be a bacterial gene promoter region? "
                f"Answer with a single word first (Yes or No), then a "
                f"1-2 sentence explanation of your reasoning.\n\n"
            )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "sequence",
                "true_label",
                "classifier_probability",
                "classifier_prediction",
                "llm_answer",
                "llm_reasoning",
            ]
        )
        for i, (record, prob) in enumerate(zip(sample, classifier_probs), start=1):
            writer.writerow(
                [
                    i,
                    record.sequence,
                    "promoter" if record.label == 1 else "non-promoter",
                    f"{prob:.3f}",
                    "promoter" if prob >= 0.5 else "non-promoter",
                    "",  # fill in after asking the chatbot: "promoter" or "non-promoter"
                    "",  # fill in: a short summary of the chatbot's stated reasoning
                ]
            )

    print(f"Wrote {len(sample)} prompts to {prompts_path}")
    print(f"Wrote scoring template to {csv_path}")
    print(
        "\nNext steps:\n"
        "1. Open prompts.txt, paste each prompt into a free chatbot "
        "(fresh chat each time).\n"
        "2. Fill in the llm_answer and llm_reasoning columns in "
        "results_template.csv.\n"
        "3. Run: python llm_comparison_score.py\n"
    )


if __name__ == "__main__":
    main()
