"""Day 5: Gen AI layer -- turn the numeric findings into a plain-language
research narrative, and (optionally) fact-check that narrative.

Design choice: explanations are generated OFFLINE by pasting the prompt
below into a free chat UI (ChatGPT/Gemini/Claude) and saving the reply as
static text -- not by calling a paid API at runtime. This keeps the whole
project at $0 ongoing cost and makes it easy to manually verify each
AI-generated claim against the literature before publishing it (the
verification step is itself part of the research process, see README).

Usage:
1. Run mutate_scan.py and compare_to_known_motifs.py first.
2. Fill in RESULTS below with the numbers those scripts printed.
3. Copy build_prompt() output into a free LLM chat.
4. Paste the AI's reply into explanations/ai_explanation_raw.txt.
5. Read it critically against a real source (e.g. a molecular biology
   textbook or the Hawley & McClure 1983 paper on E. coli promoter
   consensus sequences) and note any inaccuracies in
   explanations/ai_explanation_reviewed.md -- that review is the actual
   research contribution of this step, not the AI text itself.
"""

from pathlib import Path

EXPLANATIONS_DIR = Path(__file__).resolve().parent.parent / "explanations"

# Fill these in from the console output of mutate_scan.py / compare_to_known_motifs.py
RESULTS = {
    "test_accuracy": None,  # e.g. 0.86
    "test_roc_auc": None,  # e.g. 0.91
    "minus_35_fold": None,  # fold-over-background importance at -35 element
    "minus_10_fold": None,  # fold-over-background importance at -10 element
    "top_positions_relative_to_tss": [],  # e.g. [-10, -9, -35, ...]
}


def build_prompt() -> str:
    return f"""I trained a logistic regression model on k-mer frequencies to
classify 57-base-pair E. coli DNA windows as promoter or non-promoter
(UCI Molecular Biology Promoter dataset, 106 sequences). Test accuracy was
{RESULTS['test_accuracy']} and ROC-AUC was {RESULTS['test_roc_auc']}.

I then ran an in-silico saturation mutagenesis scan: for each promoter
sequence, I mutated every position to each alternate base and measured how
much the model's promoter-probability dropped. Averaging across all
promoter sequences gave a per-position importance profile.

The known -35 element (consensus TTGACA, centered -35 to -30 relative to
the transcription start site) showed {RESULTS['minus_35_fold']}x the
background importance. The known -10 Pribnow box (consensus TATAAT,
centered -10 to -7) showed {RESULTS['minus_10_fold']}x background.
The single most important positions relative to the TSS were:
{RESULTS['top_positions_relative_to_tss']}.

Please explain, for a high-school-level audience, (1) what a promoter and
the -10/-35 elements are and why they matter for gene expression, and
(2) what it means, scientifically, that a simple ML model trained with no
positional information rediscovered these regions on its own. Be precise
about what this does and does not prove, and flag any claims that would
need a citation to a real paper.
"""


def save_prompt():
    EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)
    prompt_path = EXPLANATIONS_DIR / "ai_prompt.txt"
    prompt_path.write_text(build_prompt(), encoding="utf-8")
    print(f"Prompt written to {prompt_path}")
    print("Paste this into a free LLM chat, then save the reply to")
    print(f"  {EXPLANATIONS_DIR / 'ai_explanation_raw.txt'}")
    print("Review it against a real source and record notes in")
    print(f"  {EXPLANATIONS_DIR / 'ai_explanation_reviewed.md'}")


if __name__ == "__main__":
    save_prompt()
