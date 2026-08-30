"""The free, public-facing website: paste in a DNA sequence and get back
candidate promoter-like regions, with the exact bases the model relied on
highlighted, and a plain rule-based label if they resemble a known motif.

Two modes, clearly separated so nobody mistakes one for the other:
  - Bacterial mode: real E. coli promoter data (UCI dataset), 57-bp windows,
    detects the -10/-35 elements.
  - TATA-box demo mode: a synthetic, programmatically generated dataset
    (no real genomic data) used to teach/demonstrate the same method on a
    eukaryotic-style motif, the TATA box, 60-bp windows.

No AI/LLM API calls happen here -- everything is the local classifier
(trained once, offline) plus simple pattern matching. That keeps hosting
$0 no matter how many people use it.

Run locally with:
    streamlit run app/app.py

Deploy for free (no server cost) at:
    https://streamlit.io/cloud  (Streamlit Community Cloud)
or
    https://huggingface.co/spaces  (Hugging Face Spaces, free tier)
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from motif_match import KNOWN_MOTIFS, TATA_MOTIFS, label_window  # noqa: E402
import scan_arbitrary_sequence as bacterial_scanner  # noqa: E402
import tata_scan_arbitrary_sequence as tata_scanner  # noqa: E402

st.set_page_config(page_title="DNA Promoter Scanner", layout="centered")

st.title("DNA Promoter Scanner")
st.write(
    "An educational tool that shows how AI can 'read' DNA: paste in a "
    "sequence, and it highlights exactly which letters matter most and "
    "why -- no coding or biology background needed."
)

with st.expander("How does this work? (click to learn the concepts)"):
    st.markdown(
        """
**1. What is a promoter, and why does it matter?**
A gene's promoter is a short DNA region right before the gene that acts
like an "on switch" -- it's where the cell's gene-reading machinery
attaches to start making that gene's product. Without a working promoter,
a gene doesn't get turned on. Different organisms use different promoter
"grammars": bacteria use the **-10 (Pribnow) box** and **-35 element**;
many eukaryotic genes (including human genes) use a **TATA box**.

**2. How does a computer turn DNA into something it can learn from?**
DNA is just a string of the letters A, C, G, T. To let a machine learning
model find patterns in it, this tool breaks each sequence into short,
overlapping chunks called **k-mers** (e.g. all 4-letter chunks) and counts
how often each one appears. Real promoters tend to contain certain k-mers
more often than random DNA -- that's the pattern the model learns.

**3. What is "mutating and re-scoring," and why does it reveal importance?**
Once a model can score "how promoter-like is this sequence?", you can ask
a deeper question: *which exact letters is it relying on?* This tool
changes one letter at a time to every other possible letter and re-checks
the score. If changing a letter causes a big drop in the score, that
position must matter a lot to the model's decision. Repeating this for
every position produces an "importance profile" for the whole sequence.
This technique is a simplified version of **in-silico saturation
mutagenesis**, a method real genomics researchers use to study DNA
function without needing a wet lab.

**4. What does the highlighting mean?**
Each letter is colored by how much the model's confidence dropped when
that letter was changed. **Darker red = more critical** to the
prediction. If the darkest-red stretch also matches a known biological
motif (shown with a green label below the sequence), that's a sign the
model's reasoning lines up with real, established biology -- not just a
guess.

*A note on scope:* this tool uses simple, transparent machine learning
(not a large language model), and any AI-assisted text on this page was
written and fact-checked by a human before publishing. Results here are
educational hypotheses to explore, not verified biological facts.
        """
    )

mode = st.radio(
    "Choose a mode:",
    [
        "Bacterial promoter (real E. coli data)",
        "TATA-box demo (synthetic data, educational)",
    ],
)

# S10 is a real E. coli promoter (from the training dataset) chosen as the
# default because it's one of the ~1-in-5 real promoters where the model's
# importance profile happens to line up cleanly with the known -10 box
# (verified: 0 mismatches). Most real promoters do NOT show this clean an
# alignment -- see PROJECT_GUIDE.md Section 6 for the honest, full picture.
# This is a curated first-impression example, not a claim that the method
# always works this well.
BACTERIAL_DEMO_SEQUENCE = (
    "acgtacgtacgtacgtacgtacgtacgtacgtacgtacgt"
    "tactagcaatacgcttgcgttcggtggttaagtatgtataatgcgcgggcttgtcgt"
    "acgtacgtacgtacgtacgtacgtacgtacgtacgtacgt"
)

# Filler + a clean TATA-box motif ("tataaaa") + filler, so the demo has an
# obvious, exact-match hit to find. The classifier only looks at k-mer
# content (not exact position within a window), so the motif just needs to
# be present somewhere in a 60-bp stretch.
TATA_DEMO_SEQUENCE = (
    "acgtacgtacgtacgtacgtacgtacgtacgtacgtacgt"
    "gctagctagctagctagctatataaaagctagctagctagctagctagctagctagc"
    "acgtacgtacgtacgtacgtacgtacgtacgtacgtacgt"
)

if mode.startswith("Bacterial"):
    st.caption(
        "Trained on a small set of 106 real E. coli promoter sequences "
        "(public UCI dataset). Highlights DNA regions that look "
        "promoter-like and the specific bases the model relies on most. "
        "Results are hypotheses to explore, not verified biological facts, "
        "and this is not a clinical, medical, or production tool. "
        "**Not applicable to human DNA** -- bacterial and human gene "
        "regulation work differently."
    )
    scanner = bacterial_scanner
    demo_sequence = BACTERIAL_DEMO_SEQUENCE
    motifs = KNOWN_MOTIFS
    window_length = 57
else:
    st.caption(
        "**Educational demo only.** This mode is trained on a "
        "programmatically GENERATED synthetic dataset (random background "
        "DNA with a TATA-box motif inserted), not on any real genome or "
        "real human DNA. It exists to show how the same AI method "
        "(train a classifier, then mutate each base to see what it relies "
        "on) can flag a eukaryotic-style regulatory element like the TATA "
        "box. Do not paste personal/real genetic data here -- it will not "
        "produce a scientifically meaningful result."
    )
    scanner = tata_scanner
    demo_sequence = TATA_DEMO_SEQUENCE
    motifs = TATA_MOTIFS
    window_length = 60

raw_input = st.text_area(
    f"Paste a DNA sequence (letters A/C/G/T only, at least {window_length} bases long):",
    value=demo_sequence,
    height=140,
)

top_n = st.slider("Number of candidate regions to show", min_value=1, max_value=5, value=2)

if st.button("Scan sequence"):
    try:
        results = scanner.analyze_sequence(raw_input, top_n=top_n)
    except ValueError as exc:
        st.error(str(exc))
        results = []

    if not results:
        st.warning("No sequence was analyzed. Check the input above and try again.")

    for i, result in enumerate(results, start=1):
        st.subheader(f"Candidate region {i}")
        st.write(
            f"Positions {result['start']}-{result['end']} in your sequence | "
            f"promoter-likelihood score: {result['promoter_probability']:.2f}"
        )

        importance = result["importance_profile"]
        max_importance = max(float(importance.max()), 1e-9)
        spans = []
        for base, score in zip(result["window_sequence"], importance):
            # Stronger red wash = more critical to the model's prediction.
            # Opacity is varied rather than lightness: on the dark theme an
            # unimportant base then fades into the page background, instead
            # of turning into a white block the way a fade-to-white scale
            # would. The letters keep the theme's text color either way.
            #
            # The red is deliberately deeper than the theme's accent color:
            # at full opacity a brighter red leaves the near-white letters
            # at only 2.8:1 contrast, and those are precisely the bases a
            # reader most needs to make out. This one holds 4.4:1 at its
            # strongest and rises from there as the wash fades.
            alpha = float(score) / max_importance
            spans.append(
                f'<span style="background-color:rgba(217,45,32,{alpha:.2f})">'
                f"{base}</span>"
            )
        st.markdown(
            f'<div style="font-family:monospace; font-size:20px; line-height:2">'
            f'{"".join(spans)}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Brighter red = higher mutation-sensitivity (more critical to the prediction)")

        labels = label_window(result["window_sequence"], importance, motifs=motifs)
        if labels:
            for label in labels:
                st.success(
                    f"Resembles a known **{label['name']}**: found "
                    f"`{label['matched_text']}` vs. consensus "
                    f"`{label['consensus']}` "
                    f"({label['mismatches']} mismatch(es))"
                )
        else:
            st.info(
                "No match to the known motifs in this tool's small reference "
                "list. This may be a false positive, a weaker/non-canonical "
                "signal, or something this tool doesn't recognize yet -- "
                "treat it as a starting point for further reading, not a "
                "conclusion."
            )

st.divider()
st.caption(
    "How it works: a small classifier scores every window of your sequence. "
    "For the highest-scoring windows, each base is mutated one at a time "
    "and re-scored to see how much the prediction depends on it -- that's "
    "the highlighting above."
)
