"""Serendipity Engine: read-only gallery of pre-generated results.

Shows the 120 pitches from the 40-seed evaluation (data/eval40_*). It does NOT run any model, so it can
be hosted for free. The live app (app.py at the repo root) needs Ollama running locally.
Run locally:  streamlit run demo/app.py
"""
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent.parent / "data"
REPO = "https://github.com/TendulkarMahadevan/serendipity-engine"
MET_URL = "https://www.metmuseum.org/art/collection/search/{}"
STRATEGIES = {"A_band": "Band", "B_cluster": "Farthest cluster", "C_random": "Random"}
ROLES = ["Close match", "Distant", "Distant"]

st.set_page_config(page_title="Serendipity Engine gallery", page_icon=":material/shuffle:", layout="centered")


@st.cache_data
def load_results() -> tuple[dict, pd.DataFrame]:
    results = json.loads((DATA / "eval40_mashups.json").read_text())
    scores = pd.read_csv(DATA / "eval40_scores.csv")
    return results, scores


def parse_record(text: str) -> dict:
    """Split 'Title: X. Culture: Y. Medium: Z. Classification: W' into fields."""
    marks = list(re.finditer(r"(?:^|\. )(Title|Culture|Period|Medium|Classification): ", text))
    out = {}
    for n, m in enumerate(marks):
        end = marks[n + 1].start() if n + 1 < len(marks) else len(text)
        out[m.group(1).lower()] = text[m.end():end].strip().rstrip(".")
    return out


results, scores = load_results()
seeds = list(dict.fromkeys(r["seed"] for r in results.values()))

st.title("Serendipity Engine gallery")
st.caption("One close match from the Met's open-access collection, plus two deliberately distant objects, "
           "tied together by an AI-written exhibition pitch.")
st.info("These are **pre-generated** results from a 40-seed evaluation, written by a small local model "
        "(Qwen 2.5 7B). This page does not run any model, so you can't type your own word here. "
        "Pitches can be generic or contain mistakes.",
        icon=":material/info:")

with st.container(horizontal=True, vertical_alignment="bottom"):
    seed = st.selectbox("Search word", seeds, key="seed", bind="query-params")
    strategy = st.segmented_control("How the two distant objects were picked", list(STRATEGIES),
                                    format_func=STRATEGIES.get, default="A_band", key="strategy",
                                    bind="query-params")
strategy = strategy or "A_band"  # segmented control returns None if the user deselects

rec = results[f"{seed}|{strategy}"]
cols = st.columns(3)
for col, role, text, oid, sim in zip(cols, ROLES, rec["texts"], rec["ids"], rec["sims"]):
    f = parse_record(text)
    with col.container(border=True):
        st.badge(role, color="green" if role == "Close match" else "orange")
        st.markdown(f"**{f.get('title', 'Untitled')}**")
        details = [f[k] for k in ("culture", "period", "medium") if f.get(k)]
        st.caption(" · ".join(details) if details else "No further details recorded")
        st.caption(f"Similarity to “{seed}”: {sim:.2f}")
        st.markdown(f"[View on the Met website :material/open_in_new:]({MET_URL.format(oid)})")

if rec["pitch"] is None:
    st.error("The model could not produce a usable pitch for this combination, even after retries. "
             "It is shown here on purpose: failures like this happened in 2 of the 120 runs.",
             icon=":material/error:")
else:
    if not rec["valid"]:
        st.warning("This pitch failed validation, so treat it with caution.", icon=":material/warning:")
    st.subheader(rec["title"])
    st.write(rec["pitch"])
    if rec["version"] == "v3" and rec["link"]:
        st.caption(f"Why these three: {rec['link']}")
    prompt = ("strict fact-checked prompt" if rec["version"] == "v3"
              else "simpler fallback prompt (the strict one failed validation)")
    st.caption(f"Written with the {prompt} · {rec['attempts']} attempt(s) · {len(rec['pitch'].split())} words")


@st.cache_data
def strategy_summary() -> pd.DataFrame:
    rows = []
    for key, name in STRATEGIES.items():
        recs = [r for r in results.values() if r["strategy"] == key]
        ok = [r for r in recs if r["pitch"] is not None]
        sub = scores[scores.strategy == key]
        rows.append({
            "Strategy": name,
            "Pitches": f"{len(ok)}/{len(recs)}",
            "Written by strict prompt": f"{sum(r['version'] == 'v3' for r in ok)}/{len(ok)}",
            "Distant items' similarity to word": round(sum(s for r in recs for s in r["sims"][1:]) / (2 * len(recs)), 3),
            "Judge novelty (1-5)": round(sub.novelty.mean(), 2),
            "Judge coherence (1-5)": round(sub.coherence.mean(), 2),
        })
    return pd.DataFrame(rows)


with st.expander("How did the three strategies compare?", icon=":material/analytics:"):
    st.dataframe(strategy_summary(), hide_index=True,
                 alt="Table comparing the band, farthest-cluster and random strategies on pitch counts, "
                     "similarity of distant items and judge scores")
    st.markdown(
        "- **No strategy could be shown to be better.** The judge was a small local model whose novelty "
        "scores agreed with themselves only 31% of the time, so the differences in this table are within noise.\n"
        "- **Farthest cluster is the weakest on measurable grounds:** its distant items came from only 5 "
        "departments (16 for the others), 18% repeated, and more of its pitches named things found in no description.\n"
        f"- Full write-up, code and method: [GitHub repository]({REPO})."
    )

with st.expander("About this project", icon=":material/help:"):
    st.markdown(
        "Normal retrieval returns the items most similar to a query. This project returns **one close match "
        "plus two distant objects** from 3,000 Met objects, then asks a local language model to write an "
        "exhibition pitch tying them together. Pitches are checked in code (cited IDs, quoted evidence, "
        "no invented names) and retried; if the strict prompt keeps failing, a simpler one is used.\n\n"
        "Built with LangChain, Chroma, Ollama and Streamlit. Object data: "
        "[The Met Open Access dataset](https://github.com/metmuseum/openaccess) (CC0). "
        f"Source code: [{REPO.split('github.com/')[1]}]({REPO})."
    )
