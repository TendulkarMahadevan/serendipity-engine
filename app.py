"""Stage 8: Streamlit demo. Run with:  streamlit run app.py"""
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from serendipity.generate import generate_with_fallback, get_llm
from serendipity.strategies import (band_sample, close_match, embed_seed, farthest_cluster_sample,
                                    fit_kmeans, load_corpus, random_sample)

st.set_page_config(page_title="Serendipity Engine", page_icon="🎲", layout="centered")


@st.cache_resource(show_spinner="Loading the collection and clustering it…")
def load():
    corpus = load_corpus()
    return corpus, fit_kmeans(corpus.X, 30)


corpus, km = load()

st.title("🎲 Serendipity Engine")
st.caption("One close match from the Met's open-access collection, plus two deliberately distant objects, "
           "tied together by an AI-written exhibition pitch.")

seed_text = st.text_input("Seed word or phrase", value="winter")
strategy = st.selectbox(
    "Strategy for the two distant objects",
    ["Band (similarity percentile band)", "Farthest cluster", "Random"],
    help="Band: random items from a similarity-rank band. Farthest cluster: random items from the "
         "KMeans cluster least like your seed. Random: any two items.")

lo, hi = 10, 30
if strategy.startswith("Band"):
    lo, hi = st.slider("Band: percentile of similarity rank, counted from the LEAST similar end "
                       "(0 = most different, 100 = most similar)", 0, 100, (10, 30))
    st.caption("The bottom 5% is always excluded; it's mostly odd outliers.")

if st.button("Pull the lever", type="primary", disabled=not seed_text.strip()):
    t0 = time.time()
    q = embed_seed(seed_text.strip())
    sims = corpus.X @ q
    top = close_match(sims)
    rng_seed = int(time.time() * 1000) % (2**31)  # fresh draw each pull

    if strategy.startswith("Band"):
        distant = band_sample(sims, lo=lo, hi=hi, seed=rng_seed)
    elif strategy.startswith("Farthest"):
        distant, cl = farthest_cluster_sample(corpus.X, q, km, seed=rng_seed)
        st.caption(f"Farthest cluster for this seed: #{cl}")
    else:
        distant = random_sample(len(corpus.ids), top, seed=rng_seed)
    idx = [top] + distant

    labels = ["Close match", "Distant", "Distant"]
    for lab, i in zip(labels, idx):
        m = corpus.metas[i]
        with st.container(border=True):
            st.markdown(f"**{lab}** · similarity to seed {sims[i]:.2f}")
            st.markdown(f"### {m['title']}")
            st.write(" · ".join(x for x in [m["culture"], m["medium"], m["date"], m["department"]] if x))
            if m.get("url"):
                st.markdown(f"[View on the Met website]({m['url'].replace('http://', 'https://')})  ·  ID {m['object_id']}")

    with st.spinner("Writing the exhibition pitch…"):
        r, version = generate_with_fallback([corpus.metas[i] for i in idx], [corpus.texts[i] for i in idx],
                                            get_llm(temperature=0.6), max_retries=2, seed_word=seed_text.strip())

    if r.mashup is None:
        st.error("The model could not produce a usable pitch after "
                 f"{r.attempts} attempts. Try pulling again. Last problem: {r.errors[-1]}")
    else:
        if not r.valid:
            st.warning("Validation failed: the pitch may cite object IDs that aren't among these three, "
                       f"or has another problem. ({r.errors[-1]}) Treat the text with caution.")
        if version == "v2":
            st.info("Fallback: the stricter, fact-checked prompt could not produce a valid pitch for these "
                    "three objects, so this one came from the simpler prompt. Details may be less reliable.")
        st.subheader(r.mashup.title)
        st.write(r.mashup.pitch)
        if r.mashup.link:
            st.caption(f"Why these three: {r.mashup.link}")
        st.caption(f"Cited IDs: {r.mashup.object_ids} · {len(r.mashup.pitch.split())} words · "
                   f"{r.attempts} attempt(s) · prompt {version} · {time.time() - t0:.0f}s")
