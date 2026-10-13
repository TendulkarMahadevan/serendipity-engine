# Serendipity Engine

**Live gallery of pre-generated results: https://serendipity-engine.streamlit.app/**

Normal search returns the items most similar to your query. This project does the opposite for
two of its three results: it returns **one close match plus two deliberately distant objects**
from the Metropolitan Museum of Art's open-access collection, then asks a local LLM to write a
short "exhibition pitch" that ties the three together.

It is a learning project in RAG (retrieval-augmented generation) built with LangChain, Chroma and
Ollama. Everything runs locally and for free.

> **Headline result:** the pipeline works end to end, but a 40-seed evaluation could **not** show
> that any distant-item strategy produces better pitches than picking at random. The LLM judge was
> too inconsistent to rank them. See [Results](#results) for what was and wasn't established.

## Try it without installing anything heavy

`demo/app.py` is a read-only **gallery** of the 120 pitches from the 40-seed evaluation: pick a
search word and a strategy and see the three objects, the pitch and why the model linked them. It
runs no model (pitches are pre-generated), so it needs only Streamlit and pandas:

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

The live app (`app.py`, below) generates new pitches but needs Ollama running locally.

## How it works

1. **Data** (`src/serendipity/data.py`): download `MetObjects.csv` (public-domain rows with a title
   and some description), sample 3,000 objects with a fixed seed, and turn each into a LangChain
   `Document` (text = title, culture, period, medium, classification).
2. **Embeddings** (`embeddings.py`): `nomic-embed-text` via Ollama, using the `search_document:` /
   `search_query:` prefixes the model was trained with.
3. **Store** (`store.py`): Chroma with cosine distance, persisted in `./chroma`, idempotent by object ID.
4. **Strategies** (`strategies.py`): pick the nearest item as the close match, then two distant items by:
   - **A. Band sampling:** random items from a percentile band of similarity rank (default 10th to 30th from the bottom).
   - **B. Farthest cluster:** KMeans (k=30), sample from the cluster whose centroid is least like the seed.
   - **C. Random:** uniform baseline.
5. **Generation** (`generate.py`): `qwen2.5:7b-instruct` writes a title, a one-sentence link and a pitch as
   structured output (Pydantic). Code then validates it:
   - cited object IDs must be among the three supplied;
   - no placeholders, hashtags, event details, leaked IDs or cut-off text;
   - the strict prompt (v3) must quote evidence phrases that are verified against the records, may only
     name materials/techniques found in the `Medium`/`Classification` fields, and may not name people or
     places that appear in no description;
   - failures are retried with the reason as feedback; if the strict prompt still fails, a simpler
     prompt (v2) is used as a fallback.
6. **Judge** (`judge.py`): a different local model (`llama3.2`) scores pitches 1-5 for novelty and
   coherence, with option order shuffled.
7. **Demo** (`app.py`): Streamlit app with a seed box, strategy menu, band sliders and a "Pull the lever" button.

## Results

Evaluation: 40 seed words x 3 strategies = 120 pitches, two shuffled judge passes per seed, bootstrap
confidence intervals over seeds. Raw data: `data/eval40_*`.

| strategy | valid pitches | novelty | coherence | departments covered (80 distant items) | repeated items | pitches naming something in no description |
|---|---|---|---|---|---|---|
| A band | 40/40 | 2.84 | 3.80 | 16 | 4% | 5/40 |
| B farthest-cluster | 39/40 | 2.77 | 4.01 | 5 | 18% | 13/39 |
| C random | 39/40 | 2.85 | 3.92 | 16 | 0% | 10/39 |

**What the evidence supports**
- **No quality winner.** Every paired difference in judged novelty has a 95% interval that includes zero
  (e.g. band minus random: -0.03, [-0.40, +0.35]).
- **The judge is the bottleneck.** Scoring the same pitch twice gave matching novelty scores only 31% of the
  time (correlation 0.15). More seeds will not fix a judge this noisy.
- **Farthest-cluster is the weakest on measurable grounds:** 5 departments across its distant items vs 16 for
  the others, the most repeats, and the most pitches with invented names (13/39 vs 5/40 for band, p = 0.034).
- **Band sampling picks sparse records more often:** 8/80 distant items have under 60 characters of text,
  vs 0 for farthest-cluster (p = 0.007).

**Things that went wrong along the way (and are handled in the code)**
- Constrained JSON output still sometimes ran on or truncated, so output length is capped and parse
  failures are retried.
- The model invented artist names, dates, venues, hashtags and "[Insert Location Here]" placeholders;
  each now has a dedicated check.
- A stricter prompt cut invented names but succeeded only ~60-70% of the time, hence the fallback.

## Limitations

- Small sample (3,000 of ~224,000 eligible objects); the sample is not balanced (Drawings and Prints is ~30%).
- Small local models for writing and judging. Pitches are often generic, and the checks verify vocabulary
  and quotes but **not** which object a claimed feature belongs to.
- One generation per seed and strategy; LLM-judge scores, not human ratings.
- **"Distant" is only moderately far.** Similarity to a seed is tightly packed (standard deviation ~0.029 across
  the 3,000 objects). Band picks sit at about the 19th percentile and random picks at the 50th, but in raw cosine
  that is only 0.502 vs 0.526. So this experiment may not test "far" hard enough. Longer seed phrases or an
  extreme-tail arm would test that; it has not been run.
- **Two prompts are mixed.** 32 of 118 pitches came from the simpler fallback prompt. Restricting to the 86
  strict-prompt pitches does not change the conclusion (a couple of borderline intervals appear among 12
  comparisons, which is what noise would produce).

## Run it

Requires Python 3.11+, [Ollama](https://ollama.com) running, and these models pulled:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b-instruct   # writer
ollama pull llama3.2              # judge
```

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/00_check_setup.py          # Ollama + embedding sanity check
python scripts/01_load_data.py            # download (~300 MB) + filter + sample
python scripts/02_embed_store.py          # embed into ./chroma (idempotent)
streamlit run app.py                      # the demo
```

Experiments, in order (each prints what it found):

| script | what it does |
|---|---|
| `03_similarity.py` | plain similarity search with scores |
| `04_mmr.py` | MMR at several `lambda_mult` values |
| `05a_checks_and_clusters.py`, `05b_run_strategies.py` | verify distances, inspect clusters, run strategies A/B/C |
| `06a`-`06c` | structured-output reliability and failure demos |
| `07_evaluate.py` | 10-seed evaluation |
| `08_prompt_compare.py` | prompt versions v1 / v2 / v3 |
| `09_evaluate_large.py` | the 40-seed evaluation behind the table above (about 70 minutes of compute; much longer wall-clock if your machine sleeps) |

Optional: `pip install watchdog` lets Streamlit reload edited helper modules without a restart.

Tested with langchain 1.4.4, langchain-chroma 1.1.0, langchain-ollama 1.1.0, chromadb 1.5.9,
streamlit 1.65, Python 3.12.

## Layout

```
app.py                  Streamlit app (live generation, needs Ollama)
demo/                   Streamlit gallery of pre-generated results (no model needed)
src/serendipity/        config, data, embeddings, store, strategies, generate, judge
scripts/                one runnable script per stage
data/eval40_*           results of the 40-seed evaluation
```

## Data and attribution

Object records come from [The Met's Open Access dataset](https://github.com/metmuseum/openaccess),
released under CC0. Generated pitches are AI-written and may contain errors; they are not museum text.
