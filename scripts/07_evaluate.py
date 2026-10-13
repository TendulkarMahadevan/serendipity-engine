"""Stage 7: generate 30 mashups (10 seeds x 3 strategies), judge them, compute extra metrics.
Results are cached in data/eval_mashups.json so reruns skip finished generations."""
import json, random, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from serendipity.generate import generate_with_fallback, get_llm
from serendipity.judge import judge
from serendipity.strategies import (band_sample, close_match, embed_seed, farthest_cluster_sample,
                                    fit_kmeans, load_corpus, random_sample)

SEEDS = ["winter", "war", "love", "ocean", "mourning", "gold", "music", "storm", "childhood", "ritual"]
STRATS = {"A_band": "A band", "B_cluster": "B farthest-cluster", "C_random": "C random"}
CACHE = ROOT / "data" / "eval_mashups.json"
C = load_corpus(); km = fit_kmeans(C.X, 30); llm = get_llm(temperature=0.6)
lens = np.array([len(t) for t in C.texts])
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

# ---- 1. generate -------------------------------------------------------------------------
for n, seed in enumerate(SEEDS):
    q = embed_seed(seed); sims = C.X @ q; top = close_match(sims)
    picks = {"A_band": band_sample(sims, seed=n),
             "B_cluster": farthest_cluster_sample(C.X, q, km, seed=n)[0],
             "C_random": random_sample(len(C.ids), top, seed=n)}
    for key, distant in picks.items():
        k = f"{seed}|{key}"
        if k in cache: continue
        idx = [top] + distant
        t0 = time.time()
        r, ver = generate_with_fallback([C.metas[i] for i in idx], [C.texts[i] for i in idx], llm, max_retries=2, seed_word=seed)
        cache[k] = {"seed": seed, "strategy": key, "idx": idx, "ids": [C.ids[i] for i in idx],
                    "texts": [C.texts[i] for i in idx], "valid": r.valid, "version": ver, "attempts": r.attempts,
                    "errors": r.errors, "title": r.mashup.title if r.mashup else None,
                    "pitch": r.mashup.pitch if r.mashup else None}
        CACHE.write_text(json.dumps(cache, indent=1))
        print(f"generated {k:<22} valid={r.valid} via={ver} attempts={r.attempts} ({time.time()-t0:.0f}s)", flush=True)

# ---- 2. judge (one call per seed, options shuffled) ---------------------------------------
rng = random.Random(7); rows = []
for seed in SEEDS:
    ok = [cache[f"{seed}|{k}"] for k in STRATS if cache[f"{seed}|{k}"]["valid"]]
    scores = {}
    if len(ok) >= 2:
        for attempt in range(3):
            try:
                scores = judge([{"key": o["strategy"], "texts": o["texts"], "title": o["title"],
                                 "pitch": o["pitch"]} for o in ok], seed, rng); break
            except Exception as e:
                print(f"judge error for {seed} (attempt {attempt+1}): {type(e).__name__}", flush=True)
    for k in STRATS:
        o = cache[f"{seed}|{k}"]; s = scores.get(k, {})
        rows.append({"seed": seed, "strategy": k, "valid": o["valid"], "version": o["version"], "attempts": o["attempts"],
                     "novelty": s.get("novelty"), "coherence": s.get("coherence"), "shown_as": s.get("shown_as"),
                     "ids": " ".join(o["ids"]), "title": o["title"]})
df = pd.DataFrame(rows); df.to_csv(ROOT / "data" / "eval_scores.csv", index=False)

# ---- 3. extra metrics + summary -------------------------------------------------------------
summ = []
for k, label in STRATS.items():
    sub = df[df.strategy == k]
    distant = [int(i) for s in SEEDS for i in cache[f"{s}|{k}"]["idx"][1:]]
    dept = {C.metas[i]["department"] for i in distant}
    summ.append({"strategy": label, "generated OK": f"{int(sub.valid.sum())}/10",
                 "novelty": round(sub.novelty.mean(), 2), "coherence": round(sub.coherence.mean(), 2),
                 "junk rate (<60 chars)": f"{np.mean([lens[i] < 60 for i in distant]):.0%}",
                 "repetition rate": f"{1 - len(set(distant)) / len(distant):.0%}",
                 "departments in 20 distant": len(dept),
                 "via v3 (strict)": f"{int((sub.version == 'v3').sum())}/10"})
print("\n" + pd.DataFrame(summ).to_string(index=False))
print("\nPer-seed scores (novelty/coherence):")
print(df.pivot(index="seed", columns="strategy", values=["novelty", "coherence"]).reindex(SEEDS).to_string())
