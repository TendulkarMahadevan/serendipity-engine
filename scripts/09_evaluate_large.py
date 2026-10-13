"""Larger evaluation: 40 seeds x 3 strategies, fallback generation, 2 shuffled judge passes,
bootstrap confidence intervals. Resumable: finished generations are cached in data/eval40_mashups.json."""
import json, random, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from serendipity.generate import CLICHES, generate_with_fallback, get_llm, ungrounded_names
from serendipity.judge import judge
from serendipity.strategies import (band_sample, close_match, embed_seed, farthest_cluster_sample,
                                    fit_kmeans, load_corpus, random_sample)

SEEDS = ["winter", "war", "love", "ocean", "mourning", "gold", "music", "storm", "childhood", "ritual",
         "silence", "harvest", "dragon", "mirror", "bridge", "night", "dance", "fire", "bird", "journey",
         "feast", "portrait", "garden", "exile", "power", "luck", "wedding", "hunting", "mask", "sleep",
         "river", "crown", "memory", "flight", "prayer", "fabric", "play", "travel", "dream", "protection"]
STRATS = ["A_band", "B_cluster", "C_random"]
CACHE = ROOT / "data" / "eval40_mashups.json"
C = load_corpus(); km = fit_kmeans(C.X, 30); llm = get_llm(temperature=0.6)
lens = np.array([len(t) for t in C.texts])
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
t_start = time.time()

# ---- 1. generate (resumable) -------------------------------------------------------------
for n, seed in enumerate(SEEDS):
    q = embed_seed(seed); sims = C.X @ q; top = close_match(sims)
    picks = {"A_band": band_sample(sims, seed=n), "B_cluster": farthest_cluster_sample(C.X, q, km, seed=n)[0],
             "C_random": random_sample(len(C.ids), top, seed=n)}
    for key, distant in picks.items():
        k = f"{seed}|{key}"
        if k in cache: continue
        idx = [top] + distant
        metas, texts = [C.metas[i] for i in idx], [C.texts[i] for i in idx]
        r, ver = generate_with_fallback(metas, texts, llm, max_retries=2, seed_word=seed)
        cache[k] = {"seed": seed, "strategy": key, "idx": idx, "ids": [C.ids[i] for i in idx], "texts": texts,
                    "sims": [float(sims[i]) for i in idx], "valid": r.valid, "version": ver, "attempts": r.attempts,
                    "errors": r.errors, "title": r.mashup.title if r.mashup else None,
                    "pitch": r.mashup.pitch if r.mashup else None, "link": r.mashup.link if r.mashup else None}
        CACHE.write_text(json.dumps(cache, indent=1))
        print(f"[{(time.time()-t_start)/60:5.1f} min] {k:<24} valid={r.valid} via={ver} attempts={r.attempts}", flush=True)

# ---- 2. judge: two passes with different shuffles --------------------------------------------
rows = []
for seed in SEEDS:
    ok = [cache[f"{seed}|{k}"] for k in STRATS if cache[f"{seed}|{k}"]["valid"]]
    passes = []
    if len(ok) >= 2:
        for p in range(2):
            rng = random.Random(1000 + 7 * SEEDS.index(seed) + p)
            for _ in range(3):
                try:
                    passes.append(judge([{"key": o["strategy"], "texts": o["texts"], "title": o["title"],
                                          "pitch": o["pitch"]} for o in ok], seed, rng)); break
                except Exception:
                    pass
    for k in STRATS:
        o = cache[f"{seed}|{k}"]
        row = {"seed": seed, "strategy": k, "valid": o["valid"], "version": o["version"], "attempts": o["attempts"],
               "n_passes": sum(k in p for p in passes)}
        for m in ("novelty", "coherence"):
            vals = [p[k][m] for p in passes if k in p]
            row[m] = float(np.mean(vals)) if vals else np.nan
            row[m + "_p1"] = vals[0] if len(vals) > 0 else np.nan
            row[m + "_p2"] = vals[1] if len(vals) > 1 else np.nan
        rows.append(row)
    print(f"judged {seed}", flush=True)
df = pd.DataFrame(rows); df.to_csv(ROOT / "data" / "eval40_scores.csv", index=False)

# ---- 3. summary -----------------------------------------------------------------------------
print("\n" + "=" * 100 + "\nSUMMARY (40 seeds x 3 strategies)\n" + "=" * 100)
summ = []
for k in STRATS:
    sub = df[df.strategy == k]; recs = [cache[f"{s}|{k}"] for s in SEEDS]
    okr = [r for r in recs if r["valid"]]
    distant = [i for r in recs for i in r["idx"][1:]]
    summ.append({"strategy": k, "valid": f"{len(okr)}/40", "via v3": f"{sum(r['version']=='v3' for r in okr)}/{len(okr)}",
                 "novelty": round(sub.novelty.mean(), 2), "coherence": round(sub.coherence.mean(), 2),
                 "junk(<60ch)": f"{np.mean([lens[i] < 60 for i in distant]):.0%}",
                 "repeat": f"{1 - len(set(distant)) / len(distant):.0%}",
                 "depts": len({C.metas[i]['department'] for i in distant}),
                 "sim to seed": round(np.mean([s for r in recs for s in r["sims"][1:]]), 3),
                 "invented names": f"{sum(bool(ungrounded_names(r['pitch'], [C.texts[i] for i in r['idx']], r['seed'] + ' ' + r['title'])) for r in okr)}/{len(okr)}",
                 "cliches/pitch": round(np.mean([sum(c in r['pitch'].lower() for c in CLICHES) for r in okr]), 2)})
print(pd.DataFrame(summ).to_string(index=False))

rng = np.random.default_rng(0)
print("\nPaired differences over seeds, mean [95% bootstrap CI over seeds]:")
for m in ("novelty", "coherence"):
    p = df.pivot(index="seed", columns="strategy", values=m)
    for a, b in [("A_band", "C_random"), ("A_band", "B_cluster"), ("B_cluster", "C_random")]:
        d = (p[a] - p[b]).dropna().values
        boots = [rng.choice(d, len(d)).mean() for _ in range(10000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  {m:<9} {a}-{b}: n={len(d)} mean={d.mean():+.2f} CI[{lo:+.2f}, {hi:+.2f}] "
              f"wins/ties/losses={(d>0).sum()}/{(d==0).sum()}/{(d<0).sum()}  {'<-- excludes 0' if lo > 0 or hi < 0 else ''}")

print("\nJudge reliability (agreement between its two passes on the same pitch):")
for m in ("novelty", "coherence"):
    a, b = df[m + "_p1"], df[m + "_p2"]; ok = a.notna() & b.notna()
    print(f"  {m:<9} n={ok.sum()} correlation={np.corrcoef(a[ok], b[ok])[0,1]:.2f}  exact agreement={(a[ok]==b[ok]).mean():.0%}")
print("\nScores by prompt version:"); print(df.dropna(subset=["novelty"]).groupby("version")[["novelty", "coherence"]].agg(["mean", "count"]).round(2))
print(f"\ntotal time {(time.time()-t_start)/60:.1f} min")
