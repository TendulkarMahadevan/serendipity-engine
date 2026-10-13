"""Stage 5b: run strategies A (band), B (farthest cluster), C (random) on the 10 seeds."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serendipity.strategies import (band_sample, close_match, embed_seed, farthest_cluster_sample,
                                    fit_kmeans, load_corpus, random_sample)

SEEDS = ["winter", "war", "love", "ocean", "mourning",
         "gold", "music", "storm", "childhood", "ritual"]
C = load_corpus()
km = fit_kmeans(C.X, 30)
lens = np.array([len(t) for t in C.texts])
t = lambda i: f"{C.metas[i]['title'][:42]} [{C.metas[i]['department'][:14]}]"

stats = {s: {"sim": [], "short": 0, "ids": [], "len": []} for s in "ABC"}
for n, seed in enumerate(SEEDS):
    q = embed_seed(seed)
    sims = C.X @ q
    top = close_match(sims)
    A = band_sample(sims, seed=n)
    B, far = farthest_cluster_sample(C.X, q, km, seed=n)
    R = random_sample(len(C.ids), top, seed=n)
    print(f"\n=== {seed} ===   close: {t(top)}  (sim {sims[top]:.3f})")
    for name, picks, extra in [("A band   ", A, ""), ("B cluster", B, f" (cluster {far})"), ("C random ", R, "")]:
        print(f"  {name}{extra}")
        for i in picks:
            print(f"      sim={sims[i]:.3f} len={lens[i]:3d}  {t(i)}")
        key = name.strip()[0]
        stats[key]["sim"] += [sims[i] for i in picks]
        stats[key]["short"] += sum(lens[i] < 50 for i in picks)
        stats[key]["len"] += [lens[i] for i in picks]
        stats[key]["ids"] += picks
    stats["B"].setdefault("far", []).append(far)

print("\n=== Summary over 10 seeds (20 distant items per strategy) ===")
print("strategy   mean sim to seed   mean text len   short(<50)   distinct distant items   most repeated")
for k, label in zip("ABC", ["A band", "B cluster", "C random"]):
    s = stats[k]
    rep = Counter(s["ids"]).most_common(1)[0][1]
    print(f"{label:<10} {np.mean(s['sim']):.3f}              {np.mean(s['len']):6.0f}         "
          f"{s['short']:>2}/20        {len(set(s['ids'])):>2}/20                    {rep}x")
print("farthest cluster chosen per seed:", stats["B"]["far"], "->", Counter(stats["B"]["far"]))
