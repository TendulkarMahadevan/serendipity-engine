"""Compare prompt v1 (old) vs v2 (concrete shared feature + seed word) on the same 10 band triples."""
import json, random, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from serendipity.generate import CLICHES, describe, generate_mashup, get_llm, ungrounded_names
from serendipity.judge import judge
from serendipity.strategies import band_sample, close_match, embed_seed, load_corpus

SEEDS = ["winter", "war", "love", "ocean", "mourning", "gold", "music", "storm", "childhood", "ritual"]
C = load_corpus(); llm = get_llm(temperature=0.6)
OUT = ROOT / "data" / "prompt_compare.json"
res = json.loads(OUT.read_text()) if OUT.exists() else {}

for n, seed in enumerate(SEEDS):
    sims = C.X @ embed_seed(seed)
    idx = [close_match(sims)] + band_sample(sims, seed=n)
    metas, texts = [C.metas[i] for i in idx], [C.texts[i] for i in idx]
    for v in ("v1", "v2", "v3"):
        k = f"{seed}|{v}"
        if k in res: continue
        r = generate_mashup(metas, texts, llm, max_retries=2, seed_word=seed, version=v)
        res[k] = {"seed": seed, "version": v, "texts": texts, "blocks": [describe(m, t) for m, t in zip(metas, texts)], "valid": r.valid, "attempts": r.attempts,
                  "errors": r.errors, "title": r.mashup.title if r.mashup else None,
                  "link": r.mashup.link if r.mashup else None, "pitch": r.mashup.pitch if r.mashup else None}
        OUT.write_text(json.dumps(res, indent=1))
        print(f"{k:<14} valid={r.valid} attempts={r.attempts}", flush=True)

# objective metrics
print("\n=== Objective metrics (valid pitches only) ===")
for v in ("v1", "v2", "v3"):
    ok = [r for r in res.values() if r["version"] == v and r["valid"]]
    words = [len(r["pitch"].split()) for r in ok]
    cl = [sum(c in r["pitch"].lower() for c in CLICHES) for r in ok]
    mention = [r["seed"] in r["pitch"].lower() for r in ok]
    inv = [bool(ungrounded_names(r["pitch"], r["blocks"], r["seed"] + " " + r["title"])) for r in ok]
    print(f"{v}: valid {len(ok)}/10 | attempts {np.mean([r['attempts'] for r in ok]):.2f} | words {np.mean(words):.0f} | "
          f"clichés/pitch {np.mean(cl):.2f} | mentions search word {sum(mention)}/{len(ok)} | "
          f"names in no description {sum(inv)}/{len(ok)}")

rng = random.Random(11)
for other in ("v1", "v2"):
    rows = []
    for seed in SEEDS:
        a, b = res[f"{seed}|{other}"], res[f"{seed}|v3"]
        if not (a["valid"] and b["valid"]): continue
        sc = None
        for _ in range(3):
            try:
                sc = judge([{"key": other, "texts": a["texts"], "title": a["title"], "pitch": a["pitch"]},
                            {"key": "v3", "texts": b["texts"], "title": b["title"], "pitch": b["pitch"]}], seed, rng); break
            except Exception:
                pass
        if sc: rows.append(sc)
    print(f"\n=== Judge (llama3.2, shuffled): v3 vs {other}, {len(rows)} paired seeds ===")
    for m in ("novelty", "coherence"):
        d = [sc["v3"][m] - sc[other][m] for sc in rows]
        print(f"{m}: v3 - {other} mean {np.mean(d):+.2f}  wins/ties/losses {sum(x>0 for x in d)}/{sum(x==0 for x in d)}/{sum(x<0 for x in d)}")
