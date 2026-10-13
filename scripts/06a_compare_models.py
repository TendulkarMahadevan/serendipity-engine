"""Stage 6a: does structured output work reliably with the local chat models?"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serendipity.generate import generate_mashup, get_llm
from serendipity.strategies import band_sample, close_match, embed_seed, load_corpus

SEEDS = ["winter", "war", "love", "ocean", "mourning", "gold", "music", "storm", "childhood", "ritual"]
MODELS = ["qwen2.5:7b-instruct", "llama3.2", "qwen3:4b"]
C = load_corpus()

triples = []
for n, s in enumerate(SEEDS):
    sims = C.X @ embed_seed(s)
    triples.append([close_match(sims)] + band_sample(sims, seed=n))

for model in MODELS:
    llm = get_llm(model)
    ok_first = ok_final = 0; secs = []; words = []; errs = []
    for idx in triples:
        metas = [C.metas[i] for i in idx]; texts = [C.texts[i] for i in idx]
        t0 = time.time()
        try:
            r = generate_mashup(metas, texts, llm)
        except Exception as e:
            errs.append(f"{type(e).__name__}: {str(e)[:100]}"); continue
        secs.append(time.time() - t0)
        ok_first += r.valid and r.attempts == 1
        ok_final += r.valid
        words.append(len(r.mashup.pitch.split()))
        errs += r.errors
    n = len(triples)
    print(f"\n{model}: valid first try {ok_first}/{n}, valid after retry {ok_final}/{n}, "
          f"crashes {n - len(secs)}, avg {sum(secs)/max(len(secs),1):.1f}s, "
          f"pitch words min/avg/max = {min(words) if words else '-'}/{sum(words)/max(len(words),1):.0f}/{max(words) if words else '-'}")
    for e in errs: print("   issue:", e)
