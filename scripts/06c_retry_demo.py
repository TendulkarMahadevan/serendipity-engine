"""Stage 6c: (1) re-run 3 real triples with the fixed prompt; (2) SYNTHETIC test of the retry path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import serendipity.generate as G
from serendipity.generate import Mashup
from serendipity.strategies import band_sample, close_match, embed_seed, load_corpus

C = load_corpus()
def triple(seed, n):
    sims = C.X @ embed_seed(seed)
    return [close_match(sims)] + band_sample(sims, seed=n)

print("##### (1) REAL model, fixed prompt #####")
llm = G.get_llm("qwen2.5:7b-instruct")
for n, seed in [(1, "war"), (8, "childhood"), (7, "storm")]:
    idx = triple(seed, n)
    metas, texts = [C.metas[i] for i in idx], [C.texts[i] for i in idx]
    r = G.generate_mashup(metas, texts, llm)
    print(f"{seed}: valid={r.valid} attempts={r.attempts} errors={r.errors} ids={r.mashup.object_ids}")
    print("   pitch ends:", "..." + r.mashup.pitch[-90:])

print("\n##### (2) SYNTHETIC stub: NOT a model result; it only exercises the validator + retry #####")
class Stub:
    """Pretends to be a chat model: first answer cites a made-up ID, second answer is fine."""
    def __init__(self, answers): self.answers, self.calls = list(answers), []
    def with_structured_output(self, schema): return self
    def invoke(self, messages):
        self.calls.append(messages[-1][1]); return self.answers.pop(0)

idx = triple("gold", 5)
metas, texts = [C.metas[i] for i in idx], [C.texts[i] for i in idx]
ids = [int(m["object_id"]) for m in metas]
bad = Mashup(title="T", pitch="A pitch.", object_ids=[ids[0], 999999])
good = Mashup(title="T", pitch="A better pitch.", object_ids=ids)

stub = Stub([bad, good]); r = G.generate_mashup(metas, texts, stub)
print(f"bad then good  -> valid={r.valid} attempts={r.attempts} errors={r.errors}")
print("   retry prompt tail:", stub.calls[1][-170:].replace("\n", " "))
stub = Stub([bad, bad]); r = G.generate_mashup(metas, texts, stub)
print(f"bad then bad   -> valid={r.valid} attempts={r.attempts}  (flagged invalid; the UI should warn)")
