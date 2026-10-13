"""Stage 6b: read real pitches, then try to provoke an invalid-ID failure."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from langchain_ollama import ChatOllama
import serendipity.generate as G
from serendipity.config import OLLAMA_URL
from serendipity.strategies import band_sample, close_match, embed_seed, load_corpus

C = load_corpus()
def triple(seed, n):
    sims = C.X @ embed_seed(seed)
    return [close_match(sims)] + band_sample(sims, seed=n)

print("##### Real pitches (qwen2.5:7b-instruct) #####")
llm = G.get_llm("qwen2.5:7b-instruct")
for n, seed in [(5, "gold"), (1, "war"), (8, "childhood")]:
    idx = triple(seed, n)
    metas, texts = [C.metas[i] for i in idx], [C.texts[i] for i in idx]
    r = G.generate_mashup(metas, texts, llm)
    print(f"\nseed={seed}  valid={r.valid} attempts={r.attempts}")
    for m in metas: print(f"   [{m['object_id']}] {m['title'][:70]}")
    print("TITLE:", r.mashup.title); print("PITCH:", r.mashup.pitch); print("IDS:", r.mashup.object_ids)

print("\n\n##### Stress: llama3.2, temp 1.5, no ID rule in the prompt, 25 tries #####")
G.SYSTEM = ("You are a museum curator. Write an imaginative exhibition pitch of at most 120 words "
            "tying together the three objects you are shown. Fill in object_ids.")
llm = ChatOllama(model="llama3.2", base_url=OLLAMA_URL, temperature=1.5)
bad = 0
for t in range(25):
    seed, n = [("gold", 5), ("war", 1), ("love", 2), ("ritual", 9)][t % 4]
    idx = triple(seed, n)
    metas, texts = [C.metas[i] for i in idx], [C.texts[i] for i in idx]
    r = G.generate_mashup(metas, texts, llm, max_retries=1)
    if r.errors:
        bad += 1
        print(f"try {t}: attempts={r.attempts} final_valid={r.valid}  errors={r.errors}")
print(f"first-attempt failures: {bad}/25")
