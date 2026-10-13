"""Stage 7: LLM judge. Scores several pitches for the same seed, shown in shuffled order."""
import random
from typing import Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from .config import OLLAMA_URL

JUDGE_MODEL = "llama3.2"  # deliberately a different model from the writer (qwen2.5)


class Score(BaseModel):
    label: Literal["X", "Y", "Z"]
    novelty: int = Field(ge=1, le=5, description="1 = obvious or generic link, 5 = surprising yet defensible link")
    coherence: int = Field(ge=1, le=5, description="1 = incoherent or invents facts, 5 = hangs together and sticks to the given facts")


class Judgement(BaseModel):
    scores: list[Score] = Field(min_length=2)


SYSTEM = (
    "You are a strict, impartial judge of museum exhibition pitches. Each option below pairs three objects "
    "with a pitch that connects them. Score every option from 1 to 5 on two criteria.\n"
    "NOVELTY: how surprising yet defensible is the connection between the three objects? Generic themes "
    "like 'human connection' or 'cultural exchange' score low.\n"
    "COHERENCE: does the pitch hang together, and does it stick to what the object descriptions say "
    "without inventing facts?\n"
    "Use the full 1-5 range and compare the options with each other. "
    "Return exactly one score entry for EACH option label shown; never return an empty list."
)


def judge(options: list[dict], seed_word: str, rng: random.Random, model: str = JUDGE_MODEL) -> dict[str, dict]:
    """options: [{'key': strategy, 'texts': [3 object texts], 'title': str, 'pitch': str}].
    Shuffles the options, shows them as X/Y/Z, and returns {strategy_key: {'novelty':n,'coherence':c}}."""
    shuffled = options[:]
    rng.shuffle(shuffled)
    labels = ["X", "Y", "Z"][: len(shuffled)]
    blocks = []
    for lab, o in zip(labels, shuffled):
        objs = "\n".join(f"  - {t}" for t in o["texts"])
        blocks.append(f"Option {lab}\nObjects:\n{objs}\nTitle: {o['title']}\nPitch: {o['pitch']}")
    user = f"Theme word used to start the search: {seed_word}\n\n" + "\n\n".join(blocks)
    llm = ChatOllama(model=model, base_url=OLLAMA_URL, temperature=0, num_predict=400,
                     client_kwargs={"timeout": 90}).with_structured_output(Judgement)
    out = llm.invoke([("system", SYSTEM), ("human", user)])
    by_label = {s.label: s for s in out.scores}
    if set(by_label) != set(labels):
        raise ValueError(f"judge returned labels {sorted(by_label)} but options were {labels}")
    return {o["key"]: {"novelty": by_label[l].novelty, "coherence": by_label[l].coherence,
                       "shown_as": l} for l, o in zip(labels, shuffled) if l in by_label}
