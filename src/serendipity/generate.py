"""Stage 6: turn 3 objects into a structured "exhibition pitch" and validate it.

Prompt versions (kept side by side so they can be compared fairly):
  v1  original: thread connecting the three objects
  v2  adds a `link` field (a concrete shared feature) and the search word
  v3  contrast-first link, quoted evidence the code can verify, and grounding checks
"""
import re
from dataclasses import dataclass, field

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from .config import CHAT_MODEL, OLLAMA_URL


# ---- result type used by the app and the evaluation -------------------------------------------
class Mashup(BaseModel):
    title: str
    pitch: str
    object_ids: list[int]
    link: str = ""
    evidence: list[str] = []


# ---- per-version schemas the LLM must fill (field order = generation order) --------------------
class MashupV1(BaseModel):
    title: str = Field(description="A short exhibition title")
    pitch: str = Field(description="An exhibition pitch of at most 120 words tying the three objects together")
    object_ids: list[int] = Field(description="The Met object IDs of the objects the pitch refers to")


class MashupV2(BaseModel):
    title: str = Field(description="A short, specific exhibition title")
    link: str = Field(description="ONE sentence naming the concrete feature (material, motif, technique, scale, "
                                  "function or date range) that connects the objects, using only facts from the descriptions")
    pitch: str = Field(description="An exhibition pitch of at most 110 words built around the link")
    object_ids: list[int] = Field(description="The Met object IDs of the objects the pitch refers to")


class MashupV3(BaseModel):
    title: str = Field(description="A short, specific exhibition title")
    evidence_1: str = Field(description="A short phrase copied letter for letter from Object 1's description")
    evidence_2: str = Field(description="A short phrase copied letter for letter from Object 2's description")
    evidence_3: str = Field(description="A short phrase copied letter for letter from Object 3's description")
    link: str = Field(description="ONE sentence of at most 35 words: the sharpest contrast (or a truly shared "
                                  "feature) between the objects, using only terms found in your evidence phrases")
    pitch: str = Field(description="An exhibition pitch of at most 100 words built on the link")
    object_ids: list[int] = Field(description="The Met object IDs of the objects the pitch refers to")


MAX_WORDS = 130       # v1/v2 asked for 110-120 words; allow slack before rejecting
MAX_WORDS_V3 = 110
MAX_LINK_WORDS = 40

CLICHES = ["human experience", "tapestry", "timeless", "universal", "enduring", "human quest",
           "human connection", "cultural exchange", "journey through time"]

_BASE_RULES = (
    "Use ONLY facts stated in the descriptions. Do NOT name artists, makers, places, people or "
    "exhibition dates or venues unless they appear in a description. Do not call anything 'anonymous'. "
    "Do not guess where an object is from: use the Culture and Department given. "
    "Plain prose only: no [brackets], hashtags, dates, locations, admission or website lines, and no "
    "object IDs in the text. In object_ids list ONLY the IDs you were given. End with a full sentence."
)

SYSTEM_V1 = (
    "You are a museum curator. You will be given three objects from the Met Museum collection. "
    "Write an imaginative exhibition pitch of at most 120 words that finds a surprising, "
    "believable thread connecting all three. Refer to each object by its title. " + _BASE_RULES
)

SYSTEM_V2 = (
    "You are a museum curator writing a short exhibition pitch. You are given three objects from the Met "
    "Museum. Object 1 is the closest match to the visitor's search word; Objects 2 and 3 were chosen because "
    "they are very different from it.\n"
    "Step 1 (field `link`): find ONE concrete, checkable feature that connects the objects: a material, a "
    "motif or subject, a technique, a scale or function, or a date range. State it in one sentence. If no "
    "feature is truly shared, use a sharp, specific contrast and say so.\n"
    "Step 2 (field `pitch`): write at most 110 words built around that feature. Name each object by its "
    "title and say what it adds. End by tying the exhibition back to the search word.\n"
    "Avoid generic phrases such as 'human experience', 'tapestry', 'timeless', 'universal' and 'enduring'. "
    + _BASE_RULES
)

SYSTEM_V3 = (
    "You are a museum curator writing a short exhibition pitch from three Met objects. Object 1 is the "
    "closest match to the visitor's search word; Objects 2 and 3 are deliberately very different from it.\n"
    "Fill the fields in this order.\n"
    "1. evidence_1, evidence_2, evidence_3: for each object, in order, copy ONE short exact phrase from its "
    "description (its title, culture, medium, date or department). Copy only the words, not the label.\n"
    "2. link: ONE sentence, at most 35 words. Default to a sharp contrast between the objects: different "
    "materials, techniques, eras or purposes. Name a shared feature only if the objects truly share it. "
    "Mention only materials, techniques, cultures and dates that appear in your evidence phrases.\n"
    "3. pitch: at most 100 words, built on the link. Name each object by its title and say what it adds. "
    "Work the search word into the exhibition's idea naturally. Never write the words 'search', "
    "'searched' or 'visitor'.\n"
    "Avoid generic phrases such as 'human experience', 'tapestry', 'timeless', 'universal' and 'enduring'. "
    + _BASE_RULES
)

VERSIONS = {"v1": (SYSTEM_V1, MashupV1), "v2": (SYSTEM_V2, MashupV2), "v3": (SYSTEM_V3, MashupV3)}


def describe(meta: dict, text: str) -> str:
    bits = [f"Object ID: {meta['object_id']}", text]
    if meta.get("date"):
        bits.append(f"Date: {meta['date']}")
    if meta.get("department"):
        bits.append(f"Department: {meta['department']}")
    return "\n".join(bits)


def build_prompt(metas: list[dict], texts: list[str], feedback: str | None = None,
                 seed_word: str | None = None, version: str = "v3") -> list[tuple[str, str]]:
    body = "\n\n".join(f"Object {n}:\n{describe(m, t)}" for n, (m, t) in enumerate(zip(metas, texts), 1))
    head = f"The visitor's search word: {seed_word}\n\n" if (seed_word and version != "v1") else ""
    user = head + body + "\n\nWrite the exhibition pitch."
    if feedback:
        user += f"\n\nYour previous answer was rejected: {feedback} Fix this."
    return [("system", VERSIONS[version][0]), ("human", user)]


def get_llm(model: str = CHAT_MODEL, temperature: float = 0.8) -> ChatOllama:
    # num_predict caps output length and client timeout caps waiting: constrained JSON decoding
    # can occasionally run on, and one runaway call would otherwise block forever.
    return ChatOllama(model=model, base_url=OLLAMA_URL, temperature=temperature,
                      num_predict=600, client_kwargs={"timeout": 90})


# ---- grounding helpers --------------------------------------------------------------------------
# Materials / techniques / media. A claim about one of these must be backed by the quoted evidence.
LEXICON = set("""gold silver bronze brass copper iron steel tin lead pewter glass ceramic earthenware porcelain
stoneware terracotta pottery stone marble limestone granite wood wooden mahogany oak walnut ivory bone silk linen
wool cotton leather paper parchment ink paint oil watercolor gouache pastel chalk graphite charcoal engraving
etching woodcut lithograph lithography print photograph photography platinum albumen drawing sculpture tapestry
embroidery lace enamel jade pearl garnet gilt""".split())

# Capitalised words that are fine even if no description contains them.
ALLOW = {"met", "museum", "object", "objects", "exhibition", "gallery"}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÿ]+", text)


def _stem(w: str) -> str:
    w = w.lower()
    return w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w


def _norm(s: str) -> str:
    """Lowercase, drop field labels like 'Medium:', and reduce to letters/digits separated by single spaces."""
    s = re.sub(r"(?i)\b(title|culture|period|medium|classification|date|department|object id)\s*:", " ", s)
    return " ".join(re.findall(r"[0-9a-zà-ÿ]+", s.lower()))


def medium_stems(blocks: list[str]) -> set[str]:
    """Stems of words in the Medium: and Classification: fields (a title mention alone doesn't count)."""
    out = set()
    for b in blocks:
        for m in re.finditer(r"(?:Medium|Classification): ([^\n]*?)(?=\. (?:Title|Culture|Period|Medium|Classification):|\n|$)", b):
            out |= {_stem(t) for t in _tokens(m.group(1))}
    return out


def ungrounded_names(text: str, source_blocks: list[str], extra_ok: str = "") -> list[str]:
    """Capitalised words (not at the start of a sentence) that appear in no description."""
    src = {_stem(t) for t in _tokens(" ".join(source_blocks) + " " + extra_ok)}
    bad = []
    for sent in re.split(r"(?<=[.!?])['\"’”)]*\s+", text):
        for w in _tokens(sent)[1:]:
            if w[0].isupper() and len(w) > 2 and _stem(w) not in src and w.lower() not in ALLOW:
                bad.append(w)
    return sorted(set(bad))


def validate(m: Mashup, allowed_ids: set[int], blocks: list[str] | None = None,
             version: str = "v3", seed_word: str | None = None) -> str | None:
    """Return an error message (fed back to the model on retry), or None if the mashup passes."""
    blocks = blocks or []
    bad = [i for i in m.object_ids if i not in allowed_ids]
    if bad:
        return f"object_ids {bad} are not among the given IDs {sorted(allowed_ids)}."
    if not m.object_ids:
        return "object_ids is empty."
    if re.search(r"\[[^\]]*\]", m.pitch):
        return "the pitch contains [bracketed placeholders]; write plain prose with no placeholders."
    if re.search(r"(^|\s)#\w", m.pitch):
        return "the pitch contains hashtags; remove them and write plain prose only."
    if re.search(r"(?i)\b(exhibition date|admission|location:|visit our|for more information)", m.pitch):
        return "the pitch contains event details (dates, location, admission); remove them."
    src = " ".join(blocks).lower()
    for w in ("anonymous", "unknown artist"):
        if w in m.pitch.lower() and w not in src:
            return f"the pitch says '{w}', which no description supports; remove invented attributions."
    n_words = len(m.pitch.split())
    if n_words > (MAX_WORDS_V3 if version == "v3" else MAX_WORDS):
        return f"the pitch is {n_words} words; it must be at most {100 if version == 'v3' else 120}. Shorten it."
    if m.pitch.rstrip()[-1:] not in ".!?\"'’”)":
        return "the pitch is cut off mid-sentence; rewrite it so it ends with a complete sentence."
    leaked = [i for i in allowed_ids if str(i) in m.pitch] + (["object_ids"] if "object_id" in m.pitch.lower() else [])
    if leaked:
        return f"the pitch text must not contain IDs or the word object_ids (found {leaked}); put IDs only in object_ids."
    if version != "v3":
        return None

    # ---- v3-only checks: step 1 evidence, step 2 link, step 3 pitch ----
    if len(m.evidence) != len(blocks):
        return f"evidence must have exactly {len(blocks)} phrases, one per object in order."
    for n, (phrase, block) in enumerate(zip(m.evidence, blocks), 1):
        if len(_norm(phrase)) < 3 or f" {_norm(phrase)} " not in f" {_norm(block)} ":
            return f"evidence phrase {n} ({phrase!r}) is not copied exactly from Object {n}'s description."
    if re.search(r"[{}\[\]]", m.link):
        return "the link contains stray brackets or braces; write one plain sentence."
    if len(m.link.split()) > MAX_LINK_WORDS or len(re.findall(r"[.!?](\s+[A-Z]|$)", m.link.strip())) > 1:
        return f"the link must be ONE sentence of at most 35 words (it has {len(m.link.split())} words)."
    ev_stems = {_stem(t) for t in _tokens(" ".join(m.evidence))}
    med = medium_stems(blocks)
    for t in _tokens(m.link):
        if _stem(t) in LEXICON and (_stem(t) not in ev_stems or _stem(t) not in med):
            return (f"the link mentions '{t}', but no object's Medium or Classification line says so (a word in a "
                    "title does not count). Only name materials and techniques you quoted from the Medium lines.")
    for t in _tokens(m.pitch):
        if _stem(t) in LEXICON and _stem(t) not in {_stem(x) for x in _tokens(src)}:
            return f"the pitch mentions '{t}', which appears in none of the descriptions; remove it."
    if re.search(r"(?i)\b(search|searched|searching|visitor|visitors)\b", m.pitch):
        return "the pitch must not use the words 'search' or 'visitor'; work the theme into the exhibition's idea."
    names = ungrounded_names(m.pitch + " " + m.link, blocks, extra_ok=(seed_word or "") + " " + m.title)
    if names:
        return f"the pitch/link names {names}, which appear in no description. Remove anything not in the descriptions."
    return None


@dataclass
class Result:
    mashup: Mashup | None
    valid: bool
    attempts: int
    errors: list[str] = field(default_factory=list)


def generate_mashup(metas, texts, llm=None, max_retries: int = 1, seed_word: str | None = None,
                    version: str = "v3") -> Result:
    """Ask for a Mashup; on a bad ID, leaked text, ungrounded claim or parse failure, retry with the reason."""
    llm = llm or get_llm()
    chain = llm.with_structured_output(VERSIONS[version][1])  # Ollama json_schema mode
    allowed = {int(m["object_id"]) for m in metas}
    blocks = [describe(m, t) for m, t in zip(metas, texts)]
    errors, feedback, mashup = [], None, None
    for attempt in range(1, max_retries + 2):
        try:
            out = chain.invoke(build_prompt(metas, texts, feedback, seed_word, version))
            d = out.model_dump()
            if version == "v3":
                d["evidence"] = [d.pop(f"evidence_{i}") for i in (1, 2, 3)]
            mashup = Mashup(**d)
            err = validate(mashup, allowed, blocks, version, seed_word)
        except Exception as e:  # runaway/truncated JSON, schema errors, timeouts
            mashup, err = None, f"the output was not valid JSON for the schema ({type(e).__name__})."
        if err is None:
            return Result(mashup, True, attempt, errors)
        errors.append(err)
        feedback = err
    return Result(mashup, False, max_retries + 1, errors)  # mashup may be None if every attempt failed to parse


def generate_with_fallback(metas, texts, llm=None, max_retries: int = 2,
                           seed_word: str | None = None) -> tuple[Result, str]:
    """Try the strict, grounded prompt (v3) first; if it can't produce a valid pitch, fall back to v2.
    Returns (result, version_used). Attempts and errors are combined so nothing is hidden."""
    llm = llm or get_llm()
    first = generate_mashup(metas, texts, llm, max_retries, seed_word, version="v3")
    if first.valid:
        return first, "v3"
    second = generate_mashup(metas, texts, llm, max_retries, seed_word, version="v2")
    combined = Result(second.mashup, second.valid, first.attempts + second.attempts,
                      [f"[v3] {e}" for e in first.errors] + [f"[v2] {e}" for e in second.errors])
    return combined, "v2"
