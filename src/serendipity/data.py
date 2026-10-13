"""Stage 1: download the Met open-access CSV and turn rows into LangChain Documents."""
import hashlib
from pathlib import Path

import pandas as pd
import requests
from langchain_core.documents import Document

from .config import DATA_DIR, RANDOM_SEED

CSV_PATH = DATA_DIR / "MetObjects.csv"
# The repo stores the CSV in Git LFS; this URL serves the real file, not the pointer.
CSV_URL = "https://media.githubusercontent.com/media/metmuseum/openaccess/master/MetObjects.csv"
POINTER_URL = "https://raw.githubusercontent.com/metmuseum/openaccess/master/MetObjects.csv"

# Columns that carry descriptive meaning (used for text and for the "enough text" filter).
DESCRIPTIVE_COLS = ["Culture", "Period", "Medium", "Classification", "Tags"]


def download_csv(path: Path = CSV_PATH) -> Path:
    """Stream the CSV to disk (it is ~318 MB) and verify it against the LFS pointer's sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pointer = requests.get(POINTER_URL, timeout=30).text
    expected = next(l.split(":", 1)[1] for l in pointer.splitlines() if l.startswith("oid sha256"))
    if path.exists():
        print(f"{path.name} already exists, skipping download")
    else:
        with requests.get(CSV_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == expected, f"sha256 mismatch: {h.hexdigest()} != {expected}"
    print(f"downloaded {path.stat().st_size:,} bytes, sha256 verified")
    return path


def load_filtered(path: Path = CSV_PATH) -> tuple[pd.DataFrame, dict]:
    """Read the CSV and keep only rows with a title plus some descriptive text. Returns (df, counts)."""
    df = pd.read_csv(path, dtype=str, low_memory=False)
    counts = {"total rows": len(df)}

    # Public domain only (column exists in the real file; see Stage 1 column printout).
    if "Is Public Domain" in df.columns:
        df = df[df["Is Public Domain"].str.lower() == "true"]
    counts["public domain"] = len(df)

    def present(col):
        return df[col].fillna("").str.strip() != ""

    df = df[present("Title")]
    counts["has title"] = len(df)

    cols = [c for c in DESCRIPTIVE_COLS if c in df.columns]
    has_desc = pd.concat([present(c) for c in cols], axis=1).any(axis=1)
    df = df[has_desc]
    counts["title + >=1 descriptive field"] = len(df)
    return df, counts


def sample(df: pd.DataFrame, n: int = 3000) -> pd.DataFrame:
    return df.sample(n=min(n, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)


def _val(row, col) -> str:
    v = row.get(col)
    return "" if pd.isna(v) else str(v).strip()


def row_to_document(row) -> Document:
    """page_content = readable text that gets embedded; metadata = fields that ride along."""
    parts = [f"Title: {_val(row, 'Title')}"]
    for label, col in [("Culture", "Culture"), ("Period", "Period"),
                       ("Medium", "Medium"), ("Classification", "Classification")]:
        if _val(row, col):
            parts.append(f"{label}: {_val(row, col)}")
    return Document(
        page_content=". ".join(parts),
        metadata={
            "object_id": int(row["Object ID"]),
            "department": _val(row, "Department"),
            "date": _val(row, "Object Date"),
            "title": _val(row, "Title"),
            "culture": _val(row, "Culture"),
            "medium": _val(row, "Medium"),
            "url": _val(row, "Link Resource"),
        },
    )


def build_documents(n: int = 3000) -> tuple[list[Document], dict]:
    df, counts = load_filtered()
    s = sample(df, n)
    counts[f"sampled (seed={RANDOM_SEED})"] = len(s)
    return [row_to_document(r) for _, r in s.iterrows()], counts
