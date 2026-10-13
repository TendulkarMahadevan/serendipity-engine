"""Stage 1: filter the Met CSV, sample 3000 rows, build Documents, show a few."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serendipity.data import build_documents

docs, counts = build_documents(3000)
print("Filtering counts")
for k, v in counts.items():
    print(f"  {k:35s} {v:>9,}")

lens = [len(d.page_content) for d in docs]
print(f"\npage_content length: min={min(lens)}, median={sorted(lens)[len(lens)//2]}, max={max(lens)} chars")

for d in docs[:5]:
    print("\n--- Document ---")
    print("page_content:", d.page_content)
    print("metadata:    ", d.metadata)
