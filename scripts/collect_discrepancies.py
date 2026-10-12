"""Collect the "Discrepancies" section of every deck's notes/sources.md into docs/notion-discrepancies.md:
the places where a Notion row disagrees with its primary sources (the slides follow the sources)."""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "data" / "manifest.json").read_text())

out = [
    "# Notion rows vs. primary sources",
    "",
    "While building each deck, the agent compared the Notion row (Core Mechanism, Impact / Measured Results) "
    "with the primary sources it read. The slides follow the sources; this page collects every difference found, "
    "per technique. Source locations for each point are in the deck's `notes/sources.md`. The rows were "
    "corrected on 2026-10-13; [`notion-corrections.md`](notion-corrections.md) has the before/after of each edit.",
    "",
]
found = 0
for e in manifest:
    src = ROOT / "decks" / e["slug"] / "notes" / "sources.md"
    if not src.exists():
        continue
    text = src.read_text()
    m = re.search(r"^(#{2,3}) [^\n]*discrepanc[^\n]*\n(.*?)(?=^#{1,2} |\Z)", text, re.S | re.I | re.M)
    if not m:
        continue
    body = m.group(2).strip()
    body = re.sub(r"^#{3,} ", "**", body, flags=re.M)  # flatten sub-headings
    if not body:
        continue
    found += 1
    out += [f"## {e['id']} · {e['technique']}", "",
            f"[deck](../decks/{e['slug']}/) · [Notion row]({e['notion_url']})", "", body, ""]
(ROOT / "docs" / "notion-discrepancies.md").write_text("\n".join(out) + "\n")
print(f"docs/notion-discrepancies.md: {found} techniques with discrepancies")
