"""Collect every decks/<slug>/notes/notion_correction.md (before/after of each Notion row edit) into
docs/notion-corrections.md, the changelog of the 2026-10-13 correction pass."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
out = [
    "# Notion row corrections (2026-10-13)",
    "",
    "The rows of *Industrial Pretraining Architecture & Infra Tech Landscape (2026)* were corrected against their "
    "primary sources, following [`notion-discrepancies.md`](notion-discrepancies.md). Each entry gives the text "
    "before and after and why; the untouched original row is kept in each deck's `notes/notion_row.md`. "
    "Only Technique, Core Mechanism, Impact / Measured Results, Source and Reference were eligible; page bodies "
    "and the other properties were not edited.",
    "",
]
n = 0
for e in manifest:
    f = ROOT / "decks" / e["slug"] / "notes" / "notion_correction.md"
    if not f.exists():
        continue
    body = f.read_text().split("\n", 1)[1].strip()          # drop the file's own H1
    body = body.replace("\n## ", "\n### ")
    if body.startswith("## "):
        body = "#" + body
    n += 1
    out += [f"## {e['id']} · {e['technique']}", "", f"[deck](../decks/{e['slug']}/) · [Notion row]({e['notion_url']})", "", body, ""]
(ROOT / "docs" / "notion-corrections.md").write_text("\n".join(out) + "\n")
print(f"docs/notion-corrections.md: {n} rows")
