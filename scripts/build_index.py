"""Regenerate README.md: an index of every deck, grouped by category, from data/manifest.json."""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "data" / "manifest.json").read_text())


def slide_count(slug):
    md = ROOT / "decks" / slug / f"{slug}.md"
    if not md.exists():
        return None
    return 1 + sum(1 for line in md.read_text().splitlines() if re.match(r"^# ", line))


def pdf_pages(slug):
    pdf = ROOT / "decks" / slug / f"{slug}.pdf"
    if not pdf.exists():
        return None
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))


by_cat = {}
for e in manifest:
    by_cat.setdefault(e["category"], []).append(e)

lines = [
    "# Industrial Pretraining Architecture & Infra Tech Landscape (2026): one talk deck per technique",
    "",
    f"{len(manifest)} conference-style explainer decks, one for each technique in the Notion database "
    "*Industrial Pretraining Architecture & Infra Tech Landscape (2026)*: sparse and linear attention, MoE routing, "
    "expert-parallel communication, low-precision training, kernels, parallelism and memory scheduling, as used in "
    "DeepSeek-V3.2/V4, Kimi K2/K3, GLM-4.5/5, Megatron Core, TransformerEngine, DeepEP, DeepGEMM, HPC-Ops, MoonEP, VeOmni and MagiAttention.",
    "",
    "Each deck is an ~8-minute talk in the USENIX OSDI/NSDI house style: a measured pain number, one insight, the "
    "mechanism walked through with redrawn diagrams, results charts whose numbers trace to the primary source, and a "
    "summary with the reference link. Every deck folder holds:",
    "",
    "| File | What it is |",
    "|---|---|",
    "| `<slug>.pdf` | the slides (Chrome render) |",
    "| `<slug>.pptx` | editable PowerPoint/Keynote/Google Slides deck with speaker notes |",
    "| `<slug>.md` | slide-markdown source |",
    "| `figures/` | hand-authored SVG diagrams + PDF/PNG renders |",
    "| `charts/` | matplotlib scripts (data at the top) + PDF/PNG renders |",
    "| `notes/` | story plan, sources with the location of every number, the Notion row |",
    "",
    "How they are made: [`docs/deck-recipe.md`](docs/deck-recipe.md). Rebuild everything with "
    "`bash scripts/build_all.sh` (needs the `usenix-talk-slides` and `ipads-arch-figures` skills, Google Chrome and `uv`).",
    "",
]
for cat in sorted(by_cat):
    lines += [f"## {cat}", "", "| # | Technique | Source | Slides | PPTX | Notion |", "|---|---|---|---|---|---|"]
    for e in by_cat[cat]:
        slug = e["slug"]
        d = f"decks/{slug}"
        n = slide_count(slug)
        pdf = f"[PDF]({d}/{slug}.pdf)" + (f" ({n} slides)" if n else "") if (ROOT / d / f"{slug}.pdf").exists() else "pending"
        pptx = f"[PPTX]({d}/{slug}.pptx)" if (ROOT / d / f"{slug}.pptx").exists() else ""
        tech = e["technique"].replace("|", "/")
        lines.append(f"| {e['id']} | [{tech}]({d}/) | {', '.join(e['source'])} | {pdf} | {pptx} | [row]({e['notion_url']}) |")
    lines.append("")
lines += [
    "## Notes",
    "",
    "- Decks summarise public papers, repositories and documentation; numbers are quoted from those sources "
    "(see each deck's `notes/sources.md`). Diagrams are redrawn; no paper figures are reproduced.",
    "- Where a Notion row disagreed with its primary source, the slides follow the source and the difference is "
    "listed under *Discrepancies* in `notes/sources.md`; all of them are collected in "
    "[`docs/notion-discrepancies.md`](docs/notion-discrepancies.md).",
    "- Every deck folder links to its Notion row; each Notion row embeds the PDF, attaches the PPTX and links back here.",
    "",
]
(ROOT / "README.md").write_text("\n".join(lines))
print("README.md written,", len(manifest), "entries")
