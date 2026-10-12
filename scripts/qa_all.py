"""Repo-wide QA: for every deck, check the deliverables exist, every referenced figure exists,
no placeholder figures, PDF is newer than its markdown, and re-run the usenix-talk-slides audit.

    uv run --with pymupdf python3 scripts/qa_all.py [slug ...]
"""
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
AUDIT = pathlib.Path.home() / ".claude/skills/usenix-talk-slides/scripts/audit_deck.py"
manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
want = set(sys.argv[1:])
rows = []
for e in manifest:
    slug = e["slug"]
    if want and slug not in want:
        continue
    d = ROOT / "decks" / slug
    md, pdf, pptx = d / f"{slug}.md", d / f"{slug}.pdf", d / f"{slug}.pptx"
    issues = []
    if not d.exists():
        rows.append((slug, "MISSING", ["no deck folder"]))
        continue
    for f in (md, pdf, pptx):
        if not f.exists():
            issues.append(f"missing {f.name}")
    for n in ("story.md", "sources.md"):
        if not (d / "notes" / n).exists():
            issues.append(f"missing notes/{n}")
    if md.exists():
        text = md.read_text()
        for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", text):
            if not (d / m.group(1)).exists():
                issues.append(f"figure not found: {m.group(1)}")
        body = re.sub(r"<!--.*?-->", "", text, flags=re.S)  # speaker notes may quote source TODOs
        if re.search(r'(?<!["“])\bTODO\b', body):
            issues.append("TODO in markdown")
        for m in re.finditer(r"%%\s*qr:\s*(\S+)", text):
            if len(m.group(1)) > 60:
                issues.append(f"long QR URL ({len(m.group(1))} chars)")
        if pdf.exists() and pdf.stat().st_mtime + 1 < md.stat().st_mtime:
            issues.append("PDF older than markdown (rebuild)")
    html = d / f"{slug}.html"
    if html.exists() and "[figure:" in html.read_text(errors="ignore"):
        issues.append("placeholder figure in HTML")
    pages = None
    if pdf.exists():
        out = subprocess.run([sys.executable, str(AUDIT), str(pdf), "--minutes", "8"],
                             capture_output=True, text=True)
        for line in out.stdout.splitlines():
            if line.strip().startswith(("FAIL", "WARN")) or " FAIL" in line[:12] or " WARN" in line[:12]:
                issues.append("audit: " + line.strip())
        try:
            import pymupdf
            pages = len(pymupdf.open(str(pdf)))
        except Exception:
            pass
    rows.append((slug, f"{pages} pages" if pages else "-", issues))

bad = 0
for slug, info, issues in rows:
    status = "OK " if not issues else "ISSUE"
    bad += bool(issues)
    print(f"{status} {slug:55s} {info}")
    for i in issues:
        print(f"      - {i}")
print(f"\n{len(rows) - bad}/{len(rows)} decks clean")
