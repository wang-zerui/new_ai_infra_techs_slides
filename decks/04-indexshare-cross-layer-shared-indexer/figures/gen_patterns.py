"""Slide 8 (mechanism 2): which of GLM-5's 78 layers keep an indexer (F) at 1/4 retention.
Patterns: uniform = 'FSSS' repeated from layer 1 (IndexCache Sec. 3.1.1); searched = IndexCache App. B (GLM-5, keep 1/4);
GLM-5.2 = HF config indexer_types (21 full, 57 shared)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
N = 78
uniform = ("FSSS" * 20)[:N]
searched = "FFSFSSSFSSFSFSSSSSSSFSSSSSSFSSSFSFSSSSFFFFFSSSFFSSSFSSSSSSSSFSSS" + "FSSSSSSFSFSSSS"
glm52 = "FFF" + "FSSS" * 18 + "S"   # placeholder, replaced below from config
import json
cfg = json.load(open(os.path.join(HERE, "..", "notes", "hf_config_GLM-5.2.json")))
glm52 = "".join("F" if t == "full" else "S" for t in cfg["indexer_types"])
assert len(searched) == N and len(glm52) == N

W, H = 870, 236
s = SVG(W, H, "indexer retention patterns, GLM-5 78 layers")
x0, cw, cg = 6, 10, 1
rows = [
    (f"Uniform every 4th layer, training-free  ·  {uniform.count('F')} indexers", uniform, False),
    (f"Greedy-searched for GLM-5  ·  {searched.count('F')} indexers", searched, True),
    (f"GLM-5.2 as shipped, trained  ·  {glm52.count('F')} indexers", glm52, True),
]
y = 4
for label, pat, good in rows:
    s.text(x0, y + 17, label, 18, INK, anchor="start", weight="bold")
    y += 26
    for i, c in enumerate(pat):
        x = x0 + i * (cw + cg)
        if c == "F":
            s.rect(x, y, cw, 28, ACC if good else "#7F7F7F", ACC if good else "#595959", 1, 1)
        else:
            s.rect(x, y, cw, 28, "#FFFFFF", "#C8C8C8", 1, 1)
    y += 28 + 18
for L in (1, 20, 40, 60, 78):
    x = x0 + (L - 1) * (cw + cg) + cw / 2
    s.line(x, y - 14, x, y - 8, MUTED, 1.4)
    s.text(x, y + 8, f"{L}", 16, MUTED)
s.text(x0 + 300, y + 8, "layer index", 16, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_patterns.svg"))
