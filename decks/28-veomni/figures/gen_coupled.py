"""fig_coupled: (a) parallel logic inside model code vs (b) VeOmni's unmodified model + separate parallel plan."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 286, "fig_coupled: coupled vs decoupled")

ROWS = [56, 122, 188]
RH = 50

# ---------------- panel (a)
s.text(10, 26, "(a) Parallelism inside model code", 20, PRB, anchor="start", weight="bold")
mods = [("ViT encoder", ["TP", "PP"]), ("LLM layers", ["TP", "PP", "EP"])]
for (name, chips), y in zip(mods, ROWS):
    s.rect(10, y, 404, RH, NEU_F, NEU, 2, 5)
    s.text(24, y + RH / 2 + 6.5, name, 19, INK, anchor="start", weight="bold")
    x = 404 + 10 - 12 - 56 * len(chips) + 4
    for c in chips:
        s.box(x, y + 9, 50, RH - 18, c, PRB_F, PRB, PRB, 17, "bold", 2, 3)
        x += 56
# new modality: must be rewritten
y = ROWS[2]
s.rect(10, y, 404, RH, "#FFFFFF", PRB, 2, 5, dash="7,5")
s.text(24, y + RH / 2 + 6.5, "Audio encoder", 19, INK, anchor="start", weight="bold")
s.text(400, y + RH / 2 + 6.5, "rewrite for TP / PP", 18, PRB, anchor="end", weight="bold")
s.text(10, 270, "one global plan for every module", 18, PRB, anchor="start", style="italic")

# divider
s.line(435, 10, 435, 276, "#BFBFBF", 1.5)

# ---------------- panel (b)
s.text(456, 26, "(b) VeOmni: model plus a plan", 20, ACC, anchor="start", weight="bold")
plan = ["FSDP2", "FSDP2 + SP + EP", "FSDP2"]
for (name, y, p) in zip(["ViT encoder", "LLM layers", "Audio encoder"], ROWS, plan):
    s.rect(456, y, 190, RH, NEU_F, NEU, 2, 5)
    s.text(551, y + RH / 2 + 6.5, name, 19, INK, weight="bold")
    s.line(684 - 2, y + RH / 2, 646 + 3, y + RH / 2, ACC, 2.4, "arrAcc")
s.rect(684, 44, 180, 206, "none", ACC, 2.2, 6)
s.add(f'<rect x="684" y="44" width="180" height="206" rx="6" fill="{ACC_F}" fill-opacity="0.55" stroke="none"/>')
for y, p in zip(ROWS, plan):
    s.text(774, y + RH / 2 + 6.5, p, 18, ACC, weight="bold")
s.text(774, 270, "ParallelPlan", 18, ACC, weight="bold", family="Menlo, monospace")
s.text(456, 270, "HF model code", 18, MUTED, anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_coupled.svg"))
