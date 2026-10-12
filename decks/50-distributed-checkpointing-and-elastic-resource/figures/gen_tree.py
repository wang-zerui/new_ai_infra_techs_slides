"""fig_tree: VeOmni checkpoint step directory and who writes each part (docs/usage/checkpoint.md)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 272, "fig_tree")
MONO = "Menlo, monospace"
rows = [
    ("model/ckpt/", "weights", "all ranks (DCP)", True),
    ("model/optimizer/", "optimizer state", "all ranks (DCP)", True),
    ("model/lr_scheduler.pt", "LR schedule", "rank 0", False),
    ("loader/rank_R.pt", "data cursor", "rank R", False),
    ("extra_state/rank_R.pt", "step and RNG", "rank R", False),
    ("checkpoint_manifest.json", "done marker", "rank 0 (last)", False),
]
s.text(16, 24, "global_step_N/", 18, INK, anchor="start", family=MONO, weight="bold")
s.text(500, 24, "holds", 18, MUTED, anchor="start", weight="bold")
s.text(740, 24, "written by", 18, MUTED, weight="bold")
Y0, RH = 40, 38
for i, (path, holds, who, dcp) in enumerate(rows):
    y = Y0 + i * RH
    if i % 2 == 0:
        s.rect(8, y, 854, RH, "#F7F7F7", "none", 0, 0)
    s.text(40, y + RH / 2 + 6, path, 17, ACC if dcp else INK, anchor="start", family=MONO,
           weight="bold" if dcp else "normal")
    s.text(500, y + RH / 2 + 6, holds, 17, INK, anchor="start")
    if dcp:
        s.box(650, y + 5, 180, RH - 10, who, ACC_F, ACC, ACC, 16, "bold", 2, 4)
    else:
        s.box(650, y + 5, 180, RH - 10, who, "#FFFFFF", NEU, INK, 16, "normal", 1.5, 4)
# tree guide
s.line(24, 34, 24, Y0 + 5 * RH + RH / 2, "#A6A6A6", 1.4)
for i in range(6):
    yy = Y0 + i * RH + RH / 2
    s.line(24, yy, 34, yy, "#A6A6A6", 1.4)
s.save(os.path.join(HERE, "fig_tree.svg"))
