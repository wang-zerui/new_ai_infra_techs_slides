"""fig_table: side-by-side shape table, DeepSeek-V3 vs Kimi K2 vs GLM-4.5.

Sources: GLM-4.5 Table 1 (arXiv 2508.06471, p.3); DeepSeek-V3 Sec. 4.2 (arXiv 2412.19437, p.22);
Kimi K2 Table 2 (arXiv 2507.20534, p.6); RoPE dims from the HF configs (qk_rope_head_dim 64 + qk_nope 128;
GLM-4.5 head_dim 128 x partial_rotary_factor 0.5).
"""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 340
s = SVG(W, H, "fig_table: shape comparison")

COLS = [(300, 490), (490, 680), (680, 870)]
HEAD_H, ROW_H = 36, 38
rows = [
    (1, "Layers (dense + MoE)", ["61 (3 + 58)", "61 (1 + 60)", "92 (3 + 89)"]),
    (1, "Hidden dim", ["7168", "7168", "5120"]),
    (1, "Routed experts (active)", ["256 (8)", "384 (8)", "160 (8)"]),
    (2, "Attention type", ["MLA", "MLA", "GQA"]),
    (2, "Query / KV heads", ["128 / 128", "64 / 64", "96 / 8"]),
    (3, "RoPE dims per QK head", ["64 of 192", "64 of 192", "64 of 128"]),
    (0, "QK-Norm (deck 66)", ["no", "no", "yes"]),
    (0, "Total / active params", ["671B / 37B", "1043B / 32B", "355B / 32B"]),
]

# GLM column background
s.rect(COLS[2][0], 0, COLS[2][1] - COLS[2][0], H, ACC_F, None, 0, rx=0)
s.rect(COLS[2][0] + 1, 1, COLS[2][1] - COLS[2][0] - 2, H - 2, "none", ACC, 2.2, rx=3)
# header
for (x0, x1), name in zip(COLS, ["DeepSeek-V3", "Kimi K2", "GLM-4.5"]):
    s.text((x0 + x1) / 2, 25, name, 20, ACC if name == "GLM-4.5" else INK, weight="bold")
s.line(0, HEAD_H, W, HEAD_H, NEU, 1.6)

prev_group = None
for i, (grp, label, vals) in enumerate(rows):
    y0 = HEAD_H + i * ROW_H
    cy = y0 + ROW_H / 2 + 0.35 * 19
    if grp and grp != prev_group:
        s.badge(13, y0 + ROW_H / 2, grp, r=11, size=15)
    prev_group = grp
    s.text(32, cy, label, 19, INK, anchor="start")
    for j, ((x0, x1), v) in enumerate(zip(COLS, vals)):
        ours = j == 2
        s.text((x0 + x1) / 2, cy, v, 19, ACC if ours else INK, weight="bold" if ours else "normal")
    if i < len(rows) - 1:
        grp_end = i in (2, 5)
        s.line(0, y0 + ROW_H, W, y0 + ROW_H, NEU if grp_end else "#D0D0D0", 1.6 if grp_end else 1)

s.save(os.path.join(HERE, "fig_table.svg"))
