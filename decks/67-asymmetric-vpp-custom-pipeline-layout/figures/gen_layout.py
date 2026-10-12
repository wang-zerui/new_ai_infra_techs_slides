"""fig_layout_1 / fig_layout_2: DeepSeek-V3 H100 layout "Et*3|(tt|)*29m|L" with PP 8 -> VPP 4 ([MC] Sec. 9.2.2).

Stage s is placed on PP rank s % 8, virtual chunk s // 8 (PipelineParallelLayerLayout.__init__).
Build 2 adds stage indices and the per-rank totals.
"""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
MONO = "Menlo, 'Courier New', monospace"
PP, VPP = 8, 4
stages = ["E+3 dense"] + ["2 MoE"] * 29 + ["MTP", "loss"]
special = {0, 30, 31}
X0, W, G = 84, 88, 8
Y0, H, RG = 66, 44, 8
totals = ["E+3d+6M", "8 MoE", "8 MoE", "8 MoE", "8 MoE", "8 MoE", "6M+MTP", "6M+loss"]

for step in (1, 2):
    s = SVG(870, 318, f"fig_layout_{step}: rank x chunk table of the H100 DeepSeek-V3 layout")
    s.text(X0 - 70, 24, "--pipeline-model-parallel-layout  Et*3|(tt|)*29m|L", 18, ACC, anchor="start", weight="bold",
           family=MONO)
    for p in range(PP):
        s.text(X0 + p * (W + G) + W / 2, Y0 - 12, f"PP {p}", 17, INK, weight="bold")
    for v in range(VPP):
        y = Y0 + v * (H + RG)
        s.text(X0 - 10, y + H / 2 + 6, f"chunk {v}", 16, INK, anchor="end")
        for p in range(PP):
            st = v * PP + p
            x = X0 + p * (W + G)
            new = st in special
            fill, stroke, col = (ACC_F, ACC, ACC) if new else (NEU_F, NEU, INK)
            s.rect(x, y, W, H, fill, stroke, 2.4 if new else 1.6, 4)
            if step == 2:
                s.text(x + 6, y + 15, f"s{st}", 15, MUTED, anchor="start")
                s.text(x + W / 2, y + 37, stages[st], 16, col, weight="bold" if new else "normal")
            else:
                s.text(x + W / 2, y + H / 2 + 6, stages[st], 16, col, weight="bold" if new else "normal")
    if step == 2:
        yt = Y0 + VPP * (H + RG) + 8
        for p in range(PP):
            x = X0 + p * (W + G)
            s.path(f"M{x + 2},{yt} v8 h{W - 4} v-8", MUTED, 1.5)
            s.text(x + W / 2, yt + 30, totals[p], 16, ACC if p in (0, 6, 7) else INK,
                   weight="bold" if p in (0, 6, 7) else "normal")
    s.save(os.path.join(HERE, f"fig_layout_{step}.svg"))
