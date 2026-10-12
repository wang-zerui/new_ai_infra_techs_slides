"""fig_models: which logit control each Muon-era MoE uses.

DeepSeek-V3: MLA, AdamW (arXiv 2412.19437 Sec. 2.1.1, 4.2). Kimi K2: MLA, MuonClip = Muon + QK-Clip (2507.20534 Sec. 2.1).
GLM-4.5: GQA, Muon, QK-Norm (2508.06471 Sec. 2.1, 2.4, Table 1). GLM-5: MLA, Muon with Muon Split, no clipping
(2602.15763 Sec. 2.1). DeepSeek-V4: CSA/HCA, Muon, RMSNorm on query heads and KV entries, no QK-Clip (2606.19348 Sec. 2.3.3).
"""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 276
s = SVG(W, H, "fig_models: attention type, optimizer and logit control per model")
COLS = [(0, 190), (190, 370), (370, 540), (540, 870)]
HEAD, RH = 36, 48
rows = [
    ("DeepSeek-V3", "MLA", "AdamW", "none"),
    ("Kimi K2", "MLA", "Muon", "QK-Clip (deck 65)"),
    ("GLM-4.5", "GQA", "Muon", "QK-Norm"),
    ("GLM-5", "MLA", "Muon", "Muon Split"),
    ("DeepSeek-V4", "CSA / HCA", "Muon", "RMSNorm on q and KV"),
]
for (x0, x1), h in zip(COLS, ["Model", "Attention", "Optimizer", "Logit control"]):
    anchor = "start" if x0 == 0 else "middle"
    s.text(x0 + 8 if x0 == 0 else (x0 + x1) / 2, 25, h, 19, MUTED, anchor=anchor, weight="bold")
s.line(0, HEAD, W, HEAD, NEU, 1.6)
for i, row in enumerate(rows):
    y0 = HEAD + i * RH
    ours = row[0] == "GLM-4.5"
    if ours:
        s.rect(2, y0 + 3, W - 4, RH - 6, ACC_F, ACC, 2.2, rx=4)
    cy = y0 + RH / 2 + 0.35 * 20
    for j, ((x0, x1), v) in enumerate(zip(COLS, row)):
        col = ACC if ours else (MUTED if v == "none" else INK)
        wt = "bold" if (ours or j == 0) else "normal"
        if j == 0:
            s.text(x0 + 8, cy, v, 20, ACC if ours else INK, anchor="start", weight="bold")
        else:
            s.text((x0 + x1) / 2, cy, v, 20, col, weight=wt)
    if i < len(rows) - 1 and not ours and rows[i + 1][0] != "GLM-4.5":
        s.line(0, y0 + RH, W, y0 + RH, "#D0D0D0", 1)
s.save(os.path.join(HERE, "fig_models.svg"))
