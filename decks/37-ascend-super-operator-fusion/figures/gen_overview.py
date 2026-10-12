"""fig_overview: where the three super-operators sit in one GLM-5 attention layer on Ascend.

Row A: the value path (MLAPO -> Sparse Flash Attention -> MoE block).
Row B: the index path (Lightning Indexer), feeding the top-2048 ids into Sparse FA.
Row C: the caches both paths stream from.
The MoE block is gray: the GLM-5 report describes no MoE fusion of its own.
"""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 300, "fig_overview: three super-operators in one GLM-5 attention layer on Ascend")

YA, YB, BH = 40, 150, 62
YC, CH = 250, 44

# ---- row A: value path
s.box(14, YA, 92, BH, ["hidden", "state"], "#FFFFFF", NEU, size=17)
s.line(106, YA + BH / 2, 138, YA + BH / 2, INK, 2.2, "arr")
s.box(140, YA, 200, BH, ["❶ MLAPO", "13 ops → 1 kernel"], ACC_F, ACC, color=ACC, size=18,
      weight="bold", sw=3)
s.text(240, YA + BH + 20, "also writes the KV-cache row", 15, MUTED, style="italic")
s.line(340, YA + BH / 2, 428, YA + BH / 2, INK, 2.2, "arr")
s.text(384, YA + BH / 2 - 10, "q", 16, INK)
s.box(430, YA, 226, BH, ["❸ Sparse Flash Attn", "gather KV · attend"], ACC_F, ACC, color=ACC,
      size=18, weight="bold", sw=3)
s.line(656, YA + BH / 2, 688, YA + BH / 2, INK, 2.2, "arr")
s.box(690, YA, 166, BH, ["MoE block", "routing · GEMM"], NEU_F, NEU, size=16)
s.text(773, YA + BH + 20, "no GLM-5 fusion claim", 15, MUTED, style="italic")

# ---- row B: index path
s.path(f"M 60 {YA + BH} L 60 {YB + 31} L 162 {YB + 31}", INK, 2.2, "arr")
s.box(164, YB, 230, BH, ["❷ Lightning Indexer", "score · ReLU · TopK"], ACC_F, ACC,
      color=ACC, size=18, weight="bold", sw=3)
s.path(f"M 394 {YB + 31} L 470 {YB + 31} L 470 {YA + BH + 4}", ACC, 2.4, "arrAcc")
s.text(480, YB + 4, "top-2048", 16, ACC, anchor="start", weight="bold")
s.text(480, YB + 24, "token ids", 16, ACC, anchor="start", weight="bold")

# ---- row C: caches
s.box(180, YC, 214, CH, ["index key cache"], NEU_F, NEU, size=16)
s.line(287, YC, 287, YB + BH + 3, MUTED, 2.2, "arrGray")
s.box(430, YC, 200, CH, ["KV cache"], NEU_F, NEU, size=16)
s.line(600, YC, 600, YA + BH + 3, MUTED, 2.2, "arrGray")

# ---- cross-deck pointers
for i, t in enumerate(["DSA / indexer: deck 02", "GPU indexer kernel: deck 10",
                       "Ascend stack: deck 23", "GPU Mega MoE: deck 35"]):
    s.text(856, YB + 18 + i * 22, t, 15, MUTED, anchor="end", style="italic")
s.save(os.path.join(HERE, "fig_overview.svg"))
