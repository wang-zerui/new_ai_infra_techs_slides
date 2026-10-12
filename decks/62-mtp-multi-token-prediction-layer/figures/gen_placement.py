"""fig_placement: where the MTP module goes in a pipeline.
(a) DeepSeek-V3 DualPipe: shallowest (embedding) and deepest (output head) layers on the same PP rank, so MTP shares
    their parameters and gradients physically (arXiv 2412.19437 §3.2).
(b) Megatron-Core flexible asymmetric VPP for DeepSeek-V3, PP=16, VPP=2: rank 14 VPP1 = MTP, rank 15 VPP1 = loss
    (arXiv 2603.07685 Table 10).
(c) GLM-5: MTP output layer co-located with the main output layer on the final stage; MTP embedding + transformer on the
    preceding stage (arXiv 2602.15763 §2.4.1)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 284
s = SVG(W, H, "fig_placement: MTP placement in pipeline parallelism, three designs")

# (a) DualPipe: one PP rank holds both ends of the model
s.rect(8, 30, 262, 206, "#FFFFFF", NEU, 1.6, 6, dash="6,4")
s.text(20, 52, "one PP rank", 17, INK, anchor="start", weight="bold")
s.box(24, 66, 230, 44, "Embedding + first layers", NEU_F, NEU, size=17)
s.box(24, 120, 230, 44, "Last layers + head", NEU_F, NEU, size=17)
s.box(24, 174, 230, 44, "MTP module", ACC_F, ACC, color=ACC, weight="bold", size=17)
s.text(139, 262, "(a) DeepSeek-V3: DualPipe", 18, INK, weight="bold")

for x in (284, 584):
    s.line(x, 20, x, 284, "#BFBFBF", 1.2)

# (b) Megatron-Core flexible VPP layout
X, CW = 292, 110
s.text(X + 66 + CW / 2 - 4, 40, "VPP 0", 16, MUTED, style="italic")
s.text(X + 66 + CW * 1.5 + 4, 40, "VPP 1", 16, MUTED, style="italic")
rows = [("rank 0", ["Emb + 3 dec", "2 dec"], 52), ("rank 14", ["2 dec", "MTP"], 134), ("rank 15", ["2 dec", "loss"], 186)]
for name, cells, y in rows:
    s.text(X + 56, y + 27, name, 16, INK, anchor="end")
    for j, c in enumerate(cells):
        hot = c == "MTP"
        s.box(X + 62 + j * (CW + 6), y, CW, 42, c, ACC_F if hot else NEU_F, ACC if hot else NEU,
              color=ACC if hot else INK, weight="bold" if hot else "normal", size=17)
s.text(X + 62 + CW + 3, 118, "⋯", 22, MUTED)
s.text(434, 262, "(b) MCore: own virtual stage", 18, INK, weight="bold")

# (c) GLM-5 split
GX, GW = 600, 262
s.rect(GX, 30, GW, 98, "#FFFFFF", NEU, 1.6, 6, dash="6,4")
s.text(GX + 10, 52, "stage N−1", 17, INK, anchor="start", weight="bold")
s.box(GX + 12, 64, GW - 24, 50, "MTP emb + block", ACC_F, ACC, color=ACC, weight="bold", size=17)
s.rect(GX, 138, GW, 98, "#FFFFFF", NEU, 1.6, 6, dash="6,4")
s.text(GX + 10, 160, "stage N (last)", 17, INK, anchor="start", weight="bold")
s.box(GX + 12, 172, GW - 24, 50, ["Output head", "main + MTP"], NEU_F, NEU, size=17)
s.text(GX + GW / 2, 262, "(c) GLM-5: split MTP", 18, INK, weight="bold")
s.save(os.path.join(HERE, "fig_placement.svg"))
