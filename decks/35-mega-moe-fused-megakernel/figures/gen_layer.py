"""fig_layer: one V4 MoE layer under EP: token's GPU -> dispatch -> top-k experts on other GPUs -> combine."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

W, H = 524, 344
s = SVG(W, H, "one MoE layer under expert parallelism")

# column headers
s.text(2, 22, "token's GPU", 16, MUTED, anchor="start", style="italic")
s.text(287, 22, "top-6 experts on other GPUs", 16, MUTED, style="italic")
s.text(W - 2, 22, "token's GPU", 16, MUTED, anchor="end", style="italic")

# token box
tx, tw, ty, th = 4, 70, 140, 64
s.box(tx, ty, tw, th, ["x", "FP8"], fill=NEU_F, stroke=NEU, sizes=[22, 16], weights=["bold", "normal"])

rows = [48, 140, 232]
gpus = ["GPU 1", "GPU 4", "GPU 6"]
rx0, rx1 = 150, 424
bw, bh = 80, 44
xs = [158, 250, 342]
for ry, g in zip(rows, gpus):
    s.rect(rx0, ry, rx1 - rx0, 64, "#FFFFFF", MUTED, 1.5, 6, dash="5,3")
    s.text(rx0 + 4, ry - 6, g, 15, MUTED, anchor="start")
    by = ry + (64 - bh) / 2
    for i, (x, lab) in enumerate(zip(xs, ["Linear 1", "SwiGLU", "Linear 2"])):
        s.box(x, by, bw - 6, bh, lab, fill=NEU_F, stroke=NEU, size=17)
        if i < 2:
            s.line(x + bw - 6, by + bh / 2, xs[i + 1] - 1, by + bh / 2, INK, 2, "arr")
    # dispatch arrow
    s.line(tx + tw, ty + th / 2, rx0 - 1, ry + 32, INK, 2.2, "arr")
    # combine arrow
    s.line(rx1, ry + 32, 455, ty + th / 2, INK, 2.2, "arr")

# sum box
s.box(456, ty, 62, th, ["Σ", "y"], fill=NEU_F, stroke=NEU, sizes=[24, 18], weights=["bold", "normal"])

# stage labels along the bottom
s.text(110, 318, "dispatch", 18, INK, weight="bold")
s.text(110, 334, "NVLink", 15, MUTED, style="italic")
s.text(287, 318, "per-expert MLP", 18, INK, weight="bold")
s.text(287, 334, "tensor cores", 15, MUTED, style="italic")
s.text(462, 318, "combine", 18, INK, weight="bold")
s.text(462, 334, "NVLink", 15, MUTED, style="italic")

s.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_layer.svg"))
