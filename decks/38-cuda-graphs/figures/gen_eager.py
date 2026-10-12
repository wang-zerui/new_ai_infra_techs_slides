"""fig_eager: eager execution (per-op Python/framework work + one launch per kernel, GPU idles between
short kernels) vs CUDA graph replay (one graph launch, kernels back to back). Schematic, not to scale.
Concept from arXiv 2603.07685 Sec. 4.3.6 Fig. 22 (redrawn)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 262, "fig_eager: eager launches vs graph replay")

X0, OP, FW, LN, KW = 64, 64, 46, 18, 26
H = 26


def lanes(y_title, y_cpu, y_gpu, title, color):
    s.text(4, y_title, title, 18, color, anchor="start", weight="bold")
    s.text(56, y_cpu + H / 2 + 6, "CPU", 17, INK, anchor="end", weight="bold")
    s.text(56, y_gpu + H / 2 + 6, "GPU", 17, INK, anchor="end", weight="bold")


# ---- eager
lanes(18, 30, 66, "eager", PRB)
for i in range(6):
    x = X0 + i * OP
    s.rect(x, 30, FW, H, NEU_F, NEU, 1.4, 2)
    s.rect(x + FW, 30, LN, H, "#595959", "#595959", 1.4, 2)
    kx = x + FW + LN
    s.box(kx, 66, KW, H, f"K{i + 1}", ACC_F, ACC, size=15)
    if i < 5:
        s.rect(kx + KW, 66, OP - KW, H, "url(#hatchPrb)", PRB, 1.4, 2)
s.rect(286, 6, 14, 14, NEU_F, NEU, 1.4, 2)
s.text(305, 18, "framework", 15, INK, anchor="start")
s.rect(390, 6, 14, 14, "#595959", "#595959", 1.4, 2)
s.text(409, 18, "launch", 15, INK, anchor="start")
s.text(X0 + 2 * OP + FW + LN + KW + (OP - KW) / 2, 112, "GPU idle", 16, PRB, weight="bold")

# ---- graph replay
lanes(146, 158, 194, "graph replay", ACC)
s.box(X0, 158, 70, H, "launch", "#595959", "#595959", color="#FFFFFF", size=15)
gx = X0 + 70
for i in range(6):
    s.box(gx + i * KW, 194, KW, H, f"K{i + 1}", ACC_F, ACC, size=15)
end_g = gx + 6 * KW
end_e = X0 + 5 * OP + FW + LN + KW
s.line(end_g + 2, 207, end_e - 2, 207, ACC, 2.2, "arrAcc", start_marker="arrAcc")
s.text((end_g + end_e) / 2, 199, "time saved", 16, ACC, weight="bold")
# time axis
s.line(X0, 244, 474, 244, INK, 1.8, "arr")
s.text(470, 260, "time", 15, INK, anchor="end")
s.save(os.path.join(HERE, "fig_eager.svg"))
