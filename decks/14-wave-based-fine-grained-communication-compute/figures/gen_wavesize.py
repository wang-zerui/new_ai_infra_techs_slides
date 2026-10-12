"""fig_wavesize: why the wave size matters (DeepGEMM @7f2a703 heuristics get_num_experts_per_wave_for_mega_moe).

Schematic: 16 SMs, one Linear-1 tile per cell; real GPUs have far more SMs and the heuristic sizes waves so
the wave's Linear-1 tiles cover 2x the SM count (imbalance slack), capped at 32 experts per wave.
"""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 224
s = SVG(W, H, "fig_wavesize: too small / right-sized / too large waves")

CW, CH, G = 56, 26, 6
PX = [14, 304, 594]


def grid(px, kind):
    for r in range(4):
        for c in range(4):
            x = px + c * (CW + G)
            y = 36 + r * (CH + G)
            i = r * 4 + c
            if kind == "small":
                if i < 3:
                    s.box(x, y, CW, CH, "E0", ACC_F, ACC, ACC, 15, sw=1.5, rx=2)
                else:
                    s.rect(x, y, CW, CH, "url(#hatchPrb)", PRB, 1.5, 2)
            elif kind == "right":
                s.box(x, y, CW, CH, f"E{i // 2}", ACC_F, ACC, ACC, 15, sw=1.5, rx=2)
            else:
                s.rect(x, y, CW, CH, "#FFFFFF", "#A6A6A6", 1.5, 2, dash="4,3")


def panel(px, title, kind, status, color, bold=False):
    s.text(px + 121, 22, title, 18, ACC if kind == "right" else INK, weight="bold")
    grid(px, kind)
    for i, t in enumerate(status):
        s.text(px + 121, 186 + i * 22, t, 17, color, weight="bold" if bold else "normal")


panel(PX[0], "1 expert per wave", "small", ["too few tiles", "SMs idle"], PRB, True)
panel(PX[1], "8 experts per wave", "right", ["smallest wave that", "fills every SM"], ACC, True)
panel(PX[2], "all 32 experts in one wave", "large", ["first GEMM waits for", "every expert's tokens"], PRB, True)
# hourglass-like marker inside the empty grid
s.rect(PX[2] + 66, 36 + 2 * (CH + G) - 21, 110, 30, "#FFFFFF", "#FFFFFF", 0, 6)
s.text(PX[2] + 121, 36 + 2 * (CH + G) - 0, "waiting", 18, MUTED, style="italic")
for x in (294, 584):
    s.line(x, 8, x, 214, "#BFBFBF", 1.2)
s.save(os.path.join(HERE, "fig_wavesize.svg"))
