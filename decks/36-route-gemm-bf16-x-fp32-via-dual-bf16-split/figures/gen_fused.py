"""fig_fused_1/2: naive two GEMMs + add (3 launches, x read twice, FP32 partials through HBM) vs the Route GEMM kernel
(warp-specialized: TMA producer streams x, W_lo, W_hi tiles; math warpgroup runs WGMMA lo then hi per K step into two
register accumulators; one epilogue store). Schematic, not to scale."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 312


def draw(step):
    s = SVG(W, H, f"fig_fused step {step}")
    # ---- top: naive
    s.text(8, 22, "Two GEMMs + add", 19, PRB, anchor="start", weight="bold")
    s.text(186, 54, "kernels", 17, INK, anchor="end")
    s.text(186, 116, "HBM", 17, INK, anchor="end")
    k = [(200, 170, "GEMM x · W_hi"), (390, 170, "GEMM x · W_lo"), (580, 120, "add")]
    for x, w, lab in k:
        s.box(x, 32, w, 38, lab, NEU_F, NEU, size=18)
    s.box(250, 98, 80, 30, "y_hi", PRB_F, PRB, color=PRB, size=17)
    s.box(440, 98, 80, 30, "y_lo", PRB_F, PRB, color=PRB, size=17)
    s.line(290, 72, 290, 96, PRB, 2, "arrPrb")
    s.line(480, 72, 480, 96, PRB, 2, "arrPrb")
    s.path("M290,130 L290,138 L650,138 L650,72", PRB, 2, "arrPrb", dash="5,3")
    s.path("M522,113 L610,113 L610,72", PRB, 2, "arrPrb", dash="5,3")
    s.text(712, 46, "x read twice", 17, PRB, anchor="start")
    s.text(712, 70, "3 launches", 17, PRB, anchor="start")
    s.text(712, 94, "FP32 partials", 17, PRB, anchor="start")
    s.text(712, 116, "through HBM", 17, PRB, anchor="start")
    if step >= 2:
        y0 = 164
        s.line(8, y0 - 8, 862, y0 - 8, "#BFBFBF", 1)
        s.text(8, y0 + 18, "Route GEMM: one kernel", 19, ACC, anchor="start", weight="bold")
        L1, L2, LH = y0 + 32, y0 + 84, 36
        s.text(186, L1 + 24, "TMA: x, W_lo, W_hi", 17, INK, anchor="end")
        s.text(186, L2 + 24, "math warpgroup", 17, INK, anchor="end")
        X, STEP = 200, 124
        for i in range(3):
            s.box(X + i * STEP, L1, STEP - 8, LH, f"k{i} tiles", "#FFFFFF", NEU, size=17)
        s.text(X + 3 * STEP + 10, L1 + 24, "…", 20, MUTED, anchor="start")
        for i in range(3):
            xs = X + (i + 1) * STEP
            hw = (STEP - 12) / 2
            s.box(xs, L2, hw, LH, "lo", ACC_F, ACC, color=ACC, size=17)
            s.box(xs + hw + 4, L2, hw, LH, "hi", ACC_F, ACC, color=ACC, size=17)
        ex = X + 4 * STEP + 4
        s.box(ex, L2, 862 - ex, LH, "epilogue + store", ACC, ACC, color="#FFFFFF", weight="bold", size=16)
        # same x tile feeds lo and hi of step 0
        s.line(X + STEP / 2 + 30, L1 + LH + 2, X + STEP + 28, L2 - 2, ACC, 1.8, "arrAcc")
        s.text(X + 4, L2 + 26, "SMEM ring", 16, MUTED, anchor="start", style="italic")
        s.line(200, H - 10, 856, H - 10, INK, 1.8, "arr")
        s.text(196, H - 4, "time", 16, INK, anchor="end")
    s.save(os.path.join(HERE, f"fig_fused_{step}.svg"))


draw(1)
draw(2)
