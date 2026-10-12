"""fig_glue_1 / fig_glue_2: forward of one DeepSeek-V3-style layer (MLA + MoE) as a strip of stages.
Build 1 (eager): GEMM/attention/communication stages in gray, glue stages as stacks of small red ops.
Build 2 (fused): each glue stack becomes one accent kernel with its mechanism badge.
Schematic: the number of small boxes per stack is illustrative, not an op count."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
stages = [
    (["QKV", "up-proj"], "gemm", 0, None),
    (["RoPE", "glue"], "glue", 6, 3),
    (["atten-", "tion"], "gemm", 0, None),
    (["router", "GEMM"], "gemm", 0, None),
    (["router", "glue"], "glue", 8, 1),
    (["permute"], "glue", 5, 2),
    (["dispatch"], "comm", 0, None),
    (["expert", "GEMMs"], "gemm", 0, None),
    (["combine"], "comm", 0, None),
    (["un-", "permute"], "glue", 5, 2),
]
W, G, X0, Y0, H = 78, 9.3, 2, 30, 110


def badge(cx, cy, n):
    s.add(f'<circle cx="{cx}" cy="{cy}" r="12" fill="{INK}"/>')
    s.text(cx, cy + 6, str(n), 16, "#FFFFFF", weight="bold")


def draw(fused):
    for i, (lab, kind, nops, mech) in enumerate(stages):
        x = X0 + i * (W + G)
        if kind == "gemm":
            s.rect(x, Y0, W, H, NEU_F, NEU, 2, 4)
        elif kind == "comm":
            s.rect(x, Y0 + 20, W, H - 40, "#FFFFFF", NEU, 2, 4, dash="5,3")
        elif not fused:
            gap = 3.0
            hh = (H - gap * (nops - 1)) / nops
            for k in range(nops):
                s.rect(x + 8, Y0 + k * (hh + gap), W - 16, hh, PRB_F, PRB, 1.6, 2)
        else:
            s.rect(x, Y0 + 26, W, H - 52, ACC_F, ACC, 2.6, 4)
            s.text(x + W / 2, Y0 + H / 2 + 6, "fused", 17, ACC, weight="bold")
            badge(x + W / 2, Y0 + 11, mech)
        for k, t in enumerate(lab):
            s.text(x + W / 2, Y0 + H + 24 + k * 19, t, 16, INK)
        if i < len(stages) - 1:
            pass
    s.line(X0, Y0 + H + 60, X0 + 10 * (W + G) - G, Y0 + H + 60, INK, 1.6, "arr")
    s.text(X0 + 10 * (W + G) - G, Y0 + H + 78, "forward of one MoE layer", 15, MUTED, anchor="end", style="italic")
    # legend
    s.rect(2, 4, 16, 14, NEU_F, NEU, 1.6, 2)
    s.text(24, 16, "GEMM / attention", 15, INK, anchor="start")
    s.rect(170, 4, 16, 14, "#FFFFFF", NEU, 1.6, 2, dash="4,2")
    s.text(192, 16, "all-to-all", 15, INK, anchor="start")
    if fused:
        s.rect(290, 4, 16, 14, ACC_F, ACC, 2, 2)
        s.text(312, 16, "fused glue kernel", 15, INK, anchor="start")
    else:
        s.rect(290, 4, 16, 14, PRB_F, PRB, 1.6, 2)
        s.text(312, 16, "eager glue op", 15, INK, anchor="start")


s = SVG(870, 224, "fig_glue_1: eager glue between GEMMs")
draw(False)
s.save(os.path.join(HERE, "fig_glue_1.svg"))
s = SVG(870, 224, "fig_glue_2: fused glue kernels")
draw(True)
s.save(os.path.join(HERE, "fig_glue_2.svg"))
