"""fig_overview: MagiAttention = host-side planning (dispatch solver -> attn solver -> overlap solver) +
GPU runtime per attention layer and overlap stage (GroupCast KV -> FFA -> GroupReduce dKV)
([BLOG] Fig. 1, Static Attn Solver: CalcMeta / CommMeta; [QS] API)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 290, "fig_overview: plan once per micro-batch, run every layer")


def badge(x, y, n):
    s.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="{INK}"/>')
    s.text(x, y + 6, str(n), 16, "#FFFFFF", weight="bold")


def comp(x, y, w, h, title, sub, n=None, fill=ACC_F, stroke=ACC):
    s.rect(x, y, w, h, fill, stroke, 2.2, 5)
    tx = x + w / 2 + (10 if n else 0)
    s.text(tx, y + h / 2 - 4, title, 19, ACC if fill == ACC_F else INK, weight="bold")
    s.text(tx, y + h / 2 + 20, sub, 17, INK)
    if n:
        badge(x + 20, y + h / 2 - 10, n)


# ---- row 1: plan
s.text(10, 22, "Plan: once per micro-batch, on the host", 18, MUTED, anchor="start", weight="bold")
Y1, H1 = 36, 76
comp(10, Y1, 150, H1, "Mask", "AttnSlice list", fill="#FFFFFF", stroke=NEU)
comp(196, Y1, 200, H1, "Dispatch solver", "balance chunks", 2)
comp(432, Y1, 196, H1, "Attn solver", "CalcMeta, CommMeta")
comp(664, Y1, 196, H1, "Overlap solver", "split into stages", 4)
for xa, xb in ((160, 196), (396, 432), (628, 664)):
    s.line(xa + 3, Y1 + H1 / 2, xb - 3, Y1 + H1 / 2, INK, 2.4, "arr")

# ---- row 2: run
s.rect(10, 160, 850, 124, "#FFFFFF", MUTED, 1.6, 6, dash="6,4")
s.text(22, 182, "Run: every attention layer, each GPU, stage by stage", 18, MUTED, anchor="start",
       weight="bold")
Y2, H2 = 196, 76
comp(40, Y2, 230, H2, "GroupCast", "fetch remote KV", 3)
comp(320, Y2, 230, H2, "FFA kernel", "partial O and LSE", 1)
comp(600, Y2, 240, H2, "GroupReduce", "return partial dKV", 3)
s.line(273, Y2 + H2 / 2, 317, Y2 + H2 / 2, INK, 2.4, "arr")
s.line(553, Y2 + H2 / 2, 597, Y2 + H2 / 2, INK, 2.4, "arr")
s.text(575, Y2 + H2 / 2 - 10, "bwd", 16, MUTED, style="italic")
# plan feeds the runtime
s.line(762, Y1 + H1 + 3, 762, 157, ACC, 2.4, "arrAcc")
s.text(752, 142, "per-stage args", 17, ACC, anchor="end", weight="bold")
s.save(os.path.join(HERE, "fig_overview.svg"))
