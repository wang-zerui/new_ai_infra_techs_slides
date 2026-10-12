"""fig_overview: loss-free routing. Sigmoid scores s; selection = Top-K(s + b); gate weights from s only;
token counts feed a sign update of b after each step (no gradient)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 292
s = SVG(W, H, "fig_overview: biased selection, unbiased weights, feedback bias update")


def badge(x, y, n):
    s.add(f'<circle cx="{x}" cy="{y}" r="12" fill="{INK}"/>')
    s.text(x, y + 6, str(n), 16, "#FFFFFF", weight="bold")


def op(x, y, sym):
    s.add(f'<circle cx="{x}" cy="{y}" r="14" fill="{ACC_F if sym == "+" else "#FFFFFF"}" '
          f'stroke="{ACC if sym == "+" else NEU}" stroke-width="2.2"/>')
    s.text(x, y + 7, sym, 22, ACC if sym == "+" else INK, weight="bold")


YA, YM, YB = 119, 181, 244          # row centres: selection, score chain, weighting
# score chain
s.box(10, YM - 23, 120, 46, "Router", NEU_F, NEU)
s.line(132, YM, 158, YM, INK, 2.2, "arr")
s.box(160, YM - 23, 120, 46, "Sigmoid", ACC_F, ACC, color=ACC, weight="bold")
badge(160, YM - 23, 3)
s.text(292, YM - 8, "s", 19, INK, style="italic", anchor="start")
s.path(f"M 282 {YM} L 305 {YM} M 305 {YA} L 305 {YB}", INK, 2.2)
s.line(305, YA, 329, YA, INK, 2.2, "arr")
s.line(305, YB, 388, YB, INK, 2.2, "arr")
# selection row
op(345, YA, "+")
badge(322, YA - 26, 1)
s.line(361, YA, 388, YA, INK, 2.2, "arr")
s.box(390, YA - 23, 110, 46, "Top-K", NEU_F, NEU)
s.line(502, YA, 638, YA, INK, 2.2, "arr")
s.text(560, YA - 9, "ids", 17, MUTED, style="italic")
s.box(640, YA - 23, 120, 46, "Experts", NEU_F, NEU)
# weighting row
s.line(445, YA + 25, 445, YB - 30, INK, 2.2, "arr")
s.text(453, 188, "ids", 17, MUTED, style="italic", anchor="start")
s.box(390, YB - 28, 170, 56, "Weights from s", "#FFFFFF", NEU)
s.line(700, YA + 25, 700, YB - 16, INK, 2.2, "arr")
s.line(562, YB, 684, YB, INK, 2.2, "arr")
op(700, YB, "×")
s.line(716, YB, 748, YB, INK, 2.2, "arr")
s.box(750, YB - 23, 110, 46, "Output", "#FFFFFF", NEU)
# feedback loop (after each step, no gradient)
FY = 36
s.rect(280, 6, 512, 60, "#FFFFFF", ACC, 1.6, 6, dash="6,4")
s.box(292, FY - 21, 100, 42, "Bias b", ACC_F, ACC, color=ACC, weight="bold")
s.box(430, FY - 21, 140, 42, "Sign update", ACC_F, ACC, color=ACC, weight="bold")
badge(430, FY - 21, 2)
s.box(604, FY - 21, 176, 42, "Load counter c", ACC_F, ACC, color=ACC, weight="bold")
s.add(f'<circle cx="620" cy="{YA}" r="4" fill="{INK}"/>')
s.line(620, YA - 4, 620, FY + 23, ACC, 2.2, "arrAcc")
s.line(602, FY, 572, FY, ACC, 2.2, "arrAcc")
s.line(428, FY, 394, FY, ACC, 2.2, "arrAcc")
s.line(345, FY + 23, 345, YA - 16, ACC, 2.2, "arrAcc")
s.text(140, 30, "after each step", 17, ACC, style="italic")
s.text(140, 52, "no gradient", 17, ACC, style="italic")
s.save(os.path.join(HERE, "fig_overview.svg"))
