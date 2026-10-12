"""fig_serial / fig_perstage / fig_overlap: one MoE layer on one EP rank, three schedules.

Schematic durations (see notes/sources.md, "Derived and analytic content" item 2): compute = 100 units,
D 30.7, L1 52.2, act 21.7, L2 26.1, C 61.3; these reproduce the paper's theoretical ratios
serial : per-stage : ideal waves = 1.92 : 1.352 : 1 (DeepSeek-V4 Fig. 5). The waves panel draws 4 waves with
double-buffered dispatch, so fill and drain stay visible. Not measurements.
"""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W = 870
X0 = 206          # timeline origin
U = 3.43          # pt per unit (192 units -> 659 pt)
LH, LG = 24, 4    # lane height, lane gap
PANEL_GAP = 16
FS = 16           # in-box label size

D, L1, A, L2, C = 30.7, 52.2, 21.7, 26.1, 61.3
NW = 4


def lane_y(top, lane):
    return top + {"d": 0, "c": 1, "m": 2}[lane] * (LH + LG)


def box(s, top, lane, t0, t1, label="", kind="comm", size=FS):
    y = lane_y(top, lane)
    x, w = X0 + t0 * U, (t1 - t0) * U
    if kind == "comm":
        s.box(x, y, w, LH, label, "#FFFFFF", NEU, INK, size, sw=1.8, rx=3)
    elif kind == "comp":
        s.box(x, y, w, LH, label, NEU_F, NEU, INK, size, sw=1.8, rx=3)
    elif kind == "wave":
        s.box(x, y, w, LH, label, ACC_F, ACC, ACC, size, weight="bold", sw=1.8, rx=3)
    elif kind == "idle":
        s.rect(x, y, w, LH, "url(#hatchPrb)", PRB, 1.8, 3)
        if label:
            s.text(x + w / 2, y + LH / 2 + 0.35 * size, label, size, PRB, weight="bold", halo=True)


def lanes(s, top, name_lines, accent=False):
    col = ACC if accent else INK
    for i, t in enumerate(name_lines):
        s.text(4, top + 20 + i * 21, t, 18, col, anchor="start", weight="bold")
    for lane, t in (("d", "dispatch"), ("c", "compute"), ("m", "combine")):
        s.text(X0 - 8, lane_y(top, lane) + LH / 2 + 5.3, t, 15, MUTED, anchor="end")


def panel_serial(s, top):
    lanes(s, top, ["(a) Serial"])
    t = 0
    box(s, top, "d", t, t + D, "FP8 tokens"); box(s, top, "c", t, t + D, "idle", "idle"); t += D
    box(s, top, "c", t, t + L1, "Linear-1", "comp"); t += L1
    box(s, top, "c", t, t + A, "SwiGLU", "comp"); t += A
    box(s, top, "c", t, t + L2, "Linear-2", "comp"); t += L2
    box(s, top, "m", t, t + C, "BF16 results"); box(s, top, "c", t, t + C, "idle", "idle")


def panel_perstage(s, top):
    lanes(s, top, ["(b) Per-stage", "     overlap"])
    box(s, top, "d", 0, D, "FP8 tokens")
    box(s, top, "c", 0, L1, "Linear-1", "comp")
    t = L1
    box(s, top, "c", t, t + A, "SwiGLU", "comp"); t += A
    box(s, top, "c", t, t + L2, "Linear-2", "comp")
    box(s, top, "m", t, t + C, "BF16 results")
    box(s, top, "c", t + L2, t + C, "idle", "idle")


def panel_waves(s, top):
    lanes(s, top, ["(c) Expert", "     waves"], accent=True)
    dw, cw, mw = D / NW, L1 / NW + A / NW + L2 / NW, C / NW
    box(s, top, "c", 0, dw, "", "idle")
    for k in range(NW):
        c0 = dw + k * cw
        # dispatch of wave k+1 is pulled while wave k computes (wave 1 before anything else)
        d0 = 0 if k == 0 else c0 - cw
        box(s, top, "d", d0, d0 + dw, f"W{k + 1}", "wave", 15)
        box(s, top, "c", c0, c0 + cw, f"W{k + 1}", "wave")
        box(s, top, "m", c0 + cw, c0 + cw + mw, f"W{k + 1}", "wave", 15)


def legend(s, with_wave):
    y = 4
    items = [("comm", "transfer"), ("comp", "compute"), ("idle", "tensor cores idle")]
    if with_wave:
        items.append(("wave", "expert wave"))
    x = X0
    for kind, label in items:
        if kind == "comm":
            s.rect(x, y, 26, 18, "#FFFFFF", NEU, 1.8, 3)
        elif kind == "comp":
            s.rect(x, y, 26, 18, NEU_F, NEU, 1.8, 3)
        elif kind == "idle":
            s.rect(x, y, 26, 18, "url(#hatchPrb)", PRB, 1.8, 3)
        else:
            s.rect(x, y, 26, 18, ACC_F, ACC, 1.8, 3)
        s.text(x + 34, y + 15, label, 16, INK, anchor="start")
        x += 34 + len(label) * 8.6 + 26


def axis(s, y):
    s.line(X0 - 2, y, W - 4, y, INK, 1.8, "arr")
    s.text(W - 6, y + 20, "time (one MoE layer, one rank)", 15, MUTED, anchor="end", style="italic")


def build(name, panels, with_wave=False):
    ph = 3 * LH + 2 * LG
    top0 = 36
    h = top0 + len(panels) * ph + (len(panels) - 1) * PANEL_GAP + 34
    s = SVG(W, h, f"{name}: schematic MoE-layer timelines")
    legend(s, with_wave)
    for i, p in enumerate(panels):
        p(s, top0 + i * (ph + PANEL_GAP))
    axis(s, top0 + len(panels) * ph + (len(panels) - 1) * PANEL_GAP + 10)
    s.save(os.path.join(HERE, name + ".svg"))


build("fig_serial", [panel_serial])
build("fig_perstage", [panel_serial, panel_perstage])
build("fig_overlap", [panel_serial, panel_perstage, panel_waves], with_wave=True)
