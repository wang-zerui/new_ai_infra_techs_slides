"""Deck 22 figures (slide scale, 1 unit = 1 pt). Run: python3 gen_figs.py, then render each SVG."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda n: os.path.join(HERE, n)
DARK = "#404040"


def badge(s, x, y, n, r=13, fill=ACC):
    s.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"/>')
    s.text(x, y + 6.3, str(n), 18, "#FFFFFF", weight="bold")


def hatch_rect(s, x, y, w, h, pattern="hatchPrb", stroke=PRB):
    s.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="url(#{pattern})" '
          f'stroke="{stroke}" stroke-width="1.8"/>')


# ------------------------------------------------------------------------------------------ fig_paths
def fig_paths():
    s = SVG(480, 330, "fig_paths: normal = same rail then NVLink; low-latency = direct, cross rail via spine")
    s.box(190, 8, 100, 36, "spine", NEU_F, NEU, size=17)
    s.box(100, 128, 110, 38, "leaf · rail 0", "#FFFFFF", NEU, size=16)
    s.box(270, 128, 110, 38, "leaf · rail 2", "#FFFFFF", NEU, size=16)
    s.rect(6, 222, 150, 100, "#FFFFFF", NEU, 1.6, rx=6)
    s.text(81, 312, "node A", 16, MUTED, weight="bold")
    s.box(24, 238, 114, 40, "GPU0", NEU_F, NEU, size=17)
    s.rect(212, 222, 262, 100, "#FFFFFF", NEU, 1.6, rx=6)
    s.text(343, 312, "node B", 16, MUTED, weight="bold")
    s.box(222, 238, 92, 40, "GPU0", NEU_F, NEU, size=17)
    s.box(372, 238, 92, 40, "GPU2", NEU_F, NEU, size=17)
    # low-latency path (dark dashed): A.GPU0 -> leaf0 -> spine -> leaf2 -> B.GPU2
    s.path("M 74,238 L 132,168", DARK, 2.4, dash="7,5")
    s.path("M 150,128 L 214,46", DARK, 2.4, dash="7,5")
    s.path("M 266,46 L 330,126", DARK, 2.4, "arr", dash="7,5")
    s.path("M 340,168 L 410,236", DARK, 2.4, "arr", dash="7,5")
    s.text(334, 80, "low-latency", 17, DARK, anchor="start", weight="bold")
    # normal path (accent): A.GPU0 -> leaf0 -> B.GPU0 -> NVLink -> B.GPU2
    s.path("M 88,238 L 146,168", ACC, 3.2)
    s.path("M 176,168 L 262,236", ACC, 3.2, "arrAcc")
    s.line(316, 258, 370, 258, ACC, 3.2, "arrAcc")
    s.text(343, 250, "NVLink", 15, ACC, style="italic")
    s.text(222, 186, "normal", 17, ACC, anchor="start", weight="bold")
    s.save(P("fig_paths.svg"))


# ------------------------------------------------------------------------------------------ fig_net_before
def fig_net_before():
    s = SVG(870, 270, "fig_net_before: (a) one shared VL -> HOL blocking; (b) static hash -> two flows on one uplink")
    # (a) one lane
    s.box(4, 34, 140, 40, "DP all-reduce", NEU_F, NEU, size=17)
    s.box(4, 116, 140, 40, "EP all-to-all", NEU_F, NEU, size=17)
    s.line(146, 54, 172, 80, INK, 2, "arr")
    s.line(146, 136, 172, 104, INK, 2, "arr")
    s.text(278, 50, "one switch queue", 17, MUTED, style="italic")
    s.rect(174, 62, 214, 56, "#FFFFFF", NEU, 2, rx=3)
    for i, x in enumerate([312, 244]):
        s.rect(x, 70, 66, 40, "#D9D9D9", NEU, 1.5, rx=2)
        s.text(x + 33, 96, "DP", 16, INK)
    for x in [182, 204, 226]:
        s.rect(x, 78, 16, 24, PRB_F, PRB, 1.8, rx=2)
    s.line(390, 90, 420, 90, INK, 2.2, "arr")
    s.text(281, 146, "EP waits behind bulk", 17, PRB, weight="bold")
    s.text(210, 258, "(a) one lane for all traffic", 18, INK, weight="bold")
    s.add('<line x1="436" y1="6" x2="436" y2="262" stroke="#BFBFBF" stroke-width="1.2"/>')
    # (b) static hashing
    s.box(500, 20, 120, 36, "spine 1", NEU_F, NEU, size=17)
    s.box(720, 20, 120, 36, "spine 2", NEU_F, NEU, size=17)
    s.box(560, 156, 140, 40, "leaf", "#FFFFFF", NEU, size=17)
    s.line(600, 154, 566, 58, PRB, 4.5, "arrPrb")
    s.text(548, 116, "A + B", 18, PRB, anchor="end", weight="bold")
    s.line(660, 154, 770, 58, "#A6A6A6", 2.2, dash="6,5")
    s.text(744, 118, "idle", 17, MUTED, anchor="start", style="italic")
    s.line(604, 236, 604, 198, INK, 2, "arr")
    s.line(656, 236, 656, 198, INK, 2, "arr")
    s.text(596, 230, "EP flow A", 16, INK, anchor="end")
    s.text(664, 230, "EP flow B", 16, INK, anchor="start")
    s.text(652, 258, "(b) static hash routing", 18, INK, weight="bold")
    s.save(P("fig_net_before.svg"))


# ------------------------------------------------------------------------------------------ fig_sm (before / after)
def fig_sm(after: bool):
    name = "fig_sm_2" if after else "fig_sm_1"
    s = SVG(870, 250, f"{name}: comm SMs during an internode dispatch, {'hook mode' if after else 'ring buffer'}")
    lanes = [("comm SMs", 34), ("NIC", 104), ("compute SMs", 174)]
    for lab, y in lanes:
        s.text(152, y + 26, lab, 18, INK, anchor="end", weight="bold")
    H = 40
    if not after:
        segs = [("send", 160, 70, "n"), ("poll", 230, 100, "p"), ("fwd", 330, 70, "n"), ("poll", 400, 110, "p"),
                ("fwd", 510, 70, "n"), ("poll", 580, 110, "p"), ("recv", 690, 80, "n")]
        for lab, x, w, k in segs:
            if k == "p":
                hatch_rect(s, x, 34, w, H)
                s.text(x + w / 2, 60, lab, 17, PRB, weight="bold", halo=True)
            else:
                s.box(x, 34, w, H, lab, NEU_F, NEU, size=17)
        s.box(196, 104, 494, H, "RDMA transfer · small ring buffer", "#FFFFFF", NEU, size=17)
        s.box(160, 174, 700, H, "GEMM on the remaining SMs", NEU_F, NEU, size=17)
    else:
        s.box(160, 34, 70, H, "send", ACC_F, ACC, color=ACC, size=17, weight="bold")
        s.box(232, 34, 456, H, "GEMM on freed SMs", "#FFFFFF", ACC, color=ACC, size=17, dash="6,4")
        s.box(690, 34, 80, H, "recv", ACC_F, ACC, color=ACC, size=17, weight="bold")
        s.box(196, 104, 488, H, "RDMA transfer · large buffer", "#FFFFFF", NEU, size=17)
        s.box(160, 174, 700, H, "GEMM", NEU_F, NEU, size=17)
        s.text(195, 26, "send phase", 16, ACC, style="italic")
        s.text(730, 26, "recv hook", 16, ACC, style="italic")
    s.line(160, 232, 860, 232, INK, 2, "arr")
    s.text(856, 248, "time", 16, INK, anchor="end")
    s.save(P(name + ".svg"))


# ------------------------------------------------------------------------------------------ fig_overview
def fig_overview():
    s = SVG(870, 318, "fig_overview: where each fix lives: GPU (SM-free RDMA), NIC (SL tag), fabric (VL, AR)")
    layers = [("GPU", 6, 96), ("NIC", 112, 80), ("fabric", 202, 110)]
    for lab, y, h in layers:
        s.rect(96, y, 770, h, "#FAFAFA", "#BFBFBF", 1.2, rx=6)
        s.text(84, y + h / 2 + 7, lab, 19, INK, anchor="end", weight="bold")
    # GPU
    s.box(114, 26, 200, 56, "compute SMs", NEU_F, NEU)
    s.box(334, 26, 250, 56, ["send phase", "+ recv hook"], ACC_F, ACC, color=ACC, weight="bold")
    s.box(604, 26, 246, 56, ["GEMM-to-send", "signals (SBO)"], ACC_F, ACC, color=ACC, weight="bold")
    badge(s, 334, 26, 3)
    badge(s, 604, 26, 3)
    # NIC
    s.box(114, 128, 300, 48, "RDMA engine · QPs", NEU_F, NEU)
    s.box(434, 128, 416, 48, "service level per QP", ACC_F, ACC, color=ACC, weight="bold")
    badge(s, 434, 128, 1)
    # fabric
    s.box(114, 222, 330, 74, ["SL → VL map", "per-VL buffers + credits"], ACC_F, ACC, color=ACC, weight="bold")
    s.box(464, 222, 386, 74, ["adaptive routing", "rail-aligned LL paths"], ACC_F, ACC, color=ACC, weight="bold")
    badge(s, 114, 222, 1)
    badge(s, 464, 222, 2)
    s.save(P("fig_overview.svg"))


# ------------------------------------------------------------------------------------------ fig_vl
def fig_vl():
    s = SVG(870, 270, "fig_vl: SL tags map EP and other traffic to separate virtual lanes with own credits")
    s.box(4, 36, 176, 52, "EP QPs · SL 1", ACC_F, ACC, color=ACC, weight="bold")
    s.box(4, 152, 176, 60, ["other QPs · SL 0", "DP · PP · storage"], NEU_F, NEU, size=17)
    s.box(216, 92, 116, 64, ["SL → VL", "table"], ACC_F, ACC, color=ACC, weight="bold")
    s.line(182, 62, 214, 104, ACC, 2.4, "arrAcc")
    s.line(182, 182, 214, 146, INK, 2.2, "arr")
    # VL1 queue (EP)
    s.text(500, 30, "VL1 · EP", 18, ACC, weight="bold")
    s.rect(380, 40, 240, 46, "#FFFFFF", ACC, 2.2, rx=3)
    for x in [560, 588]:
        s.rect(x, 50, 22, 26, ACC_F, ACC, 1.8, rx=2)
    # VL0 queue (other), congested
    s.rect(380, 158, 240, 46, "#FFFFFF", NEU, 2.2, rx=3)
    for x in [386, 446, 506, 566]:
        s.rect(x, 164, 50, 34, "#D9D9D9", NEU, 1.5, rx=2)
    s.text(500, 230, "VL0 · other", 18, INK, weight="bold")
    s.text(500, 254, "congested · out of credits", 16, PRB, style="italic")
    s.text(500, 128, "own buffer + credits per VL", 16, MUTED, style="italic")
    s.line(334, 112, 378, 66, ACC, 2.4, "arrAcc")
    s.line(334, 140, 378, 178, INK, 2.2, "arr")
    # arbiter + link
    s.box(664, 96, 104, 60, ["VL", "arbiter"], NEU_F, NEU, size=17)
    s.line(622, 63, 662, 110, ACC, 2.4, "arrAcc")
    s.line(622, 181, 662, 144, "#A6A6A6", 2.2, "arrGray", dash="5,4")
    s.line(770, 126, 862, 126, INK, 2.6, "arr")
    s.text(816, 116, "link", 17, INK)
    s.save(P("fig_vl.svg"))


# ------------------------------------------------------------------------------------------ fig_ar
def fig_ar():
    s = SVG(870, 280, "fig_ar: (a) adaptive routing spreads packets across uplinks; (b) LL-Layered rail-aligned dispatch")
    # (a)
    s.box(24, 18, 130, 36, "spine 1", NEU_F, NEU, size=17)
    s.box(272, 18, 130, 36, "spine 2", NEU_F, NEU, size=17)
    s.box(148, 168, 130, 40, "leaf", "#FFFFFF", NEU, size=17)
    s.line(190, 166, 100, 56, ACC, 3, "arrAcc")
    s.line(236, 166, 326, 56, ACC, 3, "arrAcc")
    for (x, y, t) in [(122, 92, "A1"), (150, 126, "B2"), (282, 92, "B1"), (254, 126, "A2")]:
        s.rect(x - 17, y - 13, 34, 24, ACC_F, ACC, 1.6, rx=2)
        s.text(x, y + 5, t, 15, ACC, weight="bold")
    s.line(190, 246, 190, 210, INK, 2, "arr")
    s.line(236, 246, 236, 210, INK, 2, "arr")
    s.text(182, 240, "EP flow A", 16, INK, anchor="end")
    s.text(244, 240, "EP flow B", 16, INK, anchor="start")
    s.text(213, 272, "(a) adaptive routing per packet", 18, INK, weight="bold")
    s.add('<line x1="436" y1="6" x2="436" y2="272" stroke="#BFBFBF" stroke-width="1.2"/>')
    # (b) rail-aligned LL dispatch
    s.box(452, 30, 84, 40, "GPU0", NEU_F, NEU, size=17)
    s.text(494, 22, "node A", 15, MUTED, weight="bold")
    s.box(580, 30, 112, 40, "leaf · rail 0", "#FFFFFF", NEU, size=16)
    s.rect(740, 18, 128, 236, "#FFFFFF", NEU, 1.6, rx=6)
    s.text(792, 246, "node B", 15, MUTED, weight="bold")
    ys = [28, 78, 128, 178]
    for i, y in enumerate(ys):
        s.box(754, y, 76, 36, f"GPU{i}", NEU_F, NEU, size=17)
    # ghost: direct cross-rail copies via spine
    s.box(560, 176, 100, 36, "spine", "#FFFFFF", "#A6A6A6", color=MUTED, size=16, dash="5,4")
    s.path("M 494,72 L 494,194 L 558,194", "#A6A6A6", 1.8, dash="5,4")
    for y in ys[1:]:
        s.line(662, 194, 752, y + 18, "#A6A6A6", 1.8, "arrGray", dash="5,4")
    s.text(610, 236, "before: one copy per GPU", 15, MUTED, style="italic")
    # accent: one copy on the rail, then NVLink fan-out (bracket on the right of node B)
    s.line(538, 50, 578, 50, ACC, 3.2, "arrAcc")
    s.line(694, 48, 752, 46, ACC, 3.2, "arrAcc")
    s.path("M 832,46 L 846,46 L 846,196", ACC, 2.2)
    for y in ys[1:]:
        s.line(846, y + 18, 833, y + 18, ACC, 2.2, "arrAcc")
    s.add('<text x="863" y="122" text-anchor="middle" font-size="15" fill="%s" font-style="italic" '
          'transform="rotate(-90 863 122)">NVLink</text>' % ACC)
    s.text(652, 272, "(b) rail-aligned LL dispatch", 18, INK, weight="bold")
    s.save(P("fig_ar.svg"))


# ------------------------------------------------------------------------------------------ fig_sbo
def fig_sbo():
    s = SVG(870, 256, "fig_sbo: combine send waits for the whole Down GEMM (a) vs streams per block_m with signals (b)")
    H = 34

    def lanes(y0, tag):
        s.text(8, y0 - 8, tag, 18, INK, anchor="start", weight="bold")
        s.text(166, y0 + 23, "Down GEMM", 17, INK, anchor="end")
        s.text(166, y0 + 23 + 46, "combine send", 17, INK, anchor="end")

    lanes(28, "(a) before")
    for i in range(4):
        s.box(176 + i * 92, 28, 90, H, f"block {i + 1}", NEU_F, NEU, size=16)
    for i in range(4):
        s.box(546 + i * 72, 74, 70, H, f"send {i + 1}", "#FFFFFF", DARK, size=15)
    lanes(150, "(b) SBO")
    for i in range(4):
        s.box(176 + i * 92, 150, 90, H, f"block {i + 1}", NEU_F, NEU, size=16)
    for i in range(4):
        x = 268 + i * 92
        s.box(x, 196, 70, H, f"send {i + 1}", ACC_F, ACC, color=ACC, size=15)
        s.add(f'<circle cx="{x - 2}" cy="{150 + H + 1}" r="6" fill="{ACC}"/>')
        s.line(x - 2, 150 + H + 7, x - 2, 194, ACC, 1.6, dash="3,3")
    s.text(258, 248, "signal per block_m tokens", 15, ACC, anchor="start", style="italic")
    # end markers
    s.line(834, 20, 834, 112, DARK, 2, dash="6,4")
    s.line(616, 142, 616, 234, ACC, 2.4, dash="6,4")
    s.save(P("fig_sbo.svg"))


if __name__ == "__main__":
    fig_paths(); fig_net_before(); fig_sm(False); fig_sm(True); fig_overview(); fig_vl(); fig_ar(); fig_sbo()
