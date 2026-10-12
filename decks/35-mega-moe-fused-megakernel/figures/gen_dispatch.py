"""fig_dispatch_{1,2}: symmetric buffer on two GPUs (same layout) and the pull-based dispatch."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED

W, H = 870, 292
HERE = os.path.dirname(os.path.abspath(__file__))

REG = [("counters", 150, 240), ("inputs: x, top-k", 248, 404), ("L1 ring", 412, 548),
       ("L2 ring", 556, 664), ("combine slots", 672, 862)]
RS, RR = 44, 190          # row tops (GPU s, GPU r)
RH = 58


def regions(s, top, hl_cell=None):
    for lab, x0, x1 in REG:
        s.rect(x0, top, x1 - x0, RH, NEU_F, NEU, 2, 4)
        s.text((x0 + x1) / 2, top + 22, lab, 17, INK)
        if lab.startswith("inputs") or lab == "L1 ring":
            n = 6
            cw = (x1 - x0 - 24) / n
            for i in range(n):
                cx = x0 + 12 + i * cw
                hl = hl_cell is not None and hl_cell == (lab, i)
                s.rect(cx + 1, top + 32, cw - 2, 16, ACC if hl else "#FFFFFF", ACC if hl else MUTED, 1.2, 1)


def build(step):
    s = SVG(W, H, f"pull dispatch, step {step}")
    s.text(506, 22, "symmetric buffer: same layout and offsets on every GPU", 17, ACC, weight="bold")
    s.text(132, RS + 26, "GPU s", 18, INK, anchor="end", weight="bold")
    s.text(132, RS + 46, "token's rank", 15, MUTED, anchor="end", style="italic")
    s.text(132, RR + 26, "GPU r", 18, INK, anchor="end", weight="bold")
    s.text(132, RR + 46, "expert's rank", 15, MUTED, anchor="end", style="italic")
    regions(s, RS, ("inputs: x, top-k", 2) if step >= 2 else None)
    regions(s, RR, ("L1 ring", 1) if step >= 2 else None)
    # address mapping
    s.box(606, 120, 256, 48, ["peer addr = own addr", "+ offset[peer]"], fill="#FFFFFF", stroke=ACC, color=ACC,
          sizes=[16, 16], family="Menlo, Courier New, monospace", sw=1.6)
    if step >= 2:
        # (1) source writes its token index + per-expert count into r's counters
        s.line(195, RS + RH, 195, RR - 1, ACC, 2.4, "arrAcc")
        s.step(195, 147, 1, color=ACC)
        s.text(214, 152, "store index", 16, ACC, anchor="start")
        # (2) r's dispatch warp pulls the FP8 row from s (TMA load from a peer address)
        x_src = 248 + 12 + 2 * (404 - 248 - 24) / 6 + 10
        x_dst = 412 + 12 + 1 * (548 - 412 - 24) / 6 + 9
        s.line(x_src, RS + RH, x_dst, RR - 1, ACC, 2.8, "arrAcc")
        s.step(382, 132, 2, color=ACC)
        s.text(400, 137, "pull FP8 row", 16, ACC, anchor="start")
        # (3) arrival bumps the M-block's counter
        s.path(f"M {x_dst} {RR + RH} Q {x_dst - 120} {RR + RH + 34} 200 {RR + RH + 1}", ACC, 2.2, "arrAcc")
        s.step(330, 274, 3, color=ACC)
        s.text(350, 280, "block counter + 1", 16, ACC, anchor="start")
    s.save(os.path.join(HERE, f"fig_dispatch_{step}.svg"))


for k in (1, 2):
    build(k)
