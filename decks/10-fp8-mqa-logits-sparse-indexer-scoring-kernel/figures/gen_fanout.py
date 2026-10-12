"""Slide 5 (insight): one shared indexer key feeds every (token, head) score of a Q tile (H = 64: 2 tokens)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 452, 280
s = SVG(W, H, "one key feeds 128 scores")
s.rect(0, 96, 120, 70, ACC_F, ACC, 2.6, 6)
s.text(60, 126, "key k", 19, INK, weight="bold")
s.text(60, 150, "128 B FP8", 16, INK)
cx, cw, ch = 250, 86, 22
groups = [("token t", 10), ("token t+1", 140)]
for lab, gy in groups:
    names = ["h0", "h1", None, "h63"]
    y = gy
    for nm in names:
        if nm is None:
            s.text(cx + cw / 2, y + 14, "⋮", 20, MUTED, weight="bold")
            y += 22
            continue
        s.rect(cx, y, cw, ch, "#FFFFFF", ACC, 1.6, 3)
        s.text(cx + cw / 2, y + 16, nm, 15, INK)
        s.line(122, 131, cx - 2, y + ch / 2, ACC, 1.3, "arrAcc")
        y += ch + 6
    top, bot = gy, y - 6
    s.path(f"M{cx + cw + 8},{top} H{cx + cw + 14} V{bot} H{cx + cw + 8}", INK, 1.6)
    s.text(cx + cw + 22, (top + bot) / 2 + 6, lab, 17, INK, anchor="start", weight="bold")
s.text(226, 274, "64 heads × 2 tokens = 128 scores per key read", 16, ACC, weight="bold")
s.save(os.path.join(HERE, "fig_fanout.svg"))
