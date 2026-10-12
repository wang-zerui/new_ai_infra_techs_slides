"""Status quo: FP8 training and RL, FP4 only after training -> the served weights are not the sampled ones."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, GRAY, GRAY_FILL, INK, RED, RED_FILL

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 474, 284
s = SVG(W, H)
rows = [("Pre-training", "FP8 weights", GRAY_FILL, GRAY), ("RL rollouts", "FP8 weights", GRAY_FILL, GRAY),
        ("Quantize to FP4", "after training", RED_FILL, RED), ("Serving", "FP4 weights", GRAY_FILL, GRAY)]
y = 0
ys = []
for t1, t2, f, st in rows:
    s.rect(0, y, 300, 56, f, st, 2.5 if st == RED else 2, 6)
    s.text(150, y + 24, t1, 18, "bold", RED if st == RED else INK)
    s.text(150, y + 46, t2, 18, "normal", INK)
    ys.append(y)
    if y > 0:
        s.arrow(150, y - 18, 150, y - 2)
    y += 76
s.path(f"M306,{ys[1] + 28} H336 V{ys[3] + 28} H306", RED, 2.5, head=False)
s.text(348, ys[2] + 22, "different", 18, "bold", RED, "start")
s.text(348, ys[2] + 46, "weights", 18, "bold", RED, "start")
s.save(os.path.join(HERE, "fig_mismatch.svg"))
