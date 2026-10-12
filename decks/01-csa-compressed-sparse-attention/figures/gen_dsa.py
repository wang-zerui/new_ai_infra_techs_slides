"""Status-quo figure: DSA (V3.2) keeps one KV entry per token and its indexer scores all of them."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, RED, RED_FILL, GRAY, GRAY_FILL, INK

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 480, 286
s = SVG(W, H)

# core attention and top-k (neutral, pre-existing)
s.box(40, 6, 300, 44, "Core attention (MQA)", GRAY_FILL, GRAY, 2, 18, "bold")
s.box(40, 82, 300, 44, "Top-2,048 selector", GRAY_FILL, GRAY, 2, 18, "bold")
s.arrow(190, 82, 190, 52)
s.box(10, 158, 364, 44, "Lightning indexer", GRAY_FILL, GRAY, 2, 18, "bold")
s.arrow(190, 158, 190, 128)

# token-level KV cache: one entry per token (problem)
n, cw, gap, x0, y0 = 9, 28, 6, 10, 234
for i in range(n):
    x = x0 + i * (cw + gap)
    s.rect(x, y0, cw, 26, RED_FILL, RED, 2, 3)
    s.arrow(x + cw / 2, y0 - 2, x + cw / 2, 204, RED, 2)
s.text(x0 + n * (cw + gap) + 14, y0 + 20, "…", 22, "bold", RED)
s.rect(x0 + n * (cw + gap) + 34, y0, cw, 26, RED_FILL, RED, 2, 3)
s.arrow(x0 + n * (cw + gap) + 34 + cw / 2, y0 - 2, x0 + n * (cw + gap) + 34 + cw / 2, 204, RED, 2)
s.text(10, 282, "one cached entry per token", 18, "normal", RED, "start")

# query token
s.box(394, 158, 80, 44, "query", "#fff", INK, 2, 18, "bold")
s.arrow(394, 180, 376, 180)
s.path("M434,158 V28 H342")
s.text(400, 228, "scores", 18, "bold", RED, "start")
s.text(400, 250, "all L", 18, "bold", RED, "start")
s.save(os.path.join(HERE, "fig_dsa.svg"))
