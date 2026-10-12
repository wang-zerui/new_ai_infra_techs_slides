"""Slide 7 (overview): DSA instantiated under MLA (V3.2 Fig. 2, redrawn). New parts in the accent fill."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 286
s = SVG(W, H, "DSA under MLA")


def two(x, y, w, h, l1, l2, new=True):
    s.rect(x, y, w, h, ACC_F if new else NEU_F, ACC if new else NEU, 2.5 if new else 2, 6)
    s.text(x + w / 2, y + h / 2 - 4, l1, 19, INK, weight="bold")
    s.text(x + w / 2, y + h / 2 + 19, l2, 17, INK)


# hidden state bar
s.rect(0, 32, 54, 250, NEU_F, NEU, 2, 6)
s.add(f'<text x="33" y="157" text-anchor="middle" font-size="19" font-weight="bold" fill="{INK}" '
      f'transform="rotate(-90 33 157)">hidden state hₜ</text>')
# legend
s.rect(626, 6, 22, 16, ACC_F, ACC, 2, 3); s.text(654, 20, "new in DSA", 17, INK, anchor="start")
s.rect(752, 6, 22, 16, NEU_F, NEU, 2, 3); s.text(780, 20, "MLA", 17, INK, anchor="start")

# indexer lane (new)
two(98, 32, 220, 54, "Indexer queries", "64 heads × 128")
two(98, 96, 220, 54, "Head weights", "64 per token")
s.rect(366, 32, 196, 108, ACC_F, ACC, 2.5, 6)
s.text(464, 82, "Lightning", 20, INK, weight="bold")
s.text(464, 106, "indexer", 20, INK, weight="bold")
s.badge(386, 52, 1)
two(366, 162, 196, 54, "Indexer key cache", "128-d FP8 / token")
two(612, 59, 206, 54, "Top-k selector", "k = 2,048")
s.arrow(56, 59, 95, 59)
s.arrow(56, 123, 95, 123)
s.arrow(320, 59, 363, 59)
s.arrow(320, 123, 363, 123)
s.arrow(464, 162, 464, 143)
s.arrow(564, 86, 609, 86)

# MLA lane (pre-existing)
two(98, 228, 220, 54, "Main queries", "128 heads")
two(366, 228, 196, 54, "Core attention", "MQA over k latents", new=False)
two(612, 228, 206, 54, "MLA latent cache", "576-d / token", new=False)
s.rect(98, 228, 220, 54, NEU_F, NEU, 2, 6)
s.text(208, 251, "Main queries", 19, INK, weight="bold"); s.text(208, 274, "128 heads", 17, INK)
s.arrow(56, 255, 95, 255)
s.arrow(320, 255, 363, 255)
s.arrow(610, 255, 565, 255)
# top-k indices gather from the latent cache
s.arrow(715, 115, 715, 225, ACC, 2.6)
s.badge(692, 172, 2)
s.text(724, 168, "gather", 18, ACC, anchor="start", weight="bold")
s.text(724, 190, "2,048 ids", 18, ACC, anchor="start", weight="bold")
s.save(os.path.join(HERE, "fig_overview.svg"))
