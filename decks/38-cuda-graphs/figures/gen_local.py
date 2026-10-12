"""fig_local: CudaGraphManager (cuda_graph_impl=local) for one layer on one pipeline stage.
Without PP microbatches run F,B,F,B... so one fwd/bwd graph pair is reused (L x 2 graphs). With PP the forward
of mb1 runs before the backward of mb0, so a shared graph would overwrite mb0's saved activations; each
microbatch gets its own pair (L x M x 2), captured in the order an eager pass recorded, into one mempool.
Sources: arXiv 2603.07685 Sec. 4.3.6 p.41-42 (Fig. 26); cuda_graphs.py (_CudagraphGlobalRecord,
CudaGraphManager.reuse_cudagraphs = pp.size() == 1)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 236, "fig_local: per-layer graphs with and without PP")

W, G, X0, H = 98, 10, 222, 50


def chip(x, y, top, bot, fwd, color=ACC):
    s.rect(x, y, W, H, ACC_F if fwd else "#FFFFFF", color, 2.2, 4)
    s.text(x + W / 2, y + 21, top, 16, INK, weight="bold")
    s.text(x + W / 2, y + 41, bot, 16, ACC)


def badge(cx, cy, n):
    s.add(f'<circle cx="{cx}" cy="{cy}" r="10" fill="{INK}"/>')
    s.text(cx, cy + 5.5, str(n), 14.5, "#FFFFFF", weight="bold")


# ---- no PP
YA = 16
s.text(4, YA + 20, "no PP", 18, INK, anchor="start", weight="bold")
s.text(4, YA + 42, "pair reused: L×2", 16, ACC, anchor="start")
seqA = [("F mb0", "Gf", 1), ("B mb0", "Gb", 0), ("F mb1", "Gf", 1), ("B mb1", "Gb", 0), ("F mb2", "Gf", 1),
        ("B mb2", "Gb", 0)]
for i, (t, g, f) in enumerate(seqA):
    chip(X0 + i * (W + G), YA, t, "graph " + g, f)

# ---- with PP
YB = 136
s.text(4, YB + 20, "with PP", 18, INK, anchor="start", weight="bold")
s.text(4, YB + 42, "pair per mb: L×M×2", 16, ACC, anchor="start")
seqB = [("F mb0", "Gf0", 1), ("F mb1", "Gf1", 1), ("B mb0", "Gb0", 0), ("F mb2", "Gf2", 1), ("B mb1", "Gb1", 0),
        ("B mb2", "Gb2", 0)]
for i, (t, g, f) in enumerate(seqB):
    x = X0 + i * (W + G)
    chip(x, YB, t, "graph " + g, f)
    badge(x + W - 2, YB - 2, i + 1)
s.text(X0, YB - 22, "badges: capture order = order recorded in an eager pass", 15, MUTED, anchor="start", style="italic")
# hazard: F mb1 runs before B mb0
x1 = X0 + 1 * (W + G) + W / 2
x2 = X0 + 2 * (W + G) + W / 2
s.path(f"M{x1},{YB + H + 4} Q{(x1 + x2) / 2},{YB + H + 34} {x2},{YB + H + 6}", PRB, 2.2, "arrPrb")
s.text((x1 + x2) / 2 + 70, YB + H + 40, "a shared graph would overwrite mb0 activations", 16, PRB, anchor="start")
s.save(os.path.join(HERE, "fig_local.svg"))
