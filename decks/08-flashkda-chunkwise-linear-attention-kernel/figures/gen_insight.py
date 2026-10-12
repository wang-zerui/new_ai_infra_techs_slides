"""fig_insight.svg: dependency structure of chunkwise KDA (K3 Eq. 4; FlashKDA kernel split): per-chunk 'prepare' work
needs no state; only the short 'recur' step consumes and produces the d_k x d_v state S."""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from svgkit import *
s = SVG(470, 280)
s.t(235, 22, "no state needed: all chunks at once", 17, "bold", ACC)
xs = [12, 128, 244, 360]
for i, x in enumerate(xs):
    s.box(x, 36, 98, 56, ACC_F, ACC, 2)
    s.t(x + 49, 60, "prepare", 17, "bold", ACC)
    s.t(x + 49, 82, f"chunk {i+1}", 16)
    s.line(x + 49, 94, x + 49, 150)
    s.box(x + 9, 152, 80, 50, GRAY_F, INK, 1.8)
    s.t(x + 49, 183, "recur", 17, "bold")
    if i < 3:
        s.line(x + 89, 177, xs[i + 1] + 7, 177, INK, 3)
        s.t(x + 103, 168, "S", 16, "bold")
s.t(235, 236, "thin serial chain: only S passes", 17, "bold", INK)
s.t(235, 262, "a few small matmuls per chunk", 16, "normal", GRAY, "middle", "italic")
s.save(pathlib.Path(__file__).with_name("fig_insight.svg"))
