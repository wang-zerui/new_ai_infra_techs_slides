"""fig_grids.svg: CTA grids of the two FlashKDA kernels for the benchmark shape (one sequence, T = 8192, H = 96, CHUNK = 16).
K1 grid = tiles x heads = (8192/16) x 96 = 49,152 CTAs (derived); K2 grid = sequences x heads = 1 x 96 = 96 CTAs (derived).
Grid definitions: FlashKDA csrc/smxx/fwd_launch.cu (grid_k1(total_tiles, H), grid_k2(N, H))."""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from svgkit import *
s = SVG(870, 285)
s.t(860, 18, "example: one 8K-token sequence with 96 heads", 16, "normal", GRAY, "end", "italic")
# K1
s.t(250, 50, "K1 prepare: tiles × heads", 20, "bold", ACC)
x0, y0, c = 70, 64, 15
for i in range(8):
    for j in range(22):
        s.box(x0 + j * (c + 3), y0 + i * (c + 3), c, c, ACC_F, ACC, 1, 1)
s.t(x0 + 22 * (c + 3) + 6, y0 + 4 * (c + 3) + 4, "⋯", 22, "bold", INK, "start")
s.t(x0 - 10, y0 + 4 * (c + 3) + 2, "96", 17, "bold", INK, "end")
s.t(x0 - 10, y0 + 4 * (c + 3) + 22, "heads", 15, "normal", GRAY, "end")
s.t(x0 + 11 * (c + 3), y0 + 8 * (c + 3) + 20, "512 tiles of 16 tokens", 17)
s.t(250, 274, "49,152 CTAs", 24, "bold", ACC)
s.raw('<line x1="520" y1="40" x2="520" y2="265" stroke="#D9D9D9" stroke-width="1.2"/>')
# K2
s.t(700, 50, "K2 recurrence: seqs × heads", 20, "bold", ACC)
x1 = 640
for i in range(8):
    s.box(x1, y0 + i * (c + 3), c, c, ACC_F, ACC, 1, 1)
s.t(x1 - 10, y0 + 4 * (c + 3) + 2, "96", 17, "bold", INK, "end")
s.t(x1 - 10, y0 + 4 * (c + 3) + 22, "heads", 15, "normal", GRAY, "end")
s.t(x1 + 8, y0 + 8 * (c + 3) + 20, "1 seq", 17)
s.t(x1 + 30, y0 + 2 * (c + 3) + 4, "one state S", 17, "normal", INK, "start")
s.t(x1 + 30, y0 + 3 * (c + 3) + 8, "per CTA", 17, "normal", INK, "start")
s.t(x1 + 30, y0 + 5 * (c + 3) + 6, "walks all 512", 17, "normal", INK, "start")
s.t(x1 + 30, y0 + 6 * (c + 3) + 10, "tiles in order", 17, "normal", INK, "start")
s.t(700, 274, "96 CTAs", 24, "bold", ACC)
s.save(pathlib.Path(__file__).with_name("fig_grids.svg"))
