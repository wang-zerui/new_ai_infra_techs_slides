"""fig_statusquo.svg: three status-quo ways to split a sequence across GPUs and why each falls short for KDA.
(a) softmax-style CP ships KV blocks that grow with length (K3 Sec. 5.1.2; Megatron CP doc);
(b) head-split all-to-all CP: every GPU runs the full-length recurrence for H/P heads (PR #691 baseline);
(c) LASP-style summation of zero-start states ignores the delta rule's multiplicative transition (K3 Sec. 5.1.2)."""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from svgkit import *
s = SVG(870, 285)
# (a)
s.t(140, 24, "Softmax-style CP", 19, "bold")
pos = [(40, 50), (160, 50), (160, 140), (40, 140)]
for i, (x, y) in enumerate(pos):
    s.box(x, y, 84, 44, GRAY_F, GRAY, 1.5, 4)
    s.t(x + 42, y + 28, f"GPU {i}", 16, "bold")
s.path("M126,72 H158", RED, 2.5, "r"); s.path("M202,96 V138", RED, 2.5, "r")
s.path("M158,162 H126", RED, 2.5, "r"); s.path("M82,138 V96", RED, 2.5, "r")
s.t(142, 64, "KV", 14, "bold", RED); s.t(214, 122, "KV", 14, "bold", RED, "start")
s.t(142, 180, "KV", 14, "bold", RED); s.t(70, 122, "KV", 14, "bold", RED, "end")
s.t(140, 228, "ships K and V blocks", 17)
s.t(140, 254, "grows with length", 17, "bold", RED)
s.raw('<line x1="290" y1="14" x2="290" y2="268" stroke="#D9D9D9" stroke-width="1.2"/>')
# (b)
s.t(435, 24, "Head-split all-to-all", 19, "bold")
for i in range(4):
    y = 66 + i * 36
    s.t(350, y + 22, f"GPU {i}", 15, "normal", GRAY, "end")
    s.box(358, y, 206, 28, "url(#hr)", RED, 1.2, 3)
s.t(461, 54, "each: all T tokens × H/4 heads", 15, "normal", INK)
s.t(435, 228, "CP degree ≤ heads", 17)
s.t(435, 254, "recurrence stays full length", 17, "bold", RED)
s.raw('<line x1="580" y1="14" x2="580" y2="268" stroke="#D9D9D9" stroke-width="1.2"/>')
# (c)
s.t(725, 24, "Sum local states", 19, "bold")
for i in range(3):
    x = 600 + i * 84
    s.box(x, 60, 60, 40, ACC_F, GRAY, 1.5, 4)
    s.t(x + 30, 86, sub("S̃", i), 17, "bold")
    if i < 2:
        s.t(x + 72, 88, "+", 22, "bold")
s.line(725, 104, 725, 134, INK, 2)
s.box(685, 136, 80, 40, "#FFFFFF", RED, 2, 4)
s.t(725, 162, sub("S", "in,3"), 17, "bold", RED)
s.t(790, 166, "✗", 30, "bold", RED, "start")
s.t(725, 228, "exact only if S just adds up", 17)
s.t(725, 254, "KDA multiplies S by M", 17, "bold", RED)
s.save(pathlib.Path(__file__).with_name("fig_statusquo.svg"))
