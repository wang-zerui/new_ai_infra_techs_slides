"""fig_context.svg: context parallelism in K3 splits one long sequence across GPUs; each KDA layer then needs only the
fixed-size state entering its segment, each Gated MLA layer needs earlier keys/values (K3 Sec. 3.4, 5.1.2, 5.2; Table 1)."""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from svgkit import *
s = SVG(470, 290)
s.t(235, 20, "one sequence of up to 1M tokens", 17, "normal", GRAY, "middle", "italic")
for i in range(4):
    x = 10 + i * 113
    s.box(x, 30, 109, 40, GRAY_F, GRAY, 1.5, 3)
    s.t(x + 54.5, 56, f"GPU {i}", 17, "bold")
s.box(10, 106, 170, 64, ACC_F, ACC, 2.5)
s.t(95, 134, "KDA layers", 19, "bold", ACC)
s.t(95, 158, "69 of 93", 16)
s.line(182, 138, 214, 138, ACC, 2.5, "b")
s.t(222, 132, "needs entry state S", 18, "bold", ACC, "start")
s.t(222, 156, "fixed size (this deck)", 16, "normal", INK, "start")
s.box(10, 200, 170, 64, GRAY_F, GRAY, 1.5)
s.t(95, 228, "Gated MLA layers", 18, "bold")
s.t(95, 252, "24 of 93", 16)
s.line(182, 232, 214, 232, INK, 2)
s.t(222, 226, "needs earlier K and V", 18, "bold", INK, "start")
s.t(222, 250, "grows with length (deck 68)", 16, "normal", INK, "start")
s.save(pathlib.Path(__file__).with_name("fig_context.svg"))
