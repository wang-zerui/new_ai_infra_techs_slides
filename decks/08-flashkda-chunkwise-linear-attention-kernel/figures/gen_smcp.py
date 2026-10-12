"""fig_smcp.svg: intra-device context parallelism for long prefill (K3 Sec. 5.1.1): split one sequence across SM groups,
evaluate each segment's state transition from S = 0 in parallel, compose them in order to get exact entry states,
then each segment produces its outputs. Schematic; the cross-GPU version is KCP (deck 69)."""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from svgkit import *
s = SVG(870, 285)
labels = [("one sequence", 46), ("step 1", 124), ("step 2", 196), ("step 3", 258)]
for lab, y in labels:
    s.t(132, y, lab, 17, "bold" if lab != "one sequence" else "normal", INK, "end")
s.t(132, 66, "SM group each", 15, "normal", GRAY, "end", "italic")
s.t(132, 144, "in parallel", 15, "normal", GRAY, "end", "italic")
s.t(132, 216, "compose", 15, "normal", GRAY, "end", "italic")
s.t(132, 278, "in parallel", 15, "normal", GRAY, "end", "italic")
X0, W, G = 150, 160, 18
for i in range(4):
    x = X0 + i * (W + G)
    s.box(x, 30, W, 34, GRAY_F, GRAY, 1.5, 3)
    s.t(x + W / 2, 53, f"segment {i+1}", 17)
    s.line(x + W / 2, 66, x + W / 2, 96)
    s.box(x, 96, W, 54, ACC_F, ACC, 2, 4)
    s.t(x + W / 2, 117, f'{sub("M", i+1)} and {sub("S̃", i+1)}', 18, "bold", ACC)
    s.t(x + W / 2, 143, "from S = 0", 15, "normal", GRAY)
    s.line(x + W / 2, 151, x + W / 2, 174)
    s.box(x + 30, 176, W - 60, 34, "#FFFFFF", ACC, 2, 4)
    s.t(x + W / 2, 199, sub("S", f"in,{i+1}"), 17, "bold", ACC)
    if i < 3:
        s.line(x + W - 30, 193, x + W + G + 28, 193, ACC, 2.5, "b")
    s.line(x + W / 2, 212, x + W / 2, 236)
    s.box(x, 238, W, 34, ACC_F, ACC, 1.5, 4)
    s.t(x + W / 2, 261, "outputs", 17)
s.save(pathlib.Path(__file__).with_name("fig_smcp.svg"))
