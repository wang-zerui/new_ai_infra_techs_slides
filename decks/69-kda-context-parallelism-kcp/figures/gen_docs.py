"""fig_docs.svg: KCP with packed documents and the ShortConv halo (FLA fla/ops/cp README: merge chains only fragments
of the same document, only the first local sequence can continue from a previous rank; causal_conv1d CP needs
conv1d_kernel_size and receives the previous rank's tail tokens; PR #691 'conv1d' section). Layout is illustrative."""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from svgkit import *
s = SVG(870, 285)
segs = [[("A", 200)], [("A", 80), ("B", 120)], [("C", 200)], [("C", 90), ("D", 110)]]
X = [20, 232, 444, 656]
for i, docs in enumerate(segs):
    s.t(X[i] + 100, 26, f"GPU {i}", 17, "bold")
    x = X[i]
    for d, w in docs:
        s.box(x, 36, w - 4, 40, GRAY_F, GRAY, 1.5, 3)
        s.t(x + (w - 4) / 2, 62, f"doc {d}", 16)
        x += w
# merge arrows (continuations only)
s.path(f"M{X[0]+150},78 C{X[0]+170},112 {X[1]+10},112 {X[1]+30},80", ACC, 2.5, "b")
s.t(X[0] + 212, 126, "fragment of doc A", 16, "bold", ACC)
s.path(f"M{X[2]+150},78 C{X[2]+170},112 {X[3]+10},112 {X[3]+30},80", ACC, 2.5, "b")
s.t(X[2] + 212, 126, "fragment of doc C", 16, "bold", ACC)
s.t(X[2] + 70, 100, "starts at S = 0", 15, "normal", GRAY, "middle", "italic")
s.raw('<line x1="20" y1="146" x2="850" y2="146" stroke="#D9D9D9" stroke-width="1.2"/>')
# conv halo zoom
s.t(20, 172, "ShortConv halo at the GPU 2 | GPU 3 boundary", 17, "bold", INK, "start")
c, gap = 30, 4
x0 = 250
for k in range(6):
    s.box(x0 + k * (c + gap), 196, c, c, ACC_F if k >= 3 else GRAY_F, ACC if k >= 3 else GRAY, 1.5 if k < 3 else 2, 2)
s.t(x0 + 1.5 * (c + gap), 250, "GPU 2 tail", 15, "normal", GRAY)
x1 = 520
for k in range(3):
    s.box(x1 + k * (c + gap), 196, c, c, ACC_F, ACC, 2, 2)
for k in range(3, 7):
    s.box(x1 + k * (c + gap), 196, c, c, "#FFFFFF", GRAY, 1.5, 2)
s.line(x0 + 6 * (c + gap) - gap + 3, 211, x1 - 3, 211, ACC, 2.5, "b")
s.t((x0 + 6 * (c + gap) + x1) / 2, 200, "copy", 15, "bold", ACC)
s.t(x1 + 1.5 * (c + gap) - 4, 250, "copied halo", 15, "bold", ACC)
s.t(x1 + 5 * (c + gap), 250, "GPU 3 tokens", 15, "normal", GRAY)
s.raw(f'<path d="M{x1},{190} V184 H{x1 + 4*(c+gap) - gap} V190" fill="none" stroke="{INK}" stroke-width="1.5"/>')
s.t(x1 + 2 * (c + gap), 178, "window W", 15, "normal", INK)
s.t(20, 278, "W − 1 tail tokens per boundary · halo gradients return in reverse", 16, "normal", GRAY, "start", "italic")
s.save(pathlib.Path(__file__).with_name("fig_docs.svg"))
