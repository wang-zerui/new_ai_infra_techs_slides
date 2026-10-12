"""fig_dial: sparsity = routed experts / routed experts per token. K2 sweep points (8,16,32,48,64; K2 p.7 Fig. 5)
and the configured sparsity of V3/GLM-5 (32), V4-Flash (42.7), K2 (48), V4-Pro (64) (derived from configs)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 286
s = SVG(W, H, "fig_dial: sparsity number line")
x = lambda v: 60 + v * 11.4
AY = 164
s.text(W / 2, 24, "sparsity = routed experts ÷ routed experts per token", 20, INK, weight="bold")
# axis
s.line(x(0), AY, x(70.5), AY, INK, 2.2, "arr")
for v in (8, 16, 32, 48, 64):
    s.add(f'<circle cx="{x(v):.1f}" cy="{AY}" r="9" fill="#FFFFFF" stroke="{MUTED}" stroke-width="2"/>')
    s.text(x(v), AY + 34, str(v), 19, MUTED)
s.add(f'<circle cx="{x(0) + 8:.1f}" cy="64" r="9" fill="#FFFFFF" stroke="{MUTED}" stroke-width="2"/>')
s.text(x(0) + 26, 70, "K2 sweep", 18, MUTED, anchor="start")
s.add(f'<circle cx="{x(0) + 8:.1f}" cy="94" r="5.5" fill="{ACC}" stroke="{ACC}" stroke-width="1.5"/>')
s.text(x(0) + 26, 100, "model config", 18, MUTED, anchor="start")

def model(v, lines, ly, colors):
    s.line(x(v), ly + 8, x(v), AY - 11, ACC, 1.6)
    s.add(f'<circle cx="{x(v):.1f}" cy="{AY}" r="5.5" fill="{ACC}" stroke="{ACC}" stroke-width="1.5"/>')
    for i, (t, c) in enumerate(zip(lines, colors)):
        s.text(x(v), ly - (len(lines) - 1 - i) * 23, t, 19, c, weight="bold")

model(32, ["DeepSeek-V3", "GLM-5"], 124, [MUTED, ACC])
model(256 / 6, ["V4-Flash 42.7"], 62, [ACC])
model(48, ["Kimi K2"], 124, [ACC])
model(64, ["V4-Pro"], 104, [ACC])

# consequences of turning the dial up
s.text(x(0), 238, "Lower loss at equal FLOPs", 19, ACC, anchor="start", weight="bold")
s.line(x(33), 232, x(70), 232, ACC, 2.6, "arrAcc")
s.text(x(0), 274, "More weights to hold and balance", 19, PRB, anchor="start", weight="bold")
s.line(x(33), 268, x(70), 268, PRB, 2.6, "arrPrb")
s.save(os.path.join(HERE, "fig_dial.svg"))
