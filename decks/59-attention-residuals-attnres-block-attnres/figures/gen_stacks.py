"""fig_stack_std: standard PreNorm residual stream (every sublayer output added with weight 1).
fig_stack_block: Kimi K3's Block AttnRes depth layout (K3 §2.2, HF config attn_res_block_size = 12, 93 layers):
embedding b0 + 8 blocks (7 x 12 layers + 1 x 9 layers); each sublayer input is a softmax mix of the cached block
summaries and the running partial sum of its own block."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
I = "italic"

# ---------------------------------------------------------------- standard residual stream (vertical)
s = SVG(470, 340, "fig_stack_std: standard residual stream")
SX = 70                                   # stream x
ys = [272, 208, 144, 80]                   # add-node y (bottom -> top)
names = ["Attention", "MoE", "Attention", "MoE"]
s.box(16, 300, 110, 34, "embedding", NEU_F, NEU, size=17)
s.line(SX, 300, SX, 34, INK, 3.0)
s.line(SX, 40, SX, 14, INK, 3.0, "arr")
s.text(SX + 12, 22, "output", 17, MUTED, anchor="start", style="italic")
for y, n in zip(ys, names):
    # tap from the stream into the layer, back into the add node
    s.path(f"M {SX} {y + 20} H 128", INK, 1.8, "arr")
    s.box(130, y, 168, 40, ["Norm → " + n], NEU_F, NEU, size=17)
    s.path(f"M 298 {y + 20} H 314 V {y - 16} H {SX + 13}", INK, 1.8, "arr")
    s.circle(SX, y - 16, 12, "#FFFFFF", INK, 2)
    s.text(SX, y - 10, "+", 18, INK, weight="bold")
    s.text(SX + 34, y - 22, "×1", 17, PRB, weight="bold")
s.text(SX, 48, "⋮", 22, INK)
s.rich(330, 150, [("h", INK, "bold", I), ("l", INK, "normal", "sub"), (" = h", INK, "normal", I),
                  ("1", INK, "normal", "sub"), (" + Σ f", INK, "normal"), ("i", INK, "normal", "sub")], 20,
       anchor="start")
s.text(330, 178, "fixed weight 1", 18, PRB, anchor="start", weight="bold")
s.text(330, 202, "for every layer", 18, PRB, anchor="start", weight="bold")
s.save(os.path.join(HERE, "fig_stack_std.svg"))

# ---------------------------------------------------------------- K3 block layout (horizontal)
s = SVG(870, 300, "fig_stack_block: K3 Block AttnRes layout")
BY, BH = 40, 92
blocks = [("embedding", None), ("block 1", "layers 1–12"), ("block 2", "layers 13–24"), ("⋯", None),
          ("block 8", "layers 85–93")]
xs = [10, 150, 340, 530, 610]
ws = [120, 170, 170, 60, 170]
for (name, sub), x, w in zip(blocks, xs, ws):
    if name == "⋯":
        s.text(x + w / 2, BY + BH / 2 + 8, "⋯", 28, INK)
        continue
    if name == "embedding":
        s.box(x, BY + 24, w, 44, "embedding", NEU_F, NEU, size=18)
        continue
    s.rect(x, BY, w, BH, "#FFFFFF", NEU, 1.8, 6, dash="6,4")
    s.text(x + w / 2, BY + 22, name, 18, INK, weight="bold")
    s.text(x + w / 2, BY + 44, sub, 16, MUTED)
    # two mini sublayers with an AttnRes op each
    for k, lab in enumerate(["α → Attn", "α → MoE"]):
        s.box(x + 10 + k * (w - 20) / 2, BY + 54, (w - 20) / 2 - 4, 30, lab, ACC_F, ACC, size=15)
# flow along depth
for (x0, w0), x1 in zip(zip(xs[:-1], ws[:-1]), xs[1:]):
    a = x0 + w0 + 3 if x0 != 530 else x0 + 44
    b = x1 - 3 if x1 != 530 else x1 + 14
    s.line(a, BY + 46, b, BY + 46, INK, 2.2, "arr")
s.line(783, BY + 46, 862, BY + 46, INK, 2.2, "arr")
s.text(824, BY + 34, "output", 16, MUTED, style="italic")

# block cache (bottom)
CY = 210
chips = [("b0", 70), ("b1", 235), ("b2", 425), ("⋯", 560), ("b8", 695)]
for lab, cx in chips:
    if lab == "⋯":
        s.text(cx, CY + 26, "⋯", 24, ACC)
        continue
    s.rect(cx - 30, CY, 60, 40, ACC_F, ACC, 2, 4)
    s.rich(cx, CY + 27, [(lab[0], ACC, "bold", I), (lab[1:], ACC, "bold", "sub")], 20)
# each block writes its sum once
for (lab, cx), x, w in zip(chips, xs, ws):
    if lab == "⋯":
        continue
    top = BY + 68 if lab == "b0" else BY + BH
    s.line(cx, top + 3, cx, CY - 3, ACC, 2.2, "arrAcc")
# reads: block 2's AttnRes ops attend over b0 and b1 (fan-in drawn for one block)
tx, ty = 340 + 10 + 38, BY + 84
for cx in (70, 235):
    s.path(f"M {cx + 12} {CY - 2} Q {cx + 60} {CY - 50} {tx - 4} {ty + 4}", ACC, 1.8, "arrAcc", dash="5,4")
s.rich(10, 284, [("cached summaries b", ACC, "bold"), ("0", ACC, "bold", "sub"), (" … b", ACC, "bold"),
                 ("8", ACC, "bold", "sub"),
                 (": each α attends over the finished ones + its own block's partial sum", INK, "normal")],
       17, anchor="start")
s.save(os.path.join(HERE, "fig_stack_block.svg"))
