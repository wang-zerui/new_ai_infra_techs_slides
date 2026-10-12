"""fig_rope: MLA RoPE, eager vs fused (DeepSeek-V3 dims per head: q = 128 nope + 64 rope; kv = 128 k_nope +
128 v; k_pe = 64 shared by all heads). Eager branch of multi_latent_attention.py: split q and kv, RoPE on q_pe
and k_pe, cat, expand k_pe to all heads, cat, contiguous. Fused (fused_mla_yarn_rope_apply.py): the Q kernel
rotates the 64-dim slice in place; the KV kernel splits kv, rotates k_pe once and writes key and value.
Bar widths: 1 pt per dimension."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 282, "fig_rope: MLA RoPE eager vs fused")


def chain(y, label, ops):
    s.text(4, y + 25, label, 17, INK, anchor="start", weight="bold")
    x = 52
    for k, (lab, w) in enumerate(ops):
        s.box(x, y, w, 40, lab, PRB_F, PRB, color=INK, size=15, sw=1.8)
        if k < len(ops) - 1:
            s.line(x + w + 2, y + 20, x + w + 16, y + 20, PRB, 1.8, "arrPrb")
        x += w + 18


s.text(4, 16, "eager", 18, PRB, anchor="start", weight="bold")
chain(26, "Q", [(["split"], 70), (["RoPE", "q_pe"], 80), (["cat"], 60), (["contiguous"], 104)])
chain(76, "KV", [(["split"], 70), (["RoPE", "k_pe"], 80), (["expand", "to heads"], 92), (["cat", "key"], 60),
                 (["contiguous", "key and value"], 130)])

# ---- fused
s.line(4, 132, 866, 132, "#BFBFBF", 1.2)
s.text(4, 158, "fused", 18, ACC, anchor="start", weight="bold")
X = 120
# Q kernel
s.text(4, 192, "Q kernel", 16, ACC, anchor="start", weight="bold")
s.box(X, 170, 128, 34, "q_nope", "#FFFFFF", "#A6A6A6", color=MUTED, size=15, dash="4,3")
s.box(X + 128, 170, 64, 34, "q_pe", ACC_F, ACC, size=15, weight="bold", sw=2.2)
s.text(X + 204, 192, "64-dim slice rotated in place", 15, INK, anchor="start")
# KV kernel
s.text(4, 240, "KV kernel", 16, ACC, anchor="start", weight="bold")
s.box(X, 218, 128, 34, "k_nope", NEU_F, NEU, size=15)
s.box(X + 128, 218, 128, 34, "v", NEU_F, NEU, size=15)
s.box(X + 266, 218, 64, 34, "k_pe", ACC_F, ACC, size=15, weight="bold", sw=2.2)
s.line(X + 334, 235, X + 370, 235, ACC, 2.4, "arrAcc")
s.text(X + 298, 272, "shared", 15, INK)
s.text(X + 128, 272, "kv input", 15, INK)
OX = X + 374
s.box(OX, 218, 128, 34, "k_nope", NEU_F, NEU, size=15)
s.box(OX + 128, 218, 64, 34, "rope", ACC_F, ACC, size=15, weight="bold", sw=2.2)
s.box(OX + 204, 218, 128, 34, "v", NEU_F, NEU, size=15)
s.text(OX + 96, 272, "key (every head)", 15, INK)
s.text(OX + 268, 272, "value", 15, INK)
s.save(os.path.join(HERE, "fig_rope.svg"))
