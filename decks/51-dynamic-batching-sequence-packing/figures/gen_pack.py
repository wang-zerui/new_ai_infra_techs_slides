"""fig_pack: VeOmni packing example (docs/usage/data_packing_and_dyn_bsz.md): S1=[A B C D], S2=[E F], S3=[G H I],
S4=[J] -> one stream; position_ids restart; next-token targets stop at boundaries; cu_seqlens [0 4 6 9 10];
varlen attention = block-diagonal causal mask (never materialized)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 282, "fig_pack")
MONO = "Menlo, monospace"
toks = list("ABCDEFGHIJ")
seg = [0, 0, 0, 0, 1, 1, 2, 2, 2, 3]
pos = [0, 1, 2, 3, 0, 1, 0, 1, 2, 0]
tgt = ["B", "C", "D", "–", "F", "–", "H", "I", "–", "–"]
CW, CG, BG, X0 = 32, 3, 11, 150
def cx(i):
    return X0 + i * (CW + CG) + seg[i] * BG
rows = [("input_ids", 44), ("position_ids", 104), ("targets", 164)]
for name, y in rows:
    s.text(X0 - 12, y + 22, name, 17, INK, anchor="end", family=MONO)
for i in range(10):
    fill = NEU_F if seg[i] % 2 == 0 else "#FFFFFF"
    s.box(cx(i), 44, CW, 32, toks[i], fill, NEU, INK, 17, "bold", 1.6, 3)
    s.text(cx(i) + CW / 2, 104 + 22, str(pos[i]), 17, ACC if pos[i] == 0 else INK, family=MONO,
           weight="bold" if pos[i] == 0 else "normal")
    s.text(cx(i) + CW / 2, 164 + 22, tgt[i], 17, MUTED if tgt[i] == "–" else INK, family=MONO)
# cu_seqlens ticks
bounds = [0, 4, 6, 9, 10]
YT = 226
s.text(X0 - 12, YT + 22, "cu_seqlens", 17, ACC, anchor="end", family=MONO, weight="bold")
for b in bounds:
    if b < 10:
        xb = cx(b) - (BG / 2 + 1 if b > 0 else 2)
    else:
        xb = cx(9) + CW + 2
    s.line(xb, 40, xb, YT + 2, ACC, 1.6, dash="4,3")
    s.text(xb, YT + 24, str(b), 17, ACC, family=MONO, weight="bold")
s.text(X0 + 175, 272, "targets “–” = IGNORE_INDEX", 15, MUTED, style="italic")

# attention grid (right)
GX, GY, C = 652, 52, 19
s.text(GX + 5 * C, 22, "varlen attention", 18, ACC, weight="bold")
for r in range(10):
    s.text(GX - 7, GY + r * C + 14, toks[r], 15, INK, anchor="end")
    s.text(GX + r * C + C / 2 - 1, GY - 7, toks[r], 15, INK)
    for c in range(10):
        ok = c <= r and seg[c] == seg[r]
        s.rect(GX + c * C, GY + r * C, C - 2, C - 2, ACC_F if ok else "#FFFFFF", ACC if ok else "#E0E0E0", 1.2, 1)
s.text(GX + 5 * C, GY + 10 * C + 24, "block-diagonal causal", 16, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_pack.svg"))
