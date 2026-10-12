"""Slide 10: the indexer's prefill path (contiguous KV, per-row [ks, ke)) vs decode path (paged KV + block table).
Shapes follow DeepGEMM tests/test_attention.py and the vLLM DeepSeek-V3.2 blog; sizes are illustrative."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 286
s = SVG(W, H, "prefill vs decode indexer paths")
MONO = "Menlo, Courier New, monospace"

# ---------------- (a) prefill
s.text(0, 20, "(a) Prefill: packed requests", 20, INK, anchor="start", weight="bold")
cs, x0, y0 = 19, 64, 54
reqA_kv, reqA_q = 10, 5
reqB_kv, reqB_q = 6, 3
ncol = reqA_kv + reqB_kv
rows = []
for i in range(reqA_q):
    rows.append((0, reqA_kv - reqA_q + i + 1, "A"))
for i in range(reqB_q):
    rows.append((reqA_kv, reqA_kv + reqB_kv - reqB_q + i + 1, "B"))
for r, (ks, ke, req) in enumerate(rows):
    for c in range(ncol):
        on = ks <= c < ke
        s.rect(x0 + c * cs, y0 + r * cs, cs - 2, cs - 2, ACC_F if on else "#FFFFFF", ACC if on else "#D0D0D0", 1.2, 1)
# request labels and column header
s.text(x0 + reqA_kv * cs / 2, y0 - 10, "KV of request A", 16, MUTED)
s.text(x0 + (reqA_kv + reqB_kv / 2) * cs, y0 - 10, "B", 16, MUTED)
s.line(x0 + reqA_kv * cs - 1, y0 - 4, x0 + reqA_kv * cs - 1, y0 + len(rows) * cs, MUTED, 1.4, dash="4,3")
s.text(x0 - 8, y0 + 2.5 * cs + 6, "A", 17, MUTED, anchor="end")
s.text(x0 - 8, y0 + 6.5 * cs + 6, "B", 17, MUTED, anchor="end")
s.text(x0 - 30, y0 + 4 * cs + 6, "q", 18, INK, anchor="end", weight="bold")
# ks / ke markers on one row of B
rb = reqA_q + 1
ksx, kex = x0 + rows[rb][0] * cs, x0 + rows[rb][1] * cs - 2
yy = y0 + len(rows) * cs + 12
s.line(ksx, yy, kex, yy, ACC, 2, "arrAcc", start_marker="arrAcc")
s.text(ksx - 14, yy + 6, "ks", 17, ACC, weight="bold", family=MONO)
s.text(kex + 16, yy + 6, "ke", 17, ACC, weight="bold", family=MONO)
s.text(0, 254, "one [ks, ke) range per query row", 17, INK, anchor="start")
s.text(0, 280, "fp8_mqa_logits", 17, INK, anchor="start", family=MONO)

# ---------------- (b) decode
bx = 440
s.line(bx - 22, 6, bx - 22, 282, "#BFBFBF", 1.2)
s.text(bx, 20, "(b) Decode: paged KV cache", 20, INK, anchor="start", weight="bold")
reqs = [("R0", [5, 2, 7]), ("R1", [0, 3]), ("R2", [6, 1, 4])]
owner = {p: name for name, pages in reqs for p in pages}
ty = 58
s.text(bx, ty - 12, "block table", 16, MUTED, anchor="start")
for i, (name, pages) in enumerate(reqs):
    y = ty + i * 40
    hl = name == "R0"
    s.box(bx, y, 64, 30, name, ACC_F if hl else "#FFFFFF", ACC if hl else NEU, INK, 17, "bold", 2, 4)
    for j, p in enumerate(pages):
        s.box(bx + 76 + j * 34, y, 30, 30, str(p), "#FFFFFF", ACC if hl else NEU, INK, 17, "normal", 1.6, 2)
# physical page pool
px, py, pw = bx + 214, 58, 22
s.text(px, py - 12, "page pool · 64 tok/page", 16, MUTED, anchor="start")
for p in range(8):
    hl = owner[p] == "R0"
    y = py + p * 0  # single row
    x = px + (p % 4) * (pw + 30)
    y = py + (p // 4) * 64
    s.rect(x, y, pw + 22, 40, ACC_F if hl else NEU_F, ACC if hl else NEU, 2, 3)
    s.text(x + (pw + 22) / 2, y + 17, f"p{p}", 15, INK, weight="bold")
    s.text(x + (pw + 22) / 2, y + 35, owner[p], 15, ACC if hl else MUTED)
s.text(bx, 230, "new token(s) of each request", 17, INK, anchor="start")
s.text(bx, 254, "pages located via block table", 17, INK, anchor="start")
s.text(bx, 280, "fp8_paged_mqa_logits", 17, INK, anchor="start", family=MONO)
s.save(os.path.join(HERE, "fig_paths.svg"))
