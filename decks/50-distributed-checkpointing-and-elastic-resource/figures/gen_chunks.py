"""fig_chunks: DCP's global view. 4 saved chunks of W are listed with their global offsets in .metadata;
a 2-GPU job asks for rows 0-3 and 4-7 and reads exactly the overlapping chunks."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 290, "fig_chunks")
MONO = "Menlo, monospace"
TX, TW, RH, TY = 380, 110, 26, 50

# metadata table (left)
s.text(160, 24, "saved by 4 GPUs", 19, INK, weight="bold")
s.rect(16, TY, 290, 8 * RH, "#F7F7F7", NEU, 1.6, 4)
for c in range(4):
    yc = TY + c * 2 * RH + RH + 6
    s.text(34, yc, f"W[{2 * c}:{2 * c + 2}]", 18, INK, anchor="start", family=MONO, weight="bold")
    s.text(290, yc, f"rank {c} file", 18, INK, anchor="end", family=MONO)
    if c:
        s.line(16, TY + c * 2 * RH, 306, TY + c * 2 * RH, "#D0D0D0", 1)
s.text(160, TY + 8 * RH + 24, ".metadata: global offsets", 17, MUTED, style="italic", family=None)

# global tensor (middle)
s.text(TX + TW / 2, 24, "global W", 19, INK, weight="bold")
for r in range(8):
    s.rect(TX, TY + r * RH, TW, RH, "#FFFFFF" if (r // 2) % 2 == 0 else NEU_F, NEU, 1.2, 0)
    s.text(TX + TW / 2, TY + r * RH + RH / 2 + 6, f"row {r}", 16, MUTED)
for c in range(4):
    s.rect(TX, TY + c * 2 * RH, TW, 2 * RH, "none", INK, 2.2, 0)
    s.line(310, TY + c * 2 * RH + RH, TX - 4, TY + c * 2 * RH + RH, "#A6A6A6", 1.4, dash="4,4")

# new GPUs (right)
GX, GW = 660, 190
s.text(GX + GW / 2, 24, "loaded on 2 GPUs", 19, ACC, weight="bold")
for g in range(2):
    y0 = TY + g * 4 * RH + 10
    s.box(GX, y0, GW, 4 * RH - 20, [f"GPU {g}", f"needs W[{4 * g}:{4 * g + 4}]"], ACC_F, ACC, ACC, 18, "bold")
    for k in range(2):
        cy = TY + (g * 4 + k * 2) * RH + RH
        s.line(TX + TW + 4, cy, GX - 3, y0 + (4 * RH - 20) / 2 + (k - 0.5) * 30, ACC, 2.4, "arrAcc")
s.text(GX + GW / 2, TY + 8 * RH + 24, "2 reads per GPU", 17, ACC, style="italic")
s.save(os.path.join(HERE, "fig_chunks.svg"))
