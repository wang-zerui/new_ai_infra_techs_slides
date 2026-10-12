"""fig_ulysses: DeepSpeed-Ulysses in VeOmni, P = 4. All-to-all turns sequence-sharded [S/P, H] into
head-sharded [S, H/P] before attention and back afterwards (VeOmni §3.2.2, App. B.6; docs ulysses.md)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 284, "fig_ulysses")
MONO = "Menlo, monospace"
TW, TH, G = 42, 34, 4
GY = 72

def grid(x0, owner, title, shape):
    s.text(x0 + 2 * (TW + G) - G / 2, 24, title, 18, INK, weight="bold")
    s.text(x0 + 2 * (TW + G) - G / 2, 50, shape, 17, ACC, family=MONO)
    for r in range(4):
        for c in range(4):
            g = owner(r, c)
            hot = g == 0
            s.box(x0 + c * (TW + G), GY + r * (TH + G), TW, TH, f"G{g}", ACC_F if hot else "#FFFFFF",
                  ACC if hot else NEU, ACC if hot else INK, 16, "bold" if hot else "normal", 2 if hot else 1.4, 3)
    w = 4 * (TW + G) - G
    s.text(x0 + w / 2, GY + 4 * (TH + G) + 18, "heads H →", 16, MUTED)
    return x0 + w

# axis label for sequence
s.add(f'<text x="22" y="{GY + 76}" text-anchor="middle" font-size="16" fill="{MUTED}" '
      f'transform="rotate(-90 22 {GY + 76})">sequence S →</text>')
ax = grid(40, lambda r, c: r, "before attention", "[S/P, H]")
bx0 = ax + 100
bx = grid(bx0, lambda r, c: c, "inside attention", "[S, H/P]")
ym = GY + 2 * (TH + G) - G / 2
s.line(ax + 8, ym, bx0 - 8, ym, ACC, 2.6, "arrAcc")
s.text((ax + bx0) / 2, ym - 12, "all-to-all", 18, ACC, weight="bold")

# attention box
AX0, AW = bx + 34, 146
s.line(bx + 6, ym, AX0 - 2, ym, INK, 2.2, "arr")
s.rect(AX0, GY, AW, 4 * (TH + G) - G, NEU_F, NEU, 2, 5)
for i, t in enumerate(["FlashAttention", "full sequence", "H/P heads"]):
    s.text(AX0 + AW / 2, ym - 22 + i * 25, t, 17, INK, weight="bold" if i == 0 else "normal")
# back
RX = AX0 + AW + 92
s.line(AX0 + AW + 6, ym, RX - 2, ym, ACC, 2.6, "arrAcc")
s.text((AX0 + AW + RX) / 2, ym - 12, "all-to-all", 18, ACC, weight="bold")
s.rect(RX, ym - 30, 862 - RX, 60, ACC_F, ACC, 2, 5)
s.text(RX + (862 - RX) / 2, ym - 4, "[S/P, H]", 17, ACC, family=MONO)
s.text(RX + (862 - RX) / 2, ym + 20, "out proj", 16, ACC)

# legend
s.rect(40, 256, 22, 18, ACC_F, ACC, 2, 3)
s.text(70, 271, "GPU 0's shard (P = 4)", 17, INK, anchor="start")
s.save(os.path.join(HERE, "fig_ulysses.svg"))
