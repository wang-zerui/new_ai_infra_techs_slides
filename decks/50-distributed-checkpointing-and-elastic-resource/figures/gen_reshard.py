"""fig_reshard: VeOmni restores the EP dim so DCP stores the expert weight as one global tensor; a job with a
different EP size then reads overlapping chunks (EP 8 x FSDP 2 on 16 GPUs -> EP 4 x FSDP 2 on 8 GPUs)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 304, "fig_reshard")
MONO = "Menlo, monospace"

# ---- left: what DCP sees
steps = [("runtime shard (EP dim dropped)", "[16, 1024, 768]", False),
         ("save: restore EP dim", "[Shard(1), Shard(0)]", True),
         ("DCP records global tensor", "[128, 2048, 768]", False)]
y = 20
for i, (title, mono, acc) in enumerate(steps):
    s.rect(10, y, 330, 64, ACC_F if acc else "#FFFFFF", ACC if acc else NEU, 2, 5)
    s.text(175, y + 26, title, 17, ACC if acc else INK, weight="bold")
    s.text(175, y + 52, mono, 16, ACC if acc else INK, family=MONO)
    if i < 2:
        s.line(175, y + 66, 175, y + 88, INK, 2.2, "arr")
    y += 92
s.text(175, 296, "load: drop EP dim again", 16, MUTED, style="italic")

# ---- right: tile grid
GX, TW, TH, TG, GY = 528, 100, 24, 3, 56
s.text(GX + TW / 2, 24, "hidden 0–1023", 15, MUTED)
s.text(GX + TW + TG + TW / 2, 24, "1024–2047", 15, MUTED)
s.text(GX + TW + TG / 2, 46, "down_proj [128, 2048, 768]", 15, INK, family=MONO)
for r in range(8):
    yy = GY + r * (TH + TG)
    s.text(GX - 10, yy + TH / 2 + 5.5, f"experts {16 * r}–{16 * r + 15}", 15, INK, anchor="end")
    for c in range(2):
        x = GX + c * (TW + TG)
        hot = (c == 0 and r < 2)
        s.box(x, yy, TW, TH, f"old {r + 8 * c}", ACC_F if hot else "#FFFFFF", NEU, ACC if hot else INK, 15,
              "bold" if hot else "normal", 1.2, 2)
# new-rank outlines (EP 4 x FSDP 2): 2 expert blocks per new rank
for c in range(2):
    for g in range(4):
        x = GX + c * (TW + TG) - 2
        yy = GY + g * 2 * (TH + TG) - 2
        s.rect(x, yy, TW + 4, 2 * TH + TG + 4, "none", ACC, 2.6, 3)
GB = GY + 8 * (TH + TG) - TG
# annotation for new rank 0
AX = GX + 2 * TW + TG + 14
s.text(AX, GY + TH - 2, "new rank 0", 17, ACC, anchor="start", weight="bold")
s.text(AX, GY + TH + 20, "2 reads", 17, ACC, anchor="start")
# legend
s.rect(372, GB + 16, 22, 16, "#FFFFFF", NEU, 1.4, 2)
s.text(400, GB + 30, "saved chunk", 15, INK, anchor="start")
s.rect(520, GB + 16, 22, 16, "none", ACC, 2.6, 2)
s.text(548, GB + 30, "new rank (EP 4 × FSDP 2)", 15, INK, anchor="start")
s.save(os.path.join(HERE, "fig_reshard.svg"))
