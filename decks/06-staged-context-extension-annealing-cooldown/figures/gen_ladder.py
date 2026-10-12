"""fig_ladder (overview): the three published schedules as stage chains, not to scale.
Columns: (1) bulk pretraining at high LR, (2) length ladder in the low-LR tail, (3) positional handling.
Sources: K2 §2.5; GLM-4.5 Fig. 3 + §2.3-2.4; K3 §3.3-3.4 (no stage budgets published)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 286, "fig_ladder: K2 / GLM-4.5 / K3 stage chains")

def badge(cx, cy, n):
    s.add(f'<circle cx="{cx}" cy="{cy}" r="12" fill="{ACC}"/>')
    s.text(cx, cy + 5.6, str(n), 16, "#FFFFFF", weight="bold")

def stage(x, y, w, l1, l2, long=False, pos=False, h=62):
    fill, stroke = (ACC_F, ACC) if long else (NEU_F, NEU)
    if pos:
        fill, stroke = "#FFFFFF", ACC
    s.rect(x, y, w, h, fill, stroke, 2, 4, dash="6,4" if pos else None)
    s.text(x + w / 2, y + 26, l1, 18, ACC if (long or pos) else INK, weight="bold")
    s.text(x + w / 2, y + 49, l2, 16, INK)

BX, BW = 98, 272      # bulk column
TX, TW = 386, 344      # tail column
PX, PW = 742, 124      # positions column
ROWS = [52, 128, 204]

# column headers
badge(BX + 12, 22, 1); s.text(BX + 30, 28.3, "Short bulk at high LR", 18, INK, anchor="start", weight="bold")
badge(TX + 12, 22, 2); s.text(TX + 30, 28.3, "Length ladder at low LR", 18, INK, anchor="start", weight="bold")
badge(PX + 12, 22, 3); s.text(PX + 30, 28.3, "Positions", 18, INK, anchor="start", weight="bold")
s.line(TX - 8, 40, TX - 8, 272, "#BFBFBF", 1.2, dash="4,4")
s.line(PX - 6, 40, PX - 6, 272, "#BFBFBF", 1.2, dash="4,4")

# row labels
for y, name in zip(ROWS, ["Kimi K2", "GLM-4.5", "Kimi K3"]):
    s.text(88, y + 37, name, 19, INK, anchor="end", weight="bold")

g = 8
# Kimi K2
y = ROWS[0]; w = (BW - g) / 2
stage(BX, y, w, "4K · 10T", "stable LR")
stage(BX + w + g, y, w, "4K · 5.5T", "cosine decay")
w = (TW - g) / 2
stage(TX, y, w, "4K · 400B", "anneal")
stage(TX + w + g, y, w, "32K · 60B", "long-context", long=True)
stage(PX, y, PW, "YaRN", "32K → 128K", pos=True)
# GLM-4.5
y = ROWS[1]; w = (BW - g) / 2
stage(BX, y, w, "4K · 15T", "general web")
stage(BX + w + g, y, w, "4K · 7T", "code & reasoning")
w = 108
stage(TX, y, w, "32K · 500B", "repo code", long=True)
stage(TX + w + g, y, w, "32K · 500B", "reasoning", long=True)
stage(TX + 2 * (w + g), y, TW - 2 * (w + g), "128K · 100B", "long + agent", long=True)
stage(PX, y, PW, "RoPE base", "10K → 1M", pos=True)
# Kimi K3
y = ROWS[2]; w = (BW - g) / 2
stage(BX, y, w, "8K", "pretraining")
stage(BX + w + g, y, w, "64K", "pretraining", long=True)
w = (TW - g) / 2
stage(TX, y, w, "256K", "cooldown", long=True)
stage(TX + w + g, y, w, "1M", "cooldown", long=True)
stage(PX, y, PW, "NoPE + KDA", "no retuning", pos=True)
s.save(os.path.join(HERE, "fig_ladder.svg"))
