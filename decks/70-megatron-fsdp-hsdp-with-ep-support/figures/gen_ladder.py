"""fig_ladder_1/2: what each of 4 DP ranks stores under the four --data-parallel-sharding-strategy values.
Schematic (segment heights not to scale; bytes printed on the left)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
SEGS = [("optimizer", "12 B", 72), ("gradient", "4 B", 44), ("weight", "2 B", 44)]   # top -> bottom
COLS = [("DDP", "no_shard", set()),
        ("Dist. optimizer", "optim", {"optimizer"}),
        ("ZeRO-2", "optim_grads", {"optimizer", "gradient"}),
        ("FSDP (ZeRO-3)", "optim_grads_params", {"optimizer", "gradient", "weight"})]
LX, CW, BW, GAP = 98, 192, 30, 8
Y0 = 72


def column(s, k):
    title, flag, sharded = COLS[k]
    x0 = LX + k * CW
    cx = x0 + CW / 2
    new = k >= 2
    s.text(cx, 24, title, 18, ACC if new else INK, weight="bold")
    s.text(cx, 46, flag, 15, MUTED, family="Menlo, monospace")
    bx0 = cx - (4 * BW + 3 * GAP) / 2
    for r in range(4):
        x = bx0 + r * (BW + GAP)
        y = Y0
        for name, _, h in SEGS:
            if name in sharded:
                sh = h / 4
                for j in range(4):
                    own = j == r
                    s.rect(x, y + j * sh, BW, sh, ACC_F if own else "#FFFFFF", ACC if own else "#BFBFBF",
                           1.6 if own else 1, 0, dash=None if own else "3,2")
            else:
                s.rect(x, y, BW, h, PRB_F, PRB, 1.6, 0)
            y += h + 4
        s.text(x + BW / 2, Y0 + sum(h + 4 for _, _, h in SEGS) + 16, f"r{r}", 15, MUTED)


def frame(step):
    s = SVG(870, 300, f"fig_ladder_{step}: 4 DP ranks x sharding strategy")
    y = Y0
    for name, b, h in SEGS:
        s.text(LX - 10, y + h / 2 - 2, name, 16, INK, anchor="end", weight="bold")
        s.text(LX - 10, y + h / 2 + 17, b, 15, MUTED, anchor="end")
        y += h + 4
    for k in range(2 if step == 1 else 4):
        column(s, k)
    # legend
    ly = 286
    s.rect(150, ly - 14, 22, 16, PRB_F, PRB, 1.6, 0)
    s.text(180, ly, "replicated on every rank", 16, INK, anchor="start")
    s.rect(420, ly - 14, 22, 16, ACC_F, ACC, 1.6, 0)
    s.text(450, ly, "this rank's shard", 16, INK, anchor="start")
    s.rect(620, ly - 14, 22, 16, "#FFFFFF", "#BFBFBF", 1, 0, dash="3,2")
    s.text(650, ly, "held by other ranks", 16, INK, anchor="start")
    s.save(os.path.join(HERE, f"fig_ladder_{step}.svg"))



frame(2)

# narrow two-column version for the status-quo slide
s = SVG(486, 330, "fig_ladder_0: DDP vs distributed optimizer on 4 ranks")
y = Y0
for name, b, h in SEGS:
    s.text(LX - 10, y + h / 2 - 2, name, 16, INK, anchor="end", weight="bold")
    s.text(LX - 10, y + h / 2 + 17, b, 15, MUTED, anchor="end")
    y += h + 4
for k in range(2):
    column(s, k)
s.rect(20, 274, 22, 16, PRB_F, PRB, 1.6, 0)
s.text(50, 288, "replicated on every rank", 16, INK, anchor="start")
s.rect(270, 274, 22, 16, ACC_F, ACC, 1.6, 0)
s.text(300, 288, "this rank's shard", 16, INK, anchor="start")
s.rect(20, 302, 22, 16, "#FFFFFF", "#BFBFBF", 1, 0, dash="3,2")
s.text(50, 316, "held by other ranks", 16, INK, anchor="start")
s.save(os.path.join(HERE, "fig_ladder_0.svg"))
