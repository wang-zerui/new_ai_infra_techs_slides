"""fig_zigzag_1/2: causal attention over 8 chunks on CP=4; contiguous split (a) vs zigzag split (b).

Work per GPU in chunk-block units (diagonal block = 1/2): contiguous 2/6/10/14, zigzag 8/8/8/8 (analytic, see
notes/sources.md).
"""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
TINT = ["#EAF2FA", "#C5DCF0", "#93BDE3", "#5B93C9"]   # GPU0..GPU3
N, C = 8, 22


def panel(s, x0, title, owner, problem):
    gx, gy = x0 + 72, 64
    s.text(x0 + 4, 24, title, 20, INK, anchor="start", weight="bold")
    s.text(gx + N * C / 2, gy - 8, "keys and values", 15, MUTED, style="italic")
    for r in range(N):
        g = owner[r]
        s.text(gx - 8, gy + r * C + C / 2 + 5.5, f"chunk {r}", 15, INK, anchor="end")
        for c in range(N):
            x, y = gx + c * C, gy + r * C
            if c < r:
                s.rect(x, y, C, C, TINT[g], NEU, 1, 0)
            else:
                s.rect(x, y, C, C, "#FFFFFF", "#D9D9D9", 1, 0)
                if c == r:
                    s.add(f'<path d="M{x:.1f},{y:.1f} L{x:.1f},{y + C:.1f} L{x + C:.1f},{y + C:.1f} z" '
                          f'fill="{TINT[g]}" stroke="{NEU}" stroke-width="1"/>')
    # work per GPU
    work = [0.0] * 4
    for r in range(N):
        work[owner[r]] += r + 0.5
    bx, by = gx + N * C + 20, gy + 10
    s.text(bx, gy - 8, "work per GPU", 15, MUTED, anchor="start", style="italic")
    for g in range(4):
        y = by + g * 40
        s.text(bx, y + 17, f"GPU{g}", 16, INK, anchor="start", weight="bold")
        w = work[g] * 5.0
        hot = problem and work[g] == max(work)
        s.rect(bx + 48, y, w, 24, PRB_F if hot else TINT[g], PRB if hot else (ACC if not problem else NEU), 1.6, 2)
        s.text(bx + 48 + w + 6, y + 18, f"{work[g]:g}", 16, PRB if hot else INK, anchor="start",
               weight="bold" if hot else "normal")


def legend(s):
    x = 300
    for g in range(4):
        s.rect(x + g * 110, 262, 22, 18, TINT[g], NEU, 1.5, 2)
        s.text(x + g * 110 + 28, 277, f"GPU{g}", 16, INK, anchor="start")


contig = [0, 0, 1, 1, 2, 2, 3, 3]
zigzag = [0, 1, 2, 3, 3, 2, 1, 0]
for step in (1, 2):
    s = SVG(870, 290, f"fig_zigzag_{step}: (a) contiguous split" + (" vs (b) zigzag split" if step == 2 else ""))
    panel(s, 0, "(a) Contiguous split", contig, True)
    if step == 2:
        s.line(437, 12, 437, 252, "#BFBFBF", 1.2)
        panel(s, 450, "(b) Zigzag split", zigzag, False)
    legend(s)
    s.save(os.path.join(HERE, f"fig_zigzag_{step}.svg"))
