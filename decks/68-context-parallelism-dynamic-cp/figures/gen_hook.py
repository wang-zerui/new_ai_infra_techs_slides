"""fig_hook: token-local ops scale with S, causal attention scales with S^2 (8 chunks shown)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 352, "fig_hook: strip of 8 token chunks (linear ops) vs 8x8 causal triangle (attention)")

N, C = 8, 26          # chunks, cell size
X0 = 52

# token-local ops: one cell per chunk
s.text(X0, 22, "Linear / LayerNorm / MLP", 19, INK, anchor="start", weight="bold")
for i in range(N):
    s.rect(X0 + i * C, 34, C, 30, NEU_F, NEU, 1.5, 0)
s.text(X0 + N * C + 18, 45, "work grows", 18, INK, anchor="start")
s.text(X0 + N * C + 18, 67, "with S", 18, INK, anchor="start")

# attention: causal triangle
YG = 128
s.text(X0, 100, "Attention (causal)", 19, INK, anchor="start", weight="bold")
s.text(X0 + N * C / 2, YG - 8, "keys and values", 16, MUTED, style="italic")
s.add(f'<text x="{X0 - 12}" y="{YG + N * C / 2}" text-anchor="middle" font-size="16" fill="{MUTED}" '
      f'font-style="italic" transform="rotate(-90 {X0 - 12} {YG + N * C / 2})">queries</text>')
for r in range(N):
    for c in range(N):
        x, y = X0 + c * C, YG + r * C
        if c < r:
            s.rect(x, y, C, C, PRB_F, PRB, 1.2, 0)
        elif c == r:
            s.rect(x, y, C, C, "#FFFFFF", "#D9D9D9", 1, 0)
            s.add(f'<path d="M{x:.1f},{y:.1f} L{x:.1f},{y + C:.1f} L{x + C:.1f},{y + C:.1f} z" '
                  f'fill="{PRB_F}" stroke="{PRB}" stroke-width="1.2"/>')
        else:
            s.rect(x, y, C, C, "#FFFFFF", "#D9D9D9", 1, 0)
s.text(X0 + N * C + 18, YG + N * C / 2 - 4, "work grows", 18, PRB, anchor="start", weight="bold")
s.text(X0 + N * C + 18, YG + N * C / 2 + 20, "with S²", 18, PRB, anchor="start", weight="bold")
s.save(os.path.join(HERE, "fig_hook.svg"))
