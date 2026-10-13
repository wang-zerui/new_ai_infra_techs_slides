"""fig_hook: MAGI-1 packs two videos (A: 3 chunks, B: 2 chunks) into one sequence (online Patch-and-Pack);
block-causal inside each video, nothing across videos -> varlen block-causal mask ([MAGI1] Fig. 1, §4.1.1)."""
import os
from svgkit import SVG, NEU, NEU_F, INK, MUTED, ACC

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(450, 360, "fig_hook: packed sequence + varlen block-causal mask")

labels = ["A0", "A1", "A2", "B0", "B1"]
video = [0, 0, 0, 1, 1]
C = 52                      # chunk cell size
X0, Y0 = 70, 92             # grid origin
ON = "#BFBFBF"

# packed sequence strip (doubles as key-axis labels)
s.text(X0, 24, "packed sequence: video A + video B", 17, INK, anchor="start", weight="bold")
for i, l in enumerate(labels):
    x = X0 + i * C
    s.box(x + 2, 36, C - 4, 36, l, "#FFFFFF", NEU, INK, 17, "bold")
s.text(X0 + 5 * C + 10, 60, "keys", 16, MUTED, anchor="start", style="italic")

# mask grid
for r in range(5):
    for c in range(5):
        on = video[r] == video[c] and c <= r
        s.rect(X0 + c * C, Y0 + r * C, C, C, ON if on else "#FFFFFF", "#7F7F7F", 1.2, 0)
for r, l in enumerate(labels):
    s.text(X0 - 10, Y0 + r * C + C / 2 + 6, l, 17, INK, anchor="end", weight="bold")
s.add(f'<text x="22" y="{Y0 + 2.5 * C:.0f}" text-anchor="middle" font-size="16" fill="{MUTED}" '
      f'font-style="italic" transform="rotate(-90 22 {Y0 + 2.5 * C:.0f})">queries</text>')

# outline the two video blocks
s.rect(X0, Y0, 3 * C, 3 * C, "none", INK, 2.4, 0)
s.rect(X0 + 3 * C, Y0 + 3 * C, 2 * C, 2 * C, "none", INK, 2.4, 0)

# annotations inside the empty (masked) regions
s.text(X0 + 4 * C, Y0 + 1.5 * C - 12, "masked", 17, MUTED, weight="bold", halo=True)
s.text(X0 + 4 * C, Y0 + 1.5 * C + 8, "across", 17, MUTED, weight="bold", halo=True)
s.text(X0 + 4 * C, Y0 + 1.5 * C + 28, "videos", 17, MUTED, weight="bold", halo=True)
s.text(X0 + 1.5 * C, Y0 + 4 * C + 2, "block-causal", 17, INK, weight="bold", halo=True)
s.text(X0 + 1.5 * C, Y0 + 4 * C + 22, "inside a video", 17, INK, halo=True)
s.line(X0 + 1.5 * C, Y0 + 4 * C - 16, X0 + 1.5 * C, Y0 + 3 * C + 4, INK, 1.8, "arr")
GX = X0 + 5 * C + 14
s.text(GX, Y0 + 3 * C + 30, "videos of", 17, INK, anchor="start")
s.text(GX, Y0 + 3 * C + 50, "any length", 17, INK, anchor="start")
s.text(GX, Y0 + 3 * C + 70, "share a pack", 17, INK, anchor="start")
s.save(os.path.join(HERE, "fig_hook.svg"))
