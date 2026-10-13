"""fig_slices: the hook's packed mask (video A: 3 chunks, video B: 2 chunks) written as 5 AttnSlices
(QRange, KRange, MaskType); every slice here is FULL because block-causal = full blocks per chunk row
([BLOG] AttnSlice Representation; [QS] block-causal example uses FULL slices)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 300, "fig_slices: mask -> AttnSlice list")
labels = ["A0", "A1", "A2", "B0", "B1"]
video = [0, 0, 0, 1, 1]
C = 44
X0, Y0 = 50, 70
s.text(X0 + 2.5 * C, 24, "packed mask", 18, INK, weight="bold")
for i, l in enumerate(labels):
    s.text(X0 + i * C + C / 2, Y0 - 10, l, 15, MUTED)
    s.text(X0 - 8, Y0 + i * C + C / 2 + 5, l, 15, MUTED, anchor="end")
for r in range(5):
    for c in range(5):
        s.rect(X0 + c * C, Y0 + r * C, C, C, "#FFFFFF", "#BFBFBF", 1, 0)
# slices = one per chunk row: Q = row chunk, K = first chunk of the video .. this chunk
slices = []
for r in range(5):
    k0 = video.index(video[r])
    slices.append((r, k0, r))
for i, (r, k0, k1) in enumerate(slices):
    x = X0 + k0 * C
    w = (k1 - k0 + 1) * C
    s.rect(x + 2, Y0 + r * C + 2, w - 4, C - 4, ACC_F, ACC, 2.2, 3)
    s.add(f'<circle cx="{x + 16:.1f}" cy="{Y0 + r * C + C / 2:.1f}" r="11" fill="{ACC}"/>')
    s.text(x + 16, Y0 + r * C + C / 2 + 6, str(i + 1), 15, "#FFFFFF", weight="bold")

# slice table
TX = 300
s.text(TX + 85, 24, "AttnSlice list", 18, ACC, weight="bold")
s.text(TX + 18, 62, "Q", 16, MUTED, weight="bold")
s.text(TX + 80, 62, "K", 16, MUTED, weight="bold")
s.text(TX + 148, 62, "type", 16, MUTED, weight="bold")
for i, (r, k0, k1) in enumerate(slices):
    y = Y0 + r * C + C / 2 + 6
    kr = labels[k0] if k0 == k1 else f"{labels[k0]}-{labels[k1]}"
    s.add(f'<circle cx="{TX - 14}" cy="{y - 6:.1f}" r="11" fill="{ACC}"/>')
    s.text(TX - 14, y, str(i + 1), 15, "#FFFFFF", weight="bold")
    s.text(TX + 18, y, labels[r], 17, INK, family="Menlo, monospace")
    s.text(TX + 80, y, kr, 17, INK, family="Menlo, monospace")
    s.text(TX + 148, y, "FULL", 17, INK, family="Menlo, monospace")
s.save(os.path.join(HERE, "fig_slices.svg"))
