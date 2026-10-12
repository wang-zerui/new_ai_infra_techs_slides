"""fig_releases: Mega MoE release timeline in DeepGEMM (dates and what changed; numbers live on the slide)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

W, H = 470, 350
s = SVG(W, H, "Mega MoE releases")

items = [
    ("Apr 2026 · PR #304", "Mega MoE released, FP8 × FP4 only"),
    ("Jun 2026 · PR #364", "BF16 path, ring buffers cut memory"),
    ("Jul 2026 · PR #377", "shared expert fused, L1/L2 interleaved"),
    ("Sep 2026 · PR #432", "V4.1 tuning, determinism opt-in"),
    ("Sep 30 2026 · PR #462", "locality-aware scheduling"),
]
x0, y0, dy = 26, 26, 72
s.line(x0, y0 - 10, x0, y0 + dy * (len(items) - 1) + 10, ACC, 3)
for i, (date, what) in enumerate(items):
    y = y0 + i * dy
    s.add(f'<circle cx="{x0}" cy="{y}" r="9" fill="{ACC}" stroke="#FFFFFF" stroke-width="2"/>')
    s.text(x0 + 22, y + 6, date, 18, ACC, anchor="start", weight="bold")
    s.text(x0 + 22, y + 30, what, 17, INK, anchor="start")

s.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_releases.svg"))
