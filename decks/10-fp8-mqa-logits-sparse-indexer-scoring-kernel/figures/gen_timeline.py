"""Slide 11: evolution of DeepGEMM's indexer-scoring kernels (README news; PRs #200, #304, #432, #462)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 262
s = SVG(W, H, "indexer kernel timeline")
ay = 126
s.line(10, ay, 862, ay, INK, 2.4, "arr")
items = [
    (112, "2025-09", ["SM90 FP8", "prefill + paged", "for V3.2 (H800)"], True),
    (330, "2026-04", ["FP4 indexer", "larger MTP support"], False),
    (548, "2026-09-10", ["Sparse Indexer", "hierarchical, V4.1", "determinism opt-in"], True),
    (758, "2026-09-30", ["Compressed logits", "SM100: MX in, BF16 out", "fp8_fp4_* API"], False),
]
for x, date, lines, up in items:
    s.add(f'<circle cx="{x}" cy="{ay}" r="7" fill="{ACC}"/>')
    s.text(x, ay + (-14 if not up else 30), date, 17, ACC, weight="bold")
    bw, bh = 196, 84
    by = ay - 24 - bh if up else ay + 40
    if not up:
        pass
    s.rect(x - bw / 2, by, bw, bh, ACC_F if up else "#FFFFFF", ACC, 2, 6)
    for i, t in enumerate(lines):
        s.text(x, by + 24 + i * 22, t, 17 if i == 0 else 15, INK, weight="bold" if i == 0 else "normal")
    s.line(x, ay - 8 if up else ay + 8, x, by + bh if up else by, ACC, 1.6)
s.save(os.path.join(HERE, "fig_timeline.svg"))
