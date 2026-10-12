"""fig_timeline: how DeepGEMM's K-grouped Wgrad evolved (from the repo history): a host loop over experts (May 2025),
one SM100 kernel on token-major operands (Jul 2025), one SM90 kernel with per-expert TMA descriptors (Nov 2025),
and the psum layout with K offsets on the GPU (Sep 2026)."""
import os
from tilekit import SVG, INK, MUTED, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 232, "fig_timeline: K-grouped Wgrad in DeepGEMM, 2025-2026")
ms = [("May 2025", ["per-expert loop", "SM90"], ["host K list", "G launches"], False),
      ("Jul 2025", ["one kernel", "SM100"], ["token-major", "operands (TN)"], True),
      ("Nov 2025", ["one kernel", "SM90"], ["TMA descriptor", "per expert"], True),
      ("Sep 2026", ["psum layout", "SM100"], ["K offsets", "on the GPU"], True)]
W, G, X0 = 196, 18, 8
s.line(X0, 52, 862, 52, INK, 2.2, "arr")
for i, (date, title, sub, new) in enumerate(ms):
    x = X0 + i * (W + G)
    s.add(f'<circle cx="{x + W / 2:.1f}" cy="52" r="6" fill="{ACC if new else PRB}"/>')
    s.text(x + W / 2, 34, date, 18, INK, weight="bold")
    s.box(x, 72, W, 62, title, ACC_F if new else PRB_F, ACC if new else PRB, ACC if new else PRB, 17, "bold")
    s.box(x, 144, W, 62, sub, NEU_F, NEU, INK, 16)
s.text(435, 226, "PRs #95 · #112 · #231 · #432", 15, MUTED)
s.save(os.path.join(HERE, "fig_timeline.svg"))
