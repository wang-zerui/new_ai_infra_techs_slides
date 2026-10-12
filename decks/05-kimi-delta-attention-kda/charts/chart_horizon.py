"""Analytic illustration: per-channel retention alpha^distance for three illustrative constant log-decays g
(alpha = exp(g)), Shows that channels with different decay rates keep
different memory horizons (the RoPE-like multi-scale role, Kimi Linear Sec. 6.1). Values are illustrative, not measured."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import numpy as np
import slidechart as sc
sc.setup()
fig, ax = sc.figure("right")
d = np.logspace(0, 5, 400)
lines = [(-0.5, "fast", "#5B9BD5", (4, 0.30), "left"), (-0.01, "medium", "#2E75B6", (180, 0.30), "left"),
         (-0.0002, "slow", sc.ACCENT, (9000, 0.30), "left")]
for g, name, col, pos, ha in lines:
    y = np.exp(g * d)
    ax.plot(d, y, color=col, lw=3.5)
    ax.text(pos[0], pos[1], name, fontsize=18, color=col, fontweight="bold", ha=ha, va="bottom")
ax.set_xscale("log")
ax.set_xlim(1, 1e5)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Distance from current token (tokens)")
ax.set_ylabel("Retained weight α^Δ")
sc.save(fig, __file__)
