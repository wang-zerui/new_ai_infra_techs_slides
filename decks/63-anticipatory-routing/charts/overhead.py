"""Analytic: run-level wall-clock overhead of anticipatory routing vs the fraction of steps it is active.
DeepSeek-V4 §4.2.3 bounds the extra wall-clock time to about 20% while the mode is on, and turns it on only after a
detected loss spike. overhead(f) = 0.20 * f. The 5% point is an illustrative fraction, not a reported one."""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

f = np.linspace(0, 100, 200)
ov = 0.20 * f

sc.setup()
fig, ax = sc.figure("right")
ax.plot(f, ov, color=sc.ACCENT, label="≈20% × active fraction")
ax.plot([100], [20], "o", color=sc.PROBLEM, markersize=12, zorder=5)
ax.annotate("always on: ≈20%", xy=(100, 20), xytext=(66, 18.5), color=sc.PROBLEM, fontsize=18, fontweight="bold",
            ha="center", va="center")
ax.plot([5], [1], "o", color=sc.ACCENT, markersize=12, zorder=5)
ax.annotate("on after spikes only\n(e.g. 5% of steps: ≈1%)", xy=(5, 1), xytext=(30, 14.5), color=sc.ACCENT,
            fontsize=17, fontweight="bold", ha="center", va="center",
            arrowprops=dict(arrowstyle="-|>", color=sc.ACCENT, lw=2.2))
ax.set_xlabel("Steps with anticipatory routing on (%)")
ax.set_ylabel("Extra wall-clock time (%)")
ax.set_xlim(0, 104)
ax.set_ylim(0, 24)
sc.better(ax, "Lower is better", where="upper left")
sc.save(fig, __file__)
