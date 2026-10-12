"""Analytic: MoE-layer time vs interconnect bandwidth, from DeepSeek-V4 §3.1 (C/B <= V_comp/V_comm).
x = B / B*, where B* = C * V_comm / V_comp is the balance point (B* = C / 6144 for V4-Pro).
Serial time / compute time = 1 + 1/x; perfectly overlapped = max(1, 1/x). No measured data."""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

x = np.linspace(0.4, 3.0, 300)
serial = 1 + 1 / x
overlap = np.maximum(1, 1 / x)

sc.setup()
fig, ax = sc.figure("right")
ax.plot(x, serial, color=sc.GRAYS[2], linestyle="--", label="serial")
ax.plot(x, overlap, color=sc.ACCENT, label="fully overlapped")
ax.axvline(1.0, color=sc.INK, linewidth=1.4, linestyle=":")
ax.annotate("balance point\nB = C / 2d", xy=(1.0, 1.0), xytext=(1.75, 2.35), color=sc.ACCENT, fontsize=18,
            fontweight="bold", ha="center", va="center",
            arrowprops=dict(arrowstyle="-|>", color=sc.ACCENT, lw=2.2))
ax.text(2.95, 1.07, "flat: extra bandwidth buys nothing", ha="right", va="bottom", fontsize=16,
        color=sc.ACCENT, style="italic")
ax.set_xlabel("Interconnect bandwidth / balance point")
ax.set_ylabel("Layer time / compute time")
ax.set_ylim(0.8, 3.6)
ax.set_xlim(0.4, 3.0)
ax.grid(axis="x", visible=False)
sc.top_legend(ax, ncol=2)
sc.better(ax, "Lower is better", where="upper right", color=sc.GOOD)
sc.save(fig, __file__)
