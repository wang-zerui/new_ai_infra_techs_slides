"""Analytic: values communicated per MoE layer per step to compute the per-expert quantile.

Raw margins: m * n (one margin per token per expert); histogram: n * B integer counts, independent of m.
n = 896 routed experts ([K3] Table 1); B = 1000 bins (the value K3 App. D uses for its error bound; the main
text says "a few hundred", so 1000 is an upper bound). Shaded region: "margins number in the millions"
([K3] Sec. 2.3.3). No measured traffic is plotted.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc
import numpy as np

n, B = 896, 1000
m = np.logspace(3, 8, 200)
sc.setup()
fig, ax = sc.figure("right")
ax.axvspan(1e6, 1e8, color="#F2F2F2", zorder=0)
ax.text(1.2e6, 6e11, "millions of\ntokens per step", fontsize=17, color="#595959", va="top", style="italic")
ax.loglog(m, m * n, color=sc.PROBLEM, lw=3.2, label="gather raw margins (m·n)")
ax.loglog(m, np.full_like(m, n * B), color=sc.ACCENT, lw=3.6, label="histogram (n·B)")
ax.set_xlim(1e3, 1e8)
ax.set_ylim(1e5, 1e12)
ax.set_xlabel("Tokens per global step (m)")
ax.set_ylabel("Values per layer per step")
ax.grid(True, which="major", axis="both", linestyle="--", color="#BFBFBF", linewidth=0.8)
ax.minorticks_off()
ax.annotate("", xy=(1e6, 1e6 * n * 0.85), xytext=(1e6, n * B * 1.2),
            arrowprops=dict(arrowstyle="<|-|>", color=sc.ACCENT, lw=2.4))
ax.text(1.25e6, 2.2e7, "1000× fewer\nat m = 10⁶", color=sc.ACCENT, fontsize=19, fontweight="bold", va="center")
sc.top_legend(ax, ncol=2, fontsize=17)
sc.better(ax, "Lower is better", where="upper left")
sc.save(fig, __file__)
