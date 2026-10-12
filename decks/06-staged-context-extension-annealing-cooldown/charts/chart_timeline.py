"""chart_timeline: training window vs cumulative training tokens (step chart), Kimi K2 and GLM-4.5.
Sources (notes/sources.md): K2 §2.5 p.9: 15.5T @4K (10T stable + 5.5T cosine), anneal 400B @4K, 60B @32K,
YaRN to 128K (no training tokens, dotted). GLM-4.5 Fig. 3 p.4: 15T @4K, 7T @4K, 500B @32K, 500B @32K, 100B @128K.
K3 publishes stage windows but no stage budgets, so it is not plotted."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

# (end_T, window_K) piecewise-constant segments
k2 = [(0, 4), (15.9, 4), (15.9, 32), (15.96, 32)]
glm = [(0, 4), (22.0, 4), (22.0, 32), (23.0, 32), (23.0, 128), (23.1, 128)]

sc.setup()
fig, ax = sc.figure("right")
gx, gy = zip(*glm); kx, ky = zip(*k2)
ax.plot(gx, gy, color=sc.GRAYS[1], lw=8, label="GLM-4.5 (23.1T)", solid_capstyle="butt")
ax.plot(kx, ky, color=sc.ACCENT, lw=3.4, label="Kimi K2 (15.96T)", solid_capstyle="butt")
ax.plot([15.96, 15.96], [32, 128], color=sc.ACCENT, lw=2.6, ls=(0, (1.2, 1.6)))
ax.set_yscale("log", base=2)
ax.set_yticks([4, 32, 128]); ax.set_yticklabels(["4K", "32K", "128K"]); ax.minorticks_off()
ax.set_ylim(2.6, 400); ax.set_xlim(0, 24.5)
ax.set_xlabel("Training tokens (T)")
ax.set_ylabel("Training window")
ax.text(15.6, 150, "YaRN\nonly", color=sc.ACCENT, fontsize=18, ha="right", va="bottom")
ax.annotate("60B at 32K", xy=(15.93, 32), xytext=(10.2, 46), color=sc.ACCENT, fontsize=18, fontweight="bold",
            ha="center", va="center", arrowprops=dict(arrowstyle="-|>", color=sc.ACCENT, lw=2, shrinkA=2, shrinkB=3))
ax.annotate("100B at 128K", xy=(23.05, 128), xytext=(19.6, 280), color=sc.GRAYS[3], fontsize=18, fontweight="bold",
            ha="center", va="center", arrowprops=dict(arrowstyle="-|>", color=sc.GRAYS[3], lw=2, shrinkA=2, shrinkB=3))
ax.annotate("1T at 32K", xy=(22.5, 32), xytext=(19.3, 10.5), color=sc.GRAYS[3], fontsize=18,
            ha="center", va="center", arrowprops=dict(arrowstyle="-|>", color=sc.GRAYS[3], lw=2, shrinkA=2, shrinkB=3))
sc.top_legend(ax, ncol=2)
sc.save(fig, __file__)
