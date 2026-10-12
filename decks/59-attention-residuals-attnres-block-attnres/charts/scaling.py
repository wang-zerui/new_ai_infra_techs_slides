"""Validation loss vs model size (AttnRes paper, Table 2): five sizes, baseline-tuned hyperparameters,
8192-token context. Block AttnRes with ~8 blocks."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sizes = ["194M", "241M", "296M", "436M", "528M"]
base = [1.931, 1.895, 1.829, 1.766, 1.719]
block = [1.909, 1.875, 1.809, 1.746, 1.693]
full = [1.899, 1.874, 1.804, 1.737, 1.692]

sc.setup()
fig, ax = sc.figure("right")
xs = list(range(len(sizes)))
ax.plot(xs, base, color=sc.GRAYS[2], marker="s", lw=3, label="PreNorm baseline")
ax.plot(xs, block, color=sc.ACCENT, marker="o", lw=3.4, label="Block AttnRes")
ax.plot(xs, full, color=sc.ACCENT_LIGHT, marker="^", lw=2.6, ls="--", label="Full AttnRes")
ax.set_xticks(xs)
ax.set_xticklabels(sizes)
ax.set_xlabel("Activated parameters")
ax.set_ylabel("Validation loss")
ax.set_ylim(1.66, 1.96)
sc.top_legend(ax, ncol=3, fontsize=16)
ax.set_xlim(-0.3, 4.6)
ax.annotate("", xy=(4.3, 1.692), xytext=(4.3, 1.720), arrowprops=dict(arrowstyle="<|-|>", color=sc.ACCENT, lw=2.2, shrinkA=0, shrinkB=0))
ax.plot([4.0, 4.3], [1.719, 1.719], color=sc.GRAYS[2], lw=1, ls=":")
ax.plot([4.0, 4.3], [1.693, 1.693], color=sc.ACCENT, lw=1, ls=":")
ax.text(3.05, 1.674, "−0.026 loss", color=sc.ACCENT, fontsize=19, fontweight="bold", ha="left", va="center")
sc.better(ax, "Lower is better", where="upper right")
sc.save(fig, __file__)
