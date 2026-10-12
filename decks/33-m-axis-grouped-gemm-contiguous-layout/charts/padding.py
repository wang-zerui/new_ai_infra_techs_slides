"""Analytic worst-case padding of the contiguous layout: (alignment - 1) / M extra rows per expert,
for alignment 128 (SM90 default) and 256 (SM100). Reference: capacity padding needed > 2x math (MegaBlocks §3).
See notes/sources.md."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

ms = [256, 512, 1024, 2048, 4096, 8192]
a128 = [100 * 127 / m for m in ms]
a256 = [100 * 255 / m for m in ms]

sc.setup()
fig, ax = sc.figure("right")
ax.axvspan(200, 300, color="#FBE5E5", zorder=0)
ax.text(236, 108, "decode", color=sc.PROBLEM, fontsize=18, ha="center", va="bottom")
ax.axvspan(3600, 9400, color="#DEEBF7", zorder=0)
ax.text(5800, 108, "prefill", color=sc.ACCENT, fontsize=18, ha="center", va="bottom")
ax.plot(ms, a256, marker="s", color=sc.ACCENT_LIGHT, label="align 256 (SM100)", markeredgecolor=sc.INK)
ax.plot(ms, a128, marker="o", color=sc.ACCENT, label="align 128 (SM90)", markeredgecolor=sc.INK)
ax.annotate("3.1%", xy=(4096, 3.1), xytext=(2600, 34), color=sc.ACCENT, fontsize=20, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=sc.ACCENT, lw=2.2), ha="center")
ax.set_xscale("log", base=2)
ax.set_xticks(ms)
ax.set_xticklabels(["256", "512", "1K", "2K", "4K", "8K"])
ax.set_xlim(200, 9400)
ax.set_ylim(0, 122)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_xlabel("Tokens per expert")
ax.set_ylabel("Worst-case padding (%)")
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.84), frameon=False)
ax.text(0.5, 0.97, "Lower is better", transform=ax.transAxes, ha="center", va="top", fontsize=16,
        fontstyle="italic", color=sc.GOOD)
sc.save(fig, __file__)
