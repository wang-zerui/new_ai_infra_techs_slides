"""Mega MoE (wave-based fused kernel) vs the non-fused legacy path, EP8, per batch size (tokens per rank).
Source: DeepGEMM PR #316 benchmark tables (V4-Flash and V4-Pro), averaged over 8 ranks."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

batches = ["1", "512", "8192", "32768"]
flash = [1.96, 1.73, 1.56, 1.62]
pro = [1.61, 1.54, 1.50, 1.54]

sc.setup()
fig, ax = sc.figure("right")
w = 0.38
xs = range(len(batches))
b1 = ax.bar([x - w / 2 for x in xs], flash, w, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2, label="V4-Flash")
b2 = ax.bar([x + w / 2 for x in xs], pro, w, color=sc.ACCENT_LIGHT, edgecolor=sc.INK, linewidth=1.2, label="V4-Pro")
for bars, vals, col in ((b1, flash, sc.ACCENT), (b2, pro, sc.INK)):
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=16,
                fontweight="bold", color=col)
ax.axhline(1.0, color=sc.GRAYS[2], linestyle="--", linewidth=2, label="non-fused")
ax.set_xticks(list(xs))
ax.set_xticklabels(batches)
ax.set_xlabel("Tokens per rank (EP8)")
ax.set_ylabel("Speedup over non-fused (×)")
ax.set_ylim(0, 2.45)
ax.set_xlim(-0.6, 3.6)
sc.top_legend(ax, ncol=3)
sc.better(ax, where="upper right")
sc.save(fig, __file__)
