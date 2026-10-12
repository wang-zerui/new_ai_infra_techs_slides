"""Decoding speed-up of Kimi Linear (3:1 KDA:MLA hybrid) over full-attention MLA vs decoding length.
Source: Kimi Linear (arXiv 2510.26692) Fig. 1b annotations: 4.8x @256K, 5.7x @512K, 6.3x @1M
(TPOT 1.84 ms vs 11.48 ms at 1M; the smaller cache allows larger batches)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc
lengths = ["256K", "512K", "1M"]
speedup = [4.8, 5.7, 6.3]
sc.setup()
fig, ax = sc.figure("right")
sc.bars(ax, lengths, speedup, ours=[0, 1, 2], labels=[f"{s:.1f}×" for s in speedup])
ax.axhline(1.0, color=sc.GRAYS[3], lw=2, ls="--", zorder=0)
ax.text(0.02, 0.86, "dashed: MLA = 1×", transform=ax.transAxes, ha="left", va="top", fontsize=16, color=sc.GRAYS[3])
ax.set_ylabel("TPOT speed-up over MLA (×)")
ax.set_xlabel("Decoding length (tokens)")
ax.set_ylim(0, 7.7)
sc.better(ax, "Higher is better", "upper left")
sc.save(fig, __file__)
