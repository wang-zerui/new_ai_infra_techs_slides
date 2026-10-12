"""Accumulated KV cache at 1M context, % of DeepSeek-V3.2 ([V4] Abstract, §1, Fig. 1)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sc.setup()
fig, ax = sc.figure("right")
sc.bars(ax, ["V3.2", "V4-Pro", "V4-Flash"], [100, 10, 7], ours=[1, 2], labels=["100%", "10%", "7%"])
ax.set_ylabel("KV cache at 1M\n(% of V3.2)")
ax.set_ylim(0, 125)
sc.better(ax, "Lower is better", "upper right")
sc.callout(ax, "9.5× smaller", xy=(1.34, 8), xytext=(1.6, 55))
sc.save(fig, __file__)
