"""Single-token inference FLOPs at 1M context, % of DeepSeek-V3.2 ([V4] Abstract, §1, Fig. 1)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sc.setup()
fig, ax = sc.figure("right")
names = ["V3.2", "V4-Pro", "V4-Flash"]
vals = [100, 27, 10]            # V4-Pro 27%, V4-Flash 10% of V3.2 (3.7x / 9.8x lower)
sc.bars(ax, names, vals, ours=[1, 2], labels=["100%", "27%", "10%"])
ax.set_ylabel("FLOPs per token at 1M\n(% of V3.2)")
ax.set_ylim(0, 125)
sc.better(ax, "Lower is better", "upper right")
sc.callout(ax, "3.7× lower", xy=(1.34, 24), xytext=(1.62, 62))
sc.save(fig, __file__)
