"""LongBench-V2 (EM, 1-shot) of the base models ([V4] Table 1)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sc.setup()
fig, ax = sc.figure("right")
names = ["V3.2-Base", "V4-Flash-Base", "V4-Pro-Base"]
vals = [40.2, 44.7, 51.5]
sc.bars(ax, names, vals, ours=[1, 2], labels=["40.2", "44.7", "51.5"])
ax.set_ylabel("LongBench-V2 (EM, %)")
ax.set_ylim(0, 65)
sc.better(ax, "Higher is better", "upper left")
sc.save(fig, __file__)
