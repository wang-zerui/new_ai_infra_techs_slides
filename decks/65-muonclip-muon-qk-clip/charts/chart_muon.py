"""chart_muon: training FLOPs to reach the same loss, relative to AdamW (compute-optimal scaling law).
Source: Moonlight, arXiv 2502.16982, abstract / §1 ("approximately 52% of the training FLOPs"; Fig. 1a "0.519x FLOPs")."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sc.setup()
fig, ax = sc.figure("half")
sc.bars(ax, ["AdamW", "Muon"], [1.0, 0.52], ours=1, labels=["1.00", "0.52"], width=0.56)
ax.set_ylabel("Training FLOPs for\nequal loss (× AdamW)")
ax.set_ylim(0, 1.25)
sc.better(ax, "Lower is better", where="upper right")
sc.save(fig, __file__)
