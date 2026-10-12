"""Pangu Ultra MoE, Table 5: cumulative MFU increase over the baseline, 718B MoE on 6K Ascend NPUs.

Measured numbers from arXiv 2505.04519 Table 5 (p.16). The last step is the fused-operator step;
58.7 - 49.2 = 9.5 points is derived. Different model and training, not GLM-5 inference.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

names = ["+ recompute\n& swap", "+ pipe\noverlap", "+ host\noptimization", "+ fused\noperators"]
values = [15.8, 28.6, 49.2, 58.7]

sc.setup()
fig, ax = sc.figure("right")
sc.bars(ax, names, values, ours=3, labels=[f"{v}%" for v in values])
ax.set_ylabel("Cumulative MFU gain (%)")
ax.set_ylim(0, 78)
sc.better(ax, "Higher is better", "upper left")
sc.callout(ax, "+9.5 pts\n(derived)", xy=(3, 54.0), xytext=(1.75, 70.0))
sc.save(fig, __file__)
