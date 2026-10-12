"""Derived: GLM-5 weight memory under three precision plans, against one Atlas 800 A3 node.

744e9 parameters (GLM-5 report Table 10); routed experts 724.8e9 of them (derivation in
notes/sources.md). BF16 = 2 B/param, INT8 = 1 B/param, hybrid = experts at 0.5 B + the rest at 1 B.
Node capacity "128GB x 8" from the vLLM-Ascend GLM-5 deployment guide. GB = 1e9 bytes.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

P, E = 744e9, 724.8e9
bf16 = P * 2 / 1e9
int8 = P / 1e9
hybrid = (E * 0.5 + (P - E)) / 1e9
NODE = 1024

sc.setup()
fig, ax = sc.figure("right")
names = ["BF16", "W8A8\neverywhere", "W4A8 experts\n+ W8A8 rest"]
vals = [bf16, int8, hybrid]
sc.bars(ax, names, vals, ours=2, problem=0,
        labels=[f"{v:,.0f} GB" for v in vals])
ax.axhline(NODE, color=sc.INK, linestyle="--", linewidth=2.0)
ax.text(2.46, NODE + 30, "one Atlas 800 A3 node", ha="right", va="bottom", fontsize=17, color=sc.INK)
ax.set_ylabel("Weight memory (GB)")
ax.set_ylim(0, 1850)
sc.better(ax, "Lower is better", "upper left")
sc.save(fig, __file__)
