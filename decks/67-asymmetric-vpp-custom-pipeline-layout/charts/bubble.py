"""Analytic pipeline-bubble fraction for the two DeepSeek-V3 recipes, with and without interleaving.

Model: PTD-P (arXiv 2104.04473, Sec. 2.2): bubble = (p - 1) / (v * m), balanced stages assumed.
Configs (arXiv 2603.07685, Table 17): H100 TP2 PP8, GB200 TP1 PP4; GBS 8192, MBS 1; 1,024 / 256 GPUs.
m = GBS / (MBS * DP); DP = GPUs / (TP * PP) = 64 for both -> m = 128. v = 4 is the VPP of the custom layouts.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

m = 128
cfg = [("H100 recipe\n(PP 8)", 8), ("GB200 recipe\n(PP 4)", 4)]
no_vpp = [100 * (p - 1) / m for _, p in cfg]
vpp4 = [100 * (p - 1) / (4 * m) for _, p in cfg]
print([round(x, 2) for x in no_vpp], [round(x, 2) for x in vpp4])

sc.setup()
fig, ax = sc.figure("right")
sc.grouped(ax, [c for c, _ in cfg], ["no VPP (v = 1)", "custom layout, VPP 4"], [no_vpp, vpp4], width=0.7)
w = 0.35
for g, (a, b) in enumerate(zip(no_vpp, vpp4)):
    ax.text(g - w / 2, a + 0.12, f"{a:.1f}%", ha="center", va="bottom", fontsize=20, color=sc.INK)
    ax.text(g + w / 2, b + 0.12, f"{b:.1f}%", ha="center", va="bottom", fontsize=20, color=sc.ACCENT, fontweight="bold")
ax.set_ylim(0, 7)
ax.set_ylabel("Analytic bubble\nfraction (%)")
sc.better(ax, "Lower is better", where="upper right")
sc.save(fig, __file__)
