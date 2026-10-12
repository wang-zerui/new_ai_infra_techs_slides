"""(a) Memory saved per GPU by fine-grained recomputation, DeepSeek-V3, PP4 x VPP4 x EP64, 256 GPUs
(arXiv 2603.07685 Table 4, p.24). (b) Extra compute: full-layer recompute ~33% (Sec. 4.1.4, p.22) vs
fine-grained "less than 5%" (p.23) -- drawn as an upper-bound bar at 5."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

sc.setup()
fig, (a, b) = sc.plt.subplots(1, 2, figsize=(11.6, 3.9), gridspec_kw={"width_ratios": [1.75, 1], "wspace": 0.42})
names = ["MLA up-proj", "LayerNorm", "SwiGLU", "Total"]
vals = [30.4, 8.2, 3.8, 42.4]
cols = [sc.ACCENT_LIGHT, sc.ACCENT_LIGHT, sc.ACCENT_LIGHT, sc.ACCENT]
ys = list(range(len(names)))[::-1]
a.barh(ys, vals, height=0.62, color=cols, edgecolor=sc.INK, linewidth=1.2)
for y, v in zip(ys, vals):
    a.text(v + 0.8, y, f"{v:.1f}", va="center", ha="left", fontsize=19,
           fontweight="bold" if v == 42.4 else "normal", color=sc.ACCENT if v == 42.4 else sc.INK)
a.set_yticks(ys); a.set_yticklabels(names)
a.set_xlim(0, 60)
a.set_xlabel("Memory saved per GPU (GB)")
a.grid(axis="x", linestyle="--", color="#BFBFBF"); a.grid(axis="y", visible=False)
a.text(0.98, 0.96, "Higher is better", transform=a.transAxes, ha="right", va="top", fontsize=16,
       fontstyle="italic", color=sc.GOOD)

bn = ["Full layer", "Fine-grained"]
bv = [33, 5]
rects = b.bar([0, 1], bv, width=0.6, color=[sc.GRAYS[1], sc.ACCENT], edgecolor=sc.INK, linewidth=1.2)
rects[0].set_hatch("//")
b.text(0, 34.5, "~33%", ha="center", va="bottom", fontsize=20, fontweight="bold", color=sc.PROBLEM)
b.text(1, 6.5, "<5%", ha="center", va="bottom", fontsize=20, fontweight="bold", color=sc.ACCENT)
b.set_xticks([0, 1]); b.set_xticklabels(bn)
b.set_ylim(0, 42)
b.set_ylabel("Extra compute (%)")
b.text(0.97, 0.96, "Lower is better", transform=b.transAxes, ha="right", va="top", fontsize=16,
       fontstyle="italic", color=sc.GOOD)
sc.save(fig, __file__)
