"""Fine-grained activation offloading, Megatron-Core MoE report (arXiv 2603.07685) Table 5 (p.25):
DeepSeek-V3 full, TP1 PP8 EP32 VPP4, MXFP8: 169 -> 151 GB (-10.7%), 945 -> 930 TF/s (-1.6%).
Qwen3-235B, TP2 -> TP1 and EP16 -> EP64 with offload: 172 -> 175 GB (+1.7%), 800 -> 920 TF/s (+15.0%)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

groups = ["DeepSeek-V3", "Qwen3-235B*"]
mem = [[169, 172], [151, 175]]
tf = [[945, 800], [930, 920]]
mem_d = ["−10.7%", "+1.7%"]
tf_d = ["−1.6%", "+15.0%"]
sc.setup()
fig, (a, b) = sc.plt.subplots(1, 2, figsize=(11.6, 4.2), gridspec_kw={"wspace": 0.32})
w = 0.36
for ax, vals, deltas, ylab, top in [(a, mem, mem_d, "Memory per GPU (GB)", 230), (b, tf, tf_d, "Throughput (TF/s per GPU)", 1250)]:
    xs = [0, 1]
    r0 = ax.bar([x - w / 2 for x in xs], vals[0], w, color=sc.GRAYS[0], edgecolor=sc.INK, linewidth=1.2, hatch="//",
                label="Baseline")
    r1 = ax.bar([x + w / 2 for x in xs], vals[1], w, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2, label="+ Offload")
    for r, v in zip(r0, vals[0]):
        ax.text(r.get_x() + r.get_width() / 2, v + top * 0.015, f"{v}", ha="center", va="bottom", fontsize=17)
    for r, v, d in zip(r1, vals[1], deltas):
        ax.text(r.get_x() + r.get_width() / 2, v + top * 0.015, f"{v}", ha="center", va="bottom", fontsize=17,
                fontweight="bold", color=sc.ACCENT)
        ax.text(r.get_x() + r.get_width() / 2, v - top * 0.03, d, ha="center", va="top", fontsize=17,
                fontweight="bold", color="white", rotation=90)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, top)
    ax.set_ylabel(ylab)
    ax.set_xlim(-0.6, 1.6)
a.text(0.02, 0.97, "Lower is better", transform=a.transAxes, ha="left", va="top", fontsize=16, fontstyle="italic",
       color=sc.GOOD)
b.text(0.02, 0.97, "Higher is better", transform=b.transAxes, ha="left", va="top", fontsize=16, fontstyle="italic",
       color=sc.GOOD)
handles, labels = a.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
sc.save(fig, __file__)
