"""DeepSeek-V3 per-GPU memory, BF16 vs FP8-cached linear-layer inputs (Megatron-Core MoE report, arXiv 2603.07685,
Table 3 and Sec. 4.1.3). Derived split: 131.0 GB activations = ~32 GB FP8-eligible linear inputs + 99 GB other;
FP8 halves the eligible part to ~16 GB. See notes/sources.md."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

cols = ["BF16", "FP8 caching\n(derived)"]
params = [68.5, 68.5]      # weights & grads 36.4 + main weights & optimizer 32.1
other = [99.0, 99.0]       # activations not eligible for FP8 storage (131 - 32)
lin = [32.0, 16.0]         # linear-layer inputs saved for backward: BF16 vs FP8

sc.setup()
fig, ax = sc.plt.subplots(figsize=(9.6, 3.75))
x = [0, 1]
w = 0.5
ax.bar(x, params, w, color=sc.GRAYS[2], edgecolor=sc.INK, linewidth=1.2, label="Weights, grads, optimizer")
ax.bar(x, other, w, bottom=params, color=sc.GRAYS[0], edgecolor=sc.INK, linewidth=1.2, hatch="//",
       label="Other activations")
b = [p + o for p, o in zip(params, other)]
ax.bar([0], [lin[0]], w, bottom=[b[0]], color=sc.GRAYS[1], edgecolor=sc.INK, linewidth=1.2, hatch="..",
       label="Linear inputs, BF16")
ax.bar([1], [lin[1]], w, bottom=[b[1]], color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2,
       label="Linear inputs, FP8")
top1 = b[1] + lin[1]
ax.bar([1], [16], w, bottom=[top1], color="white", edgecolor=sc.ACCENT, linewidth=1.8, linestyle="--")
ax.text(1.29, top1 + 8, "−16 GB", ha="left", va="center", fontsize=20, fontweight="bold", color=sc.ACCENT)
ax.text(0, 199.5 + 5, "199.5 GB", ha="center", va="bottom", fontsize=20, fontweight="bold", color=sc.INK)
ax.text(1, top1 + 21, "183.5 GB", ha="center", va="bottom", fontsize=20, fontweight="bold", color=sc.ACCENT)
ax.axhline(80, color=sc.PROBLEM, lw=2.4, ls="--", label="H100 memory: 80 GB")
ax.set_xticks(x); ax.set_xticklabels(cols)
ax.set_xlim(-0.6, 1.8)
ax.set_ylim(0, 245)
ax.set_ylabel("Memory per GPU (GB)")
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=18, handlelength=1.4)
sc.better(ax, "Lower is better", where="upper right")
sc.save(fig, __file__)
