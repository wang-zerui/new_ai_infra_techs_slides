"""Per-GPU memory, Qwen2-7B text model, FSDP, 128 GPUs, micro-batch 1, global batch 128, full activation checkpointing.
Source: VeOmni arXiv 2508.02317 App. A Table 1 (p.15). TorchTitan OOMs at 64K and 128K.
2.3x at 32K is derived: 71.81 / 31.41 = 2.29."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

seq = ["8K", "16K", "32K", "64K", "128K"]
titan = [23.26, 39.58, 71.81, None, None]
veomni = [21.91, 24.45, 31.41, 43.00, 70.26]

sc.setup()
fig, ax = sc.figure("right")
w = 0.38
xs = range(len(seq))
for x, v in zip(xs, titan):
    if v is None:
        ax.bar(x - w / 2, 80, w, color="none", edgecolor=sc.PROBLEM, linewidth=1.6, linestyle="--")
        ax.text(x - w / 2, 40, "OOM", rotation=90, ha="center", va="center", color=sc.PROBLEM, fontsize=18,
                fontweight="bold")
    else:
        ax.bar(x - w / 2, v, w, color=sc.GRAYS[0], edgecolor=sc.INK, linewidth=1.2, hatch="//")
ax.bar([x + w / 2 for x in xs], veomni, w, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2)
# legend proxies
ax.bar([0], [0], color=sc.GRAYS[0], edgecolor=sc.INK, hatch="//", label="TorchTitan")
ax.bar([0], [0], color=sc.ACCENT, edgecolor=sc.INK, label="VeOmni")
sc.top_legend(ax, ncol=2)
ax.set_xticks(list(xs))
ax.set_xticklabels(seq)
ax.set_xlabel("Sequence length")
ax.set_ylabel("Memory per GPU (GB)")
ax.set_ylim(0, 90)
sc.callout(ax, "2.3×", xy=(2 - w / 2, 72.5), xytext=(1.0, 78), color=sc.ACCENT, size=22)
ax.text(2 + w / 2, 31.41 + 2, "31", ha="center", va="bottom", fontsize=17, color=sc.ACCENT, fontweight="bold")
ax.text(2 - w / 2, 71.81 + 2, "72", ha="center", va="bottom", fontsize=17, color=sc.INK, fontweight="bold")
sc.better(ax, "Lower is better", where="upper left")
sc.save(fig, __file__)
