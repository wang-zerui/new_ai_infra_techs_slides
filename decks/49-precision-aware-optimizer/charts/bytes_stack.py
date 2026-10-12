"""Bytes per parameter by component (analytic, no sharding). BF16 weight 2 B and FP32 main grad 4 B fixed.
FP32 Adam: master 4 + m 4 + v 4; BF16 moments: 4 + 2 + 2 (MCore 'typical', arXiv 2603.07685 Sec. 4.1.6);
+ remainder: master stored as int16 low half, 2 B (Megatron store_param_remainders default, TE >= 2.1);
FP8 moments + remainder: 2 + 1 + 1. See notes/sources.md.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

cfgs = ["FP32\nAdam", "BF16\nmoments", "+ master\nremainder", "FP8\nmoments"]
weight = [2, 2, 2, 2]
grad = [4, 4, 4, 4]
master = [4, 4, 2, 2]
mom = [8, 4, 4, 2]          # m + v

sc.setup()
fig, ax = sc.figure("right")
xs = range(len(cfgs))
b1 = ax.bar(xs, [a + b for a, b in zip(weight, grad)], 0.62, color="#FFFFFF", edgecolor=sc.INK, linewidth=1.2,
            label="weight + grad")
bot = [a + b for a, b in zip(weight, grad)]
ax.bar(xs, master, 0.62, bottom=bot, color=sc.ACCENT_LIGHT, edgecolor=sc.INK, linewidth=1.2, label="master")
bot2 = [a + b for a, b in zip(bot, master)]
ax.bar(xs, mom, 0.62, bottom=bot2, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2, label="moments m + v")
for x, m, mo, tot in zip(xs, master, mom, [a + b for a, b in zip(bot2, mom)]):
    ax.text(x, tot + 0.4, f"{m + mo} B", ha="center", va="bottom", fontsize=20, fontweight="bold",
            color=sc.ACCENT if x else sc.INK)
ax.set_xticks(list(xs)); ax.set_xticklabels(cfgs)
ax.set_ylabel("Bytes per parameter")
ax.set_ylim(0, 22)
sc.top_legend(ax, ncol=3, fontsize=16)
sc.save(fig, __file__)
