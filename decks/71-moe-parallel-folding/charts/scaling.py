"""Strong scaling, fine-grained Mixtral-8x22B-G8T8 (64 experts, top-8), GBS 1024, BF16, H100.

Source: arXiv 2504.14960, Table 4 (p.18-19). The PDF prints "18,4%" for MCore at 256 GPUs (read as 18.4).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

gpus = [128, 256, 512, 1024]
mcore = [19.8, 18.4, 16.3, 13.4]
fold = [30.0, 29.3, 26.7, 25.5]

sc.setup()
fig, ax = sc.figure("right")
ax.plot(gpus, mcore, marker="s", color=sc.GRAYS[2], label="MCore, shared mapping", linestyle="--")
ax.plot(gpus, fold, marker="o", color=sc.ACCENT, label="MCore + Folding")
ax.set_xscale("log", base=2)
ax.set_xticks(gpus)
ax.set_xticklabels([str(g) for g in gpus])
ax.set_xlabel("H100 GPUs")
ax.set_ylabel("MFU (%)")
ax.set_ylim(0, 38)
ax.text(1024 * 1.06, 25.5, "25.5", va="center", ha="left", fontsize=18, color=sc.ACCENT, fontweight="bold")
ax.text(1024 * 1.06, 13.4, "13.4", va="center", ha="left", fontsize=18, color=sc.GRAYS[3])
ax.text(128 * 0.94, 30.0, "30.0", va="center", ha="right", fontsize=18, color=sc.ACCENT, fontweight="bold")
ax.text(128 * 0.94, 19.8, "19.8", va="center", ha="right", fontsize=18, color=sc.GRAYS[3])
ax.set_xlim(128 / 1.9, 1024 * 1.5)
sc.top_legend(ax, ncol=2)
sc.better(ax, "Higher is better", where="upper right")
sc.save(fig, __file__)
