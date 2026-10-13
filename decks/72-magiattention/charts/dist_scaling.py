"""Distributed attention throughput per GPU vs CP size, varlen causal mask, H100 (8K tokens per GPU, so
64K-512K tokens in total). Values pixel-read from the MagiAttention CP-benchmark blog, Fig. 50 (fwd) and
Fig. 51 (bwd) (notes/chart_readings.txt). MagiAttention = native group-collective backend.
Not shown (in notes): Ring AllGather, LoongTrain, Megatron HybridCP."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

gpus = [8, 16, 32, 64]
data = {
    "fwd": {"Ring P2P": [89, 69, 48, 29], "Ulysses": [278, 200, 186, 163], "USP": [261, 251, 255, 192],
            "MagiAttention": [459, 427, 316, 238]},
    "bwd": {"Ring P2P": [97, 86, 57, 29], "Ulysses": [332, 295, 277, 249], "USP": [325, 309, 285, 196],
            "MagiAttention": [361, 371, 334, 265]},
}
style = {"Ring P2P": (sc.GRAYS[1], "s", "--"), "Ulysses": (sc.GRAYS[2], "^", ":"),
         "USP": (sc.GRAYS[3], "D", "-."), "MagiAttention": (sc.ACCENT, "o", "-")}

sc.setup()
fig, axes = sc.figure("full", 1, 2, sharey=True)
for ax, (pas, series) in zip(axes, data.items()):
    for name, ys in series.items():
        c, m, ls = style[name]
        ax.plot(range(4), ys, color=c, marker=m, linestyle=ls, label=name,
                linewidth=3.4 if name == "MagiAttention" else 2.4)
    ax.set_xticks(range(4))
    ax.set_xticklabels([str(g) for g in gpus])
    ax.set_xlabel("GPUs (H100)")
    ax.set_title("Forward" if pas == "fwd" else "Backward", fontsize=20, fontweight="bold", color=sc.INK)
    ax.set_ylim(0, 520)
    m = series["MagiAttention"][-1]
    ax.set_xlim(-0.25, 3.3)
    ax.annotate(f"{m}", xy=(3, m), xytext=(2.6, m + 85), ha="center", color=sc.ACCENT, fontsize=20,
                fontweight="bold", arrowprops=dict(arrowstyle="-|>", color=sc.ACCENT, lw=2))
axes[0].set_ylabel("TFLOPs/s per GPU")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False)
sc.better(axes[1], "Higher is better", where="upper right")
fig.tight_layout()
sc.save(fig, __file__)
