"""64-GPU throughput per GPU on the varlen causal mask: best baseline vs MagiAttention with all-to-all-v
group collectives vs native (DeepEP-based) group collectives. Pixel-read from the MagiAttention CP-benchmark
blog Fig. 50, 51 (H100) and Fig. 58, 59 (B200); notes/chart_readings.txt.
Best baseline: H100 fwd USP/LoongTrain 193, H100 bwd Ulysses 249, B200 fwd Ring AllGather 295, B200 bwd
Ulysses 404. On B200 baselines run cuDNN attention, MagiAttention runs FFA_FA4.
Native/a2av: 238/208 = +14%, 265/243 = +9%, 563/398 = +41%, 554/449 = +23%."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

groups = ["H100 fwd", "H100 bwd", "B200 fwd", "B200 bwd"]
best = [193, 249, 295, 404]
a2av = [208, 243, 398, 449]
native = [238, 265, 563, 554]

sc.setup()
fig, ax = sc.figure("right")
w = 0.27
xs = list(range(4))
ax.bar([x - w for x in xs], best, w, color=sc.GRAYS[1], edgecolor=sc.INK, linewidth=1.2, hatch="//",
       label="Best baseline")
ax.bar(xs, a2av, w, color=sc.ACCENT_LIGHT, edgecolor=sc.INK, linewidth=1.2, label="Magi a2a-v")
ax.bar([x + w for x in xs], native, w, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2, label="Magi native")
for x, a, n in zip(xs, a2av, native):
    ax.text(x + w, n + 10, f"+{round(100 * (n / a - 1))}%", ha="center", va="bottom", fontsize=18,
            color=sc.ACCENT, fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels(["H100\nfwd", "H100\nbwd", "B200\nfwd", "B200\nbwd"])
ax.set_ylim(0, 680)
ax.set_xlim(-0.6, 3.6)
ax.set_ylabel("TFLOPs/s per GPU")
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=17,
          handlelength=1.2, columnspacing=0.8, borderaxespad=0.2)
sc.better(ax, "Higher is better", where="upper left")
sc.save(fig, __file__)
