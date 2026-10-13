"""FFA vs FA3 vs FlashMask forward throughput on H100 at 32K tokens (b=1, GQA 64:8:8, hd 128, BF16).
Values pixel-read from the MagiAttention CP-benchmark blog kernel charts (Fig. 20, 22, 26, 30;
notes/chart_readings.txt). FA3 cannot run the varlen block-causal mask (marked unsupported in the chart).
Ratios: 634/701 = 0.90, 622/680 = 0.91, 548/584 = 0.94; 575/159 = 3.6."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

groups = ["Full", "Causal", "Varlen\ncausal", "Varlen\nblock-causal"]
flashmask = [360, 352, 241, 159]
fa3 = [701, 680, 584, 0]
ffa = [634, 622, 548, 575]

sc.setup()
fig, ax = sc.figure("full")
w = 0.26
xs = range(len(groups))
ax.bar([x - w for x in xs], flashmask, w, color=sc.GRAYS[0], edgecolor=sc.INK, linewidth=1.2, hatch="//",
       label="FlashMask")
ax.bar([x for x in xs][:3], fa3[:3], w, color=sc.GRAYS[2], edgecolor=sc.INK, linewidth=1.2, hatch="\\\\",
       label="FA3")
ax.bar([x + w for x in xs], ffa, w, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2, label="FFA")
ax.text(3, 12, "n/a", ha="center", va="bottom", fontsize=18, color=sc.INK)
for x, v, f3 in zip(xs, ffa, fa3):
    ax.text(x + w, v + 12, f"{v}", ha="center", va="bottom", fontsize=18, color=sc.ACCENT, fontweight="bold")
for x in range(3):
    ax.text(x, fa3[x] + 12, f"{fa3[x]}", ha="center", va="bottom", fontsize=18, color=sc.INK)
ax.text(3, 680, "3.6× FlashMask", ha="center", va="bottom", fontsize=22, color=sc.ACCENT, fontweight="bold")
ax.set_xticks(list(xs))
ax.set_xticklabels(groups)
ax.set_ylim(0, 860)
ax.set_ylabel("Forward TFLOPs/s")
sc.top_legend(ax, ncol=3)
sc.better(ax, "Higher is better", where="upper left")
sc.save(fig, __file__)
