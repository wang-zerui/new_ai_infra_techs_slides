"""FlashKDA speed-up over FLA Triton chunk_kda by batch shape, T=8192 total tokens, H=96, D=128.
Sources: FlashKDA BENCHMARK_H20.md and BENCHMARK_GB200.md, table 'T=8192, H=96, D=128', column 'Speedup vs chunk_kda'.
Cases: Fixed; Varlen seq_lens=[1300, 547, 2048, 963, 271, 3063]; Varlen seq_lens=1024 x 8."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc
cases = ["1 × 8K\n(fixed)", "6 mixed\nlengths", "8 × 1K"]
h20 = [1.85, 2.06, 2.29]
gb200 = [2.31, 2.71, 3.27]
sc.setup()
fig, ax = sc.figure("right")
w = 0.36
xs = range(len(cases))
ax.bar([x - w / 2 for x in xs], h20, w, color=sc.ACCENT_LIGHT, edgecolor=sc.INK, linewidth=1.2, label="H20")
ax.bar([x + w / 2 for x in xs], gb200, w, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2, label="GB200")
for x, a, b in zip(xs, h20, gb200):
    ax.text(x - w / 2, a + 0.05, f"{a:.2f}×", ha="center", va="bottom", fontsize=17, color=sc.INK)
    ax.text(x + w / 2, b + 0.05, f"{b:.2f}×", ha="center", va="bottom", fontsize=18, fontweight="bold", color=sc.ACCENT)
ax.axhline(1.0, color=sc.GRAYS[3], lw=2, ls="--", zorder=0)
ax.set_xticks(list(xs)); ax.set_xticklabels(cases)
ax.set_ylabel("Speed-up over Triton KDA (×)")
ax.set_ylim(0, 3.9)
sc.top_legend(ax, ncol=2)
sc.better(ax, "Higher is better", "upper left")
sc.save(fig, __file__)
