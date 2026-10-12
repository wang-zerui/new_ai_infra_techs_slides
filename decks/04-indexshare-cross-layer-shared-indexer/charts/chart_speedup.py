"""IndexCache speedup over standard DSA, 30B DSA model on one H100 node (SGLang, DP attention 8).
Source: IndexCache, arXiv 2603.12201v1, Table 1 and Fig. 3(a)(b): prefill time and per-request decode throughput."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

ctx = ["10K", "60K", "120K", "200K"]
pre_half, pre_quarter = [1.21, 1.18, 1.30, 1.42], [1.27, 1.31, 1.51, 1.82]
dec_half, dec_quarter = [1.15, 1.19, 1.22, 1.26], [1.24, 1.33, 1.40, 1.48]
sc.setup()
fig, axes = sc.figure("full", 1, 2)
for ax, half, quarter, tag in ((axes[0], pre_half, pre_quarter, "(a) Prefill latency"),
                               (axes[1], dec_half, dec_quarter, "(b) Decode, per request")):
    w = 0.38
    xs = range(len(ctx))
    ax.bar([x - w / 2 for x in xs], half, w, color=sc.ACCENT_LIGHT, edgecolor=sc.INK, lw=1.2, label="keep 1/2 indexers")
    ax.bar([x + w / 2 for x in xs], quarter, w, color=sc.ACCENT, edgecolor=sc.INK, lw=1.2, label="keep 1/4 indexers")
    ax.axhline(1.0, color=sc.GRAYS[3], ls="--", lw=2)
    ax.text(3 + w / 2, quarter[-1] + 0.03, f"{quarter[-1]:.2f}×", ha="center", va="bottom", fontsize=20,
            fontweight="bold", color=sc.ACCENT)
    ax.set_xticks(list(xs)); ax.set_xticklabels(ctx)
    ax.set_ylim(0.8, 2.0)
    ax.set_xlabel(f"Context length\n{tag}")
axes[0].set_ylabel("Speedup over DSA (×)")
sc.better(axes[0], "Higher is better", where="upper left")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False)
fig.subplots_adjust(wspace=0.18)
sc.save(fig, __file__)
