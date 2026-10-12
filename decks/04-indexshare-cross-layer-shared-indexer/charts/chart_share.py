"""Indexer share of total latency vs context length, 30B DSA model (GLM-4.7-Flash-based, 47 layers).
Source: IndexCache, arXiv 2603.12201v1, §1 figure 'Indexer Time Proportion (%)' (values printed on the figure)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

ctx = ["10K", "60K", "120K", "200K"]
prefill = [27, 50, 68, 81]
decode = [27, 31, 38, 41]
sc.setup()
fig, ax = sc.figure("right")
xs = range(len(ctx))
ax.plot(xs, prefill, color=sc.PROBLEM, marker="o", lw=3.4, label="Prefill")
ax.plot(xs, decode, color=sc.GRAYS[3], marker="s", ls="--", lw=3, label="Decode")
for x, v in zip(xs, prefill):
    ax.text(x, v + 4, f"{v}%", ha="center", va="bottom", fontsize=19, fontweight="bold", color=sc.PROBLEM)
for x, v in zip(xs, decode):
    ax.text(x, v - 5, f"{v}%", ha="center", va="top", fontsize=18, color=sc.GRAYS[3])
ax.set_xticks(list(xs)); ax.set_xticklabels(ctx)
ax.set_ylim(0, 100); ax.set_xlim(-0.35, 3.35)
ax.set_ylabel("Indexer share of time (%)")
ax.set_xlabel("Context length")
sc.top_legend(ax, ncol=2)
sc.save(fig, __file__)
