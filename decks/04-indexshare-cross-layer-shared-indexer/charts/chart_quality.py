"""Change in long-context average (5 benchmarks) vs the original DSA model, 30B DSA model.
Sources: IndexCache, arXiv 2603.12201v1, Table 2 (training-free; DSA baseline 50.2) and Table 3 (training-aware;
its own DSA baseline 51.0). Deltas derived: value minus the matching baseline."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

groups = ["keep 1/2", "keep 1/4", "keep 1/8"]
uniform_free = [47.4 - 50.2, 43.0 - 50.2, 35.3 - 50.2]
searched_free = [50.3 - 50.2, 49.9 - 50.2, 46.1 - 50.2]
trained_uniform = [51.6 - 51.0, 50.6 - 51.0, None]
sc.setup()
fig, ax = sc.figure("right")
w = 0.26
xs = range(len(groups))
series = [("uniform, training-free", uniform_free, sc.GRAYS[1], "//"),
          ("searched, training-free", searched_free, sc.ACCENT_LIGHT, None),
          ("uniform, training-aware", trained_uniform, sc.ACCENT, None)]
for k, (name, vals, color, hatch) in enumerate(series):
    pos = [x + (k - 1) * w for x in xs]
    pv = [(p, v) for p, v in zip(pos, vals) if v is not None]
    ax.bar([p for p, _ in pv], [v for _, v in pv], w, color=color, edgecolor=sc.INK, lw=1.2, hatch=hatch, label=name)
    for p, v in pv:
        ax.text(p, v + (0.4 if v >= 0 else -0.4), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=16, fontweight="bold", color=sc.PROBLEM if v < -1 else sc.INK)
ax.text(2 + w, -0.4, "n/a", ha="center", va="top", fontsize=16, color=sc.GRAYS[3])
ax.axhline(0, color=sc.INK, lw=1.4)
ax.set_xticks(list(xs)); ax.set_xticklabels(groups)
ax.set_ylim(-17.5, 5)
ax.set_xlim(-0.5, 2.55)
sc.better(ax, 'Higher is better', where='upper right')
ax.set_ylabel("Δ long-context avg (points)")
ax.legend(loc="lower left", frameon=False, fontsize=16)
sc.save(fig, __file__)
