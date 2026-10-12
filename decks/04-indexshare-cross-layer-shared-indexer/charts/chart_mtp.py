"""GLM-5.2 MTP acceptance length ablation (coding scenarios, GLM-5.1 backbone and data, 7 MTP steps).
Source: Z.ai GLM-5.2 blog, 'MTP with IndexShare and KVShare' table. Each bar adds one technique cumulatively."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

names = ["Baseline", "+IndexShare\n+KVShare", "+rejection\nsampling", "+TV loss"]
vals = [4.56, 5.10, 5.29, 5.47]
sc.setup()
fig, ax = sc.figure("right")
colors = [sc.GRAYS[0], sc.ACCENT, sc.ACCENT_LIGHT, sc.ACCENT_LIGHT]
hatches = ["//", None, None, None]
for i, (v, c, h) in enumerate(zip(vals, colors, hatches)):
    ax.bar(i, v, 0.62, color=c, edgecolor=sc.INK, lw=1.2, hatch=h)
    ax.text(i, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=18, fontweight="bold",
            color=sc.ACCENT if i == 1 else sc.INK)
ax.text(1, 3.0, "+12%", ha="center", fontsize=18, fontweight="bold", color="#FFFFFF")
ax.text(3, 3.0, "+20%", ha="center", fontsize=18, fontweight="bold", color=sc.INK)
ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=17)
ax.set_ylim(0, 6.4)
ax.set_ylabel("Acceptance length (tokens)")
sc.save(fig, __file__)
