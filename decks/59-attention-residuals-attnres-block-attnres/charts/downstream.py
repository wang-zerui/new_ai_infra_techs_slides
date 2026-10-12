"""Kimi Linear 48B total / 3B active, 1.4T tokens, baseline vs Block AttnRes (AttnRes paper, Table 3), subset."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

groups = ["GPQA-\nDiamond", "Math", "Human-\nEval", "C-Eval", "BBH", "MMLU"]
base = [36.9, 53.5, 59.1, 79.6, 76.3, 73.5]
attn = [44.4, 57.1, 62.2, 82.5, 78.0, 74.6]

sc.setup()
fig, ax = sc.figure("right")
sc.grouped(ax, groups, ["PreNorm baseline", "AttnRes"], [base, attn], ours=1, width=0.74)
w = 0.74 / 2
for i, (b, a) in enumerate(zip(base, attn)):
    ax.text(i + w / 2, a + 1.0, f"+{a - b:.1f}", ha="center", va="bottom", fontsize=16, fontweight="bold",
            color=sc.ACCENT)
ax.set_ylim(30, 95)
ax.set_ylabel("Accuracy (%)")
ax.tick_params(axis="x", labelsize=16)
sc.better(ax, "Higher is better", where="upper left")
sc.save(fig, __file__)
