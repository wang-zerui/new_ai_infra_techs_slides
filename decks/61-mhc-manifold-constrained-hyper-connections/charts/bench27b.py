"""27B downstream results ([mHC] Table 4): gain over the 27B baseline in points, HC vs mHC (derived differences).
Baseline / HC / mHC: BBH 43.8/48.9/51.0, DROP 47.0/51.6/53.9, GSM8K 46.7/53.2/53.8, HellaSwag 73.7/74.3/74.7,
MATH 22.0/26.4/26.0, MMLU 59.0/63.0/63.4, PIQA 78.5/79.9/80.5, TriviaQA 54.3/56.3/57.6."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

base = [43.8, 47.0, 46.7, 73.7, 22.0, 59.0, 78.5, 54.3]
hc = [48.9, 51.6, 53.2, 74.3, 26.4, 63.0, 79.9, 56.3]
mhc = [51.0, 53.9, 53.8, 74.7, 26.0, 63.4, 80.5, 57.6]
groups = ["BBH", "DROP", "GSM8K", "Hella-\nSwag", "MATH", "MMLU", "PIQA", "Trivia-\nQA"]
sc.setup()
fig, ax = sc.figure("full")
d_hc = [round(h - b, 1) for h, b in zip(hc, base)]
d_mhc = [round(m - b, 1) for m, b in zip(mhc, base)]
sc.grouped(ax, groups, ["HC", "mHC"], [d_hc, d_mhc], ours=1, width=0.74)
for g, v in enumerate(d_mhc):
    ax.text(g + 0.185, max(v, d_hc[g]) + 0.15, f"+{v}", ha="center", va="bottom", fontsize=18, fontweight="bold", color=sc.ACCENT)
ax.set_ylabel("Gain over baseline (pts)")
ax.set_ylim(0, 9)
sc.better(ax, "Higher is better", where="upper right")
sc.save(fig, __file__)
