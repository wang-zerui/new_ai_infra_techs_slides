"""DeepSeek-V3 (BF16, PP4 x VPP4 x EP64, 256 GPUs): main weights + optimizer states per GPU.
Baseline 32.1 GB (arXiv 2603.07685 Table 3). BF16 moments save 10-12 GB (Sec. 4.1.6) -> 20.1-22.1 GB left (derived).
State offloading saves 15-20 GB at 0.1-0.2 s/iteration (Sec. 4.1.6, GB200) -> 12.1-17.1 GB left (derived).
Bars show the remaining range (darker part = guaranteed, lighter cap = range).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

names = ["FP32\nstates", "BF16\nmoments", "CPU\noffload"]
lo = [32.1, 20.1, 12.1]
hi = [32.1, 22.1, 17.1]

sc.setup()
fig, ax = sc.figure("right")
cols = [sc.GRAYS[0], sc.ACCENT, sc.GRAYS[1]]
hat = ["//", None, "\\\\"]
for x, l, h, c, ht in zip(range(3), lo, hi, cols, hat):
    ax.bar(x, l, 0.6, color=c, edgecolor=sc.INK, linewidth=1.2, hatch=ht)
    if h > l:
        ax.bar(x, h - l, 0.6, bottom=l, color="#FFFFFF", edgecolor=sc.INK, linewidth=1.2, hatch="..")
lab = ["32.1 GB", "20–22 GB", "12–17 GB"]
for x, h, t in zip(range(3), hi, lab):
    ax.text(x, h + 0.8, t, ha="center", va="bottom", fontsize=20, fontweight="bold",
            color=sc.ACCENT if x == 1 else sc.INK)
ax.text(2, hi[2] + 5.0, "+0.1–0.2 s/iter", ha="center", va="bottom", fontsize=17, color=sc.PROBLEM,
        fontweight="bold")
ax.text(1, hi[1] + 5.0, "no slowdown", ha="center", va="bottom", fontsize=17, color=sc.GOOD, fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(names)
ax.set_ylabel("Main weights + optimizer\nstate per GPU (GB)")
ax.set_ylim(0, 40)
sc.better(ax, "Lower is better", where="upper right")
sc.save(fig, __file__)
