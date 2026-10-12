"""Analytic: significant bits of an FP32 weight kept by each representation (and the relative-error bound).
BF16 8, TF32 11, BF16 hi+lo split 16 (error <= 2^-16 |w|), FP32 24. See notes/sources.md."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sc.setup()
import matplotlib as mpl
mpl.rcParams["mathtext.default"] = "regular"
fig, ax = sc.figure("right")
names = ["BF16", "TF32", "BF16 hi + lo", "FP32"]
bits = [8, 11, 16, 24]
err = [r"err ≤ $2^{-8}$", r"err ≤ $2^{-11}$", r"err ≤ $2^{-16}$", "reference"]
colors = [sc.GRAYS[0], sc.GRAYS[1], sc.ACCENT, sc.GRAYS[2]]
hatches = ["//", "\\\\", "", ""]
ys = range(len(names))
bars = ax.barh(list(ys), bits, height=0.62, color=colors, edgecolor=sc.INK, linewidth=1.2)
for b, h in zip(bars, hatches):
    if h:
        b.set_hatch(h)
ax.set_yticks(list(ys)); ax.set_yticklabels(names)
ax.invert_yaxis()
for i, (b, e) in enumerate(zip(bits, err)):
    ax.text(b + 0.4, i, f"{b} bits · {e}", va="center", ha="left", fontsize=18,
            fontweight="bold" if i == 2 else "normal", color=sc.ACCENT if i == 2 else sc.INK)
ax.set_xlim(0, 41)
ax.set_xticks([0, 8, 16, 24])
ax.set_xlabel("Significant bits of w kept (analytic)")
ax.grid(axis="y", visible=False); ax.grid(axis="x", linestyle="--", color="#BFBFBF")
sc.better(ax, "More is better", where="upper right")
sc.save(fig, __file__)
