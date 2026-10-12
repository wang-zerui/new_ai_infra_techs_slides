"""MLA decoding on H100, speedup over PyTorch (TileLang paper, arXiv 2504.17577, Fig. 14a).
TileLang reaches up to 98% of hand-written FlashMLA with ~70 lines of Python (§5.2)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

names = ["Torch", "Triton", "FlashInfer", "TileLang", "FlashMLA\n(hand-written)"]
vals = [1.0, 311.8, 798.8, 1075.9, 1097.2]
labels = ["1×", "312×", "799×", "1076×", "1097×"]

sc.setup()
fig, ax = sc.figure("right")
cols = [sc.GRAYS[0], sc.GRAYS[1], sc.GRAYS[1], sc.ACCENT, sc.GRAYS[3]]
hat = ["//", "\\\\", "xx", "", ""]
rects = ax.bar(range(5), vals, width=0.62, color=cols, edgecolor=sc.INK, linewidth=1.2)
for r, h in zip(rects, hat):
    if h:
        r.set_hatch(h)
ax.set_yscale("log")
ax.set_ylim(0.5, 20000)
for i, (r, t) in enumerate(zip(rects, labels)):
    ax.text(r.get_x() + r.get_width() / 2, r.get_height() * 1.25, t, ha="center", va="bottom", fontsize=17,
            fontweight="bold", color=sc.ACCENT if i == 3 else sc.INK)
ax.set_xticks(range(5))
ax.set_xticklabels(names, fontsize=16)
ax.set_ylabel("Speedup over Torch (×, log)")
ax.grid(axis="y", which="minor", visible=False)
sc.better(ax)
sc.save(fig, __file__)
