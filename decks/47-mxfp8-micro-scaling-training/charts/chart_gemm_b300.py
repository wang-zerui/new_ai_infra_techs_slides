"""GEMM speedup over BF16, 5B model on B300, autocast mode (TE docs, "GEMM Speedups Across Precisions").
See notes/sources.md."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

names = ["BF16", "FP8\ncurrent", "MXFP8", "FP8\ndelayed", "NVFP4"]
vals = [1.00, 1.41, 1.44, 1.61, 2.03]
labels = ["1.00×", "1.41×", "1.44×", "1.61×", "2.03×"]

sc.setup()
fig, ax = sc.plt.subplots(figsize=(6.5, 3.75))
rects = sc.bars(ax, names, vals, ours=2, labels=labels)
rects[4].set_facecolor("white"); rects[4].set_hatch(""); rects[4].set_linestyle("--")
ax.set_ylabel("GEMM speedup over BF16 (×)")
ax.set_ylim(0, 2.5)
sc.better(ax, "Higher is better", where="upper left")
sc.save(fig, __file__)
