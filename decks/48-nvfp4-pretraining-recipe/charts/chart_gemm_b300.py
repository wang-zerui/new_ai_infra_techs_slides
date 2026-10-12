"""GEMM speedup over BF16, 5B model on B300 (TE docs, "GEMM Speedups Across Precisions"): autocast includes
per-step quantization; raw kernel = pre-quantized inputs. See notes/sources.md."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

names = ["BF16", "MXFP8", "NVFP4", "NVFP4\nraw kernel"]
vals = [1.00, 1.44, 2.03, 3.55]
labels = ["1.00×", "1.44×", "2.03×", "3.55×"]

sc.setup()
fig, ax = sc.plt.subplots(figsize=(6.5, 3.75))
rects = sc.bars(ax, names, vals, ours=2, labels=labels)
rects[3].set_facecolor("white"); rects[3].set_hatch(""); rects[3].set_linestyle("--")
ax.set_ylabel("GEMM speedup over BF16 (×)")
ax.set_ylim(0, 4.3)
sc.better(ax, "Higher is better", where="upper left")
sc.save(fig, __file__)
