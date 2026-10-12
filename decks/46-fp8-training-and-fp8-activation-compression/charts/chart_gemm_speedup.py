"""GEMM speedup over BF16, 5B model on H200 NVL (TE docs, "GEMM Speedups Across Precisions").
Autocast = includes per-step quantization; raw FP8 kernel = pre-quantized inputs. See notes/sources.md."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

names = ["BF16", "FP8\nblockwise", "FP8\ncurrent", "FP8\ndelayed", "raw FP8\nkernel"]
vals = [1.00, 1.41, 1.57, 1.69, 1.92]
labels = ["1.00×", "1.41×", "1.57×", "1.69×", "1.92×"]

sc.setup()
fig, ax = sc.plt.subplots(figsize=(6.5, 3.75))
rects = sc.bars(ax, names, vals, ours=[1, 2, 3], labels=labels)
# raw kernel = upper bound, not a training recipe: hollow dashed bar
rects[4].set_facecolor("white"); rects[4].set_hatch(""); rects[4].set_linestyle("--")
rects[0].set_facecolor(sc.GRAYS[0]); rects[0].set_hatch("//")
for r in rects[1:4]:
    r.set_facecolor(sc.ACCENT)
ax.set_ylabel("GEMM speedup over BF16 (×)")
ax.set_ylim(0, 2.35)
sc.better(ax, "Higher is better", where="upper left")
sc.save(fig, __file__)
