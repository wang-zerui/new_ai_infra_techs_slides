"""Normal-SMFree (DeepEP PR #347): comm-kernel execution time, native ring-buffer mode vs hook mode.

Setup (PR #347 §3.1): 2/4 nodes x 8 H20, 4096 tokens per batch, hidden 7168, top-8.
Hook-mode time = send kernel + receive kernel; the network phase is overlapped by a GEMM (PR #347 §3.2.1).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

ops = ["FP8\ndispatch", "BF16\ndispatch", "BF16\ncombine"]
native16 = [1358, 2530, 2590]
hook16 = [765, 1381, 1425]
native32 = [3535, 6780, 6850]
hook32 = [884, 2468, 2250]

sc.setup()
fig, ax = sc.figure("right")
w = 0.2
xs = list(range(len(ops)))
series = [(native16, "native, EP16", sc.GRAYS[0], "//"), (hook16, "hook, EP16", sc.ACCENT_LIGHT, None),
          (native32, "native, EP32", sc.GRAYS[2], "//"), (hook32, "hook, EP32", sc.ACCENT, None)]
for k, (vals, lab, col, ht) in enumerate(series):
    pos = [x + (k - 1.5) * w for x in xs]
    ax.bar(pos, [v / 1000 for v in vals], w, color=col, edgecolor=sc.INK, linewidth=1.1, hatch=ht, label=lab)
for x, n, h in zip(xs, native16, hook16):
    ax.text(x - w, n / 1000 + 0.12, f"{n / h:.1f}×", ha="center", va="bottom", fontsize=17, color=sc.ACCENT,
            fontweight="bold")
for x, n, h in zip(xs, native32, hook32):
    ax.text(x + w, n / 1000 + 0.12, f"{n / h:.1f}×", ha="center", va="bottom", fontsize=17, color=sc.ACCENT,
            fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels(ops)
ax.set_ylim(0, 9.0)
ax.set_ylabel("Comm-kernel time (ms)")
sc.top_legend(ax, ncol=2, fontsize=16)
sc.better(ax, "Lower is better", where="upper left")
sc.save(fig, __file__)
