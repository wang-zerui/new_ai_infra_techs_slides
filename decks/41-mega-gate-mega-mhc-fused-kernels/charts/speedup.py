"""Published speedup ranges of Mega Gate and Mega mHC over their multi-kernel baselines.

Source: DeepGEMM PR #432 (github.com/deepseek-ai/DeepGEMM/pull/432): "Mega Gate ... 30-75% speedups",
"Mega mHC ... 45-85% speedups". Speedup = baseline kernel time / fused kernel time (tests at the PR #432
merge commit). Only the ranges are published, so the bars span lo..hi.
Baselines: Mega Gate vs cuBLASLt GEMM + TileLang top-k gate; Mega mHC vs TileLang mHC kernels
(4 kernels normal mode, 7 shifted) with DeepGEMM's TF32 prenorm GEMM.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

names = ["Mega Gate\nvs GEMM + top-k", "Mega mHC\nvs 4–7 kernels"]
lo = [1.30, 1.45]
hi = [1.75, 1.85]
labels = ["1.30–1.75×", "1.45–1.85×"]

sc.setup()
fig, ax = sc.figure("right")
xs = range(len(names))
for x, l, h in zip(xs, lo, hi):
    ax.bar(x, h - l, bottom=l, width=0.55, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2)
    ax.text(x, h + 0.04, labels[x], ha="center", va="bottom", fontsize=20, fontweight="bold", color=sc.ACCENT)
ax.axhline(1.0, color=sc.GRAYS[3], linestyle="--", linewidth=2.2, label="multi-kernel baseline = 1.0×")
ax.set_xticks(list(xs))
ax.set_xticklabels(names)
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(0, 2.3)
ax.set_ylabel("Speedup over baseline (×)")
sc.top_legend(ax, ncol=1)
sc.better(ax, "Higher is better", where="upper right")
sc.save(fig, __file__)
