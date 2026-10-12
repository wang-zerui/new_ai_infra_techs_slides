"""Mega MoE speedup over the five-kernel ("legacy") path, EP8 on SM100, averaged over 8 ranks.

Source: DeepGEMM PR #316 (github.com/deepseek-ai/DeepGEMM/pull/316), "Speedup (vs legacy)" columns.
Legacy = DeepEP dispatch + DeepGEMM grouped FP8xFP4 GEMM + TileLang SwiGLU/FP8 + grouped GEMM + DeepEP
combine, timed as a CUDA graph (tests/test_mega_moe.py at the PR #316 merge commit).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

batches = ["1", "512", "8192", "32768"]
flash = [1.96, 1.73, 1.56, 1.62]
pro = [1.61, 1.54, 1.50, 1.54]

sc.setup()
fig, ax = sc.figure("right")
w = 0.38
xs = range(len(batches))
ax.bar([x - w / 2 for x in xs], flash, w, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2, label="V4-Flash")
ax.bar([x + w / 2 for x in xs], pro, w, color=sc.ACCENT_LIGHT, edgecolor=sc.INK, linewidth=1.2, label="V4-Pro")
for x, f, p in zip(xs, flash, pro):
    ax.text(x - w / 2, f + 0.04, f"{f:.2f}", ha="center", va="bottom", fontsize=17, fontweight="bold", color=sc.ACCENT)
    ax.text(x + w / 2, p + 0.04, f"{p:.2f}", ha="center", va="bottom", fontsize=17, color=sc.INK)
ax.axhline(1.0, color=sc.GRAYS[3], linestyle="--", linewidth=2.2, label="five kernels")
ax.set_xticks(list(xs))
ax.set_xticklabels(batches)
ax.set_xlabel("Tokens per rank (EP8)")
ax.set_ylabel("Speedup vs five kernels (×)")
ax.set_ylim(0, 2.3)
sc.top_legend(ax, ncol=3)
sc.better(ax, "Higher is better", where="upper right")
sc.save(fig, __file__)
