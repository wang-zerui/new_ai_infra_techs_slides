"""HPC-Ops fused FP8 MoE speedup over the best of SGLang, vLLM Triton and vLLM CUTLASS, Hy3 (Hunyuan-V3) shapes
on H20. Source: LMSYS blog 'HPC-Ops x SGLang' (2026-08-07), Table 4 (TP8/EP1) and Table 5 (TP1/EP8),
column 'Speedup vs. best'. Means stated in the blog: 1.08x (TP8/EP1), 1.21x (TP1/EP8).
Per notes/sources.md: TP8/EP1 (intermediate per rank 192) dispatches to the cp.async low-latency path;
TP1/EP8 (intermediate 1536 > 512) dispatches to the TMA path."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc
import numpy as np

tp_bs = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
tp_sp = [1.03, 1.11, 1.12, 1.11, 1.13, 1.18, 1.07, 1.06, 1.03, 1.01, 1.00]
ep_bs = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
ep_sp = [1.18, 1.25, 1.32, 1.40, 1.60, 1.18, 1.18, 0.98, 1.01, 1.01]

sc.setup()
fig, ax = sc.figure("right")
ax.plot(np.log2(ep_bs), ep_sp, color=sc.GRAYS[3], marker="s", lw=2.6, ls="--")
ax.plot(np.log2(tp_bs), tp_sp, color=sc.ACCENT, marker="o", lw=3.2)
ax.axhline(1.0, color=sc.INK, lw=1.4)
ax.set_xticks([2, 4, 6, 8, 10, 12, 14])
ax.set_xticklabels(["4", "16", "64", "256", "1K", "4K", "16K"])
ax.set_xlabel("Tokens per call (batch)")
ax.set_ylabel("Speedup over best\nbaseline (×)")
ax.set_ylim(0.9, 1.72)
ax.text(np.log2(64), 1.645, "1.60×", ha="center", va="bottom", fontsize=18, fontweight="bold", color=sc.GRAYS[3])
ax.text(np.log2(512), 1.19, "1.18×", ha="center", va="bottom", fontsize=18, fontweight="bold", color=sc.ACCENT)
ax.text(11.4, 1.30, "TP8 · cp.async path\nmean 1.08×", ha="center", fontsize=17, color=sc.ACCENT, fontweight="bold")
ax.text(9.6, 1.47, "EP8 · TMA path\nmean 1.21×", ha="left", fontsize=17, color=sc.GRAYS[3])
sc.better(ax, "Higher is better", where="upper right")
sc.save(fig, __file__)
