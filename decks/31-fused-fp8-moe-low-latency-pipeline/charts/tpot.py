"""Hy3-FP8 TPOT on 8x H20 (TP8, FP8 KV cache, 8K input / 4K output), SGLang default vs HPC-Ops Attention + MoE
enabled together. Source: LMSYS blog 'HPC-Ops x SGLang' (2026-08-07), Table 7."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

bs = ["1", "4", "8", "16", "32", "64"]
base = [7.56, 11.10, 14.29, 22.90, 35.33, 40.70]
hpc = [7.31, 9.42, 10.76, 13.09, 18.09, 23.81]
imp = ["3.3%", "15.1%", "24.7%", "42.8%", "48.8%", "41.5%"]

sc.setup()
fig, ax = sc.figure("right")
sc.grouped(ax, bs, ["SGLang default", "HPC-Ops attn + MoE"], [base, hpc], ours=1, width=0.72)
ax.text(4, base[4] + 1.2, "−48.8%", ha="center", va="bottom", fontsize=20, fontweight="bold", color=sc.ACCENT)
ax.set_xlabel("Batch size")
ax.set_ylabel("TPOT (ms)")
ax.set_ylim(0, 47)
sc.better(ax, "Lower is better", where="upper left")
sc.save(fig, __file__)
