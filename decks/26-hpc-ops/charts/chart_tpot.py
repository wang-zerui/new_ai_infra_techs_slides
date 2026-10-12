"""Hy3-FP8 TPOT on 8x H20 (TP8, FP8 KV cache, 8K input / 4K output), SGLang default vs HPC-Ops Attention + MoE.
Source: LMSYS blog "HPC-Ops x SGLang" (2026-08-07), Table 7."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

BATCH = ["1", "4", "8", "16", "32", "64"]
SGLANG = [7.56, 11.10, 14.29, 22.90, 35.33, 40.70]   # ms
HPCOPS = [7.31, 9.42, 10.76, 13.09, 18.09, 23.81]    # ms
sc.setup()
fig, ax = sc.figure("right")
sc.grouped(ax, BATCH, ["SGLang default", "HPC-Ops"], [SGLANG, HPCOPS], ours=1)
ax.set_xlabel("Batch size")
ax.set_ylabel("TPOT (ms)")
ax.set_ylim(0, 50)
i = 4
sc.callout(ax, "−48.8%", xy=(i + 0.2, HPCOPS[i] + 0.6), xytext=(i - 0.9, 42), color=sc.ACCENT, size=22)
sc.better(ax, "Lower is better", where="upper left")
sc.save(fig, __file__)
