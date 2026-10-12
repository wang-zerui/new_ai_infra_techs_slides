"""Dynamic decode attention vs FlashInfer and FlashAttention, FP8 KV cache decode on H20.
Source: LMSYS blog "HPC-Ops x SGLang" (2026-08-07), Table 1; 2.25x = blog's average over the six cases of
min(FlashInfer, FlashAttention) / HPC-Ops dynamic (re-derived: mean of 1.92, 2.88, 2.65, 2.94, 1.55, 1.54)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

CASES = ["64×0.5K", "64×4K", "32×0.125K\n+32×4K", "2×32K\n+30×4K", "1×64K\n+15×4K", "1×128K\n+31×4K"]
FLASHINFER = [0.050, 0.221, 0.119, 0.169, 0.118, 0.220]
FLASHATTN = [0.025, 0.095, 0.053, 0.094, 0.065, 0.097]
DYNAMIC = [0.013, 0.033, 0.020, 0.032, 0.042, 0.063]
sc.setup()
fig, ax = sc.figure("full")
sc.grouped(ax, CASES, ["FlashInfer", "FlashAttention", "HPC-Ops dynamic"], [FLASHINFER, FLASHATTN, DYNAMIC],
           ours=2, width=0.78)
ax.set_ylabel("Latency (ms)")
ax.set_ylim(0, 0.30)
ax.set_xlabel("Decode batch (requests × KV length)")
ax.text(0.01, 0.97, "Lower is better; label = faster library / HPC-Ops", transform=ax.transAxes, ha="left", va="top", fontsize=16,
        fontstyle="italic", color=sc.GOOD)
# takeaway device: derived ratio min(FlashInfer, FlashAttention) / dynamic above each HPC-Ops bar
RATIO = ["1.92×", "2.88×", "2.65×", "2.94×", "1.55×", "1.54×"]   # min(FI, FA) / dynamic, derived
for i, (a, b, r) in enumerate(zip(FLASHINFER, FLASHATTN, RATIO)):
    ax.text(i, max(a, b) + 0.004, r, ha="center", va="bottom", fontsize=19, fontweight="bold", color=sc.ACCENT)
sc.save(fig, __file__)
