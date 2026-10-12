"""HPC-Ops decode attention, static split-KV vs dynamic task map, FP8 KV cache on H20.
Source: LMSYS blog "HPC-Ops x SGLang" (2026-08-07), Table 1. Ratios are the blog's printed values."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

CASES = ["64×0.5K", "64×4K", "32×0.125K\n+32×4K", "2×32K\n+30×4K", "1×64K\n+15×4K", "1×128K\n+31×4K"]
STATIC = [0.013, 0.043, 0.033, 0.056, 0.097, 0.186]   # ms
DYNAMIC = [0.013, 0.033, 0.020, 0.032, 0.042, 0.063]  # ms
RATIO = ["1.00×", "1.32×", "1.59×", "1.76×", "2.32×", "2.95×"]
sc.setup()
fig, ax = sc.figure("full")
sc.grouped(ax, CASES, ["Static split-KV", "Dynamic task map"], [STATIC, DYNAMIC], ours=1, width=0.7)
for i, (st, r) in enumerate(zip(STATIC, RATIO)):
    ax.text(i, st + 0.006, r, ha="center", va="bottom", fontsize=19, fontweight="bold",
            color=sc.ACCENT)
ax.set_ylabel("Latency (ms)")
ax.set_ylim(0, 0.225)
ax.set_xlabel("Decode batch (requests × KV length); skew grows to the right")
ax.text(0.01, 0.95, "Lower is better; label = static / dynamic", transform=ax.transAxes, ha="left", va="top",
        fontsize=16, fontstyle="italic", color=sc.GOOD)
sc.save(fig, __file__)
