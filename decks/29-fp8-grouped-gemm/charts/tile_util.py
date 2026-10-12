"""Analytic: share of MMA rows that hold real tokens vs tokens per expert (uniform groups).

HPC-Ops token tile from src/group_gemm/sm90/group_gemm_pertensor_fp8.cu (group_gemm_fp8_async):
  avg <= 8 -> 8, <= 16 -> 16, <= 32 -> 32, <= 48 -> 48, <= 64 -> 64, <= 96 -> 48, <= 128 -> 32, <= 144 -> 48, else 64.
Token-major baselines: 64-row tile (DeepGEMM SM90 masked layout, smallest BLOCK_M candidate) and 128-row segments
(DeepGEMM contiguous layout, default M alignment 128) -- csrc/jit_kernels/heuristics/{sm90,runtime}.hpp.
utilization(t, tile) = t / (ceil(t / tile) * tile).
"""
import math, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc


def hpc_tile(avg):
    for lim, tile in ((8, 8), (16, 16), (32, 32), (48, 48), (64, 64), (96, 48), (128, 32), (144, 48)):
        if avg <= lim:
            return tile
    return 64


def util(t, tile):
    return 100.0 * t / (math.ceil(t / tile) * tile)


T = [1, 2, 4, 8, 16, 32, 64, 128, 256]
series = [
    ("128-row segment", [util(t, 128) for t in T], sc.GRAYS[1], "s", "--"),
    ("64-row tile", [util(t, 64) for t in T], sc.GRAYS[3], "^", "--"),
    ("HPC-Ops swapped tile", [util(t, hpc_tile(t)) for t in T], sc.ACCENT, "o", "-"),
]
for name, v, *_ in series:
    print(name, [round(x, 1) for x in v])

sc.setup()
fig, ax = sc.figure("right")
for name, v, c, m, ls in series:
    ax.plot(range(len(T)), v, color=c, marker=m, linestyle=ls, label=name, lw=3.2 if c == sc.ACCENT else 2.6)
ax.set_xticks(range(len(T)))
ax.set_xticklabels([str(t) for t in T])
ax.set_xlabel("Tokens per expert")
ax.set_ylabel("MMA rows holding\nreal tokens (%)")
ax.set_ylim(0, 118)
ax.axvspan(-0.3, 3.3, color="#EAF1F8", zorder=0)
ax.text(1.5, 72, "decode", ha="center", fontsize=18, color=sc.ACCENT, fontstyle="italic")
ax.text(5.0, 106, "HPC-Ops swapped tile", ha="center", fontsize=18, fontweight="bold", color=sc.ACCENT)
ax.text(4.8, 57, "64-row tile", ha="right", fontsize=18, color=sc.GRAYS[3])
ax.text(5.35, 7, "128-row segment", ha="left", fontsize=18, color=sc.GRAYS[2])
sc.better(ax, "Higher is better", where="upper left")
sc.save(fig, __file__)
