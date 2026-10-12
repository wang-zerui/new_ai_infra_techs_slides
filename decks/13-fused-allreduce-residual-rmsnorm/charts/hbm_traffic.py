"""Analytic HBM traffic per token per GPU for RMSNorm(AllReduce(x) + residual), TP = 8, BF16.

Unit = one hidden vector (2H bytes). Counted from the kernel code (see notes/sources.md, "Derived numbers"):
  NCCL all_reduce + flashinfer fused_add_rmsnorm (HPC-Ops benchmark baseline), visible traffic only:
      AR reads x (1) + writes reduced x (1); norm reads x (1) + residual (1), writes residual (1) + output (1)
      = 6, a lower bound (NCCL-internal buffers not counted).
  HPC-Ops low-latency (twoshotAllreduceKernel + rmsNormLamport):
      read input 1, scatter stores landing 1, owner reads slots 1, clear scatter 1, broadcast landing 1,
      clear broadcast 1, norm kernel reads broadcast 1 + residual 1, writes residual 1 + output 1 = 10
      (polling re-reads not counted).
  HPC-Ops high-throughput (fused_allreduce_rmsnorm_kernel), P = 8:
      x read once via multimem.ld_reduce 1, normalized output lands via multimem.st 1,
      residual read + write on the owned 1/P slice only 2/P = 2.25.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

P = 8
names = ["NCCL AllReduce\n+ add-norm", "HPC-Ops\nLL mode", "HPC-Ops\nHT mode"]
vals = [6, 10, 2 + 2 / P]
labels = ["≥ 6", "≥ 10", "2.25"]

sc.setup()
fig, ax = sc.figure("right")
xs = range(len(vals))
colors = [sc.GRAYS[0], sc.ACCENT_LIGHT, sc.ACCENT]
hatches = [sc.HATCHES[0], "", ""]
rects = ax.bar(xs, vals, width=0.6, color=colors, edgecolor=sc.INK, linewidth=1.2)
for r, h in zip(rects, hatches):
    if h:
        r.set_hatch(h)
for x, v, t in zip(xs, vals, labels):
    ax.text(x, v + 0.25, t, ha="center", va="bottom", fontsize=20, fontweight="bold",
            color=sc.ACCENT if x == 2 else sc.INK)
ax.set_xticks(list(xs))
ax.set_xticklabels(names)
ax.set_ylim(0, 12.5)
ax.set_ylabel("HBM traffic per token\nper GPU (hidden vectors)")
sc.better(ax, "Lower is better", where="upper left")
sc.save(fig, __file__)
