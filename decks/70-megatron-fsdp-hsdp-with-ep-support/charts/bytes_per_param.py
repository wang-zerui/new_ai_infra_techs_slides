"""Model-state bytes per parameter per GPU vs data-parallel size d (analytic).

Assumptions (Megatron-LM dist_optimizer.md, bf16 params + fp32 grads row): BF16 weight 2 B, FP32 main grad 4 B,
FP32 master weight + Adam moments 12 B. Excludes the two transient un-sharded FSDP units and activations.
  DDP (no_shard)                18
  Distributed optimizer (optim) 6 + 12/d          (exact formula in dist_optimizer.md)
  HSDP, 8-GPU inner group       18 / min(d, 8)    (derived)
  FSDP (optim_grads_params)     18 / d            (derived)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

d = [1, 2, 4, 8, 16, 32, 64, 128, 256]
ddp = [18 for _ in d]
dopt = [6 + 12 / x for x in d]
hsdp = [18 / min(x, 8) for x in d]
fsdp = [18 / x for x in d]

sc.setup()
fig, ax = sc.figure("right")
ax.plot(d, ddp, color=sc.GRAYS[2], ls="--", marker="s", label="DDP")
ax.plot(d, dopt, color=sc.PROBLEM, marker="o", label="Dist. optimizer")
ax.plot(d, hsdp, color=sc.ACCENT_LIGHT, marker="^", label="HSDP (8 per node)")
ax.plot(d, fsdp, color=sc.ACCENT, marker="D", label="FSDP")
ax.set_xscale("log", base=2)
ax.set_xticks(d)
ax.set_xticklabels([str(x) for x in d])
ax.set_ylim(0, 20)
ax.set_xlabel("Data-parallel size d")
ax.set_ylabel("Model-state bytes per\nparameter per GPU")
ax.axhline(6, color=sc.PROBLEM, lw=1.2, ls=":")
ax.text(256, 7.0, "6-byte floor", color=sc.PROBLEM, fontsize=18, ha="right", fontweight="bold")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False, fontsize=17,
          handlelength=1.6, columnspacing=1.0)
ax.text(256, 12.0, "Lower is better", ha="right", va="center", fontsize=16, fontstyle="italic", color=sc.GOOD)
sc.save(fig, __file__)
