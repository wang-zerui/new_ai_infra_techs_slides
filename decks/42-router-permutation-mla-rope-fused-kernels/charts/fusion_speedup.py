"""Kernel-level speedups of fused MoE glue vs unfused execution.
Rows 1-2: TransformerEngine PR #1921 (fused permute+pad: 1.1x-1.6x; fused unpermute+unpad: 1.7x-3x across the
test_permutation.py benchmark configurations) - drawn as the measured range.
Rows 3-4: NVIDIA blog "Boosting MoE Training Throughput with Advanced Fusion Kernels" (fused GroupGEMM +
activation + quantize: forward up to 1.3x, backward up to 2.1x) - only the maximum is published, drawn from 1.0
with a hatch to mark "up to"."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

rows = [("permute + pad", 1.1, 1.6, "1.1–1.6×", False),
        ("unpermute + unpad", 1.7, 3.0, "1.7–3×", False),
        ("GroupGEMM+act+quant fwd", 1.0, 1.3, "up to 1.3×", True),
        ("GroupGEMM+dact+quant bwd", 1.0, 2.1, "up to 2.1×", True)]
sc.setup()
fig, ax = sc.figure("full")
ys = list(range(len(rows)))[::-1]
for y, (name, lo, hi, lab, upto) in zip(ys, rows):
    ax.barh(y, hi - lo, left=lo, height=0.55, color=sc.ACCENT_LIGHT if upto else sc.ACCENT,
            edgecolor=sc.INK, linewidth=1.2, hatch="//" if upto else None)
    ax.text(hi + 0.04, y, lab, va="center", ha="left", fontsize=20, fontweight="bold", color=sc.ACCENT)
ax.axvline(1.0, color=sc.INK, linewidth=1.6)
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(0.9, 3.5)
ax.set_xlabel("Kernel speedup over the unfused path (×)")
ax.grid(axis="x", linestyle="--", color="#BFBFBF")
ax.grid(axis="y", visible=False)
sc.better(ax, "Higher is better", where="upper right")
sc.save(fig, __file__)
