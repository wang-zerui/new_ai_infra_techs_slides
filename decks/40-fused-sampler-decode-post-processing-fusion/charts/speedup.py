"""Fused sampler speedup ranges over batch 1-512 (BF16 logits, V = 120,832).
Source: hpc-ops PR #51 benchmark line ("batch 1-512: 4.6x-8.5x vs vLLM, 2.1x-5.2x vs FlashInfer"); README "up to 8.5x"."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sc.setup()
fig, ax = sc.figure("right")
rows = [("vs vLLM-style\nPyTorch", 4.6, 8.5), ("vs FlashInfer", 2.1, 5.2)]
for i, (name, lo, hi) in enumerate(rows):
    ax.barh(i, hi - lo, left=lo, height=0.5, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2)
    ax.text(lo + 0.15, i, f"{lo}×", ha="left", va="center", fontsize=20, fontweight="bold", color="white")
    ax.text(hi + 0.15, i, f"{hi}×", ha="left", va="center", fontsize=20, fontweight="bold", color=sc.ACCENT)
ax.set_yticks([0, 1]); ax.set_yticklabels([r[0] for r in rows])
ax.invert_yaxis()
ax.axvline(1, color=sc.GRAYS[2], ls="--", lw=1.5)
ax.text(1.1, 1.42, "1× = parity", color=sc.GRAYS[3], fontsize=16, va="center")
ax.set_xlim(0, 10)
ax.set_ylim(1.6, -0.6)
ax.set_xlabel("Speedup range over batch 1–512 (×)")
ax.grid(axis="y", visible=False); ax.grid(axis="x", linestyle="--", color="#BFBFBF")
sc.better(ax, "Higher is better", where="upper right")
sc.save(fig, __file__)
