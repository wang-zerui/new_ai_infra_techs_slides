"""HPC-Ops Stem mask generation speedup over the Triton baseline vs sequence length.
Source: HPC-Ops PR #50 figure "HPC-Stem Speedup vs Triton Baseline" (printed bar labels)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

LENS = ["8K", "16K", "32K", "64K", "128K", "256K"]
SPEEDUP = [7.2, 8.0, 9.6, 16.3, 27.0, 44.0]
sc.setup()
fig, ax = sc.figure("right")
sc.bars(ax, LENS, SPEEDUP, ours=list(range(len(LENS))), labels=[f"{v:.1f}×" for v in SPEEDUP], label_size=19)
ax.set_xlabel("Sequence length")
ax.set_ylabel("Speedup over Triton (×)")
ax.text(0.03, 0.95, "Higher is better", transform=ax.transAxes, ha="left", va="top", fontsize=16,
        fontstyle="italic", color=sc.GOOD)
sc.save(fig, __file__)
