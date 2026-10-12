"""Packed sequences vs padding, RL training with thinking models (Megatron-Core MoE report, arXiv 2603.07685
§6.4.1): memory -40..60%, throughput 1.5..2x. Bars show the published ranges relative to padding = 1.0
(memory 0.4-0.6 of padding; throughput 1.5-2.0x)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sc.setup()
fig, (ax1, ax2) = sc.figure("right", 1, 2)
fig.subplots_adjust(wspace=0.55)
for ax, lo, hi, lab, ylab, better in [
        (ax1, 0.4, 0.6, "0.4–0.6×", "Memory\n(vs padding)", "Lower is better"),
        (ax2, 1.5, 2.0, "1.5–2×", "Throughput\n(vs padding)", "Higher is better")]:
    ax.bar(0, 1.0, 0.6, color=sc.GRAYS[0], edgecolor=sc.INK, linewidth=1.2, hatch="//")
    ax.bar(1, hi - lo, 0.6, bottom=lo, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2)
    ax.bar(1, lo, 0.6, color="none", edgecolor=sc.ACCENT, linewidth=1.2, linestyle="--")
    ax.text(0, 1.04, "1.0×", ha="center", va="bottom", fontsize=19, fontweight="bold", color=sc.INK)
    ax.text(1, hi + 0.04, lab, ha="center", va="bottom", fontsize=19, fontweight="bold", color=sc.ACCENT)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pad", "pack"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 2.45)
    ax.set_ylabel(ylab)
    ax.text(0.5, -0.17, better, transform=ax.transAxes, ha="center", va="top", fontsize=16, fontstyle="italic",
            color=sc.GOOD)
sc.save(fig, __file__)
