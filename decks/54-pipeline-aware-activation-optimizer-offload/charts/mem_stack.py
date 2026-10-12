"""DeepSeek-V3 per-GPU memory in BF16 (PP4 x VPP4 x EP64, 256 GPUs): Megatron-Core MoE report,
arXiv 2603.07685, Table 3 (p.20). H100 capacity 80 GB from the same page. Shared family visual of
decks 52/53/54; this deck highlights both offload targets: activations and optimizer state."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

parts = [("Weights + grads", 36.4, sc.GRAYS[2], ""),
         ("Main weights + optimizer", 32.1, "#E8A0A0", "//"),
         ("Activations", 131.0, sc.PROBLEM, "")]
sc.setup()
fig, ax = sc.plt.subplots(figsize=(11.6, 2.9))
left = 0
for name, v, c, h in parts:
    ax.barh([0], [v], left=[left], height=0.56, color=c, edgecolor=sc.INK, linewidth=1.2, hatch=h or None)
    if c == sc.PROBLEM:
        ax.text(left + v / 2 + 14, 0, f"{name}: {v:.1f} GB", ha="center", va="center", fontsize=21,
                color="white", fontweight="bold")
    else:
        yy = 0.66 if left == 0 else 0.34
        ax.plot([left + v / 2, left + v / 2], [0.29, yy], color=sc.INK, lw=1.2)
        ax.text(left + v / 2 - 1.5, yy + 0.02, f"{name}: {v:.1f} GB", ha="left", va="bottom", fontsize=18,
                color=sc.INK)
    left += v
ax.plot([80, 80], [-0.6, 0.3], color=sc.INK, lw=2.4, ls="--")
ax.text(81.5, -0.36, "H100: 80 GB", ha="left", va="top", fontsize=18, color=sc.INK, fontweight="bold")
ax.text(199.5 + 3, 0, "199.5 GB", ha="left", va="center", fontsize=20, fontweight="bold", color=sc.INK)
ax.set_xlim(0, 240)
ax.set_ylim(-0.62, 1.0)
ax.set_yticks([])
ax.set_xlabel("Memory per GPU (GB), DeepSeek-V3, BF16")
ax.grid(axis="x", linestyle="--", color="#BFBFBF")
ax.grid(axis="y", visible=False)
sc.save(fig, __file__)
