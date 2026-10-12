"""Per-checkpoint training stall (checkpointing overhead, excluding the first checkpoint), Llama 3 405B on
54 A3 Ultra VMs (432 H200 GPUs). Source: PyTorch blog "Distributed Checkpoint: Efficient checkpointing in
large-scale jobs" (2025-09-11), results table. Each column adds to the previous one.
12x is derived: 18.5 / 1.5 = 12.3."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

names = ["baseline\nasync", "+ plan &\nmeta cache", "+ process\n+ pinned", "+ local\nckpt"]
vals = [18.5, 5.5, 1.5, 2.3]
sc.setup()
fig, ax = sc.figure("right")
rects = sc.bars(ax, names, vals, ours=[2, 3], labels=["18.5 s", "5.5 s", "1.5 s", "2.3 s"])
rects[3].set_color(sc.ACCENT_LIGHT); rects[3].set_edgecolor(sc.INK)
ax.set_ylabel("Training stall per save (s)")
sc.callout(ax, "12× less", xy=(2, 3.2), xytext=(2.6, 12.5), color=sc.ACCENT, size=22)
sc.better(ax, "Lower is better", where="upper right")
sc.save(fig, __file__)
