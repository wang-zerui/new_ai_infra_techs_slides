"""Put-with-signal ping-pong round-trip latency for small messages (4-128 B) between two H100 GPUs on NVIDIA EOS.
Source: Hamidouche et al., "GPU-Initiated Networking for NCCL", arXiv 2511.15076v2, Sec. V-A (NCCL 2.28, NVSHMEM 3.4.5)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

names = ["NVSHMEM\nIBGDA", "NVSHMEM\nIBRC", "GIN\nProxy", "GIN\nGDAKI"]
vals = [24.3, 16.0, 18.0, 16.7]
sc.setup()
fig, ax = sc.figure("right")
rects = ax.bar(range(4), vals, 0.62, color=[sc.GRAYS[1], sc.GRAYS[0], sc.ACCENT_LIGHT, sc.ACCENT], edgecolor=sc.INK, linewidth=1.2)
rects[0].set_hatch("//"); rects[1].set_hatch("\\\\")
for r, v, c in zip(rects, vals, [sc.INK, sc.INK, sc.ACCENT, sc.ACCENT]):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=20, fontweight="bold", color=c)
ax.set_xticks(range(4)); ax.set_xticklabels(names)
ax.set_ylabel("Round-trip latency (µs)")
ax.set_ylim(0, 30)
sc.better(ax, "Lower is better", where="upper right")
ax.text(3, 8.3, "−31%\nvs\nIBGDA", color="white", fontsize=19, fontweight="bold", ha="center", va="center")
sc.save(fig, __file__)
