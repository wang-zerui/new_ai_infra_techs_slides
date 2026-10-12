"""engram_offload: Engram paper (arXiv 2601.07372) sec. 6.4, Table 4: inference throughput with a 100B-parameter
Engram table entirely in host DRAM (H800, nano-vLLM, 512 sequences, length U(100, 1024)).
Not a DeepEP measurement. Drops: 4B 1.9% (derived), 8B 2.8% (quoted in the paper)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

groups = ["4B dense", "8B dense"]
base = [9031.62, 6315.52]
off = [8858.28, 6140.02]

sc.setup()
fig, ax = sc.figure("right")
sc.grouped(ax, groups, ["baseline", "+100B Engram in host DRAM"], [[b / 1000 for b in base], [o / 1000 for o in off]],
           ours=1, width=0.72)
ax.set_ylabel("Throughput (k tokens/s)")
ax.set_ylim(0, 11.5)
for g, (b, o) in enumerate(zip(base, off)):
    drop = (1 - o / b) * 100
    ax.text(g + 0.18, o / 1000 + 0.25, f"−{drop:.1f}%", ha="center", va="bottom", fontsize=20, fontweight="bold",
            color=sc.ACCENT)
sc.better(ax, "Higher is better", "upper right")
sc.save(fig, __file__)
