"""Accept length with 4 speculative steps on GLM's private prompt set (arXiv 2602.15763, Table 2):
DeepSeek-V3.2 2.55 vs GLM-5 2.76 (GLM-5 trains its MTP with parameters shared across 3 MTP layers).
Cross-model comparison, not an ablation. +8% = 2.76 / 2.55 - 1 (derived)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sc.setup()
fig, ax = sc.figure("half")
sc.bars(ax, ["DeepSeek-V3.2", "GLM-5"], [2.55, 2.76], ours=1, labels=["2.55", "2.76"])
ax.set_ylim(0, 3.4)
ax.set_ylabel("Accept length\n(tokens per step)")
sc.better(ax, "Higher is better", where="upper left")
sc.save(fig, __file__)
