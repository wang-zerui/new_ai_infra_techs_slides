"""MTP accept length: GLM-5 vs DeepSeek-V3.2 at the same number of speculative steps.

Numbers from the GLM-5 technical report (arXiv 2602.15763), Table 2 (p.5): accept length 2.76 for
GLM-5 and 2.55 for DeepSeek-V3.2, measured with 4 speculative steps on the authors' private prompt
set. GLM-5 shares the parameters of 3 MTP layers during training (same draft-model memory as a
single-layer MTP).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

NAMES = ["DeepSeek-V3.2", "GLM-5"]
ACCEPT = [2.55, 2.76]

sc.setup()
fig, ax = sc.figure("right")
sc.bars(ax, NAMES, ACCEPT, ours=1, labels=["2.55", "2.76"])
ax.set_ylabel("Accept length (tokens/step)")
ax.set_ylim(0, 3.6)
sc.better(ax, "Higher is better", "upper left")
sc.save(fig, __file__)
