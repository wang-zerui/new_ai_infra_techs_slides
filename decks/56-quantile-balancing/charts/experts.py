"""Routed experts per MoE layer (and experts active per token).

Sources: loss-free balancing study 64 routed / 6 active ([LFB] arXiv 2408.15664, Table 5);
DeepSeek-V3 256 / 8 ([DSV3] arXiv 2412.19437, Sec. 4.2); Kimi K2 384 / 8 and Kimi K3 896 / 16
([K3] arXiv 2607.24653, Table 1).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

names = ["loss-free\nstudy", "DeepSeek\nV3", "Kimi\nK2", "Kimi\nK3"]
experts = [64, 256, 384, 896]
labels = ["64\ntop-6", "256\ntop-8", "384\ntop-8", "896\ntop-16"]

sc.setup()
fig, ax = sc.figure("right")
sc.bars(ax, names, experts, ours=3, labels=labels, label_size=19)
ax.set_ylabel("Routed experts per layer")
ax.set_ylim(0, 1150)
sc.save(fig, __file__)
