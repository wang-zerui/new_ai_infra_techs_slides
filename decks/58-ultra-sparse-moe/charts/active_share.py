"""Activated / total parameters (derived from published configs; see notes/sources.md):
DeepSeek-V3 37/671, GLM-5 40/744, V4-Flash 13/284, Kimi K2 32.6/1040, V4-Pro 49/1600."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sc.setup()
fig, ax = sc.figure("right")
names = ["V3\n(2024)", "GLM-5", "V4-\nFlash", "Kimi\nK2", "V4-\nPro"]
act = [37, 40, 13, 32.6, 49]
tot = [671, 744, 284, 1040, 1600]
vals = [a / t * 100 for a, t in zip(act, tot)]
sc.bars(ax, names, vals, ours=[1, 2, 3, 4], labels=[f"{v:.1f}%" for v in vals])
ax.set_ylabel("Active ÷ total params (%)")
ax.set_ylim(0, 7.5)
sc.better(ax, "Lower = sparser", where="upper right", color=sc.INK)
sc.save(fig, __file__)
