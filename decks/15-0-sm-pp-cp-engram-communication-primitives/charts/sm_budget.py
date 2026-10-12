"""sm_budget: SMs held by communication per GPU.
Data (see notes/sources.md):
  DeepSeek-V3 report, H800: 20 of 132 SMs                 [DSV3 sec. 3.5.1 p.20]
  DeepEP V1, V3-like legacy training: 24 SMs               [DeepEP README @ b306af06af l.22]
  DeepEP V2 EP, same setting: 4-6 SMs (bar drawn to 6)     [same]
  DeepEP V2 PP / CP all-gather / Engram transfer: 0 SMs    [same, l.23-25 "0 SM ..."]
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

names = ["V3 report\n(H800)", "DeepEP\nV1 EP", "DeepEP\nV2 EP", "V2 PP·CP\n·Engram"]
values = [20, 24, 6, 0]
labels = ["20", "24", "4–6", "0"]

sc.setup()
fig, ax = sc.figure("right")
sc.bars(ax, names, values, ours=3, labels=labels)
ax.set_ylabel("Communication SMs per GPU")
ax.plot([2.69, 3.31], [0.12, 0.12], color=sc.ACCENT, lw=7, solid_capstyle="butt")
ax.set_ylim(0, 30)
sc.better(ax, "Lower is better", "upper right")
sc.save(fig, __file__)
