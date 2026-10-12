"""Theoretical speedup of one MoE layer over serial execution, DeepSeek-V4-Flash configuration.
Source: DeepSeek-V4 (arXiv 2606.19348) Fig. 5: Comet-style per-stage overlap 1.42x, expert waves 1.92x."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

names = ["Serial", "Per-stage\noverlap", "Expert\nwaves"]
vals = [1.00, 1.42, 1.92]

sc.setup()
fig, ax = sc.figure("right")
sc.bars(ax, names, vals, ours=2, labels=["1.00×", "1.42×", "1.92×"])
ax.set_ylabel("Theoretical speedup (×)")
ax.set_ylim(0, 2.3)
sc.better(ax)
sc.save(fig, __file__)
