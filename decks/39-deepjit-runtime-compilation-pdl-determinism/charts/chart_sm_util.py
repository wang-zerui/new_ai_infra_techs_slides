"""Analytic SM occupancy of a 7168-wide GEMM as the N tile changes (M = 256, BLOCK_M = 128, 132 SMs).

Derivation (see sources.md, "Derived number used on a chart"):
    CTAs      = ceil(256 / 128) * ceil(7168 / BLOCK_N)
    waves     = ceil(CTAs / 132)
    occupancy = CTAs / (132 * waves)
The two anchor points (BLOCK_N = 128 -> 112 CTAs, BLOCK_N = 112 -> 128 CTAs) are published in the
DeepGEMM README @ commit 8002b769, section "Unaligned block sizes".
"""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

NUM_SMS = 132
BLOCK_NS = [64, 80, 96, 112, 128, 144, 160]

occupancy, labels = [], []
for bn in BLOCK_NS:
    ctas = 2 * math.ceil(7168 / bn)
    waves = math.ceil(ctas / NUM_SMS)
    occupancy.append(100.0 * ctas / (NUM_SMS * waves))
    labels.append("")

chosen = BLOCK_NS.index(112)
labels[chosen] = "128 SMs"
labels[BLOCK_NS.index(128)] = "112 SMs"

sc.setup()
fig, ax = sc.figure("right")
sc.bars(ax, [str(bn) for bn in BLOCK_NS], occupancy, ours=chosen, problem=BLOCK_NS.index(128),
        labels=labels, width=0.66, label_size=18)
ax.set_ylabel("SM occupancy (%)")
ax.set_xlabel("BLOCK_N")
ax.set_ylim(0, 128)
sc.better(ax, "Higher is better", where="upper left")
sc.save(fig, __file__)
