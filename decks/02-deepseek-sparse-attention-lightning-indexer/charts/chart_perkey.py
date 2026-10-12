"""Analytic multiply-adds per cached key, per query token, one layer (DeepSeek-V3.2 shapes).

indexer: 64 heads x 128 dims (FP8)                       = 8,192
MLA decode (MQA mode): 128 heads x (576 QK + 512 PV)      = 139,264 (17.0x)
MLA dense prefill (MHA mode, V3.1-Terminus): 128 x (192 + 128) = 40,960 (5.0x)
Derivation and sources: notes/analytic.py, notes/sources.md.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

names = ["Indexer\n(FP8)", "Prefill\nattention", "Decode\nattention"]
vals = [8192 / 1000, 40960 / 1000, 139264 / 1000]
sc.setup()
fig, ax = sc.figure("half")
rects = sc.bars(ax, names, vals, ours=0, labels=["8.2K", "41K (5×)", "139K (17×)"], label_size=19)
ax.set_ylabel("Multiply-adds per key (K)")
ax.set_ylim(0, 165)
sc.save(fig, __file__)
