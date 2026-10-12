"""Qwen3-30B-A3B on 32 H800: EP8 vs Megatron-FSDP + EP8 (user report, Megatron-LM issue #2772, comment of
2026-01-14, after the TE FusedAdam fix). Max reserved memory 45,552 -> 23,548 MB; throughput 137.8 -> 108.0 TFLOPS.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

names = ["EP8", "FSDP\n+ EP8"]
mem = [45.552, 23.548]
tf = [137.8, 108.0]

sc.setup()
fig, (a1, a2) = sc.figure("right", ncols=2)
fig.subplots_adjust(wspace=0.55)
r = a1.bar([0, 1], mem, width=0.62, color=[sc.GRAYS[0], sc.ACCENT], edgecolor=sc.INK, linewidth=1.2)
r[0].set_hatch("//")
a1.set_xticks([0, 1]); a1.set_xticklabels(names)
a1.set_ylabel("Max reserved (GB)")
a1.set_ylim(0, 58)
for x, v in zip([0, 1], mem):
    a1.text(x, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=19, fontweight="bold",
            color=sc.ACCENT if x else sc.INK)
a1.text(1, mem[1] + 9, "−48%", ha="center", va="bottom", fontsize=21, fontweight="bold", color=sc.ACCENT)
r = a2.bar([0, 1], tf, width=0.62, color=[sc.GRAYS[0], sc.GRAYS[2]], edgecolor=sc.INK, linewidth=1.2)
r[0].set_hatch("//")
a2.set_xticks([0, 1]); a2.set_xticklabels(names)
a2.set_ylabel("TFLOPS per GPU")
a2.set_ylim(0, 175)
for x, v in zip([0, 1], tf):
    a2.text(x, v + 4, f"{v:.1f}", ha="center", va="bottom", fontsize=19, fontweight="bold", color=sc.INK)
a2.text(1, tf[1] + 26, "−22%", ha="center", va="bottom", fontsize=21, fontweight="bold", color=sc.PROBLEM)
sc.save(fig, __file__)
