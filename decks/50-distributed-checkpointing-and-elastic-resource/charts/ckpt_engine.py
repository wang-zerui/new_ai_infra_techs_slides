"""Broadcast weight-update time with MoonshotAI checkpoint-engine (vLLM v0.10.2rc1, FP8, H20, TP16 instances).
Source: checkpoint-engine README benchmark table. 16 GPUs = 1 TP16 instance; 256 GPUs = 16 TP16 instances.
Kimi-K2-Instruct: 14.36 s -> 16.04 s; DeepSeek-V3.1: 10.19 s -> 11.33 s."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

groups = ["DeepSeek-V3.1\n(FP8)", "Kimi K2\n(1T, FP8)"]
series = ["16 GPUs (1 replica)", "256 GPUs (16 replicas)"]
vals = [[10.19, 14.36], [11.33, 16.04]]
sc.setup()
fig, ax = sc.figure("right")
sc.grouped(ax, groups, series, vals, ours=1, width=0.7)
w = 0.35
for g in range(2):
    for s_ in range(2):
        x = g - 0.35 + w * (s_ + 0.5)
        ax.text(x, vals[s_][g] + 0.4, f"{vals[s_][g]:.2f}", ha="center", va="bottom", fontsize=18,
                fontweight="bold", color=sc.ACCENT if s_ == 1 else sc.INK)
ax.set_ylim(0, 21)
ax.set_ylabel("Weight update time (s)")
sc.better(ax, "Lower is better", where="upper left")
sc.save(fig, __file__)
