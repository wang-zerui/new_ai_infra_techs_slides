"""Cost of naive batch-invariant inference (Thinking Machines, "Defeating Nondeterminism in LLM Inference", 2025):
Qwen3-8B on one GPU, 1000 requests with 90-110 output tokens. Ratios 55/26 = 2.1x and 42/26 = 1.6x are derived."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

names = ["vLLM\ndefault", "Batch-invariant\n(unoptimized)", "+ better\nattention kernel"]
vals = [26, 55, 42]

sc.setup()
fig, ax = sc.figure("right")
sc.bars(ax, names, vals, ours=[], problem=1, labels=["26 s", "55 s", "42 s"])
ax.set_ylabel("Time for 1000 requests (s)")
ax.set_ylim(0, 68)
sc.callout(ax, "2.1× slower", xy=(0.67, 44), xytext=(0.05, 50), color=sc.PROBLEM)
sc.better(ax, "Lower is better", where="upper right")
sc.save(fig, __file__)
