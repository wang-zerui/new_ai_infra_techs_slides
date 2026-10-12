"""Hybrid-EP (DeepEP hybrid-ep branch) on H100: FP8 dispatch time relative to BF16 dispatch, derived.

Published kernel-only IB algorithm bandwidth, 16 SMs, 4096 tokens, h 7168, top-8 ([HE-DOC] H100 table):
  EP16 FP8 68.48 / BF16 71.71 GB/s; EP32 55.35 / 57.69; EP64 46.52 / 49.97.
The benchmark counts FP8 bytes as BF16 bytes x (1 + 4/128)/2 ([HE-TEST]), so
  time_FP8 / time_BF16 = 0.516 x GB/s_BF16 / GB/s_FP8.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

f = (1 + 4 / 128) / 2
groups = ["16 GPUs", "32 GPUs", "64 GPUs"]
fp8_bw = [68.48, 55.35, 46.52]
bf16_bw = [71.71, 57.69, 49.97]
rel = [f * b / a for a, b in zip(fp8_bw, bf16_bw)]
print([round(r, 3) for r in rel])

sc.setup()
fig, ax = sc.figure("right")
w = 0.36
xs = range(len(groups))
ax.bar([x - w / 2 for x in xs], [1.0] * 3, w, color=sc.GRAYS[0], edgecolor=sc.INK, linewidth=1.2, hatch="//",
       label="BF16 dispatch")
ax.bar([x + w / 2 for x in xs], rel, w, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2, label="FP8 dispatch")
for x, r in zip(xs, rel):
    ax.text(x + w / 2, r + 0.03, f"{r:.2f}", ha="center", va="bottom", fontsize=20, fontweight="bold",
            color=sc.ACCENT)
ax.axhline(f, color=sc.PROBLEM, lw=2, ls="--", zorder=0, label="byte ratio 0.52")
ax.set_xticks(list(xs))
ax.set_xticklabels(groups)
ax.set_xlim(-0.6, 2.65)
ax.set_ylim(0, 1.32)
ax.set_ylabel("Dispatch time\n(BF16 = 1, derived)")
sc.top_legend(ax, ncol=3, fontsize=17)
sc.better(ax, "Lower is better", where="upper right")
sc.save(fig, __file__)
