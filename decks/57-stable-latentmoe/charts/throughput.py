"""LatentMoE (arXiv 2601.18089) Table 5: tokens/s/GPU, Hybrid-73BT-8BA, two H100, vLLM, FP8 per-tensor,
LatentMoE (l-MoE_acc) vs standard MoE at concurrency 1/4/16/64/128. Deltas derived (LatentMoE/standard - 1)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

conc = ["1", "4", "16", "64", "128"]
std = [206.6, 509.8, 1204.6, 1549.3, 1725.9]
lat = [181.6, 528.5, 1130.8, 1569.6, 1625.8]

sc.setup()
fig, ax = sc.figure("right")
sc.grouped(ax, conc, ["standard MoE", "LatentMoE"], [std, lat], ours=1, width=0.74)
w = 0.74 / 2
for i, (s_, l_) in enumerate(zip(std, lat)):
    d = l_ / s_ - 1
    ax.text(i + w / 2, max(s_, l_) + 30, f"{d * 100:+.0f}%".replace("-", "−"), ha="center", va="bottom", fontsize=17,
            fontweight="bold", color=sc.PROBLEM if d < -0.1 else sc.INK)
ax.set_ylim(0, 2050)
ax.set_xlabel("Concurrent requests")
ax.set_ylabel("Tokens/s per GPU")
sc.better(ax, "Higher is better", where="upper left")
sc.save(fig, __file__)
