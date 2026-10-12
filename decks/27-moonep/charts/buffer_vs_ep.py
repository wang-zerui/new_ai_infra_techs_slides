"""Analytic: per-GPU token-buffer size vs EP size (BF16). MoonEP [NvS, H] vs the copy-free worst case S*K*R*H.
Config = README comm benchmark: S=8192 tokens/rank, K=8, H=7168, E=384, token_padding=128 (sources.md, slide 10)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

S, K, H, E, PAD = 8192, 8, 7168, 384, 128
EPS = [8, 16, 32, 64]


def align(x, a):
    return -(-x // a) * a


moon = [align(S * K + (PAD - 1) * 2 * (E // r), PAD) * H * 2 / 1e9 for r in EPS]   # moonep/api.py NvS formula
worst = [S * K * r * H * 2 / 1e9 for r in EPS]                                     # Kimi K3 sec. 5.2.1: S x K x R

sc.setup()
fig, ax = sc.figure("right")
groups = [f"EP={r}" for r in EPS]
sc.grouped(ax, groups, ["Worst case S×K×R", "MoonEP: S×K + padding"], [worst, moon], ours=1)
w = 0.4
for g, (a, b) in enumerate(zip(worst, moon)):
    ax.text(g - w / 2, a * 1.12, f"{a:.1f}", ha="center", va="bottom", fontsize=18, color=sc.INK)
    ax.text(g + w / 2, b * 1.12, f"{b:.1f}", ha="center", va="bottom", fontsize=18, color=sc.ACCENT, fontweight="bold")
ax.set_yscale("log")
ax.set_ylim(0.5, 200)
ax.set_yticks([1, 10, 100]); ax.set_yticklabels(["1", "10", "100"])
ax.set_ylabel("Token buffer per GPU (GB)")
sc.better(ax, "Lower is better", "upper left")
sc.save(fig, __file__)
