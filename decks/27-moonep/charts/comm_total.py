"""Per-layer EP communication time (dispatch fwd + dispatch bwd + combine fwd + combine bwd), H20, EP=8,
E=384, H=7168, K=8, S=8192/rank, 32 SMs. Sums of the labeled bars in MoonEP README figure/comm_vs_deepep.png;
MoonEP bars include its planning and weight-prefetch kernels (sources.md, slide 11)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

MAXVIO = ["0.2", "1", "10", "20"]
moon = {"d_f": [2598, 2644, 2629, 2413], "d_b": [2526, 2548, 2550, 2340],
        "c_f": [1975, 1932, 1955, 1954], "c_b": [1945, 1944, 1971, 1963]}
deep = {"d_f": [2296, 2299, 2408, 2604], "d_b": [2311, 2310, 2421, 2624],
        "c_f": [2126, 2142, 2306, 2468], "c_b": [2137, 2151, 2318, 2479]}
m = [sum(v[i] for v in moon.values()) / 1000 for i in range(4)]
d = [sum(v[i] for v in deep.values()) / 1000 for i in range(4)]

sc.setup()
fig, ax = sc.figure("right")
sc.grouped(ax, MAXVIO, ["DeepEP v2", "MoonEP"], [d, m], ours=1)
w = 0.4
for g in range(4):
    ax.text(g - w / 2, d[g] + 0.15, f"{d[g]:.1f}", ha="center", va="bottom", fontsize=17, color=sc.INK)
    ax.text(g + w / 2, m[g] + 0.15, f"{m[g]:.1f}", ha="center", va="bottom", fontsize=17, color=sc.ACCENT, fontweight="bold")
ax.set_ylim(0, 13.5)
ax.set_xlabel("Routing imbalance (maxvio)")
ax.set_ylabel("EP comm per layer (ms)")
sc.better(ax, "Lower is better", "upper left")
saving = (d[3] - m[3]) / d[3] * 100
sc.callout(ax, f"−{saving:.0f}%", xy=(3 + w / 2, m[3] + 0.75), xytext=(3 + w / 2, 12.4), color=sc.ACCENT)
sc.save(fig, __file__)
