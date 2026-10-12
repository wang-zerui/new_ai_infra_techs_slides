"""Analytic: bytes each GPU receives per layer per attention head for context parallelism over P = 8 GPUs.
Softmax CP (KV all-gather, BF16 K and V, head dim d = 128, MHA-equivalent comparator): (P-1)/P * T * 2 * d * 2 bytes.
KCP (all-gather of S~ (d_k x d_v) and M (d_k x d_k), FP32 as in FLA chunk_delta_h.py, d_k = d_v = 128):
(P-1) * d_k * (d_k + d_v) * 4 bytes. Payload formula CP x H x K x (K + V) from FLA PR #691; head dim 128 is the only
shape FlashKDA supports (README). Ratio = T / (256 P) -> 512x at T = 1M, P = 8. Not a measurement."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import numpy as np
import slidechart as sc
P, d = 8, 128
T = np.array([32, 64, 128, 256, 512, 1024]) * 1024
soft = (P - 1) / P * T * 2 * d * 2 / 2**20
kcp = np.full_like(T, (P - 1) * d * (d + d) * 4, dtype=float) / 2**20
sc.setup()
fig, ax = sc.figure("right")
ax.plot(T, soft, color=sc.GRAYS[3], lw=3.5, marker="s", label="Softmax CP: K and V")
ax.plot(T, kcp, color=sc.ACCENT, lw=3.5, marker="o", label="KCP: S̃ and M")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(T); ax.set_xticklabels(["32K", "64K", "128K", "256K", "512K", "1M"])
ax.set_xlabel("Sequence length (tokens), 8 GPUs")
ax.set_ylabel("MiB per GPU, layer, head")
ax.set_ylim(0.3, 2000)
ax.annotate("", xy=(T[-1], kcp[-1] * 1.15), xytext=(T[-1], soft[-1] / 1.15),
            arrowprops=dict(arrowstyle="<->", color=sc.ACCENT, lw=2.2))
ax.text(T[-1] / 1.12, np.sqrt(soft[-1] * kcp[-1]), "512×", ha="right", va="center", fontsize=22, fontweight="bold", color=sc.ACCENT)
sc.top_legend(ax, ncol=2)
for t in ax.get_legend().get_texts(): t.set_fontsize(17)
sc.better(ax, "Lower is better", "upper left")
sc.save(fig, __file__)
print("soft@1M MiB", soft[-1], "kcp MiB", kcp[-1], "ratio", soft[-1] / kcp[-1])
