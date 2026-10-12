"""Analytic: bytes each GPU receives per layer per attention head vs CP degree P at T = 1M tokens.
Softmax CP (BF16 K and V, head dim 128, MHA-equivalent): (P-1)/P * T * 2 * 128 * 2 bytes.
KCP (FP32 S~ and M, d_k = d_v = 128): (P-1) * 128 * 256 * 4 bytes (payload formula CP x H x K x (K+V), FLA PR #691).
Shows that the KCP payload grows linearly with P but stays far below the KV volume. Not a measurement."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import numpy as np
import slidechart as sc
T, d = 2**20, 128
P = np.array([2, 4, 8, 16, 32, 64])
soft = (P - 1) / P * T * 2 * d * 2 / 2**20
kcp = (P - 1) * d * (d + d) * 4 / 2**20
sc.setup()
fig, ax = sc.figure("half")
ax.plot(P, soft, color=sc.GRAYS[3], lw=3.5, marker="s", label="Softmax CP")
ax.plot(P, kcp, color=sc.ACCENT, lw=3.5, marker="o", label="KCP")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(P); ax.set_xticklabels([str(p) for p in P])
ax.set_xlabel("CP degree (GPUs), T = 1M")
ax.set_ylabel("MiB per GPU, layer, head")
ax.set_ylim(0.05, 3000)
ax.text(64, kcp[-1] * 1.6, f"{kcp[-1]:.1f}", ha="center", va="bottom", fontsize=17, color=sc.ACCENT, fontweight="bold")
ax.text(2.35, kcp[0], f"{kcp[0]:.3f}", ha="left", va="center", fontsize=17, color=sc.ACCENT, fontweight="bold")
sc.top_legend(ax, ncol=2)
sc.better(ax, "Lower is better", "upper left")
sc.save(fig, __file__)
print(list(zip(P.tolist(), soft.round(1).tolist(), kcp.round(3).tolist())))
