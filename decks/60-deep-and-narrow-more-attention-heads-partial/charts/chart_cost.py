"""Analytic systems cost of the GLM-4.5 shape (from published configs, not measured).

(a) BF16 KV cache per token of context:
    GLM-4.5 GQA: 92 layers x 2 (K,V) x 8 KV heads x 128 dims x 2 B = 376,832 B = 368 KiB
    DeepSeek-V3 / Kimi K2 MLA: 61 layers x (512 latent + 64 RoPE key) x 2 B = 70,272 B = 68.6 KiB
    (GLM-4.5 Table 1; HF configs; DeepSeek-V3 Sec. 2.1.1: only c_KV and k_R are cached)
(b) EP all-to-alls per forward pass = 2 (dispatch + combine) x MoE layers = 2x58, 2x60, 2x89 (GLM-4.5 Table 1)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

models = ["DeepSeek-V3", "Kimi K2", "GLM-4.5"]
kv_kib = [61 * 576 * 2 / 1024, 61 * 576 * 2 / 1024, 92 * 2 * 8 * 128 * 2 / 1024]
a2a = [2 * 58, 2 * 60, 2 * 89]
colors = [sc.GRAYS[0], sc.GRAYS[1], sc.ACCENT]
hatches = [sc.HATCHES[0], sc.HATCHES[1], None]

sc.setup()
fig, (a, b) = plt.subplots(1, 2, figsize=(11.8, 3.9), gridspec_kw={"wspace": 0.32})
for ax, vals, fmt, top, ylab, xl in (
        (a, kv_kib, "{:.1f}", 470, "KV cache per token (KiB)", "(a) BF16 KV cache, all layers"),
        (b, a2a, "{:d}", 230, "All-to-alls per forward pass", "(b) EP dispatch + combine")):
    for x, v, c, h in zip(range(3), vals, colors, hatches):
        ax.bar(x, v, 0.62, color=c, edgecolor=sc.INK, linewidth=1.2, hatch=h)
        lab = fmt.format(round(v, 1) if isinstance(v, float) else v)
        if isinstance(v, float) and abs(v - round(v)) < 1e-9:
            lab = f"{int(v)}"
        ax.text(x, v + top * 0.02, lab, ha="center", va="bottom", fontsize=20, fontweight="bold",
                color=sc.ACCENT if x == 2 else sc.INK)
    ax.set_xticks([])
    ax.set_ylim(0, top)
    ax.set_ylabel(ylab)
    ax.set_xlabel(xl, labelpad=10)
    sc.better(ax, "Lower is better", where="upper left")

a.text(1.0, 240, "5.4×", ha="center", fontsize=26, fontweight="bold", color=sc.PROBLEM)
b.text(1.0, 160, "+53%", ha="center", fontsize=26, fontweight="bold", color=sc.PROBLEM)

handles = [Patch(facecolor=c, edgecolor=sc.INK, hatch=h, label=m) for m, c, h in zip(models, colors, hatches)]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
sc.save(fig, __file__)
