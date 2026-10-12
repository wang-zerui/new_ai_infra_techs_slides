"""Max attention logit: measured explosions vs QK-Clip's cap vs QK-Norm's analytic bound (log scale).

Sources: 8B ViT without QK normalization, logits > 50,000 (Dehghani et al., arXiv 2302.05442, App. B p.23);
53B-total MoE with vanilla Muon, max logit > 1,000 (Kimi K2, arXiv 2507.20534, Sec. 2.1, Fig. 2 left);
Kimi K2 with QK-Clip, cap tau = 100 (same, Fig. 2 right);
QK-Norm analytic bound sqrt(d) * g_q * g_k = sqrt(128) = 11.3 at unit gains (derivation in notes/sources.md).
The first two bars are lower bounds of measured values from different models; the last two are caps/bounds.
"""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc
import matplotlib.pyplot as plt

names = ["8B ViT, Adam\nno fix", "53B MoE, Muon\nno fix", "Kimi K2\nQK-Clip cap", "QK-Norm bound\nd = 128, g = 1"]
vals = [50000, 1000, 100, math.sqrt(128)]
labels = ["> 50,000", "> 1,000", "100", "11.3"]
colors = [sc.PROBLEM, sc.PROBLEM, sc.GRAYS[1], sc.ACCENT]
hatches = ["//", "//", "\\\\", None]

sc.setup()
fig, ax = plt.subplots(figsize=(11.6, 4.0))
for x, (v, c, h, t) in enumerate(zip(vals, colors, hatches, labels)):
    ax.bar(x, v, 0.6, color="white" if h == "//" else c, edgecolor=c if h == "//" else sc.INK,
           linewidth=1.6, hatch=h)
    ax.text(x, v * 1.35, t, ha="center", va="bottom", fontsize=21, fontweight="bold",
            color=c if c != sc.GRAYS[1] else sc.INK)
ax.set_yscale("log")
ax.set_ylim(1, 1e6)
ax.set_xticks(range(4))
ax.set_xticklabels(names)
ax.set_ylabel("Max attention logit\n(log scale)")
sc.better(ax, "Lower is safer", where="upper right")
sc.save(fig, __file__)
