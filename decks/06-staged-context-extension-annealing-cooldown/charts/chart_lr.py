"""chart_lr: learning-rate schedule vs cumulative training tokens, Kimi K2 (WSD + anneal) vs GLM-4.5 (cosine).
Sources (notes/sources.md): K2 §2.5 p.9 (500-step warm-up, 2e-4 const for 10T, cosine to 2e-5 over 5.5T,
anneal 2e-5 -> 7e-6 over 400B@4K + 60B@32K; batch 67M); GLM-4.5 §2.4 p.5 (warm-up to 2.5e-4, cosine to 2.5e-5
at end of mid-training, 23.1T total). K2 warm-up = 500 x 67M = 33.5B tokens (derived). The K2 anneal shape is not
published: drawn as a straight dashed segment. GLM-4.5 warm-up length not published: omitted."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc
import numpy as np

# Kimi K2 (T tokens, LR in 1e-4)
wu = 500 * 67e6 / 1e12
k2_x = [0, wu, 10.0]
k2_y = [0, 2.0, 2.0]
cx = np.linspace(10.0, 15.5, 200)
cy = 0.2 + 0.5 * (2.0 - 0.2) * (1 + np.cos(np.pi * (cx - 10.0) / 5.5))
ann_x = [15.5, 15.96]; ann_y = [0.2, 0.07]
# GLM-4.5
gx = np.linspace(0, 23.1, 400)
gy = 0.25 + 0.5 * (2.5 - 0.25) * (1 + np.cos(np.pi * gx / 23.1))

sc.setup()
fig, ax = sc.figure("full")
ax.plot(gx, gy, color=sc.GRAYS[2], lw=3.0, ls="-", label="GLM-4.5: cosine")
ax.plot(k2_x + list(cx), k2_y + list(cy), color=sc.ACCENT, lw=3.4, label="Kimi K2: WSD + anneal")
ax.plot(ann_x, ann_y, color=sc.ACCENT, lw=3.4, ls=(0, (2, 1.5)))
ax.set_xlim(0, 24.2); ax.set_ylim(0, 2.95)
ax.set_xlabel("Training tokens (T)")
ax.set_ylabel("Learning rate (×10⁻⁴)")
# direct labels
ax.text(4.6, 1.78, "K2 stable: 2e-4 for 10T", color=sc.ACCENT, fontsize=18, fontweight="bold", ha="center", va="center")
ax.text(9.0, 0.55, "K2 cosine: 5.5T", color=sc.ACCENT, fontsize=18, ha="left", va="center")
ax.annotate("K2 anneal to 7e-6\n+460B tokens, 4K then 32K", xy=(15.85, 0.12), xytext=(16.3, 1.8), color=sc.ACCENT,
            fontsize=18, fontweight="bold", ha="left", va="center",
            arrowprops=dict(arrowstyle="-|>", color=sc.ACCENT, lw=2, shrinkA=4, shrinkB=2))
ax.text(3.0, 2.68, "GLM-4.5 peak 2.5e-4", color=sc.GRAYS[3], fontsize=18, ha="left", va="center")
ax.annotate("GLM-4.5: 2.5e-5\nat 23.1T", xy=(23.05, 0.28), xytext=(21.4, 1.05), color=sc.GRAYS[3], fontsize=18,
            ha="center", va="center", arrowprops=dict(arrowstyle="-|>", color=sc.GRAYS[3], lw=2, shrinkA=4, shrinkB=2))
sc.top_legend(ax, ncol=2)
sc.save(fig, __file__)
