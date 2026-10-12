"""DeepSeek-V3 per-GPU memory, BF16, PP4 x VPP4 x EP64 / 256 GPUs (arXiv 2603.07685 Table 3, p.20), with and
without memory-efficient permutation (~26.3 GB saved, Sec. 4.1.2 p.21). Derived split: activations 131.0 =
26.3 (expert outputs kept for the router-weight gradient) + 104.7 other; total after = 199.5 - 26.3 = 173.2."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

params = 36.4 + 32.1
other = 131.0 - 26.3
mep = 26.3
sc.setup()
fig, ax = sc.plt.subplots(figsize=(10.4, 3.9))
x = [0, 1]
w = 0.5
ax.bar(x, [params] * 2, w, color=sc.GRAYS[2], edgecolor=sc.INK, linewidth=1.2, label="Weights, grads, optimizer")
ax.bar(x, [other] * 2, w, bottom=[params] * 2, color=sc.GRAYS[0], edgecolor=sc.INK, linewidth=1.2, hatch="//",
       label="Other activations")
ax.bar([0], [mep], w, bottom=[params + other], color=sc.PROBLEM, edgecolor=sc.INK, linewidth=1.2,
       label="Expert outputs kept for ∂L/∂p")
ax.bar([1], [mep], w, bottom=[params + other], color="white", edgecolor=sc.ACCENT, linewidth=2.0, linestyle="--")
ax.text(1.29, params + other + mep / 2, "−26.3 GB", ha="left", va="center", fontsize=20, fontweight="bold",
        color=sc.ACCENT)
ax.text(0, 199.5 + 5, "199.5 GB", ha="center", va="bottom", fontsize=20, fontweight="bold", color=sc.INK)
ax.text(1, params + other + mep + 5, "173.2 GB", ha="center", va="bottom", fontsize=20, fontweight="bold",
        color=sc.ACCENT)
ax.axhline(80, color=sc.INK, lw=2.2, ls="--", label="H100 memory: 80 GB")
ax.set_xticks(x)
ax.set_xticklabels(["Standard", "Mem-efficient\npermutation (derived)"])
ax.set_xlim(-0.6, 1.8)
ax.set_ylim(0, 245)
ax.set_ylabel("Memory per GPU (GB)")
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=18, handlelength=1.4)
sc.better(ax, "Lower is better", where="upper right")
sc.save(fig, __file__)
