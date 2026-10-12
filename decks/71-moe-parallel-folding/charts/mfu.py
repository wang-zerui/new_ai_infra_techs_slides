"""MFU of Megatron-Core with a shared mapping vs with MoE Parallel Folding (BF16, H100, GBS 256).

Source: arXiv 2504.14960, Table 1 (p.8); best config per method (Table 3, p.18).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

models = ["Mixtral\n8x22B", "Llama3\n8x70B", "Qwen2\n57B-A14B", "Mixtral\nG8T8"]
gpus = [128, 256, 64, 128]
mcore = [46.3, 38.8, 35.3, 17.1]
fold = [49.3, 41.6, 39.0, 28.8]

sc.setup()
fig, ax = sc.figure("right")
sc.grouped(ax, [f"{m}\n({g} GPUs)" for m, g in zip(models, gpus)],
           ["MCore, shared mapping", "MCore + Folding"], [mcore, fold])
w = 0.4
for g, (a, b) in enumerate(zip(mcore, fold)):
    ax.text(g - w / 2, a + 1, f"{a:.1f}", ha="center", va="bottom", fontsize=16, color=sc.INK)
    ax.text(g + w / 2, b + 1, f"{b:.1f}", ha="center", va="bottom", fontsize=16, color=sc.ACCENT, fontweight="bold")
ax.set_ylim(0, 62)
ax.set_ylabel("MFU (%)")
ax.tick_params(axis="x", labelsize=16)
sc.callout(ax, "1.68×", xy=(3.2, 34.5), xytext=(3.0, 49), color=sc.ACCENT)
sc.save(fig, __file__)
