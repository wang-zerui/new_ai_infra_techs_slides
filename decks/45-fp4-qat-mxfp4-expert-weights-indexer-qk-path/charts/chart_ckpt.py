"""Released checkpoint bytes per parameter: sum of *.safetensors sizes on Hugging Face (2026-10-13) divided by
the stated total parameter counts ([V4] Abstract, Table 1). Derived; parameter counts are rounded."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "notes"))
import slidechart as sc
import analytic as an

sc.setup()
fig, ax = sc.figure("right")
keys = ["DeepSeek-V3.2", "DeepSeek-V4-Flash", "DeepSeek-V4-Pro"]
vals = [an.CKPT_GB[k][0] * 1e9 / an.CKPT_GB[k][1] for k in keys]
sc.bars(ax, ["V3.2\n(FP8)", "V4-Flash", "V4-Pro"], vals, ours=[1, 2], labels=[f"{v:.2f}" for v in vals])
ax.set_ylabel("Checkpoint bytes\nper parameter")
ax.set_ylim(0, 1.3)
ax.axhline(an.MXFP4, color="#595959", lw=2.2, ls="--", zorder=3)
ax.text(-0.42, an.MXFP4 - 0.07, "pure MXFP4: 0.53", color="#595959", fontsize=18, fontweight="bold", va="top",
        bbox=dict(facecolor="white", edgecolor="none", pad=2), zorder=4)
sc.better(ax, "Lower is better", "upper right")
sc.save(fig, __file__)
