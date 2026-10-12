"""MTP loss weight schedules (production configs, not results).
DeepSeek-V3 (arXiv 2412.19437 §4.2 p.23): 0.3 for the first 10T tokens, 0.1 for the remaining 4.8T (14.8T total).
GLM-4.5 (arXiv 2508.06471 §2.4): 0.3 for the first 15T tokens, 0.1 for the remaining tokens (23T total, abstract)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

rows = [("GLM-4.5", [(0, 15, 0.3), (15, 23, 0.1)]),
        ("DeepSeek-V3", [(0, 10, 0.3), (10, 14.8, 0.1)])]
sc.setup()
fig, ax = sc.figure("right")
for y, (name, segs) in enumerate(rows):
    for x0, x1, lam in segs:
        dark = lam == 0.3
        ax.barh(y, x1 - x0, left=x0, height=0.55, color=sc.ACCENT if dark else sc.ACCENT_LIGHT,
                edgecolor=sc.INK, linewidth=1.2)
        ax.text((x0 + x1) / 2, y, f"λ = {lam}", ha="center", va="center", fontsize=19, fontweight="bold",
                color="white" if dark else sc.INK)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(0, 24)
ax.set_ylim(-0.6, 1.6)
ax.set_xlabel("Training tokens (T)")
ax.grid(axis="y", visible=False)
ax.grid(axis="x", linestyle="--", color="#BFBFBF", linewidth=0.8)
sc.save(fig, __file__)
