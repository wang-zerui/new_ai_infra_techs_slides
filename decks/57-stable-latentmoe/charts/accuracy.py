"""LatentMoE (NVIDIA, arXiv 2601.18089) Table 4: Hybrid Mamba-Attention MoE 73B total / 8B active, 1T tokens,
baseline (8.09B active, 72.6B total) vs l-MoE_acc (8.02B active, 72.8B total, d/l = 4).
Plain LatentMoE: no RMSNorm/SiTU/QB (those are K3's additions)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

groups = ["MMLU-Pro", "MMLU", "Code", "Math", "Common-\nsense"]
base = [48.30, 70.10, 51.95, 78.32, 81.73]
lat = [52.87, 72.11, 55.14, 80.19, 82.10]

sc.setup()
fig, ax = sc.figure("right")
sc.grouped(ax, groups, ["standard MoE", "LatentMoE (4× narrower)"], [base, lat], ours=1, width=0.74)
ax.set_ylim(40, 90)
ax.set_ylabel("Accuracy (%)")
w = 0.74 / 2
ax.text(0 + w / 2, lat[0] + 0.8, "+4.6", ha="center", va="bottom", fontsize=19, fontweight="bold", color=sc.ACCENT)
sc.better(ax, "Higher is better", where="upper left")
sc.save(fig, __file__)
