"""Published maximum speedups of HPC-Ops FP8 GroupGEMM over DeepGEMM on H20 (HPC-Ops README Performance table:
'GroupGEMM FP8 | Expert grouped matmul, prefill, decode | DeepGEMM | Up to 1.1x prefill, 1.88x decode';
same values in the January 2026 README table 'Maximum observed speedup per operator')."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sc.setup()
fig, ax = sc.figure("right")
sc.grouped(ax, ["prefill", "decode"], ["DeepGEMM", "HPC-Ops"], [[1.0, 1.0], [1.1, 1.88]], ours=1, width=0.64)
for x, v in ((0.16, 1.1), (1.16, 1.88)):
    ax.text(x, v + 0.04, f"{v:g}×", ha="center", va="bottom", fontsize=20, fontweight="bold", color=sc.ACCENT)
ax.set_ylim(0, 2.25)
ax.set_ylabel("Max speedup over\nDeepGEMM (×)")
sc.better(ax, "Higher is better", where="upper left")
sc.save(fig, __file__)
