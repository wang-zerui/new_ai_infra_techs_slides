"""GLM-5 base model at 128K: MLA vs DSA on four long-context benchmarks.
Source: GLM-5 technical report, arXiv 2602.15763v2, Table 3 (p.5)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

groups = ["MQ-NIAH", "MV-NIAH", "SQuAD", "HotpotQA"]
mla = [100.0, 95.5, 79.7, 66.3]
dsa = [100.0, 97.0, 86.0, 63.0]
sc.setup()
fig, ax = sc.figure("half")
sc.grouped(ax, groups, ["MLA (dense)", "DSA"], [mla, dsa], ours=1)
for i, (a, b) in enumerate(zip(mla, dsa)):
    ax.text(i + 0.2, b + 1.5, f"{b:.0f}" if b == int(b) else f"{b:.1f}", ha="center", va="bottom", fontsize=16,
            color=sc.ACCENT, fontweight="bold")
ax.set_ylim(0, 115)
ax.set_ylabel("Score at 128K")
ax.tick_params(axis="x", labelsize=16)
sc.save(fig, __file__)
