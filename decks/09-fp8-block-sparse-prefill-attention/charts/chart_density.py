"""HPC-BSA FP8 prefill latency relative to dense FP8 vs block density.
Source: HPC-Ops PR #45 benchmark figure "HPC-BSA Kernel Latency vs Sparsity" (batch=4, seq=65536, Hq=4, Hkv=1,
D=128, H20). Values are the percentage labels printed on that figure (latency as % of dense)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

DENSITY = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]          # % of causal KV tiles kept
BSA_REL = [101, 91, 81, 71, 61, 51, 41, 32, 22, 11]          # % of dense FP8 latency (printed labels)
sc.setup()
fig, ax = sc.figure("full")
ax.plot(DENSITY, [100] * len(DENSITY), color=sc.GRAYS[2], lw=3, ls="--", label="Dense FP8 (HPC, FA3)")
ax.plot(DENSITY, DENSITY, color=sc.GRAYS[1], lw=2.2, ls=":", label="Ideal: latency = density")
ax.plot(DENSITY, BSA_REL, color=sc.ACCENT, lw=3.4, marker="o", ms=10, label="HPC-Ops block-sparse FP8")
for d, r in zip(DENSITY, BSA_REL):
    ax.text(d, r + 6, f"{r}%", ha="center", va="bottom", fontsize=17, fontweight="bold", color=sc.ACCENT,
            bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))
ax.set_xlim(104, 6)
ax.set_ylim(0, 132)
ax.set_xticks(DENSITY)
ax.set_xticklabels([f"{d}%" for d in DENSITY])
ax.set_xlabel("Block density (share of causal KV tiles kept)")
ax.set_ylabel("Latency (% of dense FP8)")
sc.top_legend(ax, ncol=3)
ax.text(0.99, 0.97, "Lower is better", transform=ax.transAxes, ha="right", va="top", fontsize=16,
        fontstyle="italic", color=sc.GOOD)
sc.save(fig, __file__)
