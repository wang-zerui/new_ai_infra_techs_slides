"""B200, EP8 (one NVLink domain), Torch-API algorithm bandwidth vs SMs given to the kernel.
DeepEP measured at 16-48 SMs; HybridEP at 16 SMs. 4096 tokens, hidden 7168, top-8, 8 local experts.
Source: DeepEP hybrid-ep branch docs/README_Hybrid-EP.md, "B200 Platform" table.
Dotted guides and the "28 / 36 SMs" labels are derived: the smallest measured DeepEP SM count whose
bandwidth reaches HybridEP's 16-SM bandwidth."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sms = [16, 24, 28, 32, 36, 40, 44, 48]
panels = [
    ("BF16 dispatch", [348, 494, 560, 619, 635, 629, 640, 646], 535.94, 28, 560),
    ("Combine", [302, 420, 477, 524, 549, 570, 577, 586], 530.86, 36, 549),
]
sc.setup()
fig, axes = sc.figure("full", 1, 2, sharey=True)
for ax, (name, dee, hep, cross_x, cross_y) in zip(axes, panels):
    ax.plot(sms, dee, "-o", color=sc.GRAYS[2], label="DeepEP, 16–48 SMs")
    ax.plot([16], [hep], "*", color=sc.ACCENT, markersize=26, label="HybridEP, 16 SMs", zorder=5)
    ax.hlines(hep, 16, cross_x, colors=sc.ACCENT, linestyles=":", linewidth=2.4)
    right = cross_x > 30
    ax.annotate(f"DeepEP needs {cross_x}", xy=(cross_x, cross_y), xytext=(cross_x + (6 if right else 1), 400),
                color=sc.ACCENT, fontsize=19, fontweight="bold", ha="right" if right else "left",
                arrowprops=dict(arrowstyle="-|>", color=sc.ACCENT, lw=2.2))
    ax.set_xlim(13, 50)
    ax.set_ylim(250, 720)
    ax.set_xticks([16, 24, 32, 40, 48])
    ax.set_xlabel(f"SMs  ({name})")
axes[0].set_ylabel("Bandwidth (GB/s)")
sc.better(axes[1], "Higher is better", where="upper left")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=2, frameon=False)
fig.tight_layout(w_pad=2.0)
sc.save(fig, __file__)
