"""Mega MoE compute rate and NVLink traffic per batch size, EP8 on SM100 (averages over 8 ranks).

Source: DeepGEMM PR #316 tables, columns "Compute (TFLOPS)" and "Interconnect (GB/s)".
Both rise together: tensor-core work and NVLink transfers run concurrently inside one kernel.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

batches = ["1", "512", "8192", "32768"]
tflops = {"V4-Flash": [5, 1056, 1928, 2038], "V4-Pro": [7, 1098, 2304, 2438]}
nvl = {"V4-Flash": [1, 266, 499, 529], "V4-Pro": [1, 182, 393, 417]}

sc.setup()
fig, axes = sc.plt.subplots(1, 2, figsize=(11.6, 4.0))  # full width, room for a callout
w = 0.38
xs = list(range(len(batches)))
for ax, data, ylab, top, fmt in [(axes[0], tflops, "Tensor-core rate (TFLOPS)", 2900, "{:d}"),
                                 (axes[1], nvl, "NVLink traffic (GB/s)", 640, "{:d}")]:
    for i, (name, color) in enumerate([("V4-Flash", sc.ACCENT), ("V4-Pro", sc.ACCENT_LIGHT)]):
        off = -w / 2 if i == 0 else w / 2
        ax.bar([x + off for x in xs], data[name], w, color=color, edgecolor=sc.INK, linewidth=1.2, label=name)
        for x, v in zip(xs, data[name]):
            if v > top * 0.3:   # tall bar: value inside, rotated
                ax.text(x + off, v - top * 0.02, fmt.format(v), ha="center", va="top", fontsize=16, rotation=90,
                        color="#FFFFFF" if i == 0 else sc.INK, fontweight="bold")
            else:
                ax.text(x + off, v + top * 0.015, fmt.format(v), ha="center", va="bottom", fontsize=16,
                        color=sc.ACCENT if i == 0 else sc.INK, fontweight="bold" if i == 0 else "normal")
    ax.set_xticks(xs)
    ax.set_xticklabels(batches)
    ax.set_xlabel("Tokens per rank (EP8)")
    ax.set_ylabel(ylab)
    ax.set_ylim(0, top)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout(rect=(0, 0, 1, 0.93))
sc.save(fig, __file__)
