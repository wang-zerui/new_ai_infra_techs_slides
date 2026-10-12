"""Family 'memory toolbox' ledger for decks 46/49/52/53/54: per-GPU savings each Megatron-Core technique
reports against the DeepSeek-V3 BF16 baseline of Table 3 (arXiv 2603.07685):
  fine-grained recompute 42.4 GB (Table 4, p.24), memory-efficient permutation ~26.3 GB (p.21),
  optimizer-state offload 15-20 GB (p.27, GB200), FP8 activations ~16 GB (p.22),
  BF16 optimizer moments ~10-12 GB (p.26). Ranges drawn as solid low + light extension to high.
HIGHLIGHT selects this deck's bar(s)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

HIGHLIGHT = {"Mem-efficient permutation (53)"}
rows = [  # name, low, high, trade-off label, '~' if the paper says 'approximately'
    ("Selective recompute (52)", 42.4, 42.4, "<5% compute", ""),
    ("Mem-efficient permutation (53)", 26.3, 26.3, "no overhead", "~"),
    ("Optimizer-state offload (54)", 15, 20, "0.1–0.2 s/iter", ""),
    ("FP8 activations (46)", 16, 16, "precision", "~"),
    ("BF16 optimizer moments (49)", 10, 12, "precision", ""),
]
sc.setup()
fig, ax = sc.plt.subplots(figsize=(11.6, 3.7))
ys = list(range(len(rows)))[::-1]
for y, (n, lo, hi, lab, approx) in zip(ys, rows):
    hl = n in HIGHLIGHT
    c = sc.ACCENT if hl else sc.GRAYS[1]
    ax.barh([y], [lo], height=0.62, color=c, edgecolor=sc.INK, linewidth=1.2)
    if hi > lo:
        ax.barh([y], [hi - lo], left=[lo], height=0.62, color=sc.ACCENT_LIGHT if hl else sc.GRAYS[0],
                edgecolor=sc.INK, linewidth=1.2, hatch="//")
    val = f"{lo:g}–{hi:g} GB" if hi > lo else f"{approx}{lo:g} GB"
    ax.text(hi + 0.8, y, f"{val}  ·  {lab}", va="center", ha="left", fontsize=18,
            fontweight="bold" if hl else "normal", color=sc.ACCENT if hl else sc.INK)
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows])
for t, r in zip(ax.get_yticklabels(), rows):
    if r[0] in HIGHLIGHT:
        t.set_color(sc.ACCENT); t.set_fontweight("bold")
ax.set_xlim(0, 66)
ax.set_xlabel("Memory saved per GPU (GB), DeepSeek-V3")
ax.grid(axis="x", linestyle="--", color="#BFBFBF"); ax.grid(axis="y", visible=False)
sc.save(fig, __file__)
