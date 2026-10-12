"""DeepSeek-V3 ablation (arXiv 2412.19437, Table 5, large MoE: 228.7B total / 20.9B active, 578B tokens):
aux-loss-free minus aux-loss-based score, per benchmark (derived differences of published numbers).
BBH 66.7->67.9, MMLU 68.3->67.2, DROP 67.1->67.1, TriviaQA 66.7->67.7, NQ 27.1->28.1,
HumanEval 40.2->46.3, MBPP 59.2->61.2, GSM8K 70.7->74.5, MATH 37.2->39.6."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc
import matplotlib.pyplot as plt

base = {"BBH": 66.7, "MMLU": 68.3, "DROP": 67.1, "TriviaQA": 66.7, "NQ": 27.1,
        "HumanEval": 40.2, "MBPP": 59.2, "GSM8K": 70.7, "MATH": 37.2}
free = {"BBH": 67.9, "MMLU": 67.2, "DROP": 67.1, "TriviaQA": 67.7, "NQ": 28.1,
        "HumanEval": 46.3, "MBPP": 61.2, "GSM8K": 74.5, "MATH": 39.6}
delta = {k: round(free[k] - base[k], 1) for k in base}
order = sorted(delta, key=lambda k: -delta[k])

sc.setup()
fig, ax = plt.subplots(figsize=(11.6, 3.9))
xs = range(len(order))
vals = [delta[k] for k in order]
for x, k, v in zip(xs, order, vals):
    if v > 0:
        ax.bar(x, v, 0.62, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2)
    else:
        ax.bar(x, v if v else 0.0, 0.62, color=sc.GRAYS[0], edgecolor=sc.INK, linewidth=1.2, hatch="//")
    txt = f"+{v:.1f}" if v > 0 else ("0.0" if v == 0 else f"−{abs(v):.1f}")
    y = v + 0.25 if v >= 0 else v - 0.25
    ax.text(x, y, txt, ha="center", va="bottom" if v >= 0 else "top", fontsize=19, fontweight="bold",
            color=sc.ACCENT if v > 0 else sc.INK)
ax.axhline(0, color=sc.INK, lw=1.4)
ax.set_xticks(list(xs))
ax.set_xticklabels([{'HumanEval': 'Human\nEval', 'TriviaQA': 'Trivia\nQA'}.get(k, k) for k in order])
ax.set_ylim(-3.0, 7.6)
ax.set_ylabel("Δ score vs.\naux loss (points)")
sc.better(ax, "Higher is better", where="upper right")
sc.save(fig, __file__)
print(delta)
