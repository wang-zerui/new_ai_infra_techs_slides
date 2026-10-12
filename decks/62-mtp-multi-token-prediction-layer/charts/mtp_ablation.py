"""DeepSeek-V3 MTP ablation (arXiv 2412.19437, Table 4, large MoE: 228.7B total / 20.9B active, 540B tokens):
w/ MTP minus baseline (derived differences of published numbers); MTP discarded at inference, so inference cost is equal.
BBH 70.0->70.7, MMLU 67.5->66.6, DROP 68.5->70.6, TriviaQA 67.0->67.3, NQ 27.2->28.5, HumanEval 44.5->53.7,
MBPP 61.6->62.2, GSM8K 72.3->74.0, MATH 38.6->39.8."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc
import matplotlib.pyplot as plt

base = {"BBH": 70.0, "MMLU": 67.5, "DROP": 68.5, "TriviaQA": 67.0, "NQ": 27.2,
        "HumanEval": 44.5, "MBPP": 61.6, "GSM8K": 72.3, "MATH": 38.6}
mtp = {"BBH": 70.7, "MMLU": 66.6, "DROP": 70.6, "TriviaQA": 67.3, "NQ": 28.5,
       "HumanEval": 53.7, "MBPP": 62.2, "GSM8K": 74.0, "MATH": 39.8}
delta = {k: round(mtp[k] - base[k], 1) for k in base}
order = sorted(delta, key=lambda k: -delta[k])

sc.setup()
fig, ax = plt.subplots(figsize=(11.6, 3.9))
for x, k in enumerate(order):
    v = delta[k]
    if v > 0:
        ax.bar(x, v, 0.62, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2)
    else:
        ax.bar(x, v, 0.62, color=sc.GRAYS[0], edgecolor=sc.INK, linewidth=1.2, hatch="//")
    txt = f"+{v:.1f}" if v > 0 else f"−{abs(v):.1f}"
    ax.text(x, v + 0.3 if v > 0 else v - 0.3, txt, ha="center", va="bottom" if v > 0 else "top",
            fontsize=19, fontweight="bold", color=sc.ACCENT if v > 0 else sc.INK)
ax.axhline(0, color=sc.INK, lw=1.4)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([{"HumanEval": "Human\nEval", "TriviaQA": "Trivia\nQA"}.get(k, k) for k in order])
ax.set_ylim(-3.2, 11.5)
ax.set_yticks([-2, 0, 2, 4, 6, 8, 10])
ax.set_ylabel("Δ score vs.\nno MTP (points)")
sc.better(ax, "Higher is better", where="upper right")
sc.save(fig, __file__)
print(delta)
