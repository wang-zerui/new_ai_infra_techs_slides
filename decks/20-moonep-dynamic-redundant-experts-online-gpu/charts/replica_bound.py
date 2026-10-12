"""Analytic: replica slots per GPU. Upper bound E/R (always sufficient, K3 App. E Thm 1) vs the worst-case need
ceil(E(R-1)/R^2) (Thm 2), for E = 384 routed experts (README benchmark shape). See notes/sources.md, slide 5."""
import math, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

E = 384
RS = [4, 8, 16, 32, 64]
upper = [E // r for r in RS]
need = [math.ceil(E * (r - 1) / r ** 2) for r in RS]

sc.setup()
fig, ax = sc.figure("right")
sc.grouped(ax, [f"R={r}" for r in RS], ["Worst routing needs", "Reserved: E/R"], [need, upper], ours=1)
w = 0.4
for g, (a, b) in enumerate(zip(need, upper)):
    ax.text(g - w / 2, a + 1.5, str(a), ha="center", va="bottom", fontsize=18, color=sc.INK)
    ax.text(g + w / 2, b + 1.5, str(b), ha="center", va="bottom", fontsize=18, color=sc.ACCENT, fontweight="bold")
ax.set_ylim(0, 112)
ax.set_xlabel("EP size (E = 384 experts)")
ax.set_ylabel("Replica slots per GPU")
sc.save(fig, __file__)
