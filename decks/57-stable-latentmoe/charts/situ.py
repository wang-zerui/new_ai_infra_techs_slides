"""Analytic: scalar response of SwiGLU vs SiTU-GLU when both branches receive the same scalar x
(as in K3 Fig. 4). SiTU-GLU (K3 Eq. 12): [b1*tanh(x/b1)*sigmoid(x)] * [b2*tanh(x/b2)], b1 = 4, b2 = 25,
bounded by b1*b2 = 100 (K3 App. B Eq. 19). SwiGLU: x*sigmoid(x) * x. No data."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc
import numpy as np

b1, b2 = 4.0, 25.0
sig = lambda x: 1 / (1 + np.exp(-x))
situ = lambda x: b1 * np.tanh(x / b1) * sig(x) * b2 * np.tanh(x / b2)
swiglu = lambda x: x * sig(x) * x

sc.setup()
fig, (a1, a2) = sc.figure("right", 1, 2, gridspec_kw=dict(width_ratios=[1, 1.5], wspace=0.32))
x = np.linspace(-4, 4, 400)
a1.plot(x, swiglu(x), color=sc.GRAYS[2], lw=3.4, label="SwiGLU")
a1.plot(x, situ(x), color=sc.ACCENT, lw=3.0, ls="--", label="SiTU-GLU")
a1.set_xlim(-4, 4); a1.set_ylim(-1, 12)
a1.set_xlabel("x (near 0)")
a1.set_ylabel("GLU output")
x2 = np.linspace(-10, 100, 600)
a2.plot(x2, swiglu(x2), color=sc.GRAYS[2], lw=3.4, label="SwiGLU")
a2.plot(x2, situ(x2), color=sc.ACCENT, lw=3.4, label="SiTU-GLU")
a2.axhline(100, color=sc.PROBLEM, lw=1.6, ls="--")
a2.text(98, 112, "bound β₁β₂ = 100", color=sc.PROBLEM, fontsize=17, ha="right", va="bottom")
a2.set_xlim(-10, 100); a2.set_ylim(-20, 400)
a2.set_xlabel("x (large)")
a2.annotate("SwiGLU = x²σ(x)\nreaches 10⁴ at x = 100", xy=(20, 380), xytext=(32, 300), fontsize=16,
            color=sc.GRAYS[3], arrowprops=dict(arrowstyle="-|>", color=sc.GRAYS[3], lw=1.6))
h, l = a2.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.55, 0.93), ncol=2, frameon=False, fontsize=18)
sc.save(fig, __file__)
