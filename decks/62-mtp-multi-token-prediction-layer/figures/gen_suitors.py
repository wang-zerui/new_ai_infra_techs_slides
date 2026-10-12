"""fig_suitors: (a) a separate draft model must be trained and served on its own;
(b) parallel independent heads (Gloeckle et al. 2024, as contrasted in DeepSeek-V3 §2.2) predict t+1..t+3 from the
same hidden state, so the t+3 head never sees t+2."""
import os
from svgkit import SVG, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 290
s = SVG(W, H, "fig_suitors: separate drafter vs parallel independent heads")

# ---------------- (a) separate draft model
s.box(20, 70, 150, 60, "Draft model", NEU_F, NEU)
s.box(250, 50, 170, 100, ["Target", "model"], NEU_F, NEU)
s.line(172, 88, 248, 88, INK, 2.2, "arr")
s.text(210, 78, "drafts", 16, MUTED, style="italic")
s.line(248, 114, 172, 114, INK, 2.2, "arr")
s.text(210, 136, "verdict", 16, MUTED, style="italic")
s.box(20, 164, 160, 38, "own training run", PRB_F, PRB, color=PRB, size=17, weight="bold")
s.box(20, 212, 160, 38, "own serving copy", PRB_F, PRB, color=PRB, size=17, weight="bold")
s.line(95, 132, 95, 162, PRB, 2, dash="4,3")
s.text(214, 282, "(a) Separate draft model", 19, INK, weight="bold")

s.line(438, 18, 438, 290, "#BFBFBF", 1.2)

# ---------------- (b) parallel independent heads
TX = 470
s.box(TX, 196, 380, 44, "Shared trunk → hᵢ", NEU_F, NEU)
heads = ["t+1", "t+2", "t+3"]
for k, lab in enumerate(heads):
    x = TX + 4 + k * 140
    s.box(x, 92, 96, 44, f"head {k + 1}", NEU_F, NEU)
    s.line(x + 48, 194, x + 48, 138, INK, 2.2, "arr")
    s.line(x + 48, 90, x + 48, 64, INK, 2.2, "arr")
    s.text(x + 48, 54, lab, 18, INK, weight="bold")
# the missing causal link t+2 -> head 3
x2, x3 = TX + 4 + 140 + 96, TX + 4 + 280
s.line(x2 + 4, 114, x3 - 4, 114, PRB, 2.4, dash="5,4")
s.text((x2 + x3) / 2, 121, "✗", 22, PRB, weight="bold", halo=True)
s.text((x2 + x3) / 2 - 20, 24, "head 3 never sees t+2", 17, PRB, weight="bold")
s.text(654, 282, "(b) Parallel independent heads", 19, INK, weight="bold")
s.save(os.path.join(HERE, "fig_suitors.svg"))
