"""fig_overview: the three scales at which GLM-5's Ascend serving stack hides communication."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 300, "fig_overview: kernel scale, layer scale, cluster scale")

W, H, X0, GAP = 266, 206, 20, 26
TITLES = [("❶", "Layer", "Split the collective"), ("❷", "Kernel", "Fuse it into the matmul"),
          ("❸", "Cluster", "Split the roles")]

for i, (badge, scope, claim) in enumerate(TITLES):
    x = X0 + i * (W + GAP)
    s.rect(x, 24, W, H, "#FFFFFF", ACC, 1.8, 6)
    s.add(f'<circle cx="{x + 26}" cy="50" r="13" fill="{ACC}"/>')
    s.text(x + 26, 56, badge, 15, "#FFFFFF", weight="bold")
    s.text(x + 48, 56, scope, 20, ACC, anchor="start", weight="bold")
    s.text(x + W / 2, 86, claim, 19, INK)

# panel 1: AllReduce -> RS .. AG
x = X0
s.box(x + 16, 104, 100, 40, "AllReduce", NEU_F, PRB, color=PRB, size=17, weight="bold")
s.line(x + 133, 124, x + 155, 124, INK, 2.2, "arr")
s.box(x + 160, 104, 90, 40, "RS + AG", ACC_F, ACC, color=ACC, size=17, weight="bold")
s.text(x + W / 2, 176, "Norm and router run", 17, INK)
s.text(x + W / 2, 198, "on sharded tokens", 17, INK)

# panel 2: tiles with comm riding along
x = X0 + (W + GAP)
for c in range(5):
    s.rect(x + 22 + c * 46, 104, 38, 30, ACC_F, ACC, 1.8, 2)
    s.rect(x + 22 + c * 46, 140, 38, 14, "#FFFFFF", ACC, 1.4, 2)
s.text(x + W / 2, 176, "Every tile ships its partial", 17, INK)
s.text(x + W / 2, 198, "inside one kernel", 17, INK)

# panel 3: prefill / decode nodes
x = X0 + 2 * (W + GAP)
s.box(x + 20, 104, 104, 46, ["Prefill", "nodes"], ACC_F, ACC, color=ACC, size=17)
s.box(x + 142, 104, 104, 46, ["Decode", "nodes"], ACC_F, ACC, color=ACC, size=17)
s.line(x + 126, 127, x + 138, 127, ACC, 2.2, "arrAcc")
s.text(x + W / 2, 176, "Prefill cannot preempt", 17, INK)
s.text(x + W / 2, 198, "a running decode", 17, INK)

s.text(435, 268, "Same goal at three scales: no rank waiting on the fabric", 19, ACC, weight="bold")
s.save(os.path.join(HERE, "fig_overview.svg"))
