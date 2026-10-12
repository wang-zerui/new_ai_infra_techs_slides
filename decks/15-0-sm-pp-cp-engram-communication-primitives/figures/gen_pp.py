"""fig_pp_1/2: DeepEP PP send/recv. TMA stage into a send slot, one thread posts an RDMA put with a remote
arrival signal, the receiver waits and copies out, then returns a release credit (step 2 of the build)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 258


def draw(step):
    s = SVG(W, H, f"fig_pp_{step}: PP send/recv over RDMA (build step {step})")
    # node containers
    for (x, title) in ((8, "stage i · node A"), (552, "stage i+1 · node B")):
        s.rect(x, 8, 310, 244, "#FFFFFF", NEU, 1.6, 8)
        s.text(x + 155, 32, title, 19, INK, weight="bold")
    # tensors
    s.box(24, 50, 160, 46, "activation x", NEU_F, NEU, size=18)
    s.box(686, 50, 160, 46, "input x", NEU_F, NEU, size=18)
    # slots
    CW, CG, CY, CH = 44, 6, 152, 40

    def slots(x0, cur, label):
        for k in range(4):
            on = k == cur
            s.rect(x0 + k * (CW + CG), CY, CW, CH, ACC_F if on else "#FFFFFF", ACC if on else "#A6A6A6", 1.6, 2,
                   dash=None if on else "4,3")
        s.text(x0 + 97, CY + CH + 24, label, 17, MUTED)
    slots(24, 1, "send slots")
    slots(652, 1, "recv slots")
    # NICs
    s.box(234, 140, 72, 80, "NIC", ACC_F, ACC, ACC, 19, "bold")
    s.box(564, 140, 72, 80, "NIC", ACC_F, ACC, ACC, 19, "bold")
    # 1: TMA copy x -> send slot
    s.line(96, 98, 96, CY - 3, ACC, 2.4, "arrAcc")
    s.badge(96, 124, 1)
    s.text(116, 130, "TMA copy", 17, INK, anchor="start")
    # slot -> NIC A (NIC reads the slot)
    s.line(218, 172, 232, 172, ACC, 2.4, "arrAcc")
    # 2: put + arrival signal over IB
    s.line(308, 165, 562, 165, ACC, 2.8, "arrAcc")
    s.badge(435, 165, 2)
    s.text(435, 150, "1 thread posts put + signal", 17, INK)
    s.line(638, 172, 650, 172, ACC, 2.4, "arrAcc")
    # 3: wait arrival, TMA copy out
    s.line(724, CY - 3, 724, 98, ACC, 2.4, "arrAcc")
    s.badge(724, 124, 3)
    s.text(704, 130, "wait + TMA copy", 17, INK, anchor="end")
    if step >= 2:
        # 4: release credit back to the sender
        s.line(562, 205, 310, 205, INK, 2.2, "arr", dash="7,5")
        s.badge(435, 205, 4, fill="#FFFFFF", color=INK)
        s.add('<circle cx="435" cy="205" r="13" fill="none" stroke="#222222" stroke-width="2"/>')
        s.text(435, 238, "release signal frees the slot", 17, INK)
    s.save(os.path.join(HERE, f"fig_pp_{step}.svg"))


draw(1)
draw(2)
