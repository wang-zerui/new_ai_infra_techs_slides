"""fig_vmm_1/2: ElasticSymmetricMemory construction (symmetric.hpp l.145-179), one row per driver call,
showing the VA range after the call."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 270
MONO = "Menlo, monospace"
BX, GW, SW, BH = 440, 160, 256, 46

rows = [
    (["cuMemAddressReserve", "(gpu + storage bytes)"], "empty"),
    (["cuMemCreate(DEVICE)", "cuMemMap(addr)"], "gpu"),
    (["cuMemCreate(HOST_NUMA)", "cuMemMap(addr + gpu)"], "host"),
    (["cuMemSetAccess", "(GPU + host NUMA)"], "access"),
]


def draw(step):
    s = SVG(W, H, f"fig_vmm_{step}: reserve once, map HBM and host pages (build step {step})")
    s.text(BX + GW / 2, 16, "GPU segment", 16, MUTED, weight="bold")
    s.text(BX + GW + SW / 2, 16, "storage", 16, MUTED, weight="bold")
    for i, (code, state) in enumerate(rows[: 2 if step == 1 else 4]):
        y = 26 + i * 62
        s.badge(22, y + BH / 2, i + 1)
        s.text(44, y + 19, code[0], 16, INK, anchor="start", family=MONO)
        s.text(44, y + 40, code[1], 16, MUTED, anchor="start", family=MONO)
        if state == "empty":
            s.rect(BX, y, GW + SW, BH, "#FFFFFF", "#A6A6A6", 1.6, 0, dash="6,4")
            s.text(BX + (GW + SW) / 2, y + 29, "addresses only (no pages)", 17, MUTED, style="italic")
        else:
            s.rect(BX, y, GW, BH, NEU_F, NEU, 1.6, 0)
            s.text(BX + GW / 2, y + 29, "GPU rw" if state == "access" else "HBM pages", 17, INK,
                   weight="bold" if state == "access" else "normal")
            if state == "gpu":
                s.rect(BX + GW, y, SW, BH, "#FFFFFF", "#A6A6A6", 1.6, 0, dash="6,4")
            else:
                s.rect(BX + GW, y, SW, BH, ACC_F, ACC, 1.8, 0)
                s.text(BX + GW + SW / 2, y + 29, "GPU and host rw" if state == "access" else "host pages", 17,
                       ACC, weight="bold" if state == "access" else "normal")
        if i > 0:
            s.line(BX - 18, y - 14, BX - 18, y + 4, MUTED, 1.6, "arrGray")
    s.save(os.path.join(HERE, f"fig_vmm_{step}.svg"))


draw(1)
draw(2)
