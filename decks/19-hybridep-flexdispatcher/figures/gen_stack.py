"""fig_stack: layered view. MoE layer -> MoEFlexTokenDispatcher (six phases, one TP x EP group) ->
backend managers selected by --moe-flex-dispatcher-backend (deepep, deepepv2, hybridep, ncclep) ->
hardware paths used by HybridEP (TMA over NVLink/MNNVL, IBGDA to the RDMA NIC).

Sources: token_dispatcher.py (MoEFlexTokenDispatcher, _DispatchManager subclasses), transformer_config.py
(moe_flex_dispatcher_backend), Hybrid-EP README/blog (TMA intra-node, IBGDA RDMA).
"""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED, MONO

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 300, "fig_stack: Flex dispatcher layered architecture")

LX = 104
for y, t in ((33, "model"), (105, "MCore"), (176, "backend"), (274, "hardware")):
    s.text(LX, y + 6, t, 17, MUTED, anchor="end", style="italic")

# ---- row 1: MoE layer stages
R1, H1 = 12, 42
xs = [116, 304, 492, 680]
names = ["router", "dispatch", "experts", "combine"]
W1 = 156
xs = [116 + 14, 304 + 14, 492 + 14, 680 + 14]
for x, n in zip(xs, names):
    if n in ("dispatch", "combine"):
        s.box(x, R1, W1, H1, n, "#FFFFFF", ACC, color=ACC, size=18, weight="bold", sw=2.4)
    else:
        s.box(x, R1, W1, H1, n, NEU_F, NEU, size=18)
for i in range(3):
    s.line(xs[i] + W1 + 2, R1 + H1 / 2, xs[i + 1] - 2, R1 + H1 / 2, INK, 2.2, "arr")

# ---- row 2: Flex dispatcher
R2, H2 = 82, 46
s.box(304, R2, 560, H2, "MoEFlexTokenDispatcher: six phases, one TP×EP group", ACC_F, ACC, color=INK,
      size=18, weight="bold", sw=2.4)
for x in (xs[1], xs[3]):
    s.line(x + W1 / 2, R1 + H1 + 2, x + W1 / 2, R2 - 2, ACC, 2.2, "arrAcc")

# ---- row 3: backends
R3, H3 = 148, 56
bx = [116, 304, 492, 680]
flags = ["deepep", "deepepv2", "hybridep", "ncclep"]
libs = ["DeepEP v1", "DeepEP v2 (deck 24)", "HybridEP", "TE NCCL EP"]
for x, f, lib in zip(bx, flags, libs):
    hot = f == "hybridep"
    s.rect(x, R3, 184, H3, ACC_F if hot else "#FFFFFF", ACC if hot else NEU, 2.6 if hot else 2, 4)
    s.text(x + 92, R3 + 23, f, 17, ACC if hot else INK, weight="bold", family=MONO)
    s.text(x + 92, R3 + 46, lib, 17, INK)
# fan-out from the dispatcher to every backend manager
s.line(116 + 92, R2 + H2 / 2, 304 - 2, R2 + H2 / 2, MUTED, 1.6)
s.line(116 + 92, R2 + H2 / 2, 116 + 92, R3 - 2, MUTED, 1.6, "arrGray")
for x in bx[1:]:
    s.line(x + 92, R2 + H2 + 2, x + 92, R3 - 2, ACC if x == 492 else MUTED, 2.2 if x == 492 else 1.6,
           "arrAcc" if x == 492 else "arrGray")

# ---- row 4: hardware paths used by HybridEP
R4, H4 = 252, 44
s.box(116, R4, 372, H4, "NVLink / multi-node NVLink", NEU_F, NEU, size=18)
s.box(504, R4, 360, H4, "RDMA NIC (InfiniBand / RoCE)", NEU_F, NEU, size=18)
hx = 492 + 92
YH = R3 + H3 + 30
s.path(f"M{hx - 30},{R3 + H3 + 2} L{hx - 30},{YH} L{360},{YH} L{360},{R4 - 2}", ACC, 2.4, "arrAcc")
s.text(440, YH - 7, "TMA", 17, ACC, weight="bold")
s.path(f"M{hx + 30},{R3 + H3 + 2} L{hx + 30},{YH} L{700},{YH} L{700},{R4 - 2}", ACC, 2.4, "arrAcc")
s.text(660, YH - 7, "IBGDA", 17, ACC, weight="bold")
s.save(os.path.join(HERE, "fig_stack.svg"))
