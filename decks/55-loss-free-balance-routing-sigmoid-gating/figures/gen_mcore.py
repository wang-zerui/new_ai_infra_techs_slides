"""fig_mcore: Megatron-Core expert-bias update (router.py _apply_expert_bias; finalize_model_grads.py
_update_router_expert_bias; moe_utils.get_updated_expert_bias). Counts accumulate in every forward of the
global batch; once per global batch the counts are all-reduced over TPxCPxDP and b moves by +-u."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 256
s = SVG(W, H, "fig_mcore: per-micro-batch counting, one all-reduce + sign update per global batch")

LANES = [("DP rank 0", 44), ("DP rank 1", 108)]
LH = 46
for name, y in LANES:
    s.text(94, y + LH / 2 + 6.3, name, 18, INK, anchor="end")
    x = 104
    for k in range(2):
        s.box(x, y, 118, LH, "fwd + count", ACC_F, ACC, color=ACC, weight="bold", size=17)
        x += 124
        s.box(x, y, 58, LH, "bwd", NEU_F, NEU, size=17)
        x += 64
    s.text(x + 10, y + LH / 2 + 6.3, "…", 20, INK)
    s.line(x + 26, y + LH / 2, 538, y + LH / 2, INK, 2.2, "arr")
s.text(300, 28, "micro-batches of one global batch", 16, MUTED, style="italic")
s.box(540, 44, 132, 110, ["all-reduce", "counts", "TP×CP×DP"], ACC_F, ACC, color=ACC, weight="bold", size=18)
s.line(674, 99, 698, 99, ACC, 2.2, "arrAcc")
s.box(700, 44, 166, 110, ["b ← b + u ·", "sign(mean − c)"], ACC_F, ACC, color=ACC, weight="bold", size=18)
s.text(703, 28, "once per global batch", 16, MUTED, style="italic")
# implementation facts, tied to the part they describe
CY = 206
s.line(351, 156, 351, CY - 2, MUTED, 1.6, dash="4,3")
s.box(104, CY, 400, 44, "int64 counters · skipped in eval", "#FFFFFF", NEU, size=18)
s.line(783, 156, 783, CY - 2, MUTED, 1.6, dash="4,3")
s.box(540, CY, 326, 44, "fp32 buffer · not a parameter", "#FFFFFF", NEU, size=18)
s.save(os.path.join(HERE, "fig_mcore.svg"))
