"""fig_sync: HybridEP dispatch_with_permute, default (blocking) vs with a token budget (non-blocking).

Default: the preprocessing kernel computes the permuted-token count, the host synchronizes to read it, then
allocates an exact-size output (Megatron _HybridEPManager: num_permuted_tokens None -> tokens_per_expert.sum()).
With a budget (moe_expert_rank_capacity_factor -> num_permuted_tokens), non_blocking=True skips the sync;
excess tokens are dropped and an overflow flag is set.
Sources: Hybrid-EP_Implementation.md Sec. 7.2; token_dispatcher.py; fused_a2a.py.
"""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 246, "fig_sync: blocking vs budgeted HybridEP dispatch")

W, G, X0, H = 104, 18, 3, 58
xs = [X0 + i * (W + G) for i in range(4)]


def row(y, title, boxes, color):
    s.text(X0, y - 10, title, 17, color, anchor="start", weight="bold")
    for i, b in enumerate(boxes):
        if b is None:
            continue
        lab, kind = b
        if kind == "prb":
            s.box(xs[i], y, W, H, lab, PRB_F, PRB, color=PRB, size=16, weight="bold", sw=2.4)
        elif kind == "ghost":
            s.box(xs[i], y, W, H, lab, "#FFFFFF", "#A6A6A6", color=MUTED, size=16, dash="5,3")
        elif kind == "acc":
            s.box(xs[i], y, W, H, lab, ACC_F, ACC, size=16, sw=2.2)
        else:
            s.box(xs[i], y, W, H, lab, NEU_F, NEU, size=16)
    for i in range(3):
        s.line(xs[i] + W + 2, y + H / 2, xs[i + 1] - 2, y + H / 2, INK, 2.2, "arr")


row(34, "default: exact sizes", [(["metadata", "preprocess"], "n"), (["D2H sync", "host waits"], "prb"),
                                 (["allocate", "exact size"], "n"), (["dispatch", "+ permute"], "n")], INK)
row(148, "with a token budget", [(["metadata", "preprocess"], "n"), (["no sync"], "ghost"),
                                 (["allocate", "to budget"], "acc"), (["dispatch", "+ permute"], "acc")], ACC)
s.text(476, 236, "excess tokens dropped and flagged", 16, MUTED, anchor="end", style="italic")
s.save(os.path.join(HERE, "fig_sync.svg"))
