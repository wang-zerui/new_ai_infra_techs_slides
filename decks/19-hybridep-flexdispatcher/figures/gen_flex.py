"""fig_flex: the six dispatcher phases (MoETokenDispatcher) and the backend-manager calls the Flex dispatcher
makes in each (MoEFlexTokenDispatcher -> _HybridEPManager). HybridEP folds permutation into dispatch/combine,
so the permuted/restored views are pass-throughs.

Source: megatron/core/transformer/moe/token_dispatcher.py (main, 2026-10-13).
"""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED, MONO

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 248, "fig_flex: six phases and the backend manager calls")

W, G, X0 = 116, 9, 3
phases = [("dispatch", "preprocess"), ("token", "dispatch"), ("dispatch", "postprocess"), ("experts",),
          ("combine", "preprocess"), ("token", "combine"), ("combine", "postprocess")]
PY, PH = 8, 54
for i, ph in enumerate(phases):
    x = X0 + i * (W + G)
    if ph == ("experts",):
        s.box(x, PY, W, PH, "experts", NEU_F, NEU, size=17)
    else:
        s.box(x, PY, W, PH, list(ph), "#FFFFFF", ACC, color=INK, size=17, sw=2.2)
for i in range(6):
    x = X0 + i * (W + G) + W
    s.line(x + 1, PY + PH / 2, x + G - 1, PY + PH / 2, INK, 1.6)

# manager container
CY, CH = 104, 114
s.rect(X0, CY, 870 - 2 * X0, CH, "#FFFFFF", ACC, 2, 6, dash="7,4")
s.text(X0 + 10, CY + CH + 22, "_HybridEPManager (one of four backend managers)", 17, ACC, anchor="start",
       weight="bold")

calls = {
    0: [("metadata", True), ("routing map", False), ("+ probs", False)],
    1: [("dispatch", True), ("permute", False), ("+ a2a", False), ("+ permute", False)],
    2: [("pass-through", False)],
    4: [("pass-through", False)],
    5: [("combine", True), ("unpermute", False), ("+ a2a", False), ("+ unpermute", False)],
}
for i, lines in calls.items():
    x = X0 + i * (W + G)
    cx = x + W / 2
    real = len(lines) > 1
    s.rect(x + 4, CY + 12, W - 8, CH - 24, ACC_F if real else "#FFFFFF", ACC if real else "#A6A6A6", 2 if real else 1.6,
           4, dash=None if real else "4,3")
    n = len(lines)
    y0 = CY + CH / 2 - 9.5 * (n - 1) + 5.5
    for k, (t, mono) in enumerate(lines):
        if mono:
            s.text(cx, y0 + k * 19, t, 17, ACC, weight="bold")
        else:
            s.text(cx, y0 + k * 19, t, 16, INK if real else MUTED)
    s.line(cx, PY + PH + 2, cx, CY + 10, ACC if real else MUTED, 2.2 if real else 1.6, "arrAcc" if real else "arrGray")
s.save(os.path.join(HERE, "fig_flex.svg"))
