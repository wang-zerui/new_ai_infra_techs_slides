"""fig_pd: the upstream prefill/decode-disaggregated reference deployment for GLM-5 on Atlas A3."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 296, "fig_pd: prefill nodes, Mooncake KV path, decode nodes, with the reference settings")

# ---------------- prefill
s.rect(16, 24, 320, 200, "#FFFFFF", ACC, 1.8, 6, dash="6,4")
s.text(176, 50, "Prefill nodes · PP2 TP16", 20, ACC, weight="bold")
s.box(32, 64, 288, 42, "DSA context parallelism", ACC_F, ACC, color=ACC, size=18)
s.box(32, 114, 288, 42, "FlashComm split", ACC_F, ACC, color=ACC, size=18, weight="bold")
s.box(32, 164, 288, 46, ["Prefix cache", "RadixCache + host RAM"], ACC_F, ACC, color=ACC, size=17)

# ---------------- transfer
s.line(344, 110, 526, 110, ACC, 2.6, "arrAcc")
s.text(435, 92, "KV via Mooncake", 18, ACC, weight="bold")
s.line(526, 160, 344, 160, NEU, 2.2, "arrGray")
s.text(435, 184, "recompute fallback", 17, MUTED, style="italic")

# ---------------- decode
s.rect(534, 24, 320, 200, "#FFFFFF", ACC, 1.8, 6, dash="6,4")
s.text(694, 50, "Decode nodes · DP8 TP4", 20, ACC, weight="bold")
s.box(550, 64, 288, 42, "MLAPO fused pre-processing", ACC_F, ACC, color=ACC, size=17)
s.box(550, 114, 288, 42, "MTP · 3 draft tokens", ACC_F, ACC, color=ACC, size=18, weight="bold")
s.box(550, 164, 288, 46, ["Async scheduling", "D2H overlapped with next step"], ACC_F, ACC, color=ACC, size=17)

s.text(16, 266, "Long prefills can no longer preempt running decodes", 18, ACC, anchor="start", weight="bold")
s.text(854, 266, "one Atlas 800 A3 node each", 17, MUTED, anchor="end", style="italic")
s.save(os.path.join(HERE, "fig_pd.svg"))
