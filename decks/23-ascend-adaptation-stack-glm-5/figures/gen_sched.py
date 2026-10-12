"""fig_sched: the serving-side scheduling pillar — split roles, reused prefixes, drafted tokens."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 282, "fig_sched: prefill nodes, KV transfer, decode nodes, prefix cache and MTP")

# request in
s.box(12, 96, 112, 58, ["Request", "200K prompt"], NEU_F, NEU, size=18)
s.line(128, 125, 164, 125, INK, 2.2, "arr")

# prefill side
s.rect(168, 24, 228, 212, "#FFFFFF", ACC, 1.8, 6, dash="6,4")
s.text(282, 48, "Prefill nodes", 20, ACC, weight="bold")
s.box(184, 62, 196, 44, "Tensor parallel", NEU_F, NEU, size=18)
s.box(184, 114, 196, 44, "FlashComm split", ACC_F, ACC, color=ACC, size=18, weight="bold")
s.box(184, 166, 196, 54, ["Prefix cache", "RadixCache + RAM"], ACC_F, ACC, color=ACC, size=17)

# KV transfer
s.line(402, 125, 490, 125, ACC, 2.4, "arrAcc")
s.text(446, 104, "KV cache", 17, ACC, weight="bold")
s.text(446, 152, "transfer", 17, ACC, weight="bold")

# decode side
s.rect(496, 24, 240, 212, "#FFFFFF", ACC, 1.8, 6, dash="6,4")
s.text(616, 48, "Decode nodes", 20, ACC, weight="bold")
s.box(512, 62, 208, 44, "DP attention + MoE EP", NEU_F, NEU, size=17)
s.box(512, 114, 208, 44, "MTP drafts", ACC_F, ACC, color=ACC, size=18, weight="bold")
s.box(512, 166, 208, 54, ["Async scheduling", "D2H overlapped"], ACC_F, ACC, color=ACC, size=17)

# tokens out
s.line(740, 125, 776, 125, INK, 2.2, "arr")
s.box(780, 96, 82, 58, ["Tokens", "out"], NEU_F, NEU, size=18)

s.text(452, 266, "Prefill and decode no longer share a node", 18, ACC, weight="bold")
s.save(os.path.join(HERE, "fig_sched.svg"))
