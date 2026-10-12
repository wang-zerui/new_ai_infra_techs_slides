"""fig_specdec: one decode step of MTP speculative decoding with one draft token (gamma = 1).
The main model runs once over [last token, draft]; if its prediction at the last token equals the draft, the draft is
accepted and the prediction at the draft position comes for free (Leviathan et al. 2023, Alg. 1); the MTP block then
takes h at the newest position plus the embedding of the newest token and drafts the next one (DeepSeek-V3 §2.2)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 286
s = SVG(W, H, "fig_specdec: verify draft + bonus token, MTP drafts next")

CY0, CH = 206, 40
cols = {"The": 30, "cat": 140, "sat": 250}
CW = 90
s.box(cols["The"], CY0, CW, CH, "The", NEU_F, NEU)
s.box(cols["cat"], CY0, CW, CH, "cat", NEU_F, NEU)
s.box(cols["sat"], CY0, CW, CH, "sat", "#FFFFFF", ACC, color=ACC, weight="bold", dash="6,4")
s.text(130, 272, "accepted text", 16, MUTED, style="italic")
s.text(295, 272, "draft", 16, ACC, style="italic", weight="bold")
# one main-model pass over [cat, sat]
s.box(140, 112, 200, 52, "Main model", NEU_F, NEU)
for c in ("cat", "sat"):
    x = cols[c] + CW / 2
    s.line(x, CY0 - 2, x, 166, INK, 2.2, "arr")
    s.line(x, 110, x, 80, INK, 2.2, "arr")
s.box(cols["cat"], 38, CW, CH, "sat ✓", ACC_F, ACC, color=ACC, weight="bold")
s.box(cols["sat"], 38, CW, CH, "on", ACC_F, ACC, color=ACC, weight="bold")
s.text(cols["cat"] + CW / 2, 26, "= draft", 16, MUTED, style="italic")
s.text(cols["sat"] + CW / 2, 26, "bonus", 16, MUTED, style="italic")
# MTP drafts the next token from h(sat) + Emb(on)
s.box(486, 112, 140, 52, "MTP block", ACC_F, ACC, color=ACC, weight="bold")
s.line(342, 138, 484, 138, ACC, 2.4, "arrAcc")
s.text(413, 130, "h(sat)", 17, ACC, style="italic")
s.path("M 342 58 L 556 58 L 556 110", ACC, 2.4, "arrAcc")
s.text(449, 50, "Emb(on)", 17, ACC, style="italic")
s.line(628, 138, 668, 138, ACC, 2.4, "arrAcc")
s.box(670, 118, 90, CH, "the", "#FFFFFF", ACC, color=ACC, weight="bold", dash="6,4")
s.text(715, 184, "next draft", 16, ACC, style="italic", weight="bold")
# outcome
s.text(860, 64, "2 tokens this pass", 19, ACC, anchor="end", weight="bold")
s.text(860, 236, "draft wrong: keep main's token", 16, MUTED, anchor="end", style="italic")
s.text(860, 258, "1 token this pass", 16, MUTED, anchor="end", style="italic")
s.save(os.path.join(HERE, "fig_specdec.svg"))
