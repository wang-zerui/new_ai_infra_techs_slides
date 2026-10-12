"""fig_stack: the main model predicts t+1; one MTP depth beside it takes h and Emb(t+1) and predicts t+2,
reusing the main model's embedding and output head (DeepSeek-V3 §2.2)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 300, "fig_stack: main model + one MTP depth")

MX, MW = 30, 200          # main column
TX, TW = 290, 170         # MTP column
# main model
s.box(MX, 236, MW, 42, "Embedding", NEU_F, NEU)
s.box(MX, 118, MW, 100, ["Transformer", "blocks × 61"], NEU_F, NEU)
s.box(MX, 54, MW, 42, "Output head", NEU_F, NEU)
s.line(MX + MW / 2, 234, MX + MW / 2, 220, INK, 2.2, "arr")
s.line(MX + MW / 2, 116, MX + MW / 2, 98, INK, 2.2, "arr")
s.line(MX + MW / 2, 52, MX + MW / 2, 30, INK, 2.2, "arr")
s.text(MX + MW / 2 + 10, 24, "next token t+1", 18, INK, anchor="start", weight="bold")
# MTP depth 1
s.box(TX, 236, TW, 42, "Emb(t+1)", "#FFFFFF", NEU, dash="6,4")
s.box(TX, 140, TW, 56, "MTP block", ACC_F, ACC, color=ACC, weight="bold")
s.box(TX, 54, TW, 42, "Output head", "#FFFFFF", NEU, dash="6,4")
s.line(TX + TW / 2, 234, TX + TW / 2, 198, INK, 2.2, "arr")
s.line(TX + TW / 2, 138, TX + TW / 2, 98, INK, 2.2, "arr")
s.line(TX + TW / 2, 52, TX + TW / 2, 30, ACC, 2.2, "arrAcc")
s.text(TX + TW / 2 + 10, 24, "t+2", 18, ACC, anchor="start", weight="bold")
# hidden state h from the last block into the MTP block
s.line(MX + MW + 2, 168, TX - 2, 168, ACC, 2.5, "arrAcc")
s.text((MX + MW + TX) / 2, 160, "h", 19, ACC, style="italic", weight="bold")
# shared weights
s.text((MX + MW + TX) / 2, 80, "shared", 15, MUTED, style="italic")
s.line(MX + MW + 4, 87, TX - 4, 87, MUTED, 1.6, dash="4,3")
s.text((MX + MW + TX) / 2, 250, "shared", 15, MUTED, style="italic")
s.line(MX + MW + 4, 257, TX - 4, 257, MUTED, 1.6, dash="4,3")
s.save(os.path.join(HERE, "fig_stack.svg"))
