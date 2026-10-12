"""fig_grammar: how the H100 DeepSeek-V3 layout string ([MC] Sec. 9.2.2) becomes 32 stages (parse_str_to_list)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
MONO = "Menlo, 'Courier New', monospace"
s = SVG(870, 286, "fig_grammar: layout string -> unrolled -> stage list -> VPP")
LX = 150
s.text(LX, 38, "layout", 18, INK, anchor="end", weight="bold")
s.text(LX + 16, 39, "Et*3|(tt|)*29m|L", 24, ACC, anchor="start", weight="bold", family=MONO)
s.line(LX + 40, 54, LX + 40, 82, INK, 2, "arr")
s.text(LX + 56, 74, "unroll x*n and (…)*n", 16, MUTED, anchor="start", style="italic")
s.text(LX, 112, "unrolled", 18, INK, anchor="end", weight="bold")
s.text(LX + 16, 113, "Ettt|tt|tt|tt| … |tt|m|L", 22, INK, anchor="start", family=MONO)
s.line(LX + 40, 128, LX + 40, 156, INK, 2, "arr")
s.text(LX + 56, 148, "split on", 16, MUTED, anchor="start", style="italic")
s.text(LX + 122, 149, "|", 18, INK, anchor="start", weight="bold", family=MONO)
s.text(LX, 192, "stages", 18, INK, anchor="end", weight="bold")
chips = [("Ettt", "s0", True), ("tt", "s1", False), ("tt", "s2", False), ("tt", "s3", False), None,
         ("tt", "s29", False), ("m", "s30", True), ("L", "s31", True)]
x = LX + 16
for c in chips:
    if c is None:
        s.text(x + 22, 192, "…", 22, INK)
        x += 44
        continue
    lab, idx, new = c
    w = 70 if len(lab) > 2 else 52
    s.box(x, 166, w, 40, lab, ACC_F if new else NEU_F, ACC if new else NEU, color=ACC if new else INK,
          size=19, weight="bold" if new else "normal")
    s.text(x + w / 2, 228, idx, 16, MUTED)
    x += w + 8
s.text(x + 18, 182, "32 stages ÷ PP 8", 18, INK, anchor="start")
s.text(x + 18, 206, "= VPP 4", 18, ACC, anchor="start", weight="bold")
s.text(LX + 16, 272, "E embedding · t transformer layer · m MTP layer · L loss", 16, MUTED, anchor="start")
s.save(os.path.join(HERE, "fig_grammar.svg"))
