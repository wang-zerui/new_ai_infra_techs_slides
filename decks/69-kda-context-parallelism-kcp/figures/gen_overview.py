"""fig_overview.svg: KCP timeline on 4 CP ranks (schematic widths). Steps from K3 Sec. 5.1.2 and FLA fla/ops/cp README:
conv1d halo from previous rank; local fragments (M, S~) from zero; one all-gather of [S~, M]; in-order merge over
preceding ranks (rank i merges i fragments); then the normal chunk kernel from the exact entry state."""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from svgkit import *
s = SVG(870, 310)
L0, LH, LS = 62, 42, 54
for i in range(4):
    y = L0 + i * LS
    s.t(70, y + 27, f"GPU {i}", 17, "bold", INK, "end")
    if i > 0:
        s.box(84, y, 36, LH, "#FFFFFF", ACC, 1.5, 3)
    s.box(130, y, 168, LH, ACC_F, ACC, 2, 3)
    s.t(214, y + 27, f'{sub("M", i)} and {sub("S̃", i)}', 17, "bold", ACC)
    for j in range(i):
        s.box(366 + j * 32, y, 28, LH, ACC_F, ACC, 1.5, 3)
    s.box(474, y, 382, LH, GRAY_F, GRAY, 1.5, 3)
    s.t(665, y + 27, f'chunk kernel from exact {sub("S", "in")}', 17)
# all-gather band
s.box(310, L0, 44, 3 * LS + LH, ACC, ACC, 1.5, 4)
s.raw(f'<text x="337" y="{L0 + 1.5 * LS + LH / 2:.1f}" font-size="17" font-weight="bold" fill="#FFFFFF" text-anchor="middle" transform="rotate(-90 337 {L0 + 1.5 * LS + LH / 2:.1f})">all-gather</text>')
# headers
s.badge(80, 40, 3, 11); s.t(96, 46, "halo", 16, "normal", INK, "start")
s.badge(160, 40, 1, 11); s.t(176, 46, "local fragments", 16, "normal", INK, "start")
s.badge(310, 18, 2, 11); s.t(326, 24, "one all-gather", 16, "normal", INK, "start")
s.t(410, 46, "merge", 16, "normal", INK, "middle")
s.t(665, 46, "same as single GPU (deck 08)", 16, "normal", GRAY, "middle", "italic")
s.line(84, 292, 856, 292, GRAY, 1.5)
s.t(856, 308, "time", 15, "normal", GRAY, "end")
s.save(pathlib.Path(__file__).with_name("fig_overview.svg"))
