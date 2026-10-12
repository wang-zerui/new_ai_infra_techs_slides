"""fig_naive.svg: two status-quo schedules for chunkwise KDA on one GPU (schematic, 4 SMs).
(a) alternate phases: all SMs do intra-chunk work, then one SM carries the state hand-off while others idle (K3 Sec. 5.1.1).
(b) one fused kernel: one CTA per (sequence, head) walks its chunks serially; with fewer CTAs than SMs the rest idle
(FlashKDA deep dive Sec. 2)."""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from svgkit import *
s = SVG(870, 285)
WORK, STATE = "#D9D9D9", "#7F7F7F"
lane_h, gap = 30, 8
def lanes(x0, y0, title):
    s.t(x0 + 205, y0 - 16, title, 19, "bold")
    for i in range(4):
        s.t(x0 + 44, y0 + i * (lane_h + gap) + 21, f"SM {i}", 16, "normal", GRAY, "end")
# (a)
xa, ya = 10, 52
lanes(xa, ya, "Alternate phases")
x = xa + 54
for c in range(3):
    for i in range(4):
        s.box(x, ya + i * (lane_h + gap), 66, lane_h, WORK, GRAY, 1.2, 2)
    s.t(x + 33, ya + 1 * (lane_h + gap) + 21, "intra", 15)
    x += 70
    s.box(x, ya, 40, lane_h, STATE, INK, 1.2, 2)
    s.t(x + 20, ya + 21, "S", 16, "bold", "#FFFFFF")
    for i in range(1, 4):
        s.box(x, ya + i * (lane_h + gap), 40, lane_h, "url(#hr)", RED, 1.2, 2)
    x += 44
s.line(xa + 54, ya + 4 * (lane_h + gap) + 6, xa + 400, ya + 4 * (lane_h + gap) + 6, GRAY, 1.5)
s.t(xa + 400, ya + 4 * (lane_h + gap) + 30, "time", 15, "normal", GRAY, "end")
s.raw(f'<line x1="440" y1="20" x2="440" y2="235" stroke="#D9D9D9" stroke-width="1.2"/>')
# (b)
xb, yb = 452, 52
lanes(xb, yb, "One fused kernel")
for i in range(4):
    y = yb + i * (lane_h + gap)
    if i < 2:
        x = xb + 54
        for c in range(4):
            s.box(x, y, 80, lane_h, WORK, GRAY, 1.2, 2)
            s.t(x + 40, y + 21, f"h{i+1} c{c+1}", 15)
            x += 84
    else:
        s.box(xb + 54, y, 332, lane_h, "url(#hr)", RED, 1.2, 2)
s.line(xb + 54, yb + 4 * (lane_h + gap) + 6, xb + 386, yb + 4 * (lane_h + gap) + 6, GRAY, 1.5)
# legend
ly = 262
s.box(70, ly - 15, 22, 18, WORK, GRAY, 1.2, 2); s.t(100, ly, "intra-chunk work", 17, "normal", INK, "start")
s.box(300, ly - 15, 22, 18, STATE, INK, 1.2, 2); s.t(330, ly, "state hand-off", 17, "normal", INK, "start")
s.box(520, ly - 15, 22, 18, "url(#hr)", RED, 1.2, 2); s.t(550, ly, "idle SM", 17, "normal", RED, "start")
s.t(690, ly, "h = head · c = chunk", 16, "normal", GRAY, "start", "italic")
s.save(pathlib.Path(__file__).with_name("fig_naive.svg"))
