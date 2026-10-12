"""fig_smgrid: H800's 132 SMs, 20 held by communication kernels in DeepSeek-V3, and the copy work they do."""
import os
from svgkit import SVG, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 294
s = SVG(W, H, "fig_smgrid: 132-SM grid, 20 communication SMs, their tasks (DeepSeek-V3 report sec. 3.5.1)")

COLS, ROWS, C, G = 12, 11, 16, 4
X0, Y0 = 20, 34
s.text(X0, 22, "one H800 GPU: 132 SMs", 18, INK, anchor="start", weight="bold")
for i in range(COLS * ROWS):
    r, c = divmod(i, COLS)
    comm = i >= COLS * ROWS - 20
    s.rect(X0 + c * (C + G), Y0 + r * (C + G), C, C, PRB_F if comm else NEU_F, PRB if comm else "#A6A6A6", 1.3, 2)
gw = COLS * (C + G) - G
# bracket around the 20 communication SMs (last 20 cells: rows 9 (cols 4..11) and 10)
ytop = Y0 + 9 * (C + G)
ybot = Y0 + ROWS * (C + G) - G
bx = X0 + gw + 12
s.path(f"M{bx},{ytop} L{bx + 8},{ytop} L{bx + 8},{ybot} L{bx},{ybot}", PRB, 2)
# legend under the grid
ly = ybot + 30
s.rect(X0, ly - 14, 16, 16, NEU_F, "#A6A6A6", 1.3, 2)
s.text(X0 + 24, ly, "compute", 18, INK, anchor="start")
s.rect(X0 + 118, ly - 14, 16, 16, PRB_F, PRB, 1.3, 2)
s.text(X0 + 142, ly, "communication", 18, PRB, anchor="start")

# task list on the right
RX, RW = 352, 500
s.line(bx + 10, (ytop + ybot) / 2, RX - 6, 150, PRB, 2.2, "arrPrb")
s.text(RX, 22, "what the communication SMs do", 19, PRB, anchor="start", weight="bold")
tasks = ["copy between RDMA buffers and tensors", "forward IB traffic into NVLink",
         "reduce expert outputs in combine", "lay out tokens per expert", "poll flags until data lands"]
for k, t in enumerate(tasks):
    y = 38 + k * 46
    s.box(RX, y, RW, 36, t, "#FFFFFF", PRB, INK, 18, sw=1.6)
s.text(RX, 282, "tensor cores on these SMs sit idle", 18, MUTED, anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_smgrid.svg"))
