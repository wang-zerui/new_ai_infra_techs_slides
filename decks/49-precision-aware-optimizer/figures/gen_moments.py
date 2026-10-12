"""fig_moments: (top) BF16 moments: cast fused into the Adam kernel; (bottom) FP16/FP8 moments: one FP32 scale per
tensor; unscale into a temporary FP32 tensor, run the kernel, then re-scale with scale = absmax / range."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 270, "fig_moments: BF16 fused path vs FP8 scaled path")

H = 52


def chain(y, items, title):
    s.text(8, y - 14, title, 18, INK, anchor="start", weight="bold")
    x = 8
    prev = None
    for w, lab, kind in items:
        fill, stroke, col = {"store": (ACC_F, ACC, ACC), "kernel": ("#FFFFFF", ACC, INK),
                             "extra": (PRB_F, PRB, PRB)}[kind]
        s.rect(x, y, w, H, fill, stroke, 2.2, 4)
        lines = lab if isinstance(lab, list) else [lab]
        n = len(lines)
        for k, t in enumerate(lines):
            s.text(x + w / 2, y + H / 2 + 6 - (n - 1) * 10 + k * 20, t, 16, col,
                   weight="bold" if kind == "store" else "normal")
        if prev is not None:
            s.line(prev + 2, y + H / 2, x - 2, y + H / 2, INK, 2, "arr")
        prev = x + w
        x += w + 30


chain(42, [(170, ["m and v", "BF16"], "store"), (480, ["one kernel: cast in registers", "FP32 Adam math  ·  cast back"],
                                                    "kernel"),
           (130, ["m and v", "BF16"], "store")], "BF16 moments: fused")
chain(166, [(150, ["m and v FP8", "+ FP32 scale"], "store"), (130, ["unscale to", "FP32 temp"], "extra"),
            (150, ["FP32 Adam", "kernel"], "kernel"), (150, ["absmax to", "new scale"], "extra"),
            (126, ["m and v FP8", "+ FP32 scale"], "store")], "FP8 or FP16 moments: scaled per tensor")
s.text(862, 256, "scale = absmax / 448 for FP8 E4M3", 16, MUTED, anchor="end", style="italic")
s.save(os.path.join(HERE, "fig_moments.svg"))
