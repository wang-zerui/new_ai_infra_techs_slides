"""fig_hsdp: HSDP on 2 nodes x 4 GPUs: model state sharded 4 ways inside each NVLink node (per-layer AG/RS),
replicated across the 2 nodes (gradient all-reduce once per optimizer step over the scale-out network)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 286, "fig_hsdp: two node replicas, sharded inside")

GW, GH = 84, 56
for n, x0 in enumerate((10, 450)):
    s.rect(x0, 8, 410, 190, "#FFFFFF", "#7F7F7F", 1.5, 6, dash="6,4")
    s.text(x0 + 12, 32, f"node {n} · replica {n}", 18, INK, anchor="start", weight="bold")
    s.text(x0 + 398, 32, "NVLink", 16, MUTED, anchor="end", style="italic")
    xs = [x0 + 18 + i * 98 for i in range(4)]
    for i, x in enumerate(xs):
        s.rect(x, 50, GW, GH, ACC_F, ACC, 2, 4)
        s.text(x + GW / 2, 73, f"GPU{4 * n + i}", 17, INK, weight="bold")
        s.text(x + GW / 2, 95, "¼ of state", 15, ACC)
        s.line(x + GW / 2, 50 + GH, x + GW / 2, 140, ACC, 2.4)
    s.line(xs[0] + GW / 2 - 1, 140, xs[-1] + GW / 2 + 1, 140, ACC, 3.2)
    s.text(x0 + 205, 172, "AG and RS every layer", 18, ACC, weight="bold")

# cross-node replica link
s.path("M215,202 L215,234 L655,234 L655,202", MUTED, 2.2, dash="7,5")
s.add(f'<path d="M215,200 l-6,10 l12,0 z" fill="{MUTED}"/>')
s.add(f'<path d="M655,200 l-6,10 l12,0 z" fill="{MUTED}"/>')
s.text(435, 262, "gradient all-reduce once per step across nodes", 17, INK)
s.save(os.path.join(HERE, "fig_hsdp.svg"))
