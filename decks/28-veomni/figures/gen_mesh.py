"""fig_mesh: Qwen3-MoE EP plan and the resulting 2D [EP, EP-FSDP] layout of one expert weight on 16 GPUs
(docs/key_features/ep_fsdp2.md worked example: 128 experts, hidden 2048, intermediate 768, EP 8, EP-FSDP 2)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 308, "fig_mesh")
MONO = "Menlo, monospace"
NB = " "

# ---- code (left)
s.text(10, 22, "qwen3_moe/parallel_plan.py", 16, MUTED, anchor="start", family=MONO)
s.rect(8, 32, 330, 120, "#F7F7F7", NEU, 1.6, 5)
code = ['ep_plan = {', NB * 2 + '"…gate_up_proj": Shard(0),', NB * 2 + '"…down_proj":' + NB * 4 + 'Shard(0),', '}']
for i, ln in enumerate(code):
    s.text(22, 60 + i * 27, ln, 16, INK, anchor="start", family=MONO)
s.text(10, 188, "experts: EP 8 on dim-0", 18, ACC, anchor="start", weight="bold")
s.text(10, 214, "+ FSDP2 2-way on dim-1", 18, ACC, anchor="start", weight="bold")
s.text(10, 240, "→ [16, 1024, 768] per rank", 17, ACC, anchor="start", family=MONO)
s.text(10, 270, "other weights: FSDP2", 18, INK, anchor="start")
s.text(10, 294, "over all 16 ranks", 18, INK, anchor="start")

# ---- tile grid (right): down_proj [128, 2048, 768]
GX, TW, TH, TG, GY = 544, 100, 24, 4, 58
s.text(GX - 12, 24, "down_proj", 17, INK, anchor="end", weight="bold", family=MONO)
s.text(GX - 12, 46, "[128, 2048, 768]", 15, MUTED, anchor="end", family=MONO)
for c, (node, rng) in enumerate([("node 0", "dim-1: 0–1023"), ("node 1", "1024–2047")]):
    cx = GX + c * (TW + TG) + TW / 2
    s.text(cx, 24, node, 17, INK, weight="bold")
    s.text(cx, 46, rng, 15, MUTED)
for r in range(8):
    y = GY + r * (TH + TG)
    s.text(GX - 12, y + TH / 2 + 5.5, f"experts {16 * r}–{16 * r + 15}", 16, INK, anchor="end")
    for c in range(2):
        x = GX + c * (TW + TG)
        rank = r + 8 * c
        hot = r == 0
        s.box(x, y, TW, TH, f"rank {rank}", ACC_F if hot else "#FFFFFF", ACC if hot else NEU, ACC if hot else INK,
              16, "bold" if hot else "normal", 2 if hot else 1.4, 3)
GB = GY + 8 * (TH + TG) - TG
s.text(GX + TW + TG / 2, GB + 22, "row = one EP-FSDP group", 16, ACC, style="italic")

# EP brace (right)
AX = GX + 2 * TW + TG + 14
s.path(f"M{AX},{GY} L{AX + 10},{GY} L{AX + 10},{GB} L{AX},{GB}", ACC, 2.2)
s.text(AX + 20, (GY + GB) / 2 - 4, "EP = 8", 19, ACC, anchor="start", weight="bold")
s.text(AX + 20, (GY + GB) / 2 + 20, "in a node", 17, ACC, anchor="start")
s.save(os.path.join(HERE, "fig_mesh.svg"))
