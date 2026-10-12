"""fig_w4a8: what W4A8 is at run time in vLLM-Ascend.

Expert weights arrive as int8 tensors that pack two 4-bit values, with a per-output-channel
scale, offset and scale_bias; activations are quantized to int8 per token; the grouped GEMM runs
in int8 on the Cube unit and the result is dequantized. Source: vllm_ascend/quantization/methods/
w4a8/w4a8.py (register_scheme("W4A8_DYNAMIC", "moe")).
"""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 296, "fig_w4a8: int4 expert weights, int8 per-token activations")

# ---- packed weight bytes
s.text(124, 34, "expert weight, int8 bytes", 17, INK, weight="bold")
CW, CH2 = 38, 34
for r in range(2):
    for c in range(5):
        x, y = 22 + c * (CW + 4), 46 + r * (CH2 + 4)
        s.rect(x, y, CW, CH2, "#FFFFFF", ACC, 1.6, 2)
        s.line(x + CW / 2, y + 2, x + CW / 2, y + CH2 - 2, ACC, 1.2, dash="3,2")
        s.text(x + CW / 4, y + CH2 / 2 + 6, "4b", 15, INK)
        s.text(x + 3 * CW / 4, y + CH2 / 2 + 6, "4b", 15, INK)
s.text(124, 142, "two int4 weights per byte", 16, MUTED)

# ---- per-channel scales
s.box(22, 158, 204, 52, ["scale · offset · bias", "per output channel"], NEU_F, NEU, size=16)

# ---- activations
s.box(22, 228, 204, 54, ["activations: int8", "scale per token"], ACC_F, ACC, color=INK, size=16,
      sw=2.5)

# ---- the GEMM
GX, GW = 300, 226
s.rect(GX, 46, GW, 236, ACC_F, ACC, 3, 6)
s.text(GX + GW / 2, 84, "grouped GEMM", 20, ACC, weight="bold")
s.text(GX + GW / 2, 114, "per expert", 17, INK)
s.box(GX + 26, 140, GW - 52, 52, ["int8 × int8 → int32"], "#FFFFFF", ACC, color=INK, size=17)
s.text(GX + GW / 2, 220, "on the Cube unit", 17, INK)
for y0 in (94, 184, 255):
    s.line(240, y0, GX - 4, y0, INK, 2.2, "arr")

# ---- dequantize and output
s.box(GX + GW + 40, 110, 210, 92, ["dequantize", "scale · offset · bias", "→ BF16"], "#FFFFFF",
      ACC, color=INK, size=17, sw=2.5)
s.line(GX + GW + 4, 156, GX + GW + 36, 156, INK, 2.2, "arr")
s.line(GX + GW + 145, 202, GX + GW + 145, 240, INK, 2.2, "arr")
s.text(GX + GW + 145, 266, "expert output", 17, INK)

# ---- scope notes
s.text(860, 34, "routed experts only", 16, MUTED, anchor="end", style="italic")
s.text(860, 56, "attention + MLP stay W8A8", 16, MUTED, anchor="end", style="italic")
s.text(860, 78, "per channel, not per group", 16, MUTED, anchor="end", style="italic")
s.save(os.path.join(HERE, "fig_w4a8.svg"))
