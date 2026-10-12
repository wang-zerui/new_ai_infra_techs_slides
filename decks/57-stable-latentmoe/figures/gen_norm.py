"""fig_norm: the aggregated latent u = sum_i p_i E_i(W_down x) changes scale with the selected experts and their
weights; Stable LatentMoE RMS-normalizes u (3584-wide) before W_up (K3 §2.3.1, Eq. 11). Schematic magnitudes."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(490, 290, "fig_norm: RMSNorm on the aggregated latent before W_up (schematic)")
I = "italic"

s.text(8, 24, "aggregated latent u", 18, INK, anchor="start", weight="bold")
s.text(484, 24, "after norm", 18, ACC, anchor="end", weight="bold")
rows = [("token A", 58, 56, "4 small experts"), ("token B", 136, 214, "large outputs")]
for name, y, w, note in rows:
    s.text(8, y + 26, name, 18, INK, anchor="start")
    fill, stroke = (NEU_F, NEU) if w < 100 else (PRB_F, PRB)
    s.rect(84, y, w, 40, fill, stroke, 2, 3)
    s.rect(392, y, 92, 40, ACC_F, ACC, 2, 3)
    s.line(84 + w + 4, y + 20, 316, y + 20, MUTED, 1.6, "arrGray", dash="4,4")
    s.line(366, y + 20, 390, y + 20, INK, 2, "arr")
s.text(84 + 214 / 2, 204, "scale set by routing", 17, PRB, weight="bold")
# RMSNorm box (vertical)
s.rect(318, 46, 46, 144, ACC_F, ACC, 2, 5)
s.add(f'<text x="341" y="118" text-anchor="middle" font-size="19" font-weight="bold" fill="{ACC}" '
      f'transform="rotate(-90 341 118)" dominant-baseline="middle">RMSNorm</text>')
# then W_up
s.line(438, 178, 438, 222, INK, 2.2, "arr")
s.rect(360, 224, 124, 52, ACC_F, ACC, 2, 5)
s.rich(422, 246, [("W", ACC, "bold"), ("↑", ACC, "bold")], 19)
s.text(422, 268, "3584 → 7168", 16, INK)
s.text(8, 262, "bar length = RMS of u (schematic)", 16, MUTED, anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_norm.svg"))
