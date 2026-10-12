"""fig_omni: an omni-modal LLM = modality encoders + LLM foundation + modality decoders (Qwen3-MoE#Omni, VeOmni §4.3)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 320, "fig_omni: encoders -> foundation -> decoders")

# column headers
s.text(72, 22, "Encoders", 20, INK, weight="bold")
s.text(240, 22, "Foundation", 20, INK, weight="bold")
s.text(408, 22, "Decoders", 20, INK, weight="bold")

EX, EW, EH = 8, 128, 60
enc = [("ViT", "image / video", 40), ("Whisper", "audio", 118), ("Embedding", "text", 196)]
for name, mod, y in enc:
    s.rect(EX, y, EW, EH, NEU_F, NEU, 2, 5)
    s.text(EX + EW / 2, y + 26, name, 19, INK, weight="bold")
    s.text(EX + EW / 2, y + 48, mod, 17, MUTED)

FX, FW, FY, FH = 182, 116, 40, 216
s.rect(FX, FY, FW, FH, NEU_F, NEU, 2, 6)
for i, t in enumerate(["Qwen3", "30B-A3B", "MoE LLM"]):
    s.text(FX + FW / 2, FY + FH / 2 - 22 + i * 24, t, 19, INK, weight="bold" if i < 2 else "normal")

DX, DW = 344, 128
dec = [("MoVQGAN", "image out", 40), ("LM head", "text out", 196)]
for name, mod, y in dec:
    s.rect(DX, y, DW, EH, NEU_F, NEU, 2, 5)
    s.text(DX + DW / 2, y + 26, name, 19, INK, weight="bold")
    s.text(DX + DW / 2, y + 48, mod, 17, MUTED)

for _, _, y in enc:
    s.line(EX + EW + 2, y + EH / 2, FX - 1, y + EH / 2, INK, 2.2, "arr")
for _, _, y in dec:
    s.line(FX + FW + 2, y + EH / 2, DX - 1, y + EH / 2, INK, 2.2, "arr")

# shared token stream
s.line(8, 284, 472, 284, MUTED, 1.6, "arrGray")
s.text(240, 310, "one interleaved token sequence", 17, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_omni.svg"))
