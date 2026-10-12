"""fig_stack: VeOmni layered architecture; accent = VeOmni-owned, gray = HF models / PyTorch / hardware; [N] = sibling deck."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 322, "fig_stack")

LX, BX, BW = 10, 150, 712
RH, GAP = 50, 14
rows = [
    ("Models", [("ViT / Whisper", 0), ("Qwen3 / DeepSeek LLMs", 0), ("MoVQGAN / DiT", 0)]),
    ("Interfaces", [("❶ lm_encode / lm_head", 1), ("❷ ParallelPlan + DeviceMesh", 1)]),
    ("Parallelism", [("FSDP2 / HSDP [70]", 1), ("EP + FSDP2 ❷", 1), ("Ulysses SP ❸ [68]", 1)]),
    ("Services", [("Dynamic batching [51]", 1), ("DCP checkpoint [50]", 1), ("Kernels / offload", 1)]),
    ("Runtime", [("PyTorch DTensor / FSDP2 / DCP", 0), ("CUDA / ROCm / Ascend / MLU", 0)]),
]
y = 8
for label, boxes in rows:
    s.text(LX + 128, y + RH / 2 + 6.5, label, 19, INK, anchor="end", weight="bold")
    n = len(boxes)
    w = (BW - GAP * (n - 1)) / n
    x = BX
    for text, ours in boxes:
        if ours:
            s.box(x, y, w, RH, text, ACC_F, ACC, ACC, 18, "bold")
        else:
            s.box(x, y, w, RH, text, NEU_F, NEU, INK, 18)
        x += w + GAP
    y += RH + GAP
s.save(os.path.join(HERE, "fig_stack.svg"))
