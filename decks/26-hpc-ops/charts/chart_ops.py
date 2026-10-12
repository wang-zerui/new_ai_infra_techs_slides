"""Per-operator maximum observed speedup, HPC-Ops README "Performance" table (Representative result column).
Each bar = "up to" value vs. the baselines the README lists for that operator; performance varies by shape."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

# (operator, max speedup, baselines per README)
DATA = [
    ("Fused MoE FP8 (TP)", 1.6),        # vLLM CUTLASS, vLLM Triton, SGLang
    ("AllReduce + RMSNorm", 1.76),      # NCCL, FlashInfer
    ("GroupGEMM FP8 decode", 1.88),     # DeepGEMM
    ("Attention FP8 decode", 2.0),      # FlashInfer, FA3, TensorRT-LLM
    ("Attention BF16 decode", 2.22),    # FlashInfer, FA2, FA3, TensorRT-LLM
    ("Dynamic decode", 2.88),           # static split-k
    ("Sparse attention FP8", 3.16),     # MIT-BSA, FlashPrefill-BSA (BF16), dense FP8
    ("Route GEMM", 3.22),               # cuBLAS FP32, cuBLAS TF32
    ("Fused sampler", 8.5),             # vLLM-style PyTorch, FlashInfer
]
sc.setup()
fig, ax = sc.figure("full")
names = [d[0] for d in DATA]; vals = [d[1] for d in DATA]
ys = range(len(DATA))
ax.barh(list(ys), vals, height=0.66, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2)
LABELS = ["1.6×", "1.76×", "1.88×", "2.0×", "2.22×", "2.88×", "3.16×", "3.22×", "8.5×"]
for y, v, t in zip(ys, vals, LABELS):
    ax.text(v + 0.08, y, t, va="center", ha="left", fontsize=19, fontweight="bold", color=sc.ACCENT)
ax.axvline(1.0, color="#595959", lw=2, ls="--")
ax.set_yticks(list(ys)); ax.set_yticklabels(names)
ax.set_xlim(0, 9.8)
ax.set_xlabel("Max observed speedup over listed baselines (×; dashed line = 1×)")
ax.grid(axis="x", linestyle="--", color="#BFBFBF"); ax.grid(axis="y", visible=False)
ax.text(0.98, 0.05, "Higher is better", transform=ax.transAxes, ha="right", va="bottom", fontsize=16, fontstyle="italic", color=sc.GOOD)
sc.save(fig, __file__)
