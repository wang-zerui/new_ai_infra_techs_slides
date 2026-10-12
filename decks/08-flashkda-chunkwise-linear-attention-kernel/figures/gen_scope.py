"""fig_scope.svg: what the public FlashKDA v1 covers vs what lives elsewhere.
Sources: FlashKDA README (SM90+, CUDA 12.9+, K = V = 128, lower_bound in [-5, 0], cu_seqlens varlen, bf16/fp32 state,
forward API under inference_mode); K3 Sec. 5.1.1 (training use), Sec. 5.4.2 (separate decode kernel)."""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from svgkit import *
s = SVG(870, 285)
def col(x, w, title, items, fill, stroke, tc, mark, mc):
    s.box(x, 10, w, 266, fill, stroke, 2, 6)
    s.t(x + w / 2, 42, title, 20, "bold", tc)
    for i, (a, b) in enumerate(items):
        y = 80 + i * 46
        s.t(x + 24, y, mark, 19, "bold", mc, "middle")
        s.t(x + 42, y, a, 18, "bold", INK, "start")
        s.t(x + 42, y + 20, b, 15, "normal", GRAY, "start")
col(10, 420, "In FlashKDA v1", [("Hopper and Blackwell", "SM90+ · CUDA 12.9+"), ("Head dims K = V = 128", "only this shape"),
                                 ("Bounded gate", "lower bound in [−5, 0]"), ("Forward pass and varlen", "prefill · cu_seqlens batches")],
    ACC_F, ACC, ACC, "✓", ACC)
col(440, 420, "Not in the public v1", [("Backward pass", "K3 reports training use"), ("Decoding", "separate fused kernel (K3 §5.4.2)"),
                                       ("Other head dims", "and pre-Hopper GPUs"), ("Unbounded gates", "the API requires a lower bound")],
    GRAY_F, GRAY, INK, "✗", RED)
s.save(pathlib.Path(__file__).with_name("fig_scope.svg"))
