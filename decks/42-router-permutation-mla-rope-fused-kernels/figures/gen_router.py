"""fig_router: TE fused_topk_with_score_function for one token, DeepSeek-V3-style routing: sigmoid scores
+ expert bias over 256 experts in 8 groups; group score = sum of top-2; keep 4 groups; top-8 experts among
them; normalize. One warp per token, scores in its shared-memory buffer; outputs probs and the routing map.
Aux-loss scores and aux loss are two more fused kernels. Expert ids are an illustrative example.
Sources: TE fused_topk_with_score_function.cu; arXiv 2603.07685 Sec. 4.3.4; DeepSeek-V3 report Sec. 2.1.2/4.2."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 268, "fig_router: one warp routes one token")

# input
s.box(2, 112, 118, 52, ["logits", "1 token"], NEU_F, NEU, size=16)
s.line(122, 138, 146, 138, ACC, 2.4, "arrAcc")
# container
CX, CY, CW, CH = 148, 22, 494, 242
s.rect(CX, CY, CW, CH, "#FFFFFF", ACC, 2.6, 6)
s.text(CX + CW / 2, 14, "one warp = one token, scores in shared memory", 17, ACC, weight="bold")
BX, BW, BG = 166, 50, 6
kept = {0, 2, 3, 6}


def groups(y, mode):
    for g in range(8):
        x = BX + g * (BW + BG)
        if mode == "all" or g in kept:
            s.rect(x, y, BW, 30, ACC_F, ACC, 1.8, 3)
        else:
            s.rect(x, y, BW, 30, "#FFFFFF", "#A6A6A6", 1.4, 3, dash="4,3")
        s.text(x + BW / 2, y + 21, f"G{g + 1}", 15, INK if (mode == "all" or g in kept) else MUTED)


s.text(BX, 50, "sigmoid + bias: 256 scores in 8 groups", 16, INK, anchor="start")
groups(58, "all")
s.text(BX, 120, "group score = top-2 sum, keep 4 groups", 16, INK, anchor="start")
groups(128, "kept")
s.text(BX, 190, "top-8 inside kept groups, normalize", 16, INK, anchor="start")
ids = ["e3", "e9", "e70", "e81", "e100", "e193", "e200", "e214"]
for k, e in enumerate(ids):
    x = BX + k * (BW + BG)
    s.box(x, 198, BW, 30, e, "#FFFFFF", ACC, color=ACC, size=15, weight="bold", sw=1.8)
s.text(BX + 4 * (BW + BG) - BG / 2, 254, "weights sum to 1", 15, MUTED, style="italic")
for yy in (90, 160):
    s.line(624, yy + 1, 624, yy + 36, INK, 1.8, "arr")

# outputs
for (lab, y) in ((["probs", "[T × 256]"], 54), (["routing map", "or top-k ids"], 124)):
    s.box(684, y, 182, 52, lab, "#FFFFFF", NEU, size=16)
    s.line(CX + CW + 2, y + 26, 682, y + 26, INK, 2.2, "arr")
s.box(684, 198, 182, 30, "aux-loss scores", ACC_F, ACC, size=15, sw=1.8)
s.box(684, 234, 182, 30, "aux loss", ACC_F, ACC, size=15, sw=1.8)
s.text(775, 192, "2 more fused kernels", 15, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_router.svg"))
