"""Figures for deck 52 (fine-grained selective recomputation). 1 user unit = 1 pt.
Run: python3 gen_figs.py  -> fig_*.svg, then render each with ipads-arch-figures/scripts/render.sh."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda n: os.path.join(HERE, n)


def chip(s, x, y, w, h, label, kind, size=17):
    """Saved-activation chip. kind: keep (gray), drop (accent dashed), pinned (problem)."""
    if kind == "keep":
        s.box(x, y, w, h, label, NEU_F, NEU, INK, size, sw=1.8, rx=9)
    elif kind == "drop":
        s.box(x, y, w, h, label, "#FFFFFF", ACC, ACC, size, weight="bold", sw=2.2, rx=9, dash="6,4")
    elif kind == "pinned":
        s.box(x, y, w, h, label, PRB_F, PRB, PRB, size, weight="bold", sw=2.2, rx=9)


# ---------------------------------------------------------------- fig_conventional
def fig_conventional():
    s = SVG(870, 286, "two ways standard checkpointing fails for MoE")
    # panel divider
    s.line(436, 6, 436, 280, "#BFBFBF", 1.2)
    # (a) full-layer checkpoint ------------------------------------------------
    s.text(218, 22, "(a) Full-layer checkpoint", 20, INK, weight="bold")
    s.text(8, 78, "fwd", 18, INK, anchor="start", weight="bold")
    s.text(8, 174, "bwd", 18, INK, anchor="start", weight="bold")
    xs = [62, 152, 252, 330]
    ws = [84, 94, 72, 94]
    names = ["Attn", "dispatch", "MoE", "combine"]
    for x, w, n in zip(xs, ws, names):
        s.box(x, 52, w, 40, n, "#FFFFFF", NEU, INK, 18)
    # saved: only the layer input
    chip(s, 62, 100, 120, 30, "layer input", "keep", 16)
    s.line(122, 131, 122, 145, MUTED, 2, "arrGray")
    # replay row
    for x, w, n in zip(xs, ws, names):
        a2a = n in ("dispatch", "combine")
        if a2a:
            s.box(x, 148, w, 40, n, PRB_F, PRB, PRB, 18, weight="bold")
        else:
            s.box(x, 148, w, 40, n, NEU_F, NEU, INK, 18)
    s.text(218, 222, "whole layer replayed in backward", 18, INK)
    s.text(218, 248, "incl. EP all-to-all", 18, PRB, weight="bold")

    # (b) checkpoint of one op ---------------------------------------------------
    s.text(653, 22, "(b) Checkpoint of one op", 20, INK, weight="bold")
    # chain: z -> [SwiGLU] -> a -> [fc2 GEMM]
    s.box(470, 52, 150, 40, "SwiGLU", NEU_F, NEU, INK, 18)
    s.text(545, 46, "checkpointed", 16, MUTED, style="italic")
    s.box(704, 52, 150, 40, "fc2 GEMM", "#FFFFFF", NEU, INK, 18)
    s.line(622, 72, 702, 72, INK, 2.2, "arr")
    s.text(662, 64, "a", 18, INK, style="italic")
    # saved chips
    chip(s, 470, 112, 150, 34, "z (own input)", "keep")
    chip(s, 704, 112, 150, 34, "a (fc2 input)", "pinned")
    s.line(545, 94, 545, 110, MUTED, 1.6, "arrGray")
    s.line(779, 94, 779, 110, PRB, 1.8, "arrPrb")
    s.text(545, 172, "kept for its own backward", 17, MUTED, style="italic")
    s.text(779, 172, "kept by the consumer", 17, PRB, style="italic")
    s.text(653, 222, "output stays alive anyway", 18, INK)
    s.text(653, 248, "nothing freed", 18, PRB, weight="bold")
    s.save(P("fig_conventional.svg"))


# ---------------------------------------------------------------- fig_overview
def fig_overview():
    s = SVG(870, 318, "one DeepSeek-V3 layer: which saved activations the three output-discarding modules drop")
    # legend
    chip(s, 470, 4, 64, 26, "", "drop")
    s.text(542, 23, "dropped and rebuilt", 17, INK, anchor="start")
    chip(s, 708, 4, 64, 26, "", "keep")
    s.text(780, 23, "kept", 17, INK, anchor="start")

    def row(y, label, ops, chips, tags):
        s.text(8, y + 28, label, 19, INK, anchor="start", weight="bold")
        for (x, w, n) in ops:
            s.box(x, y, w, 44, n, "#FFFFFF", NEU, INK, 17)
        for i in range(len(ops) - 1):
            x1 = ops[i][0] + ops[i][1]
            x2 = ops[i + 1][0]
            s.line(x1 + 1, y + 22, x2 - 1, y + 22, INK, 2, "arr")
        for (cx, w, lab, kind, src) in chips:
            # chip below, with a stem from the arrow/op that produced it
            s.line(src, y + 23, src, y + 66, ACC if kind == "drop" else MUTED, 1.8, dash=None if kind == "keep" else "4,3")
            chip(s, cx - w / 2, y + 66, w, 32, lab, kind, 16)
        for (cx, t) in tags:
            s.text(cx, y + 120, t, 16, ACC, weight="bold", family="Menlo, monospace")

    # attention row
    ops_a = [(118, 70, "LN"), (226, 110, "down-proj"), (374, 128, "up-proj+RoPE"),
             (540, 112, "core attn"), (690, 96, "out-proj")]
    chips_a = [(207, 78, "LN out", "drop", 207), (355, 74, "latents", "keep", 355),
               (521, 82, "Q K V", "drop", 521), (671, 74, "Q K V", "keep", 671)]
    # the core-attn input chip is the same tensor: draw only once (as dropped) -> remove duplicate
    chips_a = chips_a[:3]
    tags_a = [(207, "layernorm"), (521, "mla_up_proj")]
    row(46, "Attn", ops_a, chips_a, tags_a)
    # MoE row
    ops_m = [(118, 70, "LN"), (226, 112, "dispatch"), (376, 96, "fc1"), (510, 108, "SwiGLU"),
             (656, 82, "fc2"), (776, 88, "combine")]
    chips_m = [(207, 78, "LN out", "drop", 207), (491, 78, "fc1 out", "keep", 491),
               (637, 74, "act out", "drop", 637)]
    tags_m = [(207, "layernorm"), (637, "moe_act")]
    row(186, "MoE", ops_m, chips_m, tags_m)
    s.save(P("fig_overview.svg"))


# ---------------------------------------------------------------- fig_discard (2 builds)
def fig_discard(step):
    s = SVG(870, 300, f"output-discarding checkpoint, build {step}")
    X0 = 150
    yC, yS = 62, 170          # compute lane, storage lane of output a
    H = 44
    s.text(138, yC + 28, "compute", 18, INK, anchor="end", weight="bold")
    s.text(138, yS + 20, "storage", 18, INK, anchor="end", weight="bold")
    s.text(138, yS + 40, "of a", 18, INK, anchor="end", weight="bold")
    # phase headers
    s.text(300, 30, "forward", 19, INK, weight="bold")
    s.line(472, 40, 472, 236, MUTED, 1.5, dash="6,5")
    # forward ops
    s.box(X0, yC, 108, H, "SwiGLU", NEU_F, NEU, INK, 18)
    s.box(X0 + 112, yC, 100, H, "fc2", "#FFFFFF", NEU, INK, 18)
    s.box(X0 + 216, yC, 100, H, "free a", ACC_F, ACC, ACC, 18, weight="bold")
    # storage lane forward: present from SwiGLU end to the free op
    s.box(X0 + 108, yS, 208, 40, "a on GPU", ACC_F, ACC, ACC, 17)
    if step == 1:
        s.box(X0 + 316, yS, 236, 40, "0 bytes · metadata kept", "#FFFFFF", MUTED, MUTED, 17, dash="6,4", sw=1.8)
    else:
        s.box(X0 + 316, yS, 256, 40, "0 bytes", "#FFFFFF", MUTED, MUTED, 17, dash="6,4", sw=1.8)
        # backward phase
        s.text(680, 30, "backward", 19, INK, weight="bold")
        s.box(486, yC, 130, H, "grad of fc2 out", "#FFFFFF", NEU, INK, 16)
        s.box(620, yC, 116, H, "recompute", ACC_F, ACC, ACC, 18, weight="bold")
        s.box(740, yC, 122, H, "fc2 bwd", "#FFFFFF", NEU, INK, 18)
        s.text(551, yC + H + 22, "hook fires", 16, ACC, style="italic")
        # storage lane backward: storage re-pointed to the recomputed data
        s.box(X0 + 572, yS, 140, 40, "a restored", ACC_F, ACC, ACC, 17)
        s.line(730, yC + H + 2, 730, yS - 2, ACC, 2.2, "arrAcc")
        s.line(818, yS - 2, 818, yC + H + 2, ACC, 2.2, "arrAcc")
    # chip: inputs kept
    s.text(X0, 252, "kept on GPU: z and the RNG state", 17, MUTED, anchor="start", style="italic")
    s.line(X0 - 4, 270, 862, 270, INK, 2, "arr")
    s.text(858, 294, "time", 18, INK, anchor="end")
    s.save(P(f"fig_discard_{step}.svg"))


# ---------------------------------------------------------------- fig_menu
def fig_menu():
    s = SVG(870, 318, "recompute modules: what each drops, keeps, and re-runs")
    cols = [(8, 196, "module"), (204, 230, "drops (output)"), (434, 220, "keeps (input)"), (654, 210, "re-runs")]
    for x, w, t in cols:
        s.text(x + 10, 26, t, 18, MUTED, anchor="start", weight="bold")
    s.line(8, 38, 864, 38, NEU, 1.5)
    rows = [
        ("layernorm", "2 norm outputs", "norm inputs", "1 norm each", "drop"),
        ("mla_up_proj", "Q K V (128 heads)", "MLA latents", "2 GEMMs + RoPE", "drop"),
        ("moe_act", "SwiGLU output", "fc1 output", "1 elementwise op", "drop"),
        ("core_attn · mlp", "inner activations", "module input", "whole submodule", "std"),
        ("moe", "inner activations", "module input", "experts + all-to-all", "prb"),
    ]
    y = 46
    for mod, drops, keeps, reruns, kind in rows:
        h = 46
        fill = ACC_F if kind == "drop" else ("#FFFFFF")
        s.rect(8, y, 856, h, fill, "#FFFFFF", 0.1, 0)
        s.text(18, y + 30, mod, 17, ACC if kind == "drop" else INK, anchor="start", weight="bold",
               family="Menlo, monospace")
        s.text(214, y + 30, drops, 18, INK, anchor="start")
        s.text(444, y + 30, keeps, 18, INK, anchor="start")
        s.text(664, y + 30, reruns, 18, PRB if kind == "prb" else INK, anchor="start",
               weight="bold" if kind == "prb" else "normal")
        y += h + 4
    # bracket labels at the right edge? keep table clean: a thin separator between groups
    s.line(8, 46 + 3 * 50 - 2, 864, 46 + 3 * 50 - 2, NEU, 1.2, dash="5,4")
    s.text(18, 310, "shaded rows: output-discarding checkpoint  ·  others: standard checkpoint", 16, MUTED,
           anchor="start", style="italic")
    s.save(P("fig_menu.svg"))


# ---------------------------------------------------------------- fig_fp8
def fig_fp8():
    s = SVG(870, 236, "exact replay: saved inputs, RNG states and FP8 recipe feed the recompute")
    s.text(140, 24, "saved in forward", 19, INK, weight="bold")
    labels = ["inputs (z)", "RNG states", "FP8 on/off + recipe"]
    ys = [40, 102, 164]
    for y, t in zip(ys, labels):
        chip(s, 8, y, 264, 42, t, "keep", 18)
    # recompute box
    s.box(356, 84, 196, 60, "recompute f", ACC_F, ACC, ACC, 19, weight="bold")
    s.text(454, 166, "TE recompute phase", 16, ACC, style="italic")
    for y in ys:
        s.path(f"M274,{y + 21} C318,{y + 21} 312,114 354,114", ACC, 2.2, "arrAcc")
    # result
    s.box(636, 84, 226, 60, "a restored in place", ACC_F, ACC, ACC, 18, weight="bold")
    s.line(554, 114, 634, 114, ACC, 2.4, "arrAcc")
    s.text(749, 168, "same storage object", 16, MUTED, style="italic")
    s.text(749, 190, "fc2 views see the data", 16, MUTED, style="italic")
    s.save(P("fig_fp8.svg"))


# ---------------------------------------------------------------- fig_platform
def fig_platform():
    s = SVG(870, 250, "which modules production configs recompute")
    cols = ["LayerNorm", "MLA up-proj", "SwiGLU", "dense MLP", "MoE down-proj"]
    rows = [("DeepSeek-V3 · H100", [1, 1, 1, 1, 0]),
            ("DeepSeek-V3 · GB200", [0, 0, 0, 1, 0]),
            ("Qwen3-235B · GB200/H100", [1, None, 1, 0, 0]),
            ("Kimi K2 · H800", [1, 1, 1, "?", 1])]
    X0, CW, Y0, RH = 288, 115, 44, 50
    for j, c in enumerate(cols):
        s.text(X0 + CW * j + CW / 2, 30, c, 16, INK, weight="bold")
    for i, (name, vals) in enumerate(rows):
        y = Y0 + i * RH
        s.text(X0 - 14, y + 29, name, 18, INK, anchor="end", weight="bold")
        for j, v in enumerate(vals):
            x = X0 + CW * j + 6
            if v == 1:
                s.box(x, y + 4, CW - 12, RH - 10, "recompute", ACC_F, ACC, ACC, 16, weight="bold")
            elif v is None:
                s.box(x, y + 4, CW - 12, RH - 10, "no MLA", "#FFFFFF", "#BFBFBF", MUTED, 15, dash="4,3", sw=1.4)
            elif v == "?":
                s.box(x, y + 4, CW - 12, RH - 10, "not stated", "#FFFFFF", "#BFBFBF", MUTED, 15, dash="4,3", sw=1.4)
            else:
                s.box(x, y + 4, CW - 12, RH - 10, "keep", "#FFFFFF", "#BFBFBF", MUTED, 15, sw=1.4)
    s.save(P("fig_platform.svg"))


# ---------------------------------------------------------------- fig_pairs (insight)
def fig_pairs():
    """Per-token values held by each (kept input, dropped output) pair in DeepSeek-V3 (analytic, TP1):
    LayerNorm 7168 -> 7168; MLA latents 1536+512+64 = 2112 -> Q,K,V 128*(192+192+128) = 65536;
    SwiGLU over 8 routed copies: fc1 out 8*4096 = 32768 -> act out 8*2048 = 16384."""
    s = SVG(480, 286, "kept input vs dropped output, values per token")
    k = 352 / 65536
    chip(s, 118, 4, 40, 24, "", "keep")
    s.text(166, 22, "kept input", 16, INK, anchor="start")
    chip(s, 290, 4, 40, 24, "", "drop")
    s.text(338, 22, "dropped output", 16, INK, anchor="start")
    groups = [("LayerNorm", 7168, "7,168", 7168, "7,168"),
              ("MLA", 2112, "2,112", 65536, "65,536"),
              ("SwiGLU ×8", 32768, "32,768", 16384, "16,384")]
    y = 44
    for name, vi, li, vo, lo in groups:
        s.text(108, y + 36, name, 18, INK, anchor="end", weight="bold")
        wi, wo = vi * k, vo * k
        s.rect(118, y, wi, 26, NEU_F, NEU, 1.6, 3)
        s.text(118 + wi + 6, y + 19, li, 16, INK, anchor="start")
        s.rect(118, y + 32, wo, 26, "#FFFFFF", ACC, 2, 3, dash="6,4")
        if wo > 120:
            s.text(118 + wo - 8, y + 51, lo, 16, ACC, anchor="end", weight="bold")
        else:
            s.text(118 + wo + 6, y + 51, lo, 16, ACC, anchor="start", weight="bold")
        y += 80
    s.text(118, 282, "values per token, DeepSeek-V3", 16, MUTED, anchor="start", style="italic")
    s.save(P("fig_pairs.svg"))


if __name__ == "__main__":
    fig_conventional()
    fig_overview()
    fig_discard(1)
    fig_discard(2)
    fig_menu()
    fig_fp8()
    fig_platform()
    fig_pairs()
