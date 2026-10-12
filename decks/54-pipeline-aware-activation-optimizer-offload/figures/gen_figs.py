"""Figures for deck 54 (pipeline-aware activation / optimizer offload). 1 user unit = 1 pt.
Run: python3 gen_figs.py, then render each fig_*.svg with ipads-arch-figures/scripts/render.sh."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda n: os.path.join(HERE, n)
HOST_F = "#FFF2CC"     # host-memory artifact (cream), one extra category with legend
HOST_S = "#BF9000"


def chip(s, x, y, w, h, label, kind, size=17):
    if kind == "keep":
        s.box(x, y, w, h, label, NEU_F, NEU, INK, size, sw=1.8, rx=9)
    elif kind == "recompute":
        s.box(x, y, w, h, label, "#FFFFFF", ACC, ACC, size, weight="bold", sw=2.2, rx=9, dash="6,4")
    elif kind == "offload":
        s.box(x, y, w, h, label, ACC_F, ACC, ACC, size, weight="bold", sw=2.2, rx=9)
    elif kind == "host":
        s.box(x, y, w, h, label, HOST_F, HOST_S, INK, size, sw=1.8, rx=4)
    elif kind == "pinned":
        s.box(x, y, w, h, label, PRB_F, PRB, PRB, size, weight="bold", sw=2.2, rx=9)


def simulate_1f1b(p, m, F=1, B=2):
    order = {}
    for s in range(p):
        w = min(p - 1 - s, m)
        seq = [("F", i) for i in range(w)]
        f, b = w, 0
        while b < m:
            if f < m:
                seq.append(("F", f)); f += 1
            seq.append(("B", b)); b += 1
        order[s] = seq
    done, t, idx, out = {}, [0] * p, [0] * p, []
    changed = True
    while changed:
        changed = False
        for s in range(p):
            if idx[s] >= len(order[s]):
                continue
            kind, i = order[s][idx[s]]
            dep = (done.get((s - 1, "F", i)) if s > 0 else 0) if kind == "F" else \
                (done.get((s + 1, "B", i)) if s < p - 1 else done.get((s, "F", i)))
            if dep is None:
                continue
            st = max(t[s], dep)
            d = F if kind == "F" else B
            done[(s, kind, i)] = st + d
            t[s] = st + d
            out.append((s, kind, i, st, st + d))
            idx[s] += 1
            changed = True
    return out


# ---------------------------------------------------------------- fig_inflight (villain)
def fig_inflight():
    s = SVG(870, 268, "1F1B: the first stage holds the most micro-batches of activations")
    p, m, U, X0 = 4, 6, 21, 112
    sched = simulate_1f1b(p, m)
    s.text(X0, 22, "1F1B schedule · 4 stages · 6 micro-batches", 17, MUTED, anchor="start", style="italic")
    s.text(790, 22, "held at peak", 17, MUTED, style="italic")
    for st in range(p):
        y = 38 + st * 52
        s.text(X0 - 12, y + 25, f"stage {st}", 18, INK, anchor="end", weight="bold")
    for st, kind, i, a, b in sched:
        y = 38 + st * 52
        if kind == "F":
            s.box(X0 + a * U, y, (b - a) * U, 38, str(i + 1), "#FFFFFF", NEU, INK, 16, sw=1.4, rx=2)
        else:
            s.box(X0 + a * U, y, (b - a) * U, 38, str(i + 1), NEU_F, NEU, INK, 16, sw=1.4, rx=2)
    # peak window on stage 0: after F4 (t=4) until B1 ends (t=12)
    # resident chips
    for st in range(p):
        y = 38 + st * 52
        n = p - st
        for k in range(n):
            x = 716 + k * 36
            kind = "pinned" if st == 0 else "keep"
            chip(s, x, y + 4, 30, 30, "", kind)
    # legend
    s.box(X0, 252 - 18, 34, 22, "", "#FFFFFF", NEU, INK, 14, sw=1.4, rx=2)
    s.text(X0 + 42, 251, "forward", 16, INK, anchor="start")
    s.box(X0 + 132, 252 - 18, 34, 22, "", NEU_F, NEU, INK, 14, sw=1.4, rx=2)
    s.text(X0 + 174, 251, "backward (2× longer)", 16, INK, anchor="start")
    s.text(790, 251, "chip = micro-batch", 16, MUTED, style="italic")
    s.save(P("fig_inflight.svg"))


# ---------------------------------------------------------------- fig_policy (overview)
def fig_policy():
    s = SVG(870, 318, "one layer: offload heavy inputs, recompute cheap outputs")
    chip(s, 360, 4, 56, 26, "", "offload")
    s.text(424, 23, "offloaded to host", 17, INK, anchor="start")
    chip(s, 600, 4, 56, 26, "", "recompute")
    s.text(664, 23, "recomputed (deck 52)", 17, INK, anchor="start")

    def row(y, label, ops, chips):
        s.text(8, y + 28, label, 19, INK, anchor="start", weight="bold")
        for (x, w, n) in ops:
            s.box(x, y, w, 44, n, "#FFFFFF", NEU, INK, 17)
        for i in range(len(ops) - 1):
            s.line(ops[i][0] + ops[i][1] + 1, y + 22, ops[i + 1][0] - 1, y + 22, INK, 2, "arr")
        for (cx, w, lab, kind, tag) in chips:
            s.line(cx, y + 23, cx, y + 66, ACC, 1.8, dash=None if kind == "offload" else "4,3")
            chip(s, cx - w / 2, y + 66, w, 32, lab, kind, 16)
            s.text(cx, y + 120, tag, 15, ACC, weight="bold", family="Menlo, monospace")

    ops_a = [(96, 64, "LN"), (190, 118, "QKV linear"), (338, 112, "core attn"), (480, 104, "attn proj")]
    chips_a = [(175, 76, "LN out", "recompute", "layernorm"), (323, 72, "Q K V", "offload", "core_attn"),
               (465, 88, "attn out", "offload", "attn_proj")]
    row(46, "Attn", ops_a, chips_a)
    ops_m = [(96, 64, "LN"), (190, 100, "dispatch"), (320, 72, "fc1"), (422, 96, "SwiGLU"), (548, 64, "fc2"),
             (642, 96, "combine")]
    chips_m = [(175, 76, "LN out", "recompute", "layernorm"), (305, 78, "tokens", "offload", "expert_fc1"),
               (407, 78, "fc1 out", "offload", "moe_act"), (533, 82, "act out", "recompute", "moe_act")]
    row(186, "MoE", ops_m, chips_m)
    s.save(P("fig_policy.svg"))


# ---------------------------------------------------------------- fig_streams (mechanism 1, two builds)
def fig_streams(step):
    s = SVG(870, 262, f"module-level offload and layer-staggered reload, build {step}")
    X0, WF, WB, GAP = 92, 54, 70, 24
    yC, yD, yH = 50, 112, 174
    s.text(X0 - 10, yC + 26, "compute", 17, INK, anchor="end", weight="bold")
    s.text(X0 - 10, yD + 26, "D2H", 17, ACC, anchor="end", weight="bold")
    s.text(X0 - 10, yH + 26, "H2D", 17, ACC, anchor="end", weight="bold")
    fwd = ["attn 1", "fc1 1", "attn 2", "fc1 2", "attn 3", "fc1 3"]
    s.text(X0 + 3 * WF, 30, "forward", 18, INK, weight="bold")
    for i, n in enumerate(fwd):
        s.box(X0 + i * WF, yC, WF, 40, n, "#FFFFFF", NEU, INK, 16, sw=1.6, rx=2)
    # D2H: each offloaded module's input copies out while the NEXT module computes; layer 3 stays on GPU
    for i in range(4):
        x = X0 + (i + 1) * WF + 3
        s.box(x, yD, WF - 6, 40, fwd[i], ACC_F, ACC, ACC, 15, sw=1.8, rx=2)
    s.text(X0 + 5.5 * WF, yD + 18, "layer 3", 15, MUTED, style="italic")
    s.text(X0 + 5.5 * WF, yD + 36, "kept", 15, MUTED, style="italic")
    xb = X0 + 6 * WF + GAP
    s.line(xb - GAP / 2, 36, xb - GAP / 2, 222, MUTED, 1.4, dash="6,5")
    if step == 2:
        s.text(xb + 3 * WB, 30, "backward", 18, INK, weight="bold")
        bwd = ["fc1 3", "attn 3", "fc1 2", "attn 2", "fc1 1", "attn 1"]
        for i, n in enumerate(bwd):
            s.box(xb + i * WB, yC, WB, 40, n, NEU_F, NEU, INK, 16, sw=1.6, rx=2)
        # H2D: the same module of the previous layer reloads while this module's neighbour runs backward
        for i in range(1, 5):
            x = xb + i * WB + 3
            s.box(x, yH, WB - 6, 40, bwd[i + 1], ACC_F, ACC, ACC, 15, sw=1.8, rx=2)
            # ready just before the module that needs it
            s.path(f"M{x + WB - 6},{yH + 8} C{x + WB + 2},{yH - 20} {x + WB + 2},{yC + 60} {x + WB + 2},{yC + 42}",
                   ACC, 1.6, "arrAcc")
    s.line(X0 - 4, 238, 862, 238, INK, 2, "arr")
    s.text(858, 258, "time", 16, INK, anchor="end")
    s.save(P(f"fig_streams_{step}.svg"))


# ---------------------------------------------------------------- fig_1f1b (mechanism 2)
def fig_1f1b():
    s = SVG(870, 270, "1F1B steady state on one stage: offload the forward just run, prefetch the next backward")
    X0, WF, WB = 128, 58, 116
    yC, yD, yH = 52, 118, 184
    s.text(X0 - 10, yC + 26, "compute", 17, INK, anchor="end", weight="bold")
    s.text(X0 - 10, yD + 18, "D2H", 17, ACC, anchor="end", weight="bold")
    s.text(X0 - 10, yD + 36, "offload", 15, ACC, anchor="end", style="italic")
    s.text(X0 - 10, yH + 18, "H2D", 17, ACC, anchor="end", weight="bold")
    s.text(X0 - 10, yH + 36, "prefetch", 15, ACC, anchor="end", style="italic")
    seq = [("F", 4), ("B", 1), ("F", 5), ("B", 2), ("F", 6), ("B", 3), ("F", 7)]
    x, pos = X0, []
    for kind, i in seq:
        w = WF if kind == "F" else WB
        s.box(x, yC, w, 40, f"{kind}{i}", "#FFFFFF" if kind == "F" else NEU_F, NEU, INK, 17, sw=1.6, rx=2)
        pos.append((kind, i, x, w))
        x += w
    s.text(X0, 30, "stage 0 in steady state · F = forward · B = backward · digit = micro-batch", 16, MUTED,
           anchor="start", style="italic")
    for k, (kind, i, x0, w) in enumerate(pos):
        if kind == "F" and k + 1 < len(pos):
            bx, bw = pos[k + 1][2], pos[k + 1][3]
            s.box(bx + 26, yD, bw - 30, 40, f"F{i} acts out", ACC_F, ACC, ACC, 15, sw=1.8, rx=2)
            s.line(x0 + w + 2, yC + 41, bx + 34, yD - 1, ACC, 1.6, "arrAcc")
        if kind == "B" and k + 2 < len(pos):
            nb = pos[k + 2]           # next backward
            st = x0 + w / 2
            en = nb[2] - 4
            s.box(st, yH, en - st, 40, f"B{nb[1]} acts in", ACC_F, ACC, ACC, 15, sw=1.8, rx=2)
            s.path(f"M{en},{yH + 20} L{nb[2] + 8},{yH + 20} L{nb[2] + 8},{yC + 42}", ACC, 1.6, "arrAcc")
    s.line(X0 - 4, 244, 862, 244, INK, 2, "arr")
    s.text(858, 264, "time", 16, INK, anchor="end")
    s.save(P("fig_1f1b.svg"))


# ---------------------------------------------------------------- fig_optimizer (mechanism 3)
def fig_optimizer():
    s = SVG(870, 296, "optimizer state offload vs CPU optimizer")
    X0 = 210
    s.text(X0 - 12, 66, "state offload", 18, ACC, anchor="end", weight="bold")
    s.text(X0 - 12, 88, "step on GPU", 16, MUTED, anchor="end", style="italic")
    s.text(X0 - 12, 196, "CPU optimizer", 18, INK, anchor="end", weight="bold")
    s.text(X0 - 12, 218, "step on host", 16, MUTED, anchor="end", style="italic")
    # (a) state offload: compute row + copy row
    s.box(X0, 40, 330, 40, "forward + backward of all micro-batches", "#FFFFFF", NEU, INK, 16, sw=1.6, rx=2)
    s.box(X0 + 450, 40, 90, 40, "step", ACC_F, ACC, ACC, 17, weight="bold", sw=1.8, rx=2)
    s.box(X0 + 340, 88, 104, 34, "reload", ACC_F, ACC, ACC, 16, sw=1.8, rx=2)
    s.box(X0 + 546, 88, 104, 34, "offload", ACC_F, ACC, ACC, 16, sw=1.8, rx=2)
    s.text(X0 + 165, 106, "HBM free of state", 16, ACC, style="italic")
    s.line(X0 + 446, 105, X0 + 494, 82, ACC, 1.6, "arrAcc")
    s.line(X0 + 540, 70, X0 + 596, 86, ACC, 1.6, "arrAcc")
    s.line(8, 146, 862, 146, "#D9D9D9", 1.2)
    # (b) CPU optimizer
    s.box(X0, 170, 330, 40, "forward + backward of all micro-batches", "#FFFFFF", NEU, INK, 16, sw=1.6, rx=2)
    s.box(X0 + 340, 218, 104, 34, "grads D2H", "#FFFFFF", NEU, INK, 16, sw=1.6, rx=2)
    s.box(X0 + 450, 218, 90, 34, "step", HOST_F, HOST_S, INK, 17, sw=1.6, rx=2)
    s.box(X0 + 546, 218, 104, 34, "params H2D", "#FFFFFF", NEU, INK, 16, sw=1.6, rx=2)
    s.text(X0 + 165, 236, "host CPU does the math", 16, MUTED, style="italic")
    # legend
    chip(s, X0, 268, 40, 22, "", "host", 14)
    s.text(X0 + 48, 285, "host side", 16, INK, anchor="start")
    s.text(X0 + 160, 285, "state = exp_avg + exp_avg_sq + FP32 master weights", 16, MUTED, anchor="start",
           style="italic")
    s.save(P("fig_optimizer.svg"))


# ---------------------------------------------------------------- fig_hide (insight)
def fig_hide():
    s = SVG(480, 262, "a copy is free when the next computation outlasts it")
    X0 = 96

    def case(y, title, mod_w, nxt_w, copy_w, hidden):
        s.text(8, y - 10, title, 17, INK, anchor="start", weight="bold")
        s.text(X0 - 8, y + 24, "compute", 15, INK, anchor="end")
        s.text(X0 - 8, y + 70, "D2H", 15, ACC, anchor="end", weight="bold")
        s.box(X0, y, mod_w, 36, "module", "#FFFFFF", NEU, INK, 15, sw=1.6, rx=2)
        s.box(X0 + mod_w, y, nxt_w, 36, "next module", NEU_F, NEU, INK, 15, sw=1.6, rx=2)
        cx = X0 + mod_w + 2
        if hidden:
            s.box(cx, y + 48, copy_w, 34, "copy", ACC_F, ACC, ACC, 15, sw=1.8, rx=2)
        else:
            vis = X0 + mod_w + nxt_w - cx
            s.box(cx, y + 48, vis, 34, "copy", ACC_F, ACC, ACC, 15, sw=1.8, rx=2)
            s.rect(cx + vis, y + 48, copy_w - vis, 34, PRB_F, PRB, 2, 2)
            s.text(cx + vis + (copy_w - vis) / 2, y + 70, "exposed", 15, PRB, weight="bold")
            s.rect(X0 + mod_w + nxt_w, y, copy_w - vis, 36, "#FFFFFF", PRB, 1.6, 2, dash="5,3")

    case(36, "compute-heavy (expert fc1 · attention)", 80, 230, 170, True)
    case(160, "compute-light (norm · activation)", 60, 90, 200, False)
    s.save(P("fig_hide.svg"))


if __name__ == "__main__":
    fig_inflight()
    fig_policy()
    fig_streams(1)
    fig_streams(2)
    fig_1f1b()
    fig_optimizer()
    fig_hide()
