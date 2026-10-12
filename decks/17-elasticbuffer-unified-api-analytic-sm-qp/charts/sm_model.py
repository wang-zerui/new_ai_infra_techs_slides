"""ANALYTIC chart: SM count chosen by DeepEP V2.5's get_theoretical_num_sms (deep_ep/buffers/ep.py, main @ 93eb6eb)
re-implemented below, for hybrid EP 8 x N with 256 experts, top-8 (DeepSeek-V3 routing), balanced gate.
Inputs: RDMA 50 GB/s (400 Gb/s CX7, rate/8 as in get_rdma_gbs), NVLink 160 GB/s (H800, V3 report Sec. 3.2.2),
per-SM HBM read/write 180/45 GB/s (code defaults), 132 SMs. All shown sizes are RDMA-bound for any NVLink >= 118 GB/s,
so the NVLink assumption does not change them. Reference line: V1 README example Buffer.set_num_sms(24).
See notes/sources.md (derivations)."""
import math, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc


def align(x, y):
    return -(-x // y) * y


def num_sms_v25(num_experts, topk, nodes, gpus_per_node=8, rdma=50.0, nvl=160.0, sr=180.0, sw=45.0, dev=132):
    n = nodes * gpus_per_node
    E = lambda g: g * (1 - math.comb(num_experts - num_experts // g, topk) / math.comb(num_experts, topk))
    k, kso = E(n), E(nodes)
    r = 1 / k + kso / k
    w = 1 / k + (1 / k) * (kso / nodes) + 1
    rd = (1 / k) * kso * (1 - 1 / nodes)
    nv = 1 - 1 / gpus_per_node
    bt, bg = (rd, rdma) if rd / rdma > nv / nvl else (nv, nvl)
    raw = max(bg / bt * r / sr, bg / bt * w / sw, bg / bt * (w - nv) / sr, bg / bt * (r + nv) / sw)
    return min(align(max(4, math.ceil(raw * 1.3)), 4), dev // 2 * 2), raw


nodes = [4, 8, 16, 32]
vals = [num_sms_v25(256, 8, m)[0] for m in nodes]
print("V2.5 model SMs:", dict(zip(nodes, vals)))

sc.setup()
fig, ax = sc.figure("right")
names = [f"EP 8×{m}\n{8 * m} GPUs" for m in nodes]
sc.bars(ax, names, vals, ours=list(range(len(vals))), labels=[str(v) for v in vals])
ax.axhline(24, color=sc.PROBLEM, lw=2.5, ls="--")
ax.text(len(vals) - 0.55, 24.8, "V1: fixed 24", color=sc.PROBLEM, fontsize=20, fontweight="bold", ha="right", va="bottom")
ax.set_ylim(0, 30)
ax.set_ylabel("Communication SMs per GPU")
sc.better(ax, "Lower is better", where="upper left")
sc.save(fig, __file__)
