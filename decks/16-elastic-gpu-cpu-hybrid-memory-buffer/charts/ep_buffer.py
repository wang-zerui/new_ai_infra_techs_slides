"""ep_buffer (analytic): bytes DeepEP V2.5 reserves for one EPBuffer vs. the bytes a rank can receive under
perfectly balanced routing. Replica of csrc/buffers/ep.hpp calculate_buffer_size (+ layout/ep/token.cuh,
kernels/ep/{dispatch,combine}.hpp), hybrid mode, 8 GPUs per node.
Settings: README V3-like benchmark (8K tokens per rank, hidden 7168, top-8, FP8 dispatch, BF16 combine).
Balanced bound: each of N ranks sends each of its T tokens to <= k ranks -> <= k*T tokens per rank on average,
times the larger (combine, BF16) token record. See notes/sources.md.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc


def al(x, a):
    return (x + a - 1) // a * a


def token_bytes(hidden_bytes, sf_bytes, topk, with_meta):
    meta = topk * 8 + ((1 + topk) * 4 if with_meta else 0)
    return al(al(hidden_bytes, 32) + al(sf_bytes, 32) + al(meta, 32), 64)


KCH = 8 * 160   # kNumMaxChannels


def buffer_bytes(N, T, hidden=7168, topk=8, nvl=8):
    so, su = max(1, N // nvl), min(N, nvl)
    dt = token_bytes(hidden, (hidden + 31) // 32 * 4, topk, True)      # FP8 + UE8M0/FP32 scale packs
    ct = token_bytes(hidden * 2, 0, topk, False)                        # BF16
    if so == 1:
        d, c = dt * N * T, ct * min(N, topk) * T
    else:
        d = dt * su * so * T + dt * T + dt * so * (T + KCH)
        c = ct * min(su, topk) * so * T + ct * min(so, topk) * T + ct * so * (T + KCH)
    return al(max(d, c), 2 << 20), ct


GiB = 2 ** 30
T, K = 8192, 8
eps = [8, 16, 32, 64, 128, 256]
res = [buffer_bytes(n, T) for n in eps]
reserved = [b / GiB for b, _ in res]
balanced = [K * T * ct / GiB for _, ct in res]
for n, r, b in zip(eps, reserved, balanced):
    print(f"EP{n}: reserved {r:.2f} GiB, balanced <= {b:.2f} GiB, ratio {r / b:.1f}x")

sc.setup()
fig, ax = sc.figure("right")
xs = list(range(len(eps)))
ax.plot(xs, reserved, color=sc.PROBLEM, marker="o", label="reserved (worst case)")
ax.plot(xs, balanced, color=sc.GRAYS[3], marker="s", linestyle="--", label="used under balanced routing (≤ k·T)")
ax.set_xticks(xs)
ax.set_xticklabels([str(n) for n in eps])
ax.set_xlabel("EP size (GPUs)")
ax.set_ylabel("Buffer per GPU (GiB)")
ax.set_ylim(0, 36)
sc.top_legend(ax, ncol=1)
i64 = eps.index(64)
sc.callout(ax, f"{reserved[i64]:.1f} GiB", xy=(i64, reserved[i64]), xytext=(i64 - 1.6, reserved[i64] + 12),
           color=sc.PROBLEM)

sc.save(fig, __file__)
