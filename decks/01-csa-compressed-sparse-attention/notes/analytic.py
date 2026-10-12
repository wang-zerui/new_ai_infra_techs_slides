"""Analytic per-query attention work (multiply-adds) per layer at a given context length.

All hyper-parameters come from the sources listed in notes/sources.md:
  V3.2 (DSA under MLA, MQA mode): 64 indexer heads x 128 dims, top-k 2048, 128 query heads,
      KV entry = 512 latent + 64 RoPE dims (QK over 576 dims, PV over 512 dims).
      [V3.2 paper §2.1; HF DeepSeek-V3.2 config.json; V3 paper §4.2 for d_c=512, d_R=64]
  V4 CSA: m=4, 64 indexer heads x 128 dims, top-k 1024 (Pro) / 512 (Flash), n_win=128,
      query heads 128 (Pro) / 64 (Flash), head dim c=512 (K = V = same 512-d entry).
      [V4 paper §2.3.1, §2.3.3, §4.2.1; HF config.json]
  V4 HCA: m'=128, dense over all compressed entries + 128 window entries. [V4 §2.3.2, §4.2.1]
Counts only the indexer QK dot products and the core-attention QK + PV dot products; ignores
projections, the compressor, softmax, top-k and the precision of each op.
"""
L = 2 ** 20  # "1M" tokens = 1,048,576

def dsa_v32(L):
    idx = 64 * 128 * L
    core = 128 * (576 + 512) * 2048
    return idx, core

def csa_v4(L, heads, topk):
    idx = 64 * 128 * (L // 4)
    core = heads * (512 + 512) * (topk + 128)
    return idx, core

def hca_v4(L, heads):
    return 0, heads * (512 + 512) * (L // 128 + 128)

if __name__ == "__main__":
    rows = {
        "V3.2 DSA layer": dsa_v32(L),
        "V4-Pro CSA layer": csa_v4(L, 128, 1024),
        "V4-Flash CSA layer": csa_v4(L, 64, 512),
        "V4-Pro HCA layer": hca_v4(L, 128),
        "V4-Flash HCA layer": hca_v4(L, 64),
    }
    for k, (i, c) in rows.items():
        print(f"{k:20s} indexer {i/1e9:7.3f} GMAC  core {c/1e9:7.3f} GMAC  total {(i+c)/1e9:7.3f}  idx/core {i/c if c else 0:6.1f}x")
    v32 = sum(dsa_v32(L)); pro = sum(csa_v4(L, 128, 1024)); fl = sum(csa_v4(L, 64, 512))
    print(f"DSA scored entries per query: {L:,}; kept: 2048; ratio {L//2048}x")
    print(f"CSA scored entries per query: {L//4:,}; kept: 1024+128 (Pro) / 512+128 (Flash)")
    print(f"V3.2 DSA / V4-Pro CSA per-layer total: {v32/pro:.2f}x ; / V4-Flash CSA: {v32/fl:.2f}x")
    print(f"V3.2 entries attended 2048 vs V4-Pro 1152: {2048/1152:.2f}x")
