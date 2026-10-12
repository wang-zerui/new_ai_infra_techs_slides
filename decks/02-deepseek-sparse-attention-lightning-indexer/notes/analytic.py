"""Analytic per-query multiply-adds (MACs) per layer for DeepSeek-V3.2 (DSA) vs dense MLA.

Hyper-parameters (see notes/sources.md):
  indexer: H_I = 64 heads x d_I = 128 dims, one shared 128-d key per token, top-k = 2048
           [HF DeepSeek-V3.2-Exp config.json index_n_heads/index_head_dim/index_topk; V3.2 paper Sec. 2.1.1]
  MLA: 128 query heads; MQA mode: QK over 512 latent + 64 RoPE = 576 dims, PV over 512 dims
       MHA mode (dense prefill in V3.1-Terminus): QK over 128 + 64 = 192 dims, PV over 128 dims
       [DeepSeek-V3 report Sec. 4.2: n_h = 128, d_h = 128, d_c = 512, d_h^R = 64; V3.2 Appendix A]
Counts only dot products against cached keys (indexer QK; attention QK + PV); ignores projections, softmax,
top-k and precision (the indexer runs in FP8, attention in BF16 in the reference code).
"""
H_I, D_I, TOPK = 64, 128, 2048
HEADS = 128
MQA_PER_KEY = HEADS * ((512 + 64) + 512)     # 139,264
MHA_PER_KEY = HEADS * ((128 + 64) + 128)     # 40,960
IDX_PER_KEY = H_I * D_I                      # 8,192


def dense_mqa(L):
    return MQA_PER_KEY * L


def dsa(L):
    return IDX_PER_KEY * L, MQA_PER_KEY * min(TOPK, L)


if __name__ == "__main__":
    print(f"per cached key: indexer {IDX_PER_KEY:,} MACs; MQA-mode attention {MQA_PER_KEY:,} "
          f"({MQA_PER_KEY / IDX_PER_KEY:.1f}x); MHA-mode attention {MHA_PER_KEY:,} ({MHA_PER_KEY / IDX_PER_KEY:.1f}x)")
    L = 128 * 1024
    i, c = dsa(L)
    d = dense_mqa(L)
    print(f"L = {L:,}: dense MQA {d / 1e9:.2f} G ; DSA indexer {i / 1e9:.3f} G + core {c / 1e9:.3f} G = {(i + c) / 1e9:.3f} G"
          f" -> {d / (i + c):.1f}x fewer MACs; indexer share of DSA = {i / (i + c):.0%}")
    print(f"keys attended per query at 128K: {TOPK} / {L} = {TOPK / L:.2%}")
    # indexer K cache vs MLA FP8 cache, bytes per token per layer
    idx_bytes = 128 * 1 + 4      # 128 FP8 values + one FP32 scale (block 128)
    mla_bytes = 656              # vLLM blog: 512 FP8 + 16 B scales + 64 BF16 RoPE
    print(f"indexer K cache {idx_bytes} B/token/layer vs MLA {mla_bytes} B -> +{idx_bytes / mla_bytes:.0%}")
    print(f"GLM-5 adaptation tokens vs DeepSeek: 943.7B / 20B = {943.7 / 20:.0f}x")
    print(f"DeepSeek warm-up tokens: 1000 x 16 x 128K = {1000 * 16 * 131072 / 1e9:.2f}B ;"
          f" sparse: 15000 x 480 x 128K = {15000 * 480 * 131072 / 1e9:.1f}B")
    print(f"GLM-5 warm-up tokens: 1000 x 14 x 202,752 = {1000 * 14 * 202752 / 1e9:.2f}B")
