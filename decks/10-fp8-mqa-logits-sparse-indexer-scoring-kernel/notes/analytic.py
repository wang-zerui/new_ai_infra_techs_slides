"""Analytic numbers for deck 10 (all labelled 'analytic' or 'derived' on the slides).

Shapes: DeepSeek-V3.2 indexer, H = 64 heads, D = 128 dims, FP8 E4M3 keys with one FP32 scale per key
[HF config; DeepGEMM tests: kv bytes per token = head_dim + 4 (FP8) or head_dim/2 + 4 (MXFP4)].
"""
H, D = 64, 128
print(f"MACs per (query, key) pair: {H*D:,}; FLOPs {2*H*D:,}")
# unfused: per-head FP32 scores [n_q, H, n_kv] vs fused FP32 logits [n_q, n_kv]
print(f"unfused bytes per (query,key): {4*H} B vs fused {4} B -> {H}x")
nq, nkv = 4096, 128 * 1024
print(f"one 4K-query chunk vs 128K keys: unfused {4*H*nq*nkv/2**30:.0f} GiB vs fused {4*nq*nkv/2**30:.0f} GiB (FP32)")
# Q tile = 128 (token, head) rows -> block_q = 128 / H
for h in (32, 64):
    print(f"H={h}: block_q = {128//h} tokens per Q tile")
# decode bytes per key and per request-layer at 128K
fp8, fp4 = D + 4, D // 2 + 4
L = 128 * 1024
bw = 4.8e12  # H200 HBM3e, NVIDIA spec
print(f"key bytes: FP8 {fp8} B, MXFP4 {fp4} B -> {fp8/fp4:.2f}x fewer")
print(f"decode read per request per layer at 128K: FP8 {fp8*L/1e6:.1f} MB, FP4 {fp4*L/1e6:.1f} MB;"
      f" at 4.8 TB/s: {fp8*L/bw*1e6:.1f} us vs {fp4*L/bw*1e6:.1f} us")
print(f"decode intensity (next_n=1): {2*H*D/fp8:.0f} FLOP/B; next_n=2: {2*2*H*D/fp8:.0f} FLOP/B; "
      f"H200 ridge {1979/4.8:.0f} FLOP/B (1,979 TFLOPS dense FP8 / 4.8 TB/s)")
# H200 FP8 dense peak = 3,958 (with sparsity) / 2
before = [1018, 1042, 941, 980]; after = [1093, 1126, 989, 1027]
print("PR #452 before/after as % of 1,979 TFLOPS:", [f"{b/1979:.0%}/{a/1979:.0%}" for b, a in zip(before, after)])
print("gains:", [f"{a/b-1:+.1%}" for b, a in zip(before, after)])
# FP32 logits for a 4K x 128K chunk
print(f"FP32 logits 4K x 128K = {4*nq*nkv/2**30:.0f} GiB; BF16 = {2*nq*nkv/2**30:.0f} GiB")
