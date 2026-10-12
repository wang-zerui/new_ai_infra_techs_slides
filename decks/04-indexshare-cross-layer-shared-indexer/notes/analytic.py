"""Analytic sanity check of the published GLM-5.2 '2.9x lower per-token FLOPs at 1M' (notes only, not on a slide).

Per generated token at position L (FLOPs = 2 x multiply-adds):
  indexer : n_idx layers x 32 heads x 128 dims x L                         [GLM-5 Table 10; HF config]
  core MLA: 78 layers x 64 heads x (576 QK + 512 PV) x 2048 selected keys  [HF config: 64 heads, kv_lora 512, rope 64]
  weights : 2 x 40B active parameters (standard 2-FLOPs-per-parameter estimate) [GLM-5 Table 10]
GLM-5.1: every one of 78 layers runs an indexer; GLM-5.2: 21 of 78 (HF config indexer_types).
Also: greedy-search forward-pass counts N(N-1)/2 [IndexCache Sec. 3.1.2].
"""
def per_token_flops(L, n_idx, layers=78):
    idx = 2 * n_idx * 32 * 128 * L
    core = 2 * layers * 64 * (576 + 512) * 2048
    w = 2 * 40e9
    return idx, core, w

for L in (32 * 1024, 128 * 1024, 1024 * 1024):
    a = sum(per_token_flops(L, 78)); b = sum(per_token_flops(L, 21))
    i51 = per_token_flops(L, 78)[0]
    print(f"L={L:>8}: GLM-5.1 {a/1e12:.3f} TFLOP (indexer share {i51/a:.0%}), GLM-5.2 {b/1e12:.3f} TFLOP, ratio {a/b:.2f}x")
for n in (47, 78):
    print(f"greedy full search, N={n}: {n*(n-1)//2} forward passes")
print(f"MTP: IndexShare+KVShare alone 5.10/4.56 = {5.10/4.56:.3f}; full recipe 5.47/4.56 = {5.47/4.56:.3f}")
