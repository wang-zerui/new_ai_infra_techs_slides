"""Analytic KV-cache and attention-work model for DeepSeek-V4's hybrid CSA/HCA stack.

Inputs (see notes/sources.md):
  KV entry: c = 512 dims, RoPE on the last 64 dims; storage BF16 for the 64 RoPE dims, FP8 for the other 448
      -> 64*2 + 448*1 = 576 bytes per entry  [V4 §2.3.3, §2.3.4]
  CSA: m = 4; indexer key c_I = 128 dims, cached in FP4 after QAT -> 64 bytes per indexer entry [V4 §4.2.1, §5.2.1]
  HCA: m' = 128, no indexer [V4 §2.3.2, §4.2.1]
  Sliding window: n_win = 128 raw entries per layer, fixed per request [V4 §2.3.3, §4.2.1]
  Layer mix from config.json compress_ratios: Pro 30 CSA + 31 HCA (61 layers); Flash 21 CSA + 20 HCA + 2 SWA (43)
  Baseline named by the paper: BF16 GQA8, head dim 128 -> 8 * 128 * 2 (K,V) * 2 B = 4096 B / token / layer [V4 §2.3.4]
Excluded: per-block quantization scale factors, the uncompressed tail buffer (< m' tokens per layer), MTP layer.
"""
L = 2 ** 20
ENTRY = 64 * 2 + 448 * 1          # 576 B
IDX = 128 // 2                    # 64 B (FP4)
CSA_PER_TOK = ENTRY / 4 + IDX / 4  # 160 B
HCA_PER_TOK = ENTRY / 128          # 4.5 B
WIN = 128 * ENTRY                  # 73,728 B per layer per request (fixed)
GQA8 = 8 * 128 * 2 * 2             # 4096 B / token / layer

def stack_bytes(n_csa, n_hca, n_swa, L=L):
    layers = n_csa + n_hca + n_swa
    return n_csa * CSA_PER_TOK * L + n_hca * HCA_PER_TOK * L + layers * WIN

if __name__ == "__main__":
    print(f"entry {ENTRY} B; CSA {CSA_PER_TOK} B/token/layer; HCA {HCA_PER_TOK} B/token/layer; GQA8 {GQA8}")
    for name, (c, h, s) in {"V4-Pro": (30, 31, 0), "V4-Flash": (21, 20, 2)}.items():
        layers = c + h + s
        v4 = stack_bytes(c, h, s)
        base = GQA8 * layers * L
        allcsa = stack_bytes(layers, 0, 0)
        allhca = stack_bytes(0, layers, 0)
        print(f"{name}: hybrid {v4/1e9:.2f} GB ({v4/L:,.0f} B/token incl. window) | GQA8 {base/1e9:.1f} GB "
              f"({GQA8*layers:,} B/token) | ratio {100*v4/base:.2f}% | all-CSA {allcsa/1e9:.2f} GB "
              f"({allcsa/L:,.0f} B/tok) | all-HCA {allhca/1e9:.2f} GB ({allhca/L:,.0f} B/tok)")
        csa_share = c * CSA_PER_TOK * L / v4
        print(f"   CSA layers' share of hybrid cache: {100*csa_share:.1f}%; window total {layers*WIN/1e6:.1f} MB")
    print("HCA entries at 1M:", L // 128, "; CSA entries:", L // 4)
    # per-query core-attention multiply-adds at 1M (Pro: 128 heads, QK 512 + PV 512 per entry)
    hca = 128 * 1024 * (L // 128 + 128); csa = 64 * 128 * (L // 4) + 128 * 1024 * (1024 + 128)
    dsa = 64 * 128 * L + 128 * (576 + 512) * 2048
    print(f"MAC/query/layer at 1M: HCA {hca/1e9:.2f} G, CSA {csa/1e9:.2f} G, V3.2 DSA {dsa/1e9:.2f} G")
