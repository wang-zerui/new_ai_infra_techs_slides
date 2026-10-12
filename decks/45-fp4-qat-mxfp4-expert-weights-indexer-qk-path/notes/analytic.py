"""Analytic numbers for deck 45 (FP4 QAT). Inputs in notes/sources.md.

Routed-expert parameters: layers x experts x 3 (SwiGLU gate, up, down) x hidden x expert_inter
  V4-Pro: 61 x 384 x 3 x 7168 x 3072 ; V4-Flash: 43 x 256 x 3 x 4096 x 2048   [V4 §4.2.1]
Storage per weight: BF16 2 B; FP8 1 B (+1 scale per 128x128 block, negligible);
  MXFP4 = 4-bit E2M1 element + one 8-bit E8M0 scale per 32 elements = 0.5 + 1/32 = 0.53125 B  [MX paper Table 1]
Checkpoint sizes: sum of *.safetensors in the Hugging Face repos (API tree listing, 2026-10-13).
Indexer vs core attention per CSA layer at 2^20 tokens: see deck 01 notes/analytic.py.
"""
PRO = 61 * 384 * 3 * 7168 * 3072
FLASH = 43 * 256 * 3 * 4096 * 2048
MXFP4 = 0.5 + 1 / 32
CKPT_GB = {"DeepSeek-V3.2": (689.5, 671e9), "DeepSeek-V4-Flash": (159.6, 284e9), "DeepSeek-V4-Pro": (864.7, 1.6e12)}

if __name__ == "__main__":
    print(f"V4-Pro routed-expert params {PRO/1e12:.3f} T = {100*PRO/1.6e12:.1f}% of 1.6T")
    print(f"V4-Flash routed-expert params {FLASH/1e9:.1f} B = {100*FLASH/284e9:.1f}% of 284B")
    for fmt, b in [("BF16", 2), ("FP8", 1), ("MXFP4", MXFP4)]:
        print(f"  Pro experts in {fmt}: {PRO*b/1e12:.2f} TB ; Flash: {FLASH*b/1e9:.0f} GB")
    print(f"FP8 / MXFP4 bytes per weight: {1/MXFP4:.2f}x")
    for k, (gb, n) in CKPT_GB.items():
        print(f"{k}: {gb} GB / {n/1e9:.0f} B params = {gb*1e9/n:.2f} bytes per parameter")
    print(f"V4-Pro vs V3.2: params {1.6e12/671e9:.2f}x, checkpoint bytes {864.7/689.5:.2f}x")
    idx, core = 64 * 128 * (2 ** 20 // 4), 128 * 1024 * (1024 + 128)
    print(f"CSA layer (Pro) at 2^20: indexer {idx/1e9:.2f} G MAC vs core {core/1e9:.3f} G -> {idx/core:.1f}x")
