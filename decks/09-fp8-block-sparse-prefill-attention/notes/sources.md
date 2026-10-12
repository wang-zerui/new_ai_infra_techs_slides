# Sources for deck 09 (FP8 block-sparse prefill attention)

Downloads: `$SCRATCH/research/09/` (PR texts `pr45.md`, `pr50.md`, `pr66.md`; PR benchmark images in `dl/`;
Stem paper `dl/stem-2603.06274.pdf` + `stem.txt`); repo sources shared in `$SCRATCH/research/26/repo/`.

## Sources read

| Tag | Source | What was used |
|---|---|---|
| [R] | HPC-Ops README (main) | "Updates / Sparse Attention" write-up; Performance row "Sparse Attention FP8 ... Up to 3.16x" with baselines MIT-BSA BF16, FlashPrefill-BSA BF16, HPC-Dense FP8, FA3-Dense FP8; catalog row (`hpc/attention.py`, `hpc/stem.py`) |
| [PR45] | GitHub PR #45 "add block-sparse fp8 prefill attention (dim=128, two quant schemes)", merged 2026-06-04, and its benchmark figure "HPC-BSA Kernel Latency vs Sparsity" | two quant schemes; figure config batch=4, seq=65536, Hq=4, Hkv=1, D=128, H20; printed labels = HPC-BSA FP8 latency as % of dense: 101, 91, 81, 71, 61, 51, 41, 32, 22, 11 % at block density 100…10 %; dense baselines HPC-Dense FP8 and FA3-Dense FP8 flat |
| [PR50] | GitHub PR #50 "Add stem scoring ops for pre-prefill", and its two figures | Stem = ICML 2026 paper arXiv 2603.06274; ops prep / OAM GEMM / TPD; figure "HPC-Stem Speedup vs Triton Baseline": 7.2×, 8.0×, 9.6×, 16.3×, 27.0×, 44.0× at 8K, 16K, 32K, 64K, 128K, 256K; figure "HPC-Stem Mask Accuracy vs FP32 Ground Truth": HPC-Stem at or above the 99.9% PASS threshold at every length (14 to 1M) |
| [PR66] | GitHub PR #66 "SM90 fp8 double-O accumulator and stem causal-offset fixes" | FP8 PV wgmma now accumulates into a fresh zeroed FP32 tile, then adds to O (tensor-core 22-bit accumulator precision); token-accurate causal block offset in Stem |
| [S] | Niu et al., "Stem: Rethinking Causal Information Flow in Sparse Attention", ICML 2026, arXiv 2603.06274v2 | Fig. 1 (H20 latency, Llama-3.1-8B-Instruct, batch 1, BF16, FlashAttention-2 dense): dense 24 / 95 / 384 / 1540 ms at 16K / 32K / 64K / 128K; Stem 10/16, 30/46, 90/121, 330/420 ms (attention kernel / total); §3.4 "Empirical Latency": 1540 → 420 ms = 3.7× at 128K, metric overhead ≈ 90 ms; §3 implementation: block size B = 128, MIT Block Sparse Attention library (BF16), 4 initial + 4 local blocks; Table 2 LongBench: Qwen3-8B dense 32.01 vs Stem 31.64 at 25% budget, Llama-3.1-8B dense 42.02 vs Stem 41.48 at 31% budget |
| [P] | `hpc/attention.py` `attention_with_kvcache_blocksparse_prefill_fp8` docstring | `block_mask` uint8 [batch, heads_q, Q tiles, KV tiles], True = compute; diagonal tile should be kept (all-skipped Q tile → softmax NaN); no deadlock regardless of mask; paged FP8 K/V; Q per-token per-head scale; K per-tensor or per-token-group per-head; V per-tensor or per-head; output BF16 |
| [ST] | `hpc/stem.py` | Stem pipeline `stem_paged_kv` = prep_paged_kv (K_flat group-summed, V_bias) → prep_varlen_q → oam_gemm (block logits + V bias, causal) → tpd (top-k mask); defaults block 128, stride 16, λ = 0.3, initial 4 blocks, window 4 blocks, k-rate 0.2 (+30) for 56–159 prompt blocks, 0.1 (+30) for ≥ 160 |
| [E] | `src/attention/entry.cc` (block-sparse entry) | FP8 E4M3 Q/K/V required; dim_qk = dim_v = 128; `kTileM = kTileN = 128`; page size must divide 128; quant_type 0/1 |
| [K] | `src/attention/prefill/sm90/warp_spec_with_kvcache_blocksparse_fp8_dim128.cu`, `kernels.cuh` | grid = #SMs (persistent), 384 threads (producer warp + 2 consumer warpgroups); wgmma `SM90_64x64x32_F32E4M3E4M3_SS_TN` (Q·Kᵀ) and `SM90_64x128x32_F32E4M3E4M3_RS_TN` (P·V); producer warp builds an "active tile list" in SMEM by warp-parallel stream compaction (`__ballot_sync` + `__popc`) of the mask row while Q's TMA load is in flight, then TMA-loads only those KV tiles page by page; consumers iterate the same list; causal masking only for tiles at/after `num_tile_full`; scores scaled by q-scale × k-scale; P × 256 (`kFp8PrefillPScale`) before FP8 cast, output × vscale / 256 in the epilogue; `get_next_tile` walks Q tiles from the last (longest causal range) first in a zig-zag order across CTAs |
| [T] | `tests/test_attention_blocksparse_*_fp8.py` | `BSA_BLOCK = 128`; random causal masks with a skip ratio |

## Numbers on slides

| Slide | Number | Source location |
|---|---|---|
| 2 | Stem keeps 25–31% of causal blocks on LongBench | [S] Table 2 "BUD." column |
| 3 | **1540 ms** dense attention at 128K; 24 ms at 16K (64× for 8× length, derived 1540/24 = 64.2) | [S] Fig. 1 and §3.4 |
| 3 (notes) | Stem end to end (paper, BF16 MIT kernel): 1540 → 420 ms at 128K | [S] §3.4 |
| 4 | dense FP8 computes every causal tile; BF16 block-sparse (MIT Block-Sparse-Attention, used by the Stem paper; FlashPrefill-BSA) skips tiles in BF16 | [R] baselines; [S] §3 implementation |
| 5 | LongBench 41.48 vs 42.02 (Llama-3.1-8B, 31% budget); notes: 31.64 vs 32.01 (Qwen3-8B, 25%), i.e. −0.54 and −0.37 points (derived) | [S] Table 2 |
| 6 | mask uint8 [batch, Q heads, Q tiles, KV tiles]; 128 × 128 tiles; 1 persistent CTA per SM, 384 threads; heaviest Q tiles first | [P]; [E]; [K] launch code + `get_next_tile` |
| 7 | Stem: 128-token blocks, stride 16, 4 initial + 4 window blocks, length-dependent top-k budget | [ST]; [S] |
| 8 | active-tile list via ballot/popc while Q's TMA is in flight; tail tile appended if outside the mask | [K] "Build active tile list in SMEM" |
| 9 | FP8 wgmma 64×64×32 (QK); P × 256 before FP8 cast, ÷256 with v-scale in the epilogue; fresh FP32 accumulator (PR #66, July 2026) | [K]; [PR66] |
| 10 | relative latency 101, 91, 81, 71, 61, 51, 41, 32, 22, 11 % at density 100…10 % (batch 4, 64K, Hq 4, Hkv 1, D 128, H20) | [PR45] figure labels |
| 11 | **up to 3.16×** vs MIT-BSA BF16, FlashPrefill-BSA BF16, HPC-Dense FP8, FA3-Dense FP8 | [R] Performance table |
| 12 | Stem mask generation 7.2, 8.0, 9.6, 16.3, 27.0, 44.0× faster than Triton (8K–256K); ≥ 99.9% mask match vs FP32 (14 to 1M tokens) | [PR50] figures |
| 12 (notes) | metric overhead ≈ 90 ms at 128K in the paper's setup | [S] §3.4 |
| 13 | dim 128 only; page size divides 128; diagonal tile must be kept; per-Q-head mask | [E]; [P] |
| 14 | summary repeats 1540 ms, 11%, 3.16×, 7.2–44× | as above |

Derived numbers: 64× = 1540 / 24 (slide 3); "≈ 9× faster than dense at 10% density" is NOT used (would be read
off a plot); slides use only the printed percentage labels.

## Discrepancies (Notion row vs primary sources)

1. **"per-tile FP8 scaling"** (Notion, echoing the README): in the code the scales are per-token·per-head for Q,
   per-tensor or per-token-group·per-head for K, per-tensor or per-head for V, plus a fixed ×256 scale on P before
   its FP8 cast. There is no separate scale computed per KV tile. Slides describe the actual scale granularity.
2. **"Up to 3.16x over MIT-BSA / FlashPrefill-BSA (BF16) and a dense FP8 baseline"**: README lists *two* dense FP8
   baselines (HPC-Dense FP8 and FA3-Dense FP8) and does not publish the per-case table; the only public
   per-sparsity data (PR #45 plot) compares against dense FP8. Slides quote "up to 3.16×" with the README's full
   baseline list and use the PR plot for the density curve.
3. **"the companion kernel that makes DSA-style sparsity deployable"**: not supported by the sources. The kernel is
   *block* sparse (128 × 128 tiles from a block mask); DeepSeek Sparse Attention selects individual tokens
   (top-2048). The repo pairs the kernel with **Stem** (Tencent, ICML 2026) as the mask generator; the Stem paper
   applies Stem on top of DSA with DSA's own kernel. Slides present Stem as the companion, not DSA.
4. README calls long-context prefill attention "memory-bandwidth bound"; the slides avoid that characterization
   (prefill attention is usually compute-bound) and only state measured latencies.
