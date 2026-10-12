# Sources — deck 10: FP8 MQA-logits / lightning-indexer scoring kernel (DeepGEMM)

All DeepGEMM code was read at commit 057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7 (main, 2026-09-30) via `gh api`.
Local copies: `$SCRATCH/research/10/src/`.

## Sources read

| Tag | Source | Where |
|---|---|---|
| [RD] | DeepGEMM README (cached `gh-deepgemm-README.md`) | "News" (2025.09.28 scoring kernels for V3.2, PR #200; 2026.04.16 FP4 Indexer, PR #304; 2026.09.10 Sparse Indexer, PR #432); "V3.2 MQA kernels for the indexer" (non-paged for prefill, paged for decode, inputs/shapes, compressed `[seq_len, max_seqlen_k]` output, reference loop) |
| [K90] | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh` | WGMMA with KV tile as A (64 keys per math warpgroup) and Q tile (BLOCK_Q × heads) as B; TMA warp + math warpgroups (register reconfig 32 / 112); per-Q-block KV span = union of [ks, ke) rounded down to 4; epilogue: `fmaxf(acc, 0) * weight`, in-thread sums, `* scale_kv`, 2 `__shfl_xor_sync`, store only inside [ks, ke) at column `kv − ks`; persistent loop `block_q_idx += kNumSMs` |
| [H90] | `csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp` | 128 specialized + 512 math threads; 3 Q stages, 3 KV stages; grid = #SMs; head_dim 32/64/128; paged: split_kv 256, math warpgroups = split_kv / 64, next_n atom 2 if next_n ≥ 2; metadata kernel: one warp, prefix sum of ceil(context_len / 256) |
| [API] | `csrc/apis/attention.hpp` | `fp8_fp4_mqa_logits`, `fp8_fp4_paged_mqa_logits`, `get_paged_mqa_logits_metadata`, sparse variants; SM90: E4M3 + FP32 per-token KV scale, FP32 weights, FP32 logits, heads 32 or 64; SM100: MXFP8/MXFP4 + packed UE8M0, BF16 weights/logits; block_kv 256 (SM90) / 384 (SM100); logits row stride 1024-byte aligned; paged: SM90 page 64, SM100 page 32/64/128 varlen; schedule meta [#SMs + 1, 2] |
| [H100K] | `csrc/jit_kernels/impls/sm100_mqa_logits.hpp` | `kMQALogitsSplitKV = 384`; "the Q tile always covers 128 (token, head) rows", `block_q = 128 / num_heads`; 3 TMEM stages; math threads = 384/128 × 128; FP4: 2 Q / 8 KV stages; contiguous scheduler cost model (split cost 2, Q-block start cost 3) |
| [K100] | `deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh` + `ptx/utils.cuh` | UMMA_M = 128 keys per math warpgroup, accumulators in TMEM; epilogue `cvt.rn.relu.bf16x2.f32` then `__hfma2` with BF16 head weights, `__hadd2` reduction; KV-ring reuse across tasks sharing splits |
| [KP90] | `impls/sm90_fp8_paged_mqa_logits.cuh`, `scheduler/sm90_paged_mqa_logits.cuh` | 4 KV groups = (1 TMA warp + 1 math warpgroup) each, one 64-token page per step; block-table lookups by TMA-warp lanes; metadata: total splits / #SMs per SM, binary search for each SM's (request, split) start |
| [T] | `tests/test_attention.py` | reference loops (`ref_mqa_logits`, `ref_paged_mqa_logits`); FP8 cache layout `head_dim + 4` bytes per token (values then FP32 scale); MXFP4 `head_dim/2 + 4`; benchmark prints TFLOPS = 2 · valid pairs · H · D / t and "relu/cyc/SM" |
| [PR] | DeepGEMM PRs #200 (merged 2025-09-29, "Add DeepSeek V3.2 indexer logit kernels for H800 (SM90)", +1,602 lines), #304 (2026-04-17: "FP4 Indexer (MQA logits) with larger MTP support"), #432 (2026-09-10: "Sparse Indexer: hierarchical sparse indexing for V4.1, with throughput close to the Lightning Indexer"; `use_deterministic_algorithms()` added, default False; previously all kernels deterministic and batch-invariant), #462 (2026-09-30: compressed logits, `clean_logits` removed, SM100 MXFP8/MXFP4 + BF16 logits, legacy `fp8_mqa_logits` / `fp8_paged_mqa_logits` removed) | PR bodies via `gh api` |
| [PR452] | DeepGEMM PR #452 (community, open, 2026-09-18), "perf: Use full-rate FADD relu in SM90 FP8 MQA logits epilogue" | H200 (132 SMs) table: before → after TFLOPS for 4 production shapes; FMNMX 64 vs FADD 128 ops/clk/SM claim; "+3.7% avg over 192 cases" |
| [IC] | IndexCache, arXiv 2603.12201, §1 figure | indexer share of prefill time 81% at 200K (27% at 10K), 30B DSA model, H100, SGLang |
| [V32] | DeepSeek-V3.2, arXiv 2512.02556, §2.1 Eq. 1 | index score formula; "small number of heads and can be implemented in FP8" |
| [HF] | DeepSeek-V3.2-Exp `config.json`, `inference/model.py`, `inference/kernel.py` | H = 64, D = 128, k = 2,048; `fp8_index` TileLang reference: ReLU, × per-head weight·q-scale, Σ heads, × k-scale |
| [VLLM] | vLLM blog 2025-09-29 | `deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)` usage; ks/ke for batched prefill; "the logits tensor is materialized before running a row-wise topk" |
| [NV] | NVIDIA H200 product page (fetched 2026-10-13) | H200 SXM FP8 Tensor Core 3,958 TFLOPS "with sparsity"; HBM bandwidth 4.8 TB/s |
| [GLM5] | GLM-5, arXiv 2602.15763v2, "DSA RL insights" | non-deterministic top-k operators degraded RL |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 64 heads × 128 dims, k = 2,048 | V3.2 indexer shape | [HF]; [V32] |
| 2025-09-28 / 2025-09-29 | README news / PR #200 merge (V3.2-Exp launched 2025-09-29) | [RD]; [PR] |
| 81% (27% at 10K) | indexer share of prefill time at 200K, 30B DSA model | [IC] |
| 256 B vs 4 B per (query, key); **64×** | unfused per-head FP32 scores (64 × 4 B) vs fused FP32 logit | analytic, `notes/analytic.py` |
| 128 GiB vs 2 GiB | one 4,096-query chunk × 131,072 keys: unfused FP32 scores vs FP32 logits | analytic, same |
| 128 rows; 2 tokens (H = 64), 4 tokens (H = 32) | Q tile = 128 (token, head) rows, block_q = 128 / H | [H100K]; [API] (`get_mqa_logits_block_q` used for both arches) |
| 256 keys = 4 × 64 (SM90); 384 = 3 × 128 (SM100) | KV tile / split per CTA step | [H90]; [K90] static assert `BLOCK_KV == kNumMathThreads / 2`; [H100K] |
| 3 stages (Q and KV), 128 + 512 threads | SM90 pipeline depth and threads | [H90] |
| 32 / 112 registers | TMA vs math warpgroup register reconfiguration (SM90) | [K90] |
| 2 shuffles | `__shfl_xor_sync` offsets 1, 2 for the cross-lane head sum | [K90] |
| 64-token pages, 256-key splits = 4 pages | SM90 paged decode | [API]; [H90]; [KP90] |
| next_n ≤ 2 | SM90 paged decode tokens per request per call | [H90] assert |
| 132 B per key (FP8: 128 + 4); 68 B (MXFP4: 64 + 4); 1.94× | indexer key bytes | [T] |
| 124 vs 412 FLOP/B | decode arithmetic intensity (2 · 64 · 128 / 132) vs H200 ridge (1,979 TFLOPS / 4.8 TB/s) | analytic, `notes/analytic.py`; [NV] |
| 17.3 MB / 3.6 µs (FP8), 8.9 MB / 1.9 µs (FP4) | indexer keys read per request per layer at 128K and the 4.8 TB/s lower bound — notes only | analytic |
| 1,018 / 1,042 / 941 / 980 TFLOPS → 1,093 / 1,126 / 989 / 1,027 | PR #452 before/after on H200: (H32, D128, SQ512, SKV140K, FP32), (H64, SQ512, SKV130,560, BF16, CP1), (H32, SQ512, SKV130,560, BF16, CP0), (H64, SQ2048, SKV8192, BF16, CP1) | [PR452] |
| +4.8% … +8.1% | gains from the ReLU instruction swap | [PR452] (ratios derived, match the PR's own column) |
| 1,979 TFLOPS | H200 dense FP8 peak = 3,958 "with sparsity" / 2 | derived from [NV] |
| 48–53% (50–57% after) | before (after) / 1,979 | derived |
| 2026-04 FP4 indexer; 2026-09 Sparse Indexer; 2026-09-30 API change | evolution | [PR] |

## Mechanism facts (no numbers) used on slides

- Reference semantics: `out_ij = Σ_h relu(q[i,h,:] · kv_j) · w[i,h]` with `kv_j = kv_fp8[j] · scale[j]` over j ∈ [ks_i, ke_i) ([RD]).
- The per-key scale multiplies the head sum once because scales are positive: ReLU(s·x) = s·ReLU(x) for s > 0 (the kernel
  multiplies `(sum[0] + sum[1]) * scale_kv`; [K90]); the TileLang reference does the same ([HF] kernel.py).
- Q-side per-head scale and the 1/√d softmax scale are folded into the head weights by the caller ([HF] model.py
  `weights = weights_proj(x) * n_heads**-0.5; weights = weights * q_scale * softmax_scale`).
- Split-KV decode writes disjoint logit columns, so no cross-CTA combine is needed (follows from [KP90]: each split stores
  its own `kv_offset` columns; there is no reduction buffer).
- Logits are materialized before top-k ([VLLM]); top-k is a separate kernel (e.g. TileLang / torch.topk; [GLM5]).

## Discrepancies (Notion row vs primary sources)

1. **"E4M3 dot products"**: true on SM90. On SM100 the current API requires MXFP8 or MXFP4 inputs with packed UE8M0 block
   scales (block of 32), BF16 weights and BF16 logits ([API], PR #462); plain per-token-scaled E4M3 is SM90-only now.
2. **"Separate non-paged (prefill) and paged (decode) versions"**: correct, plus a third family since 2026-09: the Sparse
   Indexer (`fp8_fp4_sparse_mqa_logits`, `..._paged_sparse_...`) that scores only selected KV blocks for V4.1 (PR #432).
3. **Function names**: the Notion row and the vLLM blog use `fp8_mqa_logits` / `fp8_paged_mqa_logits`; since PR #462
   (2026-09-30) these are removed in favour of `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits`.
4. **Impact has no number**: DeepSeek publishes no benchmark table for these kernels. The only measured throughput used
   on slides is a community PR (#452, unmerged) on H200; slides label it as such.
