# Sources — deck 40: Fused Sampler (decode post-processing fusion)

Repository state read: Tencent/hpc-ops `main` at HEAD 2a2e265 (2026-09-07). Local copies in
`$SCRATCH/research/40/src/` and `$SCRATCH/research/36/hpcops_prs.md` (PR #51 text).

## Sources read

| Tag | Source | Where |
|---|---|---|
| [RM] | HPC-Ops README (cached `research/cache/gh-hpc-ops-README.md`), https://github.com/Tencent/hpc-ops | "Updates / Sampler" write-up; Operator Catalog row "Sampling"; Performance row "Fused sampler" |
| [PR51] | hpc-ops PR #51 "add fused sampler ops (topk/topp/temperature + rejection sampling)", merged 2026-06-09, https://github.com/Tencent/hpc-ops/pull/51 | Summary + Benchmark line |
| [API] | `hpc/sampler.py` (`SoftmaxPolicy`, `fused_sampler` docstring and fast-path dispatch) | full file |
| [K] | `src/sampler/fused_sampler.cu` | header comment l.3–13; K1 `fused_scan_topk_kernel` l.161–289; K2 `stage2_kernel` l.296–580; `pick_n_blocks_per_row` l.609–627; launcher l.632–676; vocab dispatch l.757–766 |
| [KT] | `src/sampler/fused_sampler_temperature.cu` | header l.1–11; kernel l.84+; launcher constants l.336–349; `pick_n_blocks_per_row` l.435–445; vocab dispatch l.539–545; concurrency note |
| [H] | `src/sampler/sampler.h` | comments on the top-`max_topk` bound and concurrency |
| [RNG] | `src/sampler/sampler_rng.cuh` | Philox, per-launch offset, `gumbel_noise_from_uniform` |
| [TST] | `tests/test_sampler.py` | bit-exact tests (`torch.equal`) vs pure-torch reference with injected Gumbel noise: `test_only_logits`, `test_topk_topp_softmax_matrix` (max_topk 32/64, all softmax policies), `test_repetition_penalty_and_writeback`, `test_bf16_logits`, `test_padded_logits_stride`, `test_fused_sampler_temperature_only_golden`; `test_error_unsupported_vocab` |
| [BEN] | `benchmark/sampler/README.md` + `benchmark_sampler.py` | default config (V=120832, BF16 logits, batches 1,8,16,32,64,128,256,512); constants l.36–40 (rp 1.1, T 1.05, top-k 20, top-p 0.9); baselines l.115–180; HPC call l.181–193 (AFTER_TOPK, max_topk 32) |
| [FI] | Ye et al., "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving", arXiv 2501.01005 (Notion body link) | read: no section on sampling kernels; not used on slides |
| [LM] | LMSYS blog "HPC-Ops × SGLang" (2026-08-07) | checked: no sampler content (0 occurrences of "sampl") |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| V = 120,832 | vocabulary size of the benchmark and the only vocab compiled into both kernels | [RM] Performance row; [BEN]; [K] l.757–766; [KT] l.539–545 |
| batch 1–512 | benchmark range | [RM]; [BEN] default batches; [PR51] |
| **~20** (21 counted) vocabulary-wide tensor ops per step | vLLM/PyTorch "full" baseline in [BEN]: penalty+temperature (l.138–152): `to(float32)`, `zeros` ×2, `scatter_` ×2, `.contiguous()` ×2 (a copy for B>1), `apply_repetition_penalties`, `div_(T)` = 9; top-k (l.158–160): `sort`, `<` compare, `masked_fill` = 3; top-p (l.161–166): `softmax`, `cumsum`, `<=` compare, `masked_fill`, `scatter_` = 5; sample (l.167–169): `softmax`, `exponential_`, `div_`, `argmax` = 4. Total 21 (19 at B=1 where the two slices are already contiguous). Plus two host→device copies of token histories. Slide says "~20", "counted from the benchmark baseline" | **derived** (count) from [BEN] |
| stage groups in fig_baseline: penalty 8 (incl. the FP32 cast), temperature 1, top-k 3, top-p 5, sample 4 | same count, grouped | **derived** from [BEN] |
| 7 stages | repetition penalty, temperature, top-k, top-p, softmax, random sampling, penalty-mask update | [RM] Sampler write-up |
| 2 kernels (full path); 1 kernel (temperature-only) | K1 `fused_scan_topk_kernel` + K2 `stage2_kernel`; temperature path = one kernel whose last block reduces | [RM]; [K] header; [KT] header |
| max_topk 32 or 64 | compile-time top-K bound; top-k must be ≤ max_topk | [API]; [H] |
| K1: 1024 threads/block, 4 logits/thread; keepers = K/4 threads (8 for K = 32) | thread roles | [K] l.35–36, 177–180 |
| N = clamp(⌊SMs / B⌋, 8, Nmax), Nmax = min(1024/K, 32) → 32 (K = 32), 16 (K = 64) | blocks per batch row in K1 | [K] l.42–46, 609–627 |
| K2 block = Nmax × K threads (1,024 for K = 32 and for K = 64) | candidate merge | [K] l.307–308, 663 |
| 1 bit per vocab token → 15,104 bytes per slot | penalty bitmask row = ceil(V/8) | **derived**: 120,832 / 8 = 15,104; layout from [API] docstring `[MAX_BS, ceil(V/8)] uint8` |
| benchmark config: repetition penalty 1.1, T = 1.05, top-k 20, top-p 0.9, AFTER_TOPK, max_topk 32 | settings behind the speedups | [BEN] l.36–40, 181–193 |
| **4.6–8.5×** vs vLLM; **2.1–5.2×** vs FlashInfer; batch 1–512 | sampler speedup ranges | [PR51] Benchmark line |
| up to 8.5× | README headline (BF16 logits, V 120,832, batch 1–512; baselines vLLM-style PyTorch, FlashInfer) | [RM] Performance table |
| example in fig_k2: top-k = 20 / top-p = 0.9 | the benchmark's settings used as running example; bar heights schematic | [BEN] |

## Mechanism facts (no numbers) used on slides

- K1 per chunk: loaders load 4 logits (vector load), apply repetition penalty (bit test in the slot's mask row;
  logit × 1/rp if > 0 else × rp) and × 1/T in registers; for BEFORE_TOPK update an online max/sum; then the whole
  block runs `cub::BlockRadixSort::SortDescending`, which places the current top-K into the keeper threads
  ([K] l.226–263, 91–113). After the loop: block max/sum (BEFORE_TOPK) and the K candidates go to small
  scratch buffers [B, Nmax, K] ([K] l.265–288).
- K2: reduce partial max/sum to a global max and 1/sum (BEFORE_TOPK: full-vocab probabilities without a second
  vocabulary pass), merge N × K candidates with one block radix sort, AFTER_TOPK local softmax over the top-k,
  top-p by warp inclusive prefix scan (keep the first token and every token whose preceding mass < p),
  Gumbel-max: key = log p + g (or logit + g with policy NONE), warp argmax with smallest-token tie-break, write
  the token and `atomicOr` the sampled token's bit into the penalty mask ([K] l.329–579).
- Noise: Philox (cuRAND) Gumbel(0) noise g = −log(−log U), fresh per launch via a host offset counter; or an
  injected `gumbel_noise` tensor for bit-exact reproducibility ([RNG]; [API]).
- Temperature-only fast path: token = argmax_v(logit/T + Gumbel); grid (N, B), 256 threads, each block reduces its
  slice to one (score, token) partial; the last block to arrive (atomic counter) reduces the N partials ([KT]).
  Supports a per-row draft-token mask for speculative decoding ([API]; [H]).
- Softmax policies: NONE (sample on logits), BEFORE_TOPK (softmax over the full vocab, computed from the online
  statistics), AFTER_TOPK (softmax over the surviving top-k); top-p requires a softmax ([API]).
- Bound: the kernel always reduces the vocabulary to the top-`max_topk` candidates before sampling; `topk = 0`
  does not mean full-vocabulary sampling ([API]; [H]).
- Penalty: repetition penalty only (multiplicative, presence bit); the bitmask records presence, not counts
  ([K] `apply_rp_temp`).
- Concurrency: the temperature path's per-device workspace is not safe for concurrent streams ([API]; [KT]).
- Baselines: the vLLM/PyTorch path rebuilds prompt/output presence masks every call from token-ID histories
  (H2D copy + scatter) and calls vLLM's `apply_repetition_penalties`; the FlashInfer path runs the same
  penalty/temperature code, then `flashinfer.sampling.top_k_top_p_sampling_from_logits` ([BEN] l.134–180).

## Discrepancies

1. **Notion "Up to 8.5x over vLLM-style PyTorch / FlashInfer"** — 8.5× is the maximum over the vLLM-style
   PyTorch path; over FlashInfer the PR reports 2.1–5.2× ([PR51]). Slides give both ranges.
2. **Notion "largest win in small-batch RL rollout"** — no source mentions RL rollout or says at which batch
   size the maximum occurs; the README only says fragmented kernels hurt "small-batch serving". Not used.
3. **Notion body "LMSYS: HPC-Ops × SGLang — sampling benchmark results"** — the blog has no sampler content.
   **Notion body "FlashInfer: sampling and attention kernels"** — arXiv 2501.01005 covers attention only.
4. **Notion / README "small top-k uses local heap/merge selection"** — the current code keeps a rolling top-K in
   keeper threads and merges it with each chunk by a block radix sort (cub `BlockRadixSort`); PR #51 also says
   "BlockRadixSort top-k path". Slides describe the code.
5. **"Folds ... into two kernels"** — true for the full path; the temperature-only path is one kernel (with an
   in-kernel last-block reduction), not merely "lighter".
6. The GPU used for the PR #51 numbers is not stated (the library targets H20/SM90); slides do not name one.
