# Sources for deck 07 (Dynamic Decode Attention)

Downloaded copies: `$SCRATCH/research/26/` (shared with decks 26 and 09): `repo/README.md`, `lmsys.txt`,
`repo/src/attention/...`, `repo/hpc/attention.py`, `repo/benchmark/attention_decode/*`.

## Sources read

| Tag | Source | What was used |
|---|---|---|
| [R] | HPC-Ops README (main), https://github.com/Tencent/hpc-ops | "Updates / Dynamic Decode Attention"; "Performance" row "Dynamic decode attention ... Up to 2.88x" |
| [L] | LMSYS blog "HPC-Ops × SGLang" (Tencent Hunyuan AI Infra + SGLang team, 2026-08-07) | "Attention: load balancing for mixed-length decode", "Scheduling around live KV work", Table 1, Table 6, Table 7, Highlights |
| [A] | `src/attention/decode/sm90/assign_task.cu` | assign kernel: `kTileN=64`; `num_tile_per_cta = max(ceil(total_tiles_all_heads / num_total_ctas), min_process_len / kTileN)`; fill order outer `ihead_kv`, inner `ibatch`; one descriptor per (head, request, chunk); last chunk flagged `is_casual_chunk` and its causal length adjusted; per-(head, request) chunk counts written to the task map |
| [T] | `src/attention/decode/sched_task_info.h` | descriptor `TaskScheduleInfo` = 12 ints (head, batch, chunk, seq start, #KV, #KV-cache, #tiles, #full tiles, causal flag, 3 pad) = 48 B; `kCtaPerSmMap {9: {4,3,3,2,2}}` (CTAs per SM by number of query tokens 1..5) |
| [P] | `hpc/attention.py` | `get_attention_decode_task_workspace` (sm90: dynamic flat-bin workspace, kTileN=64, min_process_len default 512), `assign_attention_decode_task`, `attention_decode_bf16/fp8(..., task_map=...)` |
| [E] | `src/attention/entry.cc` | static policy: BF16 `splitk = 16` if batch ≤ 32 else 4; FP8 `splitk = 4` with `splitk_min_len` 512 / 4096; dynamic: `splitk = kCtaPerSmMap[num_seq_q-1] * sm_count`; MTP 0/1/2; page size 32/64 (BF16 check); dispatch pairs list 90 only |
| [K] | `src/attention/decode/sm90/dynamic/smallm_bf16_dim128_dynamic*.cu(h)` | grid = `num_total_ctas` (persistent), block = 128 math threads + 32 load threads; `kTileN=64`, `kStage=2`; wgmma atom `SM90_64x8x16` (64 KV tokens on M, 8 query heads on N for `num_seq_q=1`); head_dim 128 only; heads per group 8 or 4; page size 16/32/64; TMA loads of paged K/V; writes FP32 partial O + LSE per task; PDL launch attribute on attention and combine |
| [C] | `src/attention/decode/sm90/splitk_combine_kernels.cuh` | combine reads per-(request, KV head) chunk count from the task map; LSE-weighted merge with global max; "light" slots one warp each, "heavy" slots (> 24 chunks = 3 × 8 warps) split the 128-wide head dim into 4 tiles |
| [B] | `benchmark/attention_decode/README.md`, `bench_attention_decode_bf16.py` | scenario definitions (`AxB` = A requests of KV length B); default 1 KV head / 8 Q heads, head_dim 128, page 64 (FlashAttention page 256); `--include-taskmap` flag (task-map build excluded from the timed region by default) |

## Numbers on slides

| Slide | Number | Source location |
|---|---|---|
| 2 | KV work ∝ live length: a 16K request ≈ 16× the KV work of a 1K request | [L] "Attention: load balancing..." paragraph 1 |
| 3 | **4.3×** (derived) = 0.186 ms / 0.043 ms: static split-KV latency on 1×128K + 31×4K vs 64×4K | [L] Table 1, rows 6 and 2 (HPC-Ops static column) |
| 3 | 258,048 vs 262,144 KV tokens (derived): 131,072 + 31 × 4,096 vs 64 × 4,096, i.e. ~2% less KV | scenario definitions [B]/[L] |
| 4 | static policies: fixed split count (BF16: 16 chunks per request if batch ≤ 32, else 4; FP8: 4) or fixed chunk size sized for the longest request | [L] two-policy paragraph; [E] |
| 5 | 1.00× on 64×0.5K (0.013 vs 0.013 ms); 64-token tiles; budget = total tiles / CTAs | [L] Table 1 row 1; [A] |
| 6 | task map built once per decode step from device-side lengths and reused across layers | [L] "Scheduling around live KV work" |
| 7 | toy example 20 + 8 + 6 + 6 = 40 tiles over 4 CTAs → budget 10 tiles; floor 512 tokens = 8 tiles | illustration of the formula in [A]; floor = `min_process_len / kTileN` with API default 512 [P] |
| 8 | descriptor = 12 ints / 48 B: (KV head, request, chunk, start, length, ...) | [T] |
| 9 | 4 persistent CTAs per SM (one query token); 1 load warp + 1 math warpgroup; wgmma 64 × 8 (KV × heads) | [T] kCtaPerSmMap; [K] |
| 9 | combine: heavy path above 24 chunks | [C] `kHeavyThresh = 3 * kCombineWarps`, `kCombineWarps = 8` |
| 10 | Table 1 latencies (ms) static / dynamic: 64×0.5K 0.013/0.013 (1.00×); 64×4K 0.043/0.033 (1.32×); 32×0.125K+32×4K 0.033/0.020 (1.59×); 2×32K+30×4K 0.056/0.032 (1.76×); 1×64K+15×4K 0.097/0.042 (2.32×); 1×128K+31×4K 0.186/0.063 (2.95×) | [L] Table 1 (ratios as printed by the blog) |
| 10 | README headline "up to 2.88x" vs static split-k | [R] Performance table |
| 11 | FlashInfer / FlashAttention (ms): 0.050/0.025, 0.221/0.095, 0.119/0.053, 0.169/0.094, 0.118/0.065, 0.220/0.097 | [L] Table 1 |
| 11 | dynamic on average **2.25×** faster than the best of FlashInfer and FlashAttention per case | [L] Highlights + text above Table 1; re-derived: mean of (1.92, 2.88, 2.65, 2.94, 1.55, 1.54) = 2.25 |
| 12 | H200 upstream validation, dynamic vs static: Hy3-FP8 output throughput +2.0%, total throughput +2.0%, median TTFT −5.3%; Qwen3 BF16 output throughput +3.0%, mean E2E latency −2.8%, mean TPOT −2.8% | [L] Table 6 |
| 12 | H200 Hy3-FP8 attention vs FlashAttention: output throughput +3.7–5.9% | [L] Table 11 |
| 12 | Hy3-FP8 on 8×H20 with HPC-Ops Attention + MoE: TPOT −15.1% to −48.8% at batch 4–64 | [L] Table 7 |
| 13 | skewed batch still 1.9× slower than uniform 64×4K under dynamic (0.063 vs 0.033 ms, derived) | [L] Table 1 |
| 13 | MTP draft tokens: BF16 decode `mtp` 0–4, FP8 decode `mtp` 0–3; FP8 page size 64 | [E] `TORCH_CHECK` lines in `attention_decode_bf16_entry` / `attention_decode_fp8_entry` |
| 11 (notes) | FlashInfer page size 64, FlashAttention page size 256 in the benchmark | [B] README "FP8" section |
| 14 | summary repeats 4.3×, 2.95×, 2.25× | as above |

## Discrepancies (Notion row vs primary sources)

1. **Headline number**: Notion says "up to 2.88x" (README table). The newer LMSYS blog Table 1 (FP8 KV, H20) reports
   **2.95×** on 1×128K + 31×4K. Slides use 2.95× (with the table) and mention the README's 2.88× in notes.
2. **"builds a task graph ... greedily bin-packs"**: the source builds a flat *task map* (per-CTA bins of task
   descriptors), not a graph. The packing is a deterministic sequential fill — every CTA gets the same tile budget
   and bins are filled in (KV head, request) order, spilling a request's remaining tiles into the next bin — rather
   than a sort-based greedy bin-packing. README itself calls it "greedy bin-packing"; slides describe the actual fill.
3. **"sharply cutting CTA tail latency"**: qualitative; the measured effect is kernel latency (Table 1). End to end
   on H200 the attention-only gain is small (+2.0% / +3.0% output throughput, Table 6); the large TPOT gains on
   H20 (−15.1% to −48.8%) are for Attention + MoE together (Table 7). Slides separate the two.
