# Notion row corrections (2026-10-13)

The rows of *Industrial Pretraining Architecture & Infra Tech Landscape (2026)* were corrected against their primary sources, following [`notion-discrepancies.md`](notion-discrepancies.md). Each entry gives the text before and after and why; the untouched original row is kept in each deck's `notes/notion_row.md`. Only Technique, Core Mechanism, Impact / Measured Results, Source and Reference were eligible; page bodies and the other properties were not edited.

## 01 · CSA compressed sparse attention

[deck](../decks/01-csa-compressed-sparse-attention/) · [Notion row](https://app.notion.com/p/e36ba1f7295647b897468dc2b4d30590)

### Core Mechanism
**Before:** Every m=4 consecutive tokens are compressed into one KV entry by a learned overlapping-window weighted average (1M -> 250K), followed by top-k (512/1024) selection plus a 128-token sliding window for local detail; MQA at the core.
**After:** Each m=4-token block becomes one KV entry: a learned softmax-weighted sum over 8 token projections (this block and the previous one, overlapping), cutting length to 1/4; an indexer then picks the top-k compressed entries (1024 Pro / 512 Flash), plus a 128-token sliding window; shared-KV MQA at the core.
**Why:**
- Compression detail from V4 §2.3.1 Eq. 9-12 (softmax over 2m = 8 projections, C^a of block i and C^b of block i-1, learned positional biases).
- "1M -> 250K" is not a number in V4; the paper states the factor (length to 1/m, m = 4), so the row now says 1/4 (sources.md discrepancy 2).
- Top-k is over compressed entries, chosen by the indexer (Eq. 13-17); 1024 for Pro, 512 for Flash (§4.2.1); MQA is shared-KV (Eq. 18-19).

### Impact / Measured Results
**Before:** Per-token FLOPs at 1M context are only 27% (Pro) / 10% (Flash) of V3.2.
**After:** Whole V4 stack (CSA + HCA + low-precision compute), no CSA-only ablation: single-token FLOPs at 1M context are 27% (Pro) / 10% (Flash) of V3.2's.
**Why:**
- V4 credits the 27% / 10% to the hybrid CSA + HCA design together with low-precision computation and storage, not to CSA alone; no CSA-only ablation is published (Abstract; §1 p.5; §2.3.4).

### Other properties
none

## 02 · DeepSeek Sparse Attention (DSA) + Lightning Indexer

[deck](../decks/02-deepseek-sparse-attention-lightning-indexer/) · [Notion row](https://app.notion.com/p/cf0641f4eac74fd9bd6c903a15b59ca3)

### Core Mechanism
**Before:** A lightweight indexer scores history tokens (multi-head ReLU dot-product weighting), top-k (about 2048) picks a KV subset, and sparse attention then runs on top of MLA; migrated through dense-to-sparse continued pretraining.
**After:** A lightweight FP8 indexer (64 heads) scores every preceding token with head-weighted ReLU dot products; top-k (k = 2048) picks the KV entries, and sparse attention runs on MLA in its MQA mode; migrated from V3.1-Terminus by a dense warm-up, then sparse continued pretraining at 128K.
**Why:**
- k is exactly 2,048, not "about" (V32 §2.1.1; HF config index_topk) (sources.md discrepancy 1).
- 64 heads and FP8 from HF config / model.py and V32 Eq. 1 ("can be implemented in FP8"); DSA is instantiated on MLA's MQA mode (V32 §2.1).
- Warm-up (main model frozen, dense) then sparse training from V3.1-Terminus on 128K sequences (V32 §2.1.1).

### Impact / Measured Results
**Before:** Large drop in long-context training and serving cost with base quality on par with V3.1; GLM-5 adopts it wholesale and completed large-scale DSA-based RL.
**After:** Per-token serving cost at 128K about 3.6x lower (prefill) / 8.7x (decode) than V3.1-Terminus (derived from V3.2 Fig. 3, H800); training cost not quantified; quality on par. GLM-5 adopts it with changes (32 indexer heads, 20B-token adaptation, deterministic top-k, frozen indexer during RL).
**Why:**
- DeepSeek measures only inference cost (V32 Fig. 3, §2.3); training-cost reduction is not measured, so it is no longer claimed (discrepancy 2). The 3.6x / 8.7x are read from the Fig. 3 vector data (notes/digitize_fig3.py), hence "derived".
- Baseline is V3.1-Terminus, not "V3.1" (V32 §2.1, §2.2: on par, +4 on AA-LCR).
- "Wholesale" is wrong: GLM-5 uses 32 indexer heads (Table 10), 20B adaptation tokens vs 943.7B (§2.1.1), deterministic torch.topk and a frozen indexer during RL ("DSA RL insights") (discrepancy 3).

### Other properties
none. Not changed (page body, out of scope): the body links Native Sparse Attention (arXiv 2502.11089) as a paper for this row, though NSA is a different, natively trained method (sources.md discrepancy 4).

## 03 · HCA heavily-compressed attention + hybrid attention stacking

[deck](../decks/03-hca-highly-compressed-attention-hybrid-attention/) · [Notion row](https://app.notion.com/p/ceae2901ccd44561a14abacc7f6574bf)

### Technique
**Before:** HCA highly-compressed attention + hybrid attention stacking
**After:** HCA heavily-compressed attention + hybrid attention stacking
**Why:** the paper names it Heavily Compressed Attention (V4 Abstract, §2.3.2).

### Core Mechanism
**Before:** High-compression attention layers with m'=128 interleaved with CSA; every layer remains attention - no SSM or Gated-DeltaNet-style linear attention is introduced.
**After:** Softmax pooling folds each 128-token block into one KV entry (m'=128), and queries attend densely to all entries plus a 128-token raw window (no indexer, no top-k); interleaved with CSA, so every layer stays softmax attention.
**Why:**
- Mechanism detail from V4 §2.3.2-2.3.3 and Eq. 20-23 (non-overlapping blocks, dense shared-KV MQA, 128-token window).
- The "no SSM / Gated-DeltaNet" framing is not argued in the paper (it even manages the window cache "as a state-space model", §3.5.1); kept only the supported fact that every layer is softmax attention.

### Impact / Measured Results
**Before:** KV cache falls to 10% (Pro) / 7% (Flash) of V3.2, making 1M context routine.
**After:** Whole V4 stack (CSA + HCA + low-precision KV storage): 1M-token KV cache is 10% (Pro) / 7% (Flash) of V3.2's; retrieval still fades past 128K (MRCR 8-needle 0.92 at 128K, 0.59 at 1M).
**Why:**
- V4 credits the 10% / 7% to the hybrid CSA + HCA design together with low-precision storage, not to HCA alone (§1 p.5, §2.3.4).
- "Routine 1M" glosses over the measured retrieval drop (V4 Fig. 9).

### Other properties
none

## 04 · IndexShare (cross-layer shared indexer)

[deck](../decks/04-indexshare-cross-layer-shared-indexer/) · [Notion row](https://app.notion.com/p/e18d1e0afe6f4d3d9c37bfdf815b0738)

### Core Mechanism
**Before:** Every four sparse-attention layers reuse the top-k selection of a single indexer, avoiding repeated indexing computation.
**After:** In GLM-5.2 the first of every four DSA layers runs the indexer and the next three reuse its top-k (the first three layers keep their own: 21 indexers in 78 layers); the IndexCache paper also uses searched, non-uniform layouts.
**Why:**
- "Every four" is the GLM-5.2 production layout (G52B; HF52 indexer_types FFF + FSSS x 18 = 21 full of 78), not a rule of the technique (sources.md discrepancy 2).
- The paper's headline training-free results use a greedy-searched, non-uniform pattern (IC §3.1, Table 2, App. B).

### Impact / Measured Results
**Before:** 2.9x lower per-token FLOPs at 1M context; the improved MTP adds up to 20% more accepted length in speculative decoding.
**After:** GLM-5.2 vs GLM-5.1: 2.9x lower per-token FLOPs at 1M (a FLOPs count, not a measured speedup); MTP accepted length up to +20% only together with KVShare, rejection sampling and a TV loss (IndexShare + KVShare alone: +12%, derived).
**Why:**
- 2.9x is a model-level FLOPs comparison, GLM-5.2 vs GLM-5.1, from the Z.ai blog figure (G52R; G52B), not a measured speedup (discrepancy 4).
- The +20% (4.56 to 5.47) stacks IndexShare + KVShare, rejection sampling and an end-to-end TV loss (G52B table, GLM-5.1 backbone, 7 MTP steps); IndexShare + KVShare alone is 4.56 to 5.10 = +12%, derived (discrepancy 3).

### Other properties
none. Title left as is: GLM-5.2's README and blog call it IndexShare; the cited paper (arXiv 2603.12201) calls it IndexCache (sources.md discrepancy 1).

## 05 · Kimi Delta Attention (KDA)

[deck](../decks/05-kimi-delta-attention-kda/) · [Notion row](https://app.notion.com/p/0a25e512d50542b08af4e090523e14b2)

### Core Mechanism
**Before:** A Gated DeltaNet refinement: the delta rule plus channel-wise forget gates maintain a fixed-size recurrent state; three KDA layers alternate with one gated-MLA layer, fully NoPE (position information comes implicitly from gated decay).
**After:** A Gated DeltaNet refinement: the delta rule plus channel-wise forget gates maintain a fixed-size recurrent state; in K3, three KDA layers alternate with one Gated MLA layer, plus one final MLA layer (69 KDA + 24 MLA), fully NoPE (position comes implicitly from gated decay).
**Why:**
- K3 places one extra Gated MLA layer at the end of the backbone (§2.1), so the stack is 69 KDA + 24 MLA = 93 layers (Table 1), not an exact 3:1 (sources.md discrepancy 2).

### Impact / Measured Results
**Before:** State does not grow with length, so 1M context needs no RoPE interpolation; with AttnRes and Stable LatentMoE it gives roughly 2.5x better overall scaling efficiency than K2.
**After:** 1M context needs no RoPE interpolation because K3 is NoPE, not because the state is fixed-size; the roughly 2.5x scaling-efficiency gain over K2 is joint (KDA, AttnRes, Stable LatentMoE, refined data and training recipes), with no KDA-only ablation.
**Why:**
- K3 §3.4 attributes the no-interpolation property to NoPE (no RoPE to rescale); the fixed state is a memory property, not the reason (discrepancy 3).
- K3 credits the ~2.5x to KDA, AttnRes, Stable LatentMoE and refined data and training recipes together; there is no KDA-only ablation at K3 scale (Abstract; §1; §3.2, Fig. 7) (discrepancy 1).

### Other properties
none

## 06 · Staged context extension + annealing / cooldown

[deck](../decks/06-staged-context-extension-annealing-cooldown/) · [Notion row](https://app.notion.com/p/ff95a3d4ac0745728425571a7344ee70)

### Core Mechanism
**Before:** Main pretraining on short sequences (K2: 4k->32k->128k with YaRN; K3: 8k->64k), stretched to 256k->1M only during cooldown / mid-training, with constant LR plus cosine decay and final annealing.
**After:** Main pretraining on short sequences, long windows only in a low-LR tail: K2 trains 4K then 32K and reaches 128K with YaRN (WSD schedule, then anneal); GLM-4.5 goes 4K to 32K to 128K in mid-training (cosine); K3 trains 8K to 64K, then 256K to 1M in cooldown (cosine).
**Why:**
- K2 trains only 4K and 32K; 128K is a YaRN extension on top of the 32K stage (K2 §2.5) (sources.md discrepancy 3).
- "Constant LR plus cosine decay and final annealing" is K2's WSD schedule only (K2 §2.5); K3 uses cosine with 1% warm-up (K3 §3.3) and GLM-4.5 uses cosine, reporting WSD underfits (GLM §2.4) (discrepancy 1).
- "Cooldown" is K3's term; GLM-4.5's long stages are "mid-training" within one cosine schedule (GLM Fig. 3, §2.3-2.4; K3 §3.3-3.4) (discrepancy 5).

### Impact / Measured Results
**Before:** Concentrates expensive long-context compute on a small tail of tokens; this is how K3 gets its 1M context.
**After:** Puts long-context compute on a small tail of tokens (K2: 60B at 32K; GLM-4.5: 100B at 128K; K3 publishes no budgets); no report ablates staging, and K3's 1M also relies on NoPE + KDA.
**Why:**
- The quantified tails come from K2 (60B at 32K, §2.5) and GLM-4.5 (100B at 128K, Fig. 3); K3 says only "small fraction of the overall training budget" (§3.4) (discrepancy 2).
- None of the three reports ablates staged vs. unstaged training, so there is no measured result (discrepancy 4).
- K3's 1M context also rests on NoPE + KDA, with no positional rescaling (K3 §2.1.2, §3.4), not on staging alone (discrepancy 2).

### Other properties
none

## 07 · Dynamic Decode Attention (dynamic task scheduling)

[deck](../decks/07-dynamic-decode-attention/) · [Notion row](https://app.notion.com/p/395fb76ae58944d0ae141c2a047478b6)

### Core Mechanism
**Before:** Splits all requests into uniform KV tiles, builds a task graph before each decode step and greedily bin-packs it across CTAs; the attention kernel consumes the task map directly, with a combine kernel merging split-k results.
**After:** Splits all requests into uniform 64-token KV tiles and, once per decode step, builds a flat task map: every persistent CTA gets the same tile budget, filled in (KV head, request) order with overflow spilling into the next CTA; the attention kernel consumes the map directly and a combine kernel merges split-k results.
**Why:**
- The source builds a flat task map of per-CTA bins of task descriptors, not a graph (assign_task.cu [A]; sched_task_info.h [T]) (sources.md discrepancy 2).
- The packing is a deterministic sequential fill with one tile budget per CTA, in (KV head, request) order, spilling a request's remaining tiles into the next bin, not a sort-based greedy bin-packing ([A]: kTileN = 64, num_tile_per_cta formula, loop order) (discrepancy 2).
- Map built once per decode step and reused across layers (LMSYS [L] "Scheduling around live KV work"); persistent CTAs from [K].

### Impact / Measured Results
**Before:** Up to 2.88x over static split-k scheduling (long context and mixed-length batches), sharply cutting CTA tail latency.
**After:** Kernel latency up to 2.95x lower than static split-k on 1x128K + 31x4K (FP8 KV, H20; the README says up to 2.88x); end to end on H200 the attention-only gain is small (+2.0% / +3.0% output throughput).
**Why:**
- The newer LMSYS Table 1 reports 2.95x (0.186 to 0.063 ms) on 1x128K + 31x4K; the README's 2.88x is kept as a pointer (discrepancy 1).
- "Sharply cutting CTA tail latency" is qualitative; the measured effect is kernel latency (Table 1), and end to end on H200 dynamic vs static gives only +2.0% (Hy3-FP8) / +3.0% (Qwen3 BF16) output throughput (Table 6). The large H20 TPOT gains (Table 7) are Attention + MoE together, so they are not credited here (discrepancy 3).

### Other properties
none

## 08 · FlashKDA (chunkwise linear-attention kernel)

[deck](../decks/08-flashkda-chunkwise-linear-attention-kernel/) · [Notion row](https://app.notion.com/p/addf5d5fcfad445081c78ebac0e50d60)

### Core Mechanism
**Before:** A CUTLASS-based chunkwise kernel that overlaps intra-chunk computation with cross-chunk recurrent state propagation instead of stalling between two alternating phases; shared by training and prefill and auto-dispatched as a flash-linear-attention backend.
**After:** A CUTLASS-based chunkwise kernel: a token-parallel first kernel does all state-independent intra-chunk work (16-token chunks), then a warp-specialized second kernel runs the serial cross-chunk state chain while TMA warps overlap memory traffic; K3 uses it for training and prefill, but the public v1 is forward-only (FLA backend, inference mode).
**Why:**
- In the public v1 the two kernels run one after the other on the same stream (SRC fwd_launch.cu); the overlap comes from hoisting state-independent work into the token-parallel K1 and from warp specialization in K2 (load/store warps vs MMA warps) (SRC fwd_kernel2.cuh; BLOG §2) (sources.md discrepancy 3).
- 16-token chunks vs FLA's 64 (BLOG §1).
- K3 §5.1.1 says training and prefill, but FlashKDA v1 exposes only flash_kda.fwd and FLA dispatches it under torch.inference_mode(); no backward kernel is public (README) (discrepancy 1).

### Impact / Measured Results
**Before:** Lets linear attention actually saturate the hardware on long training sequences - a prerequisite for using KDA in a 2.8T model.
**After:** Forward pass at T = 8K, 96 heads: 1.85x (H20) / 2.31x (GB200) faster than FLA's Triton chunk_kda at fixed length, up to 3.27x on varlen batches (GB200); no hardware-utilization or training numbers are published.
**Why:**
- No utilization or long-sequence training numbers exist; the only public measurements are forward timings at T = 8K (BENCH-H20, BENCH-GB200: 2.6220 vs 4.8388 ms; 1.0087 vs 2.3271 ms; varlen 1024 x 8 = 3.27x) (discrepancy 4).
- "Prerequisite for using KDA in a 2.8T model" is not stated by any source; K3 says only that FlashKDA "substantially outperforms the Triton reference" (discrepancy 4).

### Other properties
none. Not changed (page body, out of scope): the body links the Kimi-Linear repo as "FlashKDA kernels"; FlashKDA lives in github.com/MoonshotAI/FlashKDA (sources.md discrepancy 2).

## 09 · FP8 block-sparse prefill attention

[deck](../decks/09-fp8-block-sparse-prefill-attention/) · [Notion row](https://app.notion.com/p/6ee53460854a4c988e083e2ba08d1f88)

### Core Mechanism
**Before:** Takes a precomputed block mask, skips masked KV tiles wholesale, and applies per-tile FP8 scaling to preserve numerical quality under sparsity (paged KV supported).
**After:** Takes a precomputed 128x128 block mask, loads and computes only the active KV tiles, and runs FP8 with Q scaled per token and head, K per tensor or per token group and head, V per tensor or per head, P by a fixed 256 (no per-tile scales); paged KV supported.
**Why:**
- There is no scale computed per KV tile: Q per-token per-head, K per-tensor or per-token-group per-head, V per-tensor or per-head ([P] docstring), and P x 256 before its FP8 cast ([K] kFp8PrefillPScale) (sources.md discrepancy 1).
- 128 x 128 tiles ([E] kTileM = kTileN = 128); the producer warp builds an active-tile list and TMA-loads only those tiles ([K]).

### Impact / Measured Results
**Before:** Up to 3.16x over MIT-BSA / FlashPrefill-BSA (BF16) and a dense FP8 baseline; the companion kernel that makes DSA-style sparsity deployable.
**After:** Up to 3.16x vs MIT-BSA / FlashPrefill-BSA (BF16) and HPC / FA3 dense FP8 (README, no per-case table); at 10% block density it takes 11% of dense FP8 latency (H20, 64K). Paired with Stem block masks, not DSA (token-level).
**Why:**
- README lists two dense FP8 baselines (HPC-Dense FP8, FA3-Dense FP8) and publishes no per-case table ([R]) (discrepancy 2).
- The only per-density data is the PR #45 plot: 11% of dense FP8 latency at 10% density (batch 4, seq 65536, Hq 4, Hkv 1, D 128, H20) ([PR45]).
- "Companion kernel that makes DSA-style sparsity deployable" is unsupported: the kernel is block-sparse, DSA picks individual tokens (top-2048); the repo pairs it with Stem as the mask generator ([PR50]; [S]) (discrepancy 3).

### Other properties
none

## 10 · FP8 MQA logits / sparse indexer scoring kernel

[deck](../decks/10-fp8-mqa-logits-sparse-indexer-scoring-kernel/) · [Notion row](https://app.notion.com/p/ac84a3f18a034bac9f7c33b63a8eb6e3)

### Core Mechanism
**Before:** Weighted-ReLU MQA logits for the lightning indexer: each query takes E4M3 dot products over KV in [start, end), then ReLU, then a head-weighted sum; separate non-paged (prefill) and paged (decode) versions, now extended to an FP4 indexer.
**After:** Weighted-ReLU MQA logits for the lightning indexer: each query takes dot products over its KV range, E4M3 on SM90 or MXFP8 / MXFP4 on SM100, then ReLU, then a head-weighted sum; non-paged (prefill) and paged (decode) versions, plus a sparse variant (2026-09) that scores only selected KV blocks.
**Why:**
- E4M3 with a per-token scale is SM90-only; on SM100 the current API requires MXFP8 or MXFP4 inputs with packed UE8M0 block scales ([API]; PR #462) (sources.md discrepancy 1). This also covers the FP4 indexer (PR #304).
- A third family exists since 2026-09: the Sparse Indexer (fp8_fp4_sparse_mqa_logits and its paged variant) that scores only selected KV blocks for V4.1 ([RD] News; PR #432) (discrepancy 2).
- "[start, end)" rewritten as "its KV range" to avoid bracket characters in Notion markdown; meaning unchanged ([RD] ks / ke).

### Impact / Measured Results
**Before:** The prerequisite for DSA sparsity to actually save compute - the indexer itself must not become the new bottleneck.
**After:** unchanged
**Why:**
- Qualitative and not contradicted: IndexCache measures the indexer at up to 81% of prefill time at 200K on a 30B DSA model ([IC]). DeepSeek publishes no benchmark for these kernels (discrepancy 4), and the row claims none, so no number is added.

### Other properties
none

## 11 · 1F1B / batch-level EP all-to-all overlap

[deck](../decks/11-1f1b-batch-level-ep-all-to-all-overlap/) · [Notion row](https://app.notion.com/p/0e6015aa2a8449ff917574f6139df4c2)

### Core Mechanism
**Before:** --overlap-moe-expert-parallel-comm with --delay-wgrad-compute hides EP all-to-all inside 1F1B pipeline slots and batch-level splits; DP/TP/PP communication overlap is supported as well.
**After:** --overlap-moe-expert-parallel-comm with --delay-wgrad-compute merges the forward of one micro-batch with the backward of another in the 1F1B steady state, so each one's EP all-to-all hides behind the other's compute (two CUDA streams, weight gradients deferred); DP/TP/PP communication overlap is supported as well.
**Why:**
- The all-to-all is hidden behind the compute of an adjacent micro-batch in merged FWD-BWD, not in pipeline slots or bubbles; the first forward and last backward stay exposed (MC p.33; CODE-PLAN run docstrings; CODE-1F1B) (sources.md discrepancy 1).
- --delay-wgrad-compute splits the backward MLP into data and weight gradients and defers the latter (MC p.33; CODE-CFG docstring).

### Impact / Measured Results
**Before:** Buries EP communication in pipeline bubbles - the default optimization for large-EP training.
**After:** DeepSeek-V3 on H100 (EP64): EP all-to-all drops from 30-40% of iteration time after DeepEP to under 5%; off by default, and left off when EP fits in NVLink (DeepSeek-V3 on GB200).
**Why:**
- Not "pipeline bubbles": see Core Mechanism (discrepancy 1).
- Measured result: 30-40% remaining after DeepEP, under 5% with 1F1B FWD-BWD overlap + W/D split, DeepSeek-V3 EP64 on H100 (MC §4.2.3 p.32-34).
- Not "the default": both flags default to False (RD lines 563-564), and the best DeepSeek-V3 GB200 config (EP64 inside NVL72) does not use it (MC Table 17; App. B.2) (discrepancy 2).

### Other properties
none

## 12 · Ascend · comm/compute fusion and PD-disaggregated scheduling

[deck](../decks/12-ascend-comm-compute-fusion-and-pd-disaggregated/) · [Notion row](https://app.notion.com/p/b1ba69567b7740fdbeead2013a32881c)

### Core Mechanism
**Before:** Splits AllReduce into ReduceScatter + AllGather and pipelines it with tiled matrix computation to mask weaker interconnect bandwidth; layered with prefill/decode-disaggregated serving, prefix caching and MTP speculative decoding.
**After:** FlashComm splits AllReduce into ReduceScatter + AllGather so the MoE path sees 1/TP of the tokens and communication hides behind computation; vLLM Ascend can also fuse the matmul with its reduce-scatter; layered with prefill/decode-disaggregated serving, prefix caching and MTP speculative decoding.
**Why:**
- The GLM-5 report says only that FlashComm "splits AllReduce operations to hide communication latency behind computation" (GLM5 §5 p.22); the RS + AG split and the 1/TP token sharding are from vLLM Ascend (VA-GLM5 §9.1 Table 2; VA-SP).
- Tile-level fusion of matmul and collective is the engine's (fuse_gemm_comms, VA-SP) and the literature's (FLUX), not the report's, so it is now attributed to vLLM Ascend (sources.md discrepancy 1).
- "To mask weaker interconnect bandwidth" dropped: no source states it for GLM-5 on Ascend, and no Ascend communication share is published (discrepancy 4).

### Impact / Measured Results
**Before:** One domestic node matches a dual-GPU node; about 50% lower long-sequence serving cost.
**After:** Whole Ascend stack, no per-technique split: about 50% lower long-sequence deployment cost; one node claimed comparable to dual-GPU clusters, with no benchmark published.
**Why:**
- The 50% is the result of the whole Ascend adaptation stack (quantization + fused kernels + scheduling), with no per-pillar breakdown (GLM5 §5 p.22) (discrepancy 3).
- The report writes "comparable to dual-GPU international clusters" with no baseline, benchmark, metric or configuration; "matches" overstated it (discrepancy 2).

### Other properties
none

## 13 · Fused AllReduce + residual + RMSNorm

[deck](../decks/13-fused-allreduce-residual-rmsnorm/) · [Notion row](https://app.notion.com/p/2a55d26c7c7a4f858f5816d7ca24b473)

### Core Mechanism
**Before:** Fuses the post-TP AllReduce, residual add and RMSNorm into an NVLink-native kernel: a CUDA multicast high-throughput mode for many tokens and a Lamport P2P dual-kernel mode with PDL overlap for few tokens, both on two-shot communication schedules.
**After:** Fuses the post-TP AllReduce, residual add and RMSNorm into one NVLink-native op: a single-kernel CUDA multicast high-throughput mode for many tokens and a two-kernel Lamport low-latency mode (P2P scatter, multicast broadcast, chained by PDL) for few tokens, both on two-shot schedules.
**Why:**
- "An NVLink-native kernel" is wrong for LL: HT is one kernel, LL is two kernels chained by PDL ([PR48]; [LL]) (sources.md discrepancy 3).
- LL is not purely P2P: only the scatter (reduce) phase uses P2P stores; the broadcast stores through the multicast address ([LL] twoshotAllreduceKernel) (discrepancy 2).

### Impact / Measured Results
**Before:** Up to 1.76x over NCCL / FlashInfer for single-node BF16 TP shapes at hidden 4096/5120/7168.
**After:** unchanged
**Why:**
- Matches the HPC-Ops README Performance table ([RD]; metric definition [BR]: best HPC-Ops mode vs best of NCCL / FlashInfer, 8xH20, BF16).

### Other properties
none. Not changed (page body, out of scope): the body cites the LMSYS HPC-Ops x SGLang blog as "vs NCCL / FlashInfer", but that blog has no AllReduce + RMSNorm result; the 1.76x is from the README (sources.md discrepancy 1).

## 14 · Wave-based fine-grained communication/compute overlap

[deck](../decks/14-wave-based-fine-grained-communication-compute/) · [Notion row](https://app.notion.com/p/0a5b3fd612d24e81a538b34ef561d06c)

### Core Mechanism
**Before:** Experts are split into waves and compute starts the moment a wave's dispatch lands, so in steady state the current wave's compute, the next wave's transfer and the return of finished results all run concurrently, fused into one pipelined kernel.
**After:** unchanged
**Why:**
- Matches V4 §3.1 p.15 (waves, compute starts when a wave's communication completes, three-way steady-state concurrency, single pipelined kernel).

### Impact / Measured Results
**Before:** Fully hides MoE communication whenever per-layer comm time < compute time; derives the hardware balance point C/B <= 2d (about 6144 FLOPs/Byte), so 1 GBps of interconnect covers 6.1 TFLOP/s; up to 1.96x on RL rollout.
**After:** Hides MoE communication in steady state when per-layer comm time is below compute time (pipeline fill and drain stay exposed); balance point C/B at most 2d = 6144 FLOPs/Byte for V4-Pro (d = expert width 3072), so each GB/s of interconnect covers 6.1 TFLOP/s; MoE-layer speedup over non-fused kernels 1.50-1.73x, up to 1.96x (V4-Flash, 1 token per rank, EP8).
**Why:**
- "Fully" overstates: the first dispatch and last combine of the pipeline stay exposed, so the hiding holds in steady state (V4 §3.1 "Communication Latency Can Be Hidden"; Fig. 5) (sources.md discrepancy 1).
- 6144 is exact and V4-Pro only, with d = expert intermediate dimension 3072, not the hidden size (V4 §3.1 p.16; §4.2.1) (discrepancy 2).
- 1.96x is a MoE-layer kernel benchmark vs the non-fused DeepEP + DeepGEMM path (V4-Flash, 1 token per rank, EP8; PR316), quoted by V4 for latency-sensitive cases such as RL rollout, not an end-to-end RL speedup; general inference gets 1.50-1.73x (V4 §3.1 p.16) (discrepancies 3, 4).

### Other properties
none. Not changed (page body, out of scope): the body lists Flux (arXiv 2406.06858) as a paper, while V4 cites Comet and FlashDMoE as inspirations (sources.md discrepancy 5).

## 15 · 0-SM PP / CP / Engram communication primitives

[deck](../decks/15-0-sm-pp-cp-engram-communication-primitives/) · [Notion row](https://app.notion.com/p/c3f064d3adbd46a983472f8d7992f155)

### Core Mechanism
**Before:** Beyond EP, experimental primitives: PP and Engram (remote memory access) go over RDMA with 0 SMs, CP uses the Copy Engine with 0 SMs; DP/TP all-gather and reduce-scatter implementations are included.
**After:** Beyond EP, experimental primitives that keep SMs off the payload path (SMs at most stage data or post work requests): PP send/recv and Engram remote reads go over RDMA, the CP all-gather over NVLink copy engines (V2.5 adds RDMA and hybrid paths); V2.5 adds CP/DP reduce-scatter and all-reduce, which still use SMs.
**Why:**
- "0 SMs" holds for the payload, not for kernels: PP launches one-warp-per-CTA TMA staging kernels and Engram/RDMA all-gather launch warps that post puts/gets; only the pure-NVLink all-gather runs no kernel ([C-PP], [C-EN25], [C-AG25]).
- Copy Engine is the NVLink path only; V2.5 adds RDMA (NIC) and hybrid paths ([C-AG25]).
- DP/TP all-gather/reduce-scatter were "on-going" at V2 ([R2] l.37); V2.5 BucketBuffer covers CP and DP, not TP, and "reduce-scatter and all-reduce use GPU SMs" ([R25] l.14, l.35, l.293, l.328).

### Impact / Measured Results
**Before:** Communication no longer competes with compute for SMs - the key direction toward "free" communication (currently experimental).
**After:** PP, CP all-gather and Engram transfers no longer occupy SMs; EP dispatch/combine still do (4-6 SMs in V2 vs 24 in V1 for V3-like training). Still experimental; no bandwidth or end-to-end numbers are published for these primitives.
**Why:**
- "Communication" overall still uses SMs: EP dispatch/combine require them ([R25] l.39 "zero-SM RDMA EP is not supported"; [R2] l.22 for 24 to 4-6).
- Dropped the unsupported "key direction toward free communication" framing; no source measures these primitives' bandwidth or end-to-end effect (discrepancy item 4).

### Other properties
none

## 16 · Elastic GPU/CPU hybrid memory buffer

[deck](../decks/16-elastic-gpu-cpu-hybrid-memory-buffer/) · [Notion row](https://app.notion.com/p/f8ca122ace3b41dba7d867f9c2731da3)

### Core Mechanism
**Before:** One contiguous virtual address range backed by a mix of GPU and CPU physical memory, transparently supporting Engram and imbalanced EP; EP replay compresses intermediate buffers.
**After:** One contiguous virtual address range: GPU memory first, then pinned host (or GPU) storage shards registered for RDMA only; shipped for Engram tables (opt-in CPU storage in V2.5 EngramBuffer, experimental), while host-backed EP buffers and EP replay remain V2 roadmap items.
**Why:**
- Host-backed storage shipped for Engram only; `EPBuffer` passes zero storage bytes, so "imbalanced EP" is not supported ([BUF]; [R2] l.35 lists it as on-going; [EB2] "currently GPU-only").
- Not "transparent": the caller sizes the storage and sets `use_cpu_rdma_storage` ([ENG], [CTX] l.75-99).
- EP replay is a V2 on-going item with no code in V2.5 ([R2] l.36).
- Mechanism detail: VA layout from [SYM] l.137-179; storage shards registered as `NCCL_WIN_GIN_ONLY` windows ([CTX] l.207-222).

### Impact / Measured Results
**Before:** Mitigates V2's larger buffer footprint so worst-case load no longer dictates the memory budget.
**After:** Not yet for EP: V2.5 EP buffers are GPU-only and still sized for worst-case load; host-backed storage has shipped only for Engram tables, letting them exceed HBM (experimental).
**Why:**
- The V2.5 EP size formula still reserves room for N x T tokens per rank and EP passes no storage bytes ([EPSZ], [BUF]).
- The only shipped user is EngramBuffer ([R25] l.17, l.384-386; [PR763]).

### Other properties
none

## 17 · ElasticBuffer unified API + analytic SM/QP budgeting

[deck](../decks/17-elasticbuffer-unified-api-analytic-sm-qp/) · [Notion row](https://app.notion.com/p/288a723bba1048b2964f7c67a24d4a3a)

### Core Mechanism
**Before:** Unifies the high-throughput and low-latency dispatch/combine APIs into ElasticBuffer (with a new GEMM layout) and derives the optimal SM and QP counts analytically from the MoE configuration, removing auto-tuning; handle caching reuses routing metadata.
**After:** Unifies the high-throughput and low-latency dispatch/combine APIs into one buffer (ElasticBuffer in V2.0, EPBuffer in V2.5) with an expanded layout for grouped expert GEMMs, and computes SM and QP counts from an analytic model of the MoE configuration and link bandwidths instead of tuned per-EP-size tables; a cached handle replays the same routing (e.g., backward), not new routing.
**Why:**
- ElasticBuffer was split into EPBuffer / EngramBuffer / PPBuffer / BucketBuffer in V2.5 ([PR763]).
- The README's name for the layout is the "expanded layout for grouped expert GEMMs" (`do_expand`, `expert_alignment`) ([R25]).
- "Optimal" overstated: the model is a balanced-gate bandwidth estimate with +30% headroom and rounding ([EP25] l.431-560); V1 used a per-EP-size config table plus cluster tuning ([V1code], [R1]).
- V2.5 narrows handle reuse: "A new routing decision needs a fresh dispatch; decoding alone does not make an old routing handle reusable" ([R25]).

### Impact / Measured Results
**Before:** Communication SMs drop from 24 to 4-6 at equal or better performance in V3-style training; 726/740 GB/s on SM100 NVLink EP8.
**After:** Communication SMs drop from 24 to 4-6 at equal or better performance in V3-style training; SM100 NVLink EP8 reaches 726/740 GB/s (dispatch/combine) at 64 SMs, 643/675 GB/s at 24 SMs.
**Why:**
- 726/740 GB/s is the 64-SM "max perf" row; the minimum-SM row is 643/675 GB/s at 24 SMs ([R2] Performance table rows 4-5).

### Other properties
none

## 18 · FP8 dispatch + BF16 combine low-precision communication

[deck](../decks/18-fp8-dispatch-bf16-combine-low-precision/) · [Notion row](https://app.notion.com/p/8f8e53cef45c40b58e4079637b461706)

### Core Mechanism
**Before:** Dispatch sends FP8 (data plus scale) and combine returns BF16; the backward dispatch is exactly the forward combine; the Hybrid-EP backend further supports NVFP4.
**After:** Dispatch sends FP8 (data plus per-128-channel scales) and combine returns BF16; in backward, dispatch's gradient is a combine and combine's a dispatch, so token-to-expert stays FP8 and expert-to-token BF16; NVFP4 is only listed for the Hybrid-EP branch, whose own docs show BF16 and FP8.
**Why:**
- The source states the mirror per operator ("The backward pass of MoE dispatch is actually a combine" and vice versa, [RD]); precisions follow the direction ([V3] §3.3.3).
- Scale granularity: 1x128 per-token groups ([V3] §3.3.2; [MATH]).
- NVFP4: main README lists it for the Hybrid-EP branch, but that branch's doc lists "BF16, FP8" and its combine is always 2-byte ([HE-DOC], [HE-CFG]).

### Impact / Measured Results
**Before:** About 3h bytes per token-expert - the premise behind DeepSeek V4's C/B balance point (about 6144 FLOPs/Byte).
**After:** About 3h bytes per token-expert (plus about 3% FP32 scales, derived); DeepSeek V4 uses 3h for its analytic hardware balance point, C/B at most 2d = 6144 FLOPs/Byte (V4-Pro, d = 3072), not a measured result.
**Why:**
- V4's 3h ignores scales; with DeepEP's FP32 scales the payload is 3h + h/32 = 3.03h (derived in sources.md).
- 6144 FLOPs/Byte is a design threshold from V4 §3.1 "Observations and Proposals" (p.16) for d = 3072 ([V4] §4.2.1), not a DeepEP measurement.

### Other properties
none

## 19 · HybridEP + FlexDispatcher

[deck](../decks/19-hybridep-flexdispatcher/) · [Notion row](https://app.notion.com/p/3e7e3ebd211b43769e3a8acb1883c7f6)

### Core Mechanism
**Before:** MCore's Flex dispatcher abstracts the DeepEP and HybridEP backends uniformly; HybridEP uses TMA/IBGDA and fewer SMs, with native MNNVL support.
**After:** MCore's Flex dispatcher abstracts four backends (deepep, deepepv2, hybridep, ncclep on main) behind one interface; HybridEP uses TMA over NVLink and IBGDA over RDMA in persistent warp-specialized kernels with fewer SMs, and is the only Flex backend with MNNVL support.
**Why:**
- On main the Flex dispatcher has four backends, not two ([MLM-CFG] `moe_flex_dispatcher_backend`; [MLM-TD] `MoEFlexTokenDispatcher`).
- TMA for NVLink, IBGDA for RDMA, persistent warp-specialized kernels ([BLOG] design section; [HEP-IMPL] §8; [HEP-RD]).
- "Currently only hybridep backend supports the MNNVL case" ([MLM-CFG]).

### Impact / Measured Results
**Before:** Higher MoE all-to-all bandwidth utilization on GB200 NVL72 / B200 / H100 (available by default in MCore v0.15).
**After:** Higher MoE all-to-all bandwidth utilization on GB200 NVL72 / B200 / H100; added in MCore v0.15 as an opt-in Flex backend marked experimental (the Flex default backend is deepep).
**Why:**
- Not a default: v0.15 added HybridEP to the Flex dispatcher ([RD] l.16), but `moe_token_dispatcher_type` defaults to allgather and `moe_flex_dispatcher_backend` defaults to deepep, marked "[Experimental]" ([RD] l.312; [MLM-CFG]).

### Other properties
none

## 20 · MoonEP · dynamic redundant experts + online GPU planning + zero-copy static shapes

[deck](../decks/20-moonep-dynamic-redundant-experts-online-gpu/) · [Notion row](https://app.notion.com/p/dfc27fd2e9d443c9a0a0efd215bdc3da)

### Core Mechanism
**Before:** Three co-designed pieces in one library: (1) replicates hot experts online according to the current batch's routing distribution so every EP rank receives exactly S x K / EP tokens, with the backward pass reducing redundant replicas' gradients back to the home rank (mathematically equivalent to no redundancy); (2) a GPU kernel solves redundant-expert placement + token assignment online for every micro-batch, with no host involvement and negligible cost relative to the all-to-all; (3) permute/unpermute is fused into dispatch/combine with a fixed static S x K buffer instead of a dynamic S x K x R shape, removing per-layer host-device syncs and intermediate copies.
**After:** Three co-designed pieces in one library: (1) replicates hot experts online according to the current batch's routing so every EP rank receives exactly S x K tokens (S = input tokens per rank), with the backward pass reducing replicas' gradients back to the home rank (mathematically equivalent to no redundancy); (2) a GPU kernel solves redundant-expert placement + token assignment online for every micro-batch (one launch, rank 0 plans), with no per-layer host involvement and a near-optimal plan; (3) permute/unpermute is fused into dispatch/combine with a static, expert-grouped buffer of S x K rows plus alignment padding instead of a dynamic S x K x R shape, removing per-layer host-device syncs and intermediate copies.
**Why:**
- Each rank receives S x K tokens, not S x K / EP ([README]; [K3] §5.2.1).
- The buffer is NvS = align(S·K + (P−1)·2·E/R, P), i.e. S x K plus per-group padding ([API]).
- One cooperative launch, rank 0 plans and multicasts ([PLAN]); host still allocates buffers once at construction, so "no host involvement" holds per layer ([API]).
- K3 calls the plan near-optimal against offline ILP references ([K3] §5.2.1 "Online planning"); cost moved to Impact.

### Impact / Measured Results
**Before:** Removes the EP dispatch long tail and makes GroupedGEMM a fully static load, with no offline EPLB statistics; online planning overhead is about zero and routing drift is tracked batch by batch; static shapes give lower memory and copy overhead, are CUDA Graph friendly, and have no synchronization bubbles.
**After:** Every rank gets the same S x K tokens, removing the EP dispatch long tail without offline EPLB statistics; routing drift is tracked batch by batch. Per-rank GroupedGEMM totals are static, but per-expert sizes still vary. Planning is negligible (README), though it triggers replica-weight prefetch that adds to dispatch time; static shapes remove per-layer host syncs and the S x K x R worst-case buffer. README chart (H20, EP8): combine stays flat as imbalance grows, 1,954 vs 2,468 µs for DeepEP v2 at maxvio 20.
**Why:**
- "Fully static load" overstated: per-expert `cu_seqlens` change every micro-batch, and K3 adds a workload-aware expert-GEMM scheduler for the skew ([K3] §5.2.1).
- "About zero" overstated: the README says "negligible" for planning, but the weight prefetch it triggers is a visible segment of the README dispatch chart ([CHART-COMM]; estimated 0.5-0.75 ms, so no number given).
- "CUDA Graph friendly" is claimed by neither [README] nor [K3]; dropped. "No synchronization bubbles" narrowed to K3's "removes per-layer host sync".
- Added the one published comparison: combine fwd labels 1,954 vs 2,468 µs at maxvio 20 ([CHART-COMM]; H20, EP8 per [README]).

### Other properties
none

## 21 · NCCL Gin backend (GPU-initiated RDMA)

[deck](../decks/21-nccl-gin-backend-gpu-initiated-rdma/) · [Notion row](https://app.notion.com/p/50c0563d46924e00830d20a62dd6cbde)

### Core Mechanism
**Before:** Moves from NVSHMEM to NCCL Gin: GPU threads issue one-sided put / put-with-signal directly, header-only, reusing existing NCCL communicators with symmetric memory window registration; the whole library is JIT-compiled.
**After:** Moves from NVSHMEM to NCCL Gin (GDAKI backend required): GPU threads post one-sided put / put-with-signal (plus get, signal, flush) straight to the NIC through NCCL's header-only device API, reusing existing NCCL communicators (or creating one) with symmetric memory window registration; GPU kernels are JIT-compiled, the host extension is built at install.
**Why:**
- "Header-only" applies to NCCL's device API headers that the JIT kernels include, not to DeepEP as a whole ([R2] vs [R25]).
- Only GPU kernels are JIT-compiled; V2.5 "Installation builds the host C++ extension" ([R25]).
- DeepEP V2.5 also uses `get` (Engram), `signal` and `flush_async` ([DEV2]).
- V2.5 requests `NCCL_GIN_TYPE_GDAKI` and asserts if unavailable; the Proxy backend is not used ([CTX25]).
- Communicator reuse falls back to a DeepEP-created communicator when PyTorch lacks `_comm_ptr` ([HDL]).

### Impact / Measured Results
**Before:** Lighter dependency chain and simpler deployment, and the basis for 0-SM communication paths.
**After:** Lighter dependency chain and simpler deployment (NVSHMEM removed); GPU-posted RDMA carries the 0-SM-payload PP and Engram paths, but SMs still post the work and EP dispatch/combine still need SMs.
**Why:**
- V2.5: "EP dispatch and combine require GPU SMs; zero-SM RDMA EP is not supported" ([R25] Notes); Gin removes the CPU proxy, not the SM that posts the work ([GIN] §III-C).
- NVSHMEM "no longer a dependency" ([R25]; [PR763]).

### Other properties
none

## 22 · Network-side optimization: VL traffic isolation / adaptive routing / SM-free RDMA

[deck](../decks/22-network-side-optimization-vl-traffic-isolation/) · [Notion row](https://app.notion.com/p/c0f6478d3ece4c3c8cef0965065c4c0a)

### Core Mechanism
**Before:** InfiniBand virtual lanes (VL) isolate EP traffic from everything else and adaptive routing is enabled; community work (AntGroup) adds Normal-SMFree (decoupling comm-kernel execution from NIC transfer), LL-SBO (overlapping Down GEMM with combine sends) and LL-Layered (rail-optimized forwarding and merging).
**After:** DeepEP sets an InfiniBand service level per buffer so the fabric's SL-to-VL mapping can put EP traffic on its own virtual lane (V2: EP vs other; V1 used three classes), and V2 recommends adaptive routing under all loads (V1: heavy load only); community work (AntGroup, experimental branch) adds Normal-SMFree (open PR: decoupling comm-kernel execution from NIC transfer), LL-SBO (overlapping Down GEMM with combine sends) and LL-Layered (same-rail RDMA then NVLink forwarding).
**Why:**
- DeepEP only sets the SL (`sl_idx`, `EP_OVERRIDE_RDMA_SL`, `EP_DEFAULT_RDMA_SL`); the SL-to-VL mapping is fabric configuration ([RD] "Network configurations").
- V1 recommended three classes (normal, low-latency, other); V2 reduced it to EP vs other ([RD-V1a], [RD-V2]).
- AR advice changed: unsafe for normal kernels (Feb 2025), heavy load only (Mar 2025 to Feb 2026), all loads (V2) ([RD-V1a], [AR-FIX], [RD-V1b], [RD-V2]).
- Normal-SMFree is PR #347, still open against `antgroup-opt` ([PR347]); LL-Layered sends to the same-rail rank, then NVLink-forwards; "merging" is not described ([PR500]).

### Impact / Measured Results
**Before:** Stops EP all-to-all from colliding with DP/checkpoint traffic and returns RDMA-path SMs to compute.
**After:** Separates EP all-to-all from other traffic (e.g. DP, storage) on its own virtual lane; returning RDMA-path SMs to compute is only an open AntGroup PR, as mainline DeepEP V2.5 still needs SMs for RDMA EP.
**Why:**
- Mainline: "EP dispatch and combine require GPU SMs; zero-SM RDMA EP is not supported" ([RD] Notes); SM-free normal kernels exist only in unmerged PR #347 ([PR347]).
- "Stops colliding" softened: VLs share each link through an arbiter, and no source quantifies the collision cost (discrepancy item 5; [FF] §VI-A1).

### Other properties
none

Left for the user (page body, not touched): the body cites the Azure DeepEP blog for "adaptive routing, traffic isolation", but that post covers NUMA/HCA affinity and PCI atomics, not VLs or AR ([AZ]); the discussion sub-page's LL-SBO "+8-10%" is the LMSYS figure for BS 20-56, while PR #390's bs-32 runs show +6.7% / +7.5% ([LMSYS-ANT], [PR390]).

## 23 · Ascend adaptation stack (GLM-5)

[deck](../decks/23-ascend-adaptation-stack-glm-5/) · [Notion row](https://app.notion.com/p/3eda454335e04ef09f6a0b6991ee5608)

### Core Mechanism
**Before:** Container entry: GLM-5's day-one full-stack engineering for domestic GPU/NPU ecosystems, built on three pillars - aggressive quantization, high-performance fused operators, and inference-engine scheduling. Sub-techniques are the rows pointing here.
**After:** Container entry: GLM-5's full-stack deployment (inference) adaptation for seven domestic chip platforms, with Ascend Atlas as the case study, built on three pillars - aggressive quantization (W4A8 experts, W8A8 elsewhere), high-performance fused operators, and inference-engine scheduling. Sub-techniques are the rows pointing here.
**Why:**
- §5 is a deployment/inference adaptation; the report never says GLM-5 was trained on domestic chips, so the scope is stated ([GLM5] §5 p.21-22, §2.4 p.9; discrepancy item 5).
- "Day-one" is not in any source read; dropped.
- Seven platforms and Ascend Atlas "as a case study" ([GLM5] §1 p.4, §5 p.21); precision split from [GLM5] §5 p.21.

### Impact / Measured Results
**Before:** A single domestic GLM-5 node matches a dual-GPU cluster, with 50% lower long-sequence serving cost.
**After:** Report claims, without naming GPU, benchmark or metric: a single Chinese node performs comparably to dual-GPU international clusters, and long-sequence deployment cost falls 50%.
**Why:**
- The report's sentence is "comparable to dual-GPU international clusters" and gives no GPU model, node count, benchmark, metric or serving configuration ([GLM5] §5 p.22); restated as the report's claim, not a measurement.

### Other properties
none

Left for the user: the row's Source is "GLM-5 series, VeOmni". None of the sources read for this deck mentions VeOmni (ByteDance's training framework, deck 28); the discrepancy notes do not flag it, so it was not changed.

## 24 · DeepEP v2 (EP communication library)

[deck](../decks/24-deepep-v2-ep-communication-library/) · [Notion row](https://app.notion.com/p/d471baa2bcdf452592855568ee99f726)

### Core Mechanism
**Before:** Container entry: DeepSeek's expert-parallel communication library, rewritten in V2. Sub-techniques are the rows whose "Parent Library / Framework" points here: ElasticBuffer unified API, NCCL Gin backend, elastic GPU/CPU buffers, 0-SM PP/CP/Engram, FP8 dispatch + BF16 combine, and network-side traffic isolation.
**After:** Container entry: DeepSeek's expert-parallel communication library, rewritten in V2 (Apr 2026) and restructured in V2.5 (Sep 2026). Sub-techniques are the rows whose "Parent Library / Framework" points here: ElasticBuffer unified API (EPBuffer since V2.5), NCCL Gin backend, elastic GPU/CPU buffers (shipped for Engram storage only), 0-SM PP/CP/Engram (experimental), FP8 dispatch + BF16 combine, and network-side traffic isolation.
**Why:**
- V2.5 split ElasticBuffer into EPBuffer / EngramBuffer / PPBuffer / BucketBuffer ([R25] News; [PR763]).
- Elastic GPU/CPU buffers were "Still on-going" in V2; V2.5 ships Engram storage on GPU or CPU, not a hybrid buffer for EP ([R2]; [R25] News).
- PP / Engram / bucket collectives are marked experimental, and V2.5 drops the "0 SM" label ([R25] features; [PR763]).

### Impact / Measured Results
**Before:** Several times lower SM usage than V1 and scale-up/out domains up to EP2048; FP8 dispatch with IB virtual-lane isolation and adaptive routing.
**After:** Up to 4x fewer SMs than V1 (24 to 4-6 for V3-like training) at up to 1.3x peak performance; V2 README claims scale-up/out domains up to EP2048, but published benchmarks stop at EP32 (8x4); FP8 dispatch with IB virtual-lane isolation and adaptive routing.
**Why:**
- Replaced "several times" with the release's own numbers ([R2] New features and Performance: "24 to 4 - 6", "up to 1.3x peak performance, while saving up to 4x SM count").
- EP2048 is a V2 README statement; V2.5 no longer gives a number and the largest published configuration is EP 8x4 ([R2], [R25]; [R2] "We omit results for larger EP configurations").
- Adaptive-routing advice matches V2 ("under all network load conditions"); kept.

### Other properties
none

## 25 · DeepGEMM (unified tensor-core kernel library)

[deck](../decks/25-deepgemm-unified-tensor-core-kernel-library/) · [Notion row](https://app.notion.com/p/c0133413563f4ee390f9e4fb55027780)

### Core Mechanism
**Before:** Container entry: collects the core compute primitives of modern LLMs (GEMM, fused MoE, indexer scoring, HyperConnection) into a single CUDA codebase with very few kernels, all JIT-compiled at runtime. Sub-techniques are the rows pointing here.
**After:** Container entry: collects the core compute primitives of modern LLMs (FP8/FP4/BF16 GEMM, fused MoE, indexer scoring, HyperConnection, fused routing gate) into a single CUDA codebase of compact kernels (a few hundred lines per kernel family), all JIT-compiled at runtime through DeepJIT. Sub-techniques are the rows pointing here.
**Why:**
- "Very few kernels" no longer holds: 19 kernel implementation files, 9,593 lines of device code (S6.2, derived); each family is a few hundred lines (S2.1 "~300 lines" in 2025; 452 / 579 lines today, S6.3).
- The 2026-09 release adds Mega Gate (fused routing GEMM) and moves JIT into the separate DeepJIT (S1.1, S1.4, S3.4); GEMM precisions from S1.1.

### Impact / Measured Results
**Before:** Matches or beats expert-tuned libraries with a lightweight implementation; native support for variable per-expert token counts and fine-grained scaling.
**After:** Matches or exceeds expert-tuned libraries per the README, up to 1550 TFLOPS on H800; its published speedups are against DeepSeek's own CUTLASS 3.6-based kernels; native support for variable per-expert token counts and fine-grained scaling.
**Why:**
- The claim is the README's qualitative sentence (S1.2); the only published speedup baseline is "our internally and carefully optimized implementation based on CUTLASS 3.6" (S2.3), not a vendor library, so the baseline is named.
- 1550 TFLOPS on H800 from README news 2025.04.18 (S1.3).

### Other properties
none

## 26 · HPC-Ops (Tencent Hunyuan inference operator library)

[deck](../decks/26-hpc-ops/) · [Notion row](https://app.notion.com/p/85f1abe2180f4b86bf1e9ea6d23705ce)

### Core Mechanism
**Before:** Container entry: a production-grade CUDA operator library for LLM inference hot paths (attention, MoE, GEMM, sampling, normalization, comm/compute fusion), deeply tuned for SM90 (H20) and upstreamed into vLLM / SGLang. Sub-techniques are the rows pointing here.
**After:** Container entry: a production-grade CUDA operator library for LLM inference hot paths (attention, MoE, GEMM, sampling, normalization, comm/compute fusion), deeply tuned for SM90 (H20); attention, router GEMM and MoE kernels are upstreamed into SGLang main, and the API is designed for vLLM / SGLang integration. Sub-techniques are the rows pointing here.
**Why:**
- Upstreaming is documented only for SGLang (Attention #30540/#32304, Router GEMM #30247/#31943, MoE #30541; [L] integration table); for vLLM the README only says the API is "designed for seamless integration" and vLLM appears as a benchmark baseline ([R]).

### Impact / Measured Results
**Before:** In Tencent's large-scale serving: about 30% QPM gain for Hunyuan models and about 17% for DeepSeek-class models; 15.1-48.8% lower end-to-end TPOT for Hy3-FP8 on 8x H20.
**After:** Tencent-reported via MarkTechPost (not in the README): about 30% QPM gain for Hunyuan models and about 17% for DeepSeek models in production serving; LMSYS: 15.1-48.8% lower end-to-end TPOT for Hy3-FP8 on 8x H20 in SGLang at batch 4-64 (3.3% at batch 1).
**Why:**
- The QPM figures appear in no README version; their only source is MarkTechPost's release article quoting Tencent ([MTP]).
- The TPOT range covers batch 4-64 only; batch 1 is -3.3% ([L] Table 7, TP8, FP8 KV, 8K in / 4K out).

### Other properties
none

## 27 · MoonEP (perfectly balanced EP communication library)

[deck](../decks/27-moonep/) · [Notion row](https://app.notion.com/p/65b197b767bb4d5c9dc17bbce5c0ca13)

### Core Mechanism
**Before:** Container entry: Moonshot's MIT-licensed EP communication library released with Kimi K3, aiming for identical compute per rank under any routing skew. Sub-techniques: dynamic redundant-expert balancing, online GPU planning kernel, zero-copy static shapes.
**After:** Container entry: Moonshot's MIT-licensed EP communication library released with Kimi K3, giving every rank exactly S x K tokens under any routing skew. Sub-techniques: dynamic redundant-expert balancing, online GPU planning kernel, zero-copy static shapes.
**Why:**
- What is equalized is the per-rank token count (S x K, S = input tokens per rank), not compute: per-expert counts inside a rank stay skewed, so K3 adds a workload-aware expert-GEMM scheduler ([README] property 1; [K3] §5.2.1).

### Impact / Measured Results
**Before:** Static shapes and no per-layer host sync; communication buffer drops from a worst-case S x K x R to a fixed S x K; increasingly outperforms DeepEP v2 as routing imbalance grows on H20/EP=8.
**After:** Static shapes and no per-layer host sync; communication buffer drops from a worst-case S x K x R to S x K plus alignment padding; on H20/EP=8 the advantage over DeepEP v2 grows with routing imbalance (end-to-end iteration time 1.9% slower at maxvio 0.2, 11.1% faster at maxvio 15; DeepEP v2 OOMs at maxvio 20).
**Why:**
- Actual allocation is NvS = align(S·K + 127·2·E/R, 128): 77,824 vs 65,536 slots at the benchmark config ([API]).
- "Outperforms" is not true at low imbalance: MoonEP is 1.9% slower end-to-end at maxvio 0.2 and its dispatch fwd is 2,598 vs 2,296 µs ([CHART-E2E], [CHART-COMM]); added the published endpoints.

### Other properties
none

## 28 · VeOmni (model-centric omni-modal training framework)

[deck](../decks/28-veomni/) · [Notion row](https://app.notion.com/p/9b9145984b6a433abf69f32d2c06db67)

### Core Mechanism
**Before:** Container entry: ByteDance Seed's PyTorch-native omni-modal training framework; its core claim is decoupling parallelism/communication logic from model definitions, providing a composable distributed recipe zoo (FSDP1/2, Ulysses SP, EP) plus dynamic batching, DCP checkpointing and multiple hardware backends.
**After:** Container entry: ByteDance Seed's PyTorch-native omni-modal training framework; its core claim is decoupling parallelism/communication logic from model definitions, providing a composable distributed recipe zoo (FSDP2/HSDP, Ulysses SP, EP; the paper also used FSDP1, since removed) plus dynamic batching, checkpointing (ByteCheckpoint in the paper, PyTorch DCP in the repo) and multiple hardware backends.
**Why:**
- The paper integrates FSDP1 and FSDP2 ([VO] §3.2.1), but the repo says FSDP1 "has since been deprecated upstream and removed from VeOmni" ([DOC-EP]).
- The paper uses ByteCheckpoint ([VO] §3.2.5); the open-source repo uses PyTorch DCP ([README], [DOC-CKPT]).

### Impact / Measured Results
**Before:** More than 2800 tokens/s/GPU for a 30B omni-modal MoE on 128 GPUs with 160K context; adding a new modality takes very little code.
**After:** Over 2,800 tokens/s/GPU for a 30B omni-modal MoE on 128 GPUs at short context; the same model scales to 160K context with larger SP/EP at much lower throughput; adding a modality needs minimal code change (repo guide: five files for an omni-modal MoE).
**Why:**
- The paper states throughput and 160K context as two separate facts; Fig. 6(a) has the highest throughput at 16K (SP1/EP1) and reaches 160K only with larger SP/EP at much lower throughput ([VO] Abstract, §4.2, Fig. 6; same pattern in Table 4: 2,853 at 16K vs 1,075 tokens/s/GPU at 128K for text-only Qwen3-30B-A3B).
- "Very little code" is the paper's unquantified "minimal code change"; the repo guide lists `__init__.py` plus 4 more files for an omni-modal MoE ([DOC-NEW]).

### Other properties
none

## 29 · FP8 grouped GEMM (per-tensor / block-wise scaling)

[deck](../decks/29-fp8-grouped-gemm/) · [Notion row](https://app.notion.com/p/4b75d7ab1bfc40239d6b9566c5d6fc14)

### Core Mechanism
**Before:** FP8 grouped GEMM for EP and grouped-expert matmuls, accepting cu_num_tokens input with unequal per-group token counts and both quantization granularities.
**After:** unchanged
**Why:**
- No real error: the API arguments are `seqlens` / `cu_seqlens` ([API]), but the README Basic Usage example names the same tensor `cu_num_tokens_per_group` ([RD]); per-tensor and block-wise FP8 paths both exist ([API], [PT], [BW]).

### Impact / Measured Results
**Before:** vs DeepGEMM: 1.1x on prefill and 1.88x on decode (H20).
**After:** vs DeepGEMM on H20: up to 1.1x on prefill and 1.88x on decode (maximum observed speedups as published in the HPC-Ops README).
**Why:**
- The README states both figures as maxima ("Up to" in the current Performance table, "Maximum observed speedup per operator" in the January 2026 version); no per-shape data is published ([RD], [RD-JAN]).

### Other properties
none (the page body's LMSYS link note "prefill/decode comparison vs DeepGEMM" is inaccurate, but the body is out of scope)

## 30 · FP8/FP4/BF16 dense GEMM + UE8M0 scaling layout

[deck](../decks/30-fp8-fp4-bf16-dense-gemm-ue8m0-scaling-layout/) · [Notion row](https://app.notion.com/p/eedb72bc30e14aa8ba49b28b4344d0b9)

### Core Mechanism
**Before:** fp8_gemm_{nt,nn,tn,tt}; SM90 supports only the NT layout with FP32 scale factors, while SM100 supports all layouts with 4x UE8M0 packed into an int; LHS scale factors must be TMA-aligned and transposed.
**After:** unchanged
**Why:**
- Matches the README word for word (S1.1, S1.2); the `fp8_gemm_*` names are kept as aliases of `fp8_fp4_gemm_*` (S3.5). Omitting the two SM90 kernels (1d1d / 1d2d, S3.1) and SM100's gran_k 32 (S2.1) is harmless.

### Impact / Measured Results
**Before:** Peaked at 1550 TFLOPS on H800, matching or beating expert-tuned libraries with a minimal implementation.
**After:** Peaked at 1550 TFLOPS on H800 (April 2025 release note); per-shape H800 results are 1.1x to 2.7x over DeepSeek's internal tuned CUTLASS 3.6 implementation, with a core kernel of about 300 lines.
**Why:**
- 1550 TFLOPS is the README's peak across shapes (S1.4, news 2025-04-18); kept, dated.
- The only measured comparison is against DeepSeek's own CUTLASS 3.6-based implementation on H800 SXM5 (1.1x to 2.7x, S6); "expert-tuned libraries" in general is not what was measured, so the baseline is named.
- "Minimal implementation" made concrete as the README's "about 300 lines" core kernel (S6).

### Other properties
none (title keeps "BF16": DeepGEMM does expose `bf16_gemm_*`, though it usually dispatches to cuBLASLt, S3.4)

## 31 · Fused FP8 MoE low-latency pipeline

[deck](../decks/31-fused-fp8-moe-low-latency-pipeline/) · [Notion row](https://app.notion.com/p/4d0436aff16240398bbe1c41ab56d5ad)

### Core Mechanism
**Before:** Fuses routing, Gate-Up GEMM, activation quantization, Down GEMM and top-k weighted reduction into a single pipeline; Gate-Up reads original tokens by routing index to drop the gather stage; shared-memory counting relieves global atomics; the low-latency mode trades warp specialization for CTA residency and chains stages with PDL to cut launch bubbles.
**After:** Chains routing count/index, Gate-Up GEMM, activation + FP8 quantization, Down GEMM and top-k weighted reduction as five PDL-linked stages (6 to 7 kernels, HBM intermediates); in the low-latency cp.async path (per-rank intermediate at most 512), Gate-Up reads original tokens by routing index instead of a gathered copy, shared-memory counting replaces per-pair global atomics, and one warpgroup per CTA (no warp specialization) raises CTA residency.
**Why:**
- "Single pipeline" read as one kernel: it is five PDL-chained stages, the count/index stage itself 2 or 3 kernels, with intermediates in HBM ([CPF], [CNT], [EN] allocations).
- The gather-free, no-warp-specialization path is only dispatched when the per-rank intermediate is at most 512 and a multiple of 64; otherwise the TMA warp-specialized path with a gather runs ([EN], [DEF]).
- Shared-memory counting removes the default path's per-(token, expert) global atomic, leaving one per expert per block ([CNT], [DEF]).

### Impact / Measured Results
**Before:** Up to 1.6x (TP) and 1.5x (EP) over vLLM CUTLASS, Triton and SGLang on DeepSeek-V3 / Hunyuan-V3 / Qwen3-235B.
**After:** H20: up to 1.6x (TP8) and 1.5x (EP8) vs the median of vLLM CUTLASS, vLLM Triton and SGLang; vs the best baseline per batch (Hy3): mean 1.08x / 1.21x, max 1.18x / 1.60x; EP8 uses the TMA path, not the low-latency one.
**Why:**
- PR #49 states the 1.6x / 1.5x relative to the median of the compared backends, not each of them ([PR49]); hardware H20 ([PR49]).
- Against the best baseline per batch, LMSYS Tables 4-5 (Hy3, H20) give mean 1.08x (TP8/EP1) and 1.21x (TP1/EP8), maxima 1.18x and 1.60x ([LMSYS]).
- EP8 mode keeps the full intermediate (1536 for Hy3) per rank, above the 512 cp.async limit, so those numbers come from the TMA path ([BR] + [EN]; derived item 3 in sources.md).
- Model list dropped to fit the caveats; the three models remain in the README row ([RD]).

### Other properties
none

## 32 · K-axis grouped GEMM (MoE weight backward)

[deck](../decks/32-k-axis-grouped-gemm-moe-weight-backward/) · [Notion row](https://app.notion.com/p/56c25e6b3cbe4425b251e2b59530b802)

### Core Mechanism
**Before:** k_grouped_fp8_gemm_tn_contiguous keeps M/N fixed and groups along K, purpose-built for MoE weight gradients (wgrad), with a matching K-grouped UE8M0 packing kernel.
**After:** k_grouped_fp8_gemm_tn_contiguous (SM100) and k_grouped_fp8_gemm_nt_contiguous (SM90) keep M/N fixed and group along K (each expert's tokens, segments aligned to 128), purpose-built for MoE weight gradients (wgrad); SM100 packs the scales to UE8M0 with a K-grouped packing kernel, SM90 keeps FP32 scales.
**Why:**
- `k_grouped_fp8_gemm_tn_contiguous` is SM100-only; Hopper has used `k_grouped_fp8_gemm_nt_contiguous` since Nov 2025 ([API] l.320-427, [V3-SM90]).
- The UE8M0 packing kernel applies to SM100 only; SM90 takes FP32 scales with gran_k = k_alignment = 128 ([LAYOUT] l.89-116, [RD] l.162).
- K is the token dimension, each expert's segment aligned to BLOCK_K = 128 ([SCHED] l.40-43, l.63-82).

### Impact / Measured Results
**Before:** Puts the MoE backward pass on the FP8 fine-grained-scaling path as well - critical for training, unlike inference-only operator libraries.
**After:** Puts MoE weight-gradient GEMMs on the FP8 fine-grained-scaling path as well; published kernel numbers cover only the May 2025 per-expert loop (900 to 1044 TFLOPS on Hopper, 4 experts); the Sept 2026 release reports about 20% faster MoE backward GEMMs.
**Why:**
- The kernel covers weight gradients, not the whole backward pass (Dgrad uses the M-grouped kernels, [RD] l.78).
- "Unlike inference-only operator libraries" has no support in the primary sources; dropped. "Critical for training" dropped as unsourced framing.
- Added the only published numbers, with their scope: PR #95 grouped wgrad table, a host loop over experts, kernel time only, GPU unnamed but Hopper-only at the time ([PR95], [V1-API], [V1-TEST] l.294); the single-kernel versions have only "about 20%" from the Sept 2026 release ([PR432]).

### Other properties
none

## 33 · M-axis grouped GEMM (contiguous layout)

[deck](../decks/33-m-axis-grouped-gemm-contiguous-layout/) · [Notion row](https://app.notion.com/p/3e1e63852a934fad8b5d53773913bce5)

### Core Mechanism
**Before:** Unlike CUTLASS generic grouped GEMM, only the M axis is grouped while N/K stay fixed (MoE experts share a shape); per-expert tokens are concatenated into one tensor with each segment aligned to the GEMM M block.
**After:** unchanged
**Why:**
- Matches the README ([RD] l.76-80). The CUTLASS comparison is the README's own framing; the measured baseline was DeepSeek's internal CUTLASS 3.6-based implementation ([RD-0325] l.10), but the row states no number against it, so it does not mislead.

### Impact / Measured Results
**Before:** Fits training forward and prefill, where per-expert token counts vary, without padding waste.
**After:** Fits training forward and prefill, where per-expert token counts vary; each expert segment is padded only up to the M alignment (128 rows by default), not to a fixed capacity.
**Why:**
- The layout does pad: every expert segment is rounded up to the M alignment (128 by default, 256 theoretical on SM100) with -1 / zero rows ([RD] l.78, [HEUR] l.12, l.57-64, [GEN] l.396-406). What it avoids is capacity-factor padding of every expert to a fixed size ([MB] §3 p.4).

### Other properties
none

## 34 · Masked grouped GEMM (CUDA Graph friendly)

[deck](../decks/34-masked-grouped-gemm-cuda-graph-friendly/) · [Notion row](https://app.notion.com/p/d8af0edc72e64994bbb502dc9a3bfcea)

### Core Mechanism
**Before:** Under decode plus CUDA Graph the CPU cannot know per-expert token counts, so a mask tensor tells the kernel to compute only the valid parts - and it consumes DeepEP low-latency kernel output directly.
**After:** Under decode plus CUDA Graph the CPU cannot know per-expert token counts, so a device-side mask (masked_m) tells the kernel to compute only the valid rows of each expert's fixed-capacity slot; it consumes DeepEP V1 low-latency dispatch output directly (DeepEP V2.5 decode instead uses the contiguous kernel's psum layout).
**Why:**
- Mask semantics from the API and scheduler: `masked_m[g]` on device, A shaped per expert at `max_m` capacity, M tiles computed from device memory ([OLD-API] l.114-136, [SCHED] l.188-204).
- The direct pairing is with DeepEP V1's `low_latency_dispatch` ([EP1-BUF]); V2.5 removed the V1 API and its decode path emits an expanded layout with GPU-side prefix sums, consumed by the contiguous kernel's psum layout ([EP2] l.19, l.163; [TEST] l.203-205).

### Impact / Measured Results
**Before:** Removes the per-MoE-layer host sync, a prerequisite for enabling CUDA Graphs in EP inference and rollout.
**After:** Together with DeepEP low-latency dispatch, removes the per-MoE-layer host sync, a prerequisite for enabling CUDA Graphs in EP decoding (inference and rollout).
**Why:**
- The sync is removed by the pair: normal dispatch still blocks the CPU and is not graph-compatible; the GEMM alone only removes the need for host-side shapes ([EP1] l.173, l.265, l.281; [LMSYS] l.163-188).
- Scoped to decoding, the only use the sources describe ([RD] l.82-86); rollout kept as decoding, with no number claimed (no source measures it).

### Other properties
none

## 35 · Mega MoE fused MegaKernel

[deck](../decks/35-mega-moe-fused-megakernel/) · [Notion row](https://app.notion.com/p/77562a16a085453aa50e5744e3382f7e)

### Core Mechanism
**Before:** Fuses EP dispatch, linear1 (FP8 x FP4 / FP8 x FP8), SwiGLU, linear2 and EP combine into one mega-kernel, using symmetric memory so NVLink communication overlaps Tensor-Core compute inside a single kernel.
**After:** unchanged
**Why:**
- Matches the current README and PR #304 ([RD] lines 112-114, [PR304]). FP8 x FP8 was added after the first FP8 x FP4-only release, but the current README lists both, so the row is right as of now ([RD], [API]).

### Impact / Measured Results
**Before:** Eliminates kernel boundaries and intermediate buffer write-backs; single-stream overlap is friendlier to numerical reproducibility than dual-stream or micro-batch splitting.
**After:** Removes kernel boundaries and the BF16 linear1 output round trip (the FP8 post-SwiGLU activation still goes through an HBM ring buffer); 1.50x to 1.96x over the five-kernel path (V4-Flash and V4-Pro, EP8, 1 to 32768 tokens per rank); in deterministic mode the test asserts bitwise-equal output to the non-fused path.
**Why:**
- Not all intermediate write-backs go away: the L1 epilogue TMA-stores FP8 activations into `l2_token_buffer` in global memory, and the test's HBM model counts "L1 output write" and "L2 acts read" ([K] lines 1066-1255, [TEST]). What disappears is the BF16 [n, 2I] linear-1 output and its re-read by a separate SwiGLU kernel ([TEST] `run_baseline`).
- The reproducibility comparison with dual-stream / micro-batch overlap appears in no primary source; replaced with what the sources show: `torch.equal(fused_y, baseline_y)` under `use_deterministic_algorithms(True)` ([TEST], [PR462]; determinism is opt-in since [PR432]).
- Added the published speedup range: eight points of PR #316, 1.50x (V4-Pro, 8192 tokens/rank) to 1.96x (V4-Flash, 1 token/rank) vs the five-kernel DeepEP + DeepGEMM + TileLang path ([PR316], [TEST] at 891d57b4).

### Other properties
none

## 36 · Route GEMM (BF16 x FP32 via dual-BF16 split)

[deck](../decks/36-route-gemm-bf16-x-fp32-via-dual-bf16-split/) · [Notion row](https://app.notion.com/p/1ca14bb9a0ce4d439cc9a0686b60a02d)

### Core Mechanism
**Before:** Splits FP32 weights into a high BF16 term plus a low residual BF16 term (fixed scale 1/256) and fuses two BF16 Tensor-Core GEMMs in one kernel with shared input gathering, register-resident accumulators and a single result write.
**After:** Splits FP32 weights into a high BF16 term plus a low residual BF16 term (fixed scale 1/256) and fuses two BF16 Tensor-Core GEMMs in one kernel that loads each input tile once for both, keeps both accumulators in registers and writes the result once.
**Why:**
- There is no gather: one TMA load of each X tile per K step feeds both WGMMAs (W_low and W_high), two register accumulators, one epilogue `acc_low * scale + acc_high` and one TMA store ([KER] lines 242-364).

### Impact / Measured Results
**Before:** Up to 3.22x over cuBLAS FP32/TF32 while keeping FP32 accuracy; suits precision-sensitive small matrices such as MoE router GEMMs and the state-compress GEMMs of linear/sparse attention.
**After:** Up to 3.22x over FP32 cuBLAS and 1.78x over TF32 cuBLAS (H20, K=4096, N=192) at close-to-FP32 accuracy (max abs error 0.00177 vs 0.06464 for TF32); suits precision-sensitive small matrices such as MoE router GEMMs and the state-compress GEMMs of linear/sparse attention.
**Why:**
- 3.22x is over FP32 cuBLAS only (M=512); over TF32 cuBLAS the maximum is 1.78x (M=96), range 1.25x to 1.78x ([LM] Table 2 and operator-benchmark paragraph; [RM] lists both baselines with one number).
- "Keeping FP32 accuracy" overstates: the split keeps about 16 of FP32's 24 significand bits, and the measured max abs error vs the FP32 cuBLAS reference is 0.00177, not zero; TF32 gives 0.06464 on the same sweep ([LM] Highlights; analytic bound in sources.md). The blog's own wording is "precision close to a full FP32 GEMM".
- Use cases kept: MoE router and sparse/linear-attention state-compress GEMMs are in the README Operator Catalog and Performance rows ([RM]).

### Other properties
none

## 37 · Ascend · super-operator fusion (MLAPO / Sparse FA / MoE Mega-Fusion)

[deck](../decks/37-ascend-super-operator-fusion/) · [Notion row](https://app.notion.com/p/5926e0ea48474805988cb4d1419bac18)

### Core Mechanism
**Before:** Coarse-grained fusion tuned to domestic NPU memory behavior: Lightning Indexer fuses score+ReLU+TopK into one op, Sparse Flash Attention fuses sparse indexing with attention, MLAPO collapses 13 small ops into one super-operator, and MoE routing + permute + GroupedGEMM become a Mega-Fusion.
**After:** GLM-5's Ascend kernel fusion: Lightning Indexer fuses score + ReLU + TopK into one kernel so the NPU overlaps compute with memory access; Sparse Flash Attention does TopK selection from the KV cache and sparse attention in one kernel; MLAPO fuses 13 small pre-processing ops into one super-operator that runs Vector and Cube units in parallel. GLM-5 reports no MoE fusion (vLLM-Ascend separately fuses grouped matmul + SwiGLU + quant).
**Why:**
- The three kernels and their stated purposes are the report's own descriptions ([GLM5] §5 p.21-22); "tuned to domestic NPU memory behavior" replaced by the stated benefits (compute/memory overlap for the indexer, Vector/Cube parallelism for MLAPO).
- "MoE routing + permute + GroupedGEMM become a Mega-Fusion" is not in the GLM-5 report, which names only these three kernels ([GLM5] §5). The Ascend stack's MoE fusion that does exist is grouped matmul + SwiGLU + quantization in vLLM-Ascend ([VA-MOE]), not attributed to GLM-5; Pangu Ultra MoE's permute and GroupedMatMul fusions are for a different model's training ([PANGU] §4.5).

### Impact / Measured Results
**Before:** Sharply reduces kernel-launch and intermediate memory traffic - the key to domestic hardware matching GPUs.
**After:** No kernel-level numbers published; GLM-5 credits quantization, fusion and scheduling jointly for one Chinese-chip node comparable to dual-GPU international clusters, at 50% lower long-sequence deployment cost.
**Why:**
- The report publishes no launch counts, memory-traffic numbers or per-kernel speedups ([GLM5] §5); the "sharply reduces" claim has no measurement behind it.
- The only end-to-end claim (single node comparable to "dual-GPU international clusters", 50% lower deployment cost in long-sequence scenarios) is attributed to all hardware-level co-optimizations together, not to fusion ([GLM5] §5 p.22 last paragraph); the GPU setup is not specified in the report.

### Other properties
none changed. Title still lists "MoE Mega-Fusion", which the GLM-5 report does not describe (its third kernel is the Lightning Indexer); left for the user to decide.

## 38 · CUDA Graphs (full-iteration / per-layer / TE graphed callables)

[deck](../decks/38-cuda-graphs/) · [Notion row](https://app.notion.com/p/7d97c06866b54afdb590597a7dc39756)

### Core Mechanism
**Before:** --cuda-graph-impl offers three capture-and-replay modes: full_iteration (FullCudaGraphWrapper), local (per-layer CudaGraphManager) and transformer_engine (make_graphed_callables, FP8- and PP-schedule-aware).
**After:** --cuda-graph-impl offers three capture-and-replay modes: full_iteration (FullCudaGraphWrapper: forward and backward of all microbatches, optimizer stays eager), local (per-layer CudaGraphManager, captured in recorded order) and transformer_engine (make_graphed_callables, captured ahead of time from the PP schedule's order); both per-layer modes handle FP8 and PP.
**Why:**
- FP8 and PP awareness is not specific to the TE path: the MCore `local` CudaGraphManager also handles FP8 (deferred amax, weight caching) and PP (recorded order); TE differs by capturing ahead of time from an explicit `_order` ([BG] FP8/PP challenges and comparison table; [CG] `_CudagraphGlobalRecord`, `TECudaGraphHelper`).
- full_iteration scope: forward and backward of all microbatches with their communication; optimizer, LR schedule and logging stay eager ([DOC]; [FCG]; [EX-LLAMA]).

### Impact / Measured Results
**Before:** Removes kernel-launch overhead in small-kernel-heavy regimes; standard in tuned DeepSeek-V3 configurations on GB200.
**After:** Cuts kernel-launch overhead in small-kernel-heavy regimes, for the captured regions only; partial graphs gave 10% end-to-end (about 7 GB extra memory) in DeepSeek-V3 GB200 training and are enabled in its final GB200 config.
**Why:**
- Launch overhead is removed only inside graphs: per-layer modes leave dispatch, expert GEMMs, combine and inter-graph code eager for dropless MoE, and full_iteration leaves the optimizer eager ([DOC] MoE scopes; [MC] p.40-41).
- "Standard in tuned DeepSeek-V3 configurations" replaced by what the paper reports: 10% end-to-end at about 7 GB extra memory from partial graphs with memory optimizations ([MC] §4.3.6 p.42), and CUDA graphs enabled in the final 256-GB200 configuration but not in the H100 one ([MC] Table 17 p.74, App. B.2 p.87).

### Other properties
none

## 39 · DeepJIT runtime compilation + PDL + determinism switches

[deck](../decks/39-deepjit-runtime-compilation-pdl-determinism/) · [Notion row](https://app.notion.com/p/1ebac796b72b428f9c32afc18e7b5fd4)

No change: the row has no real error, only omissions.

### Core Mechanism
**Before:** All kernels are JIT-compiled at runtime (no CUDA compilation at install time) through a low-CPU-overhead C++ JIT module with on-disk cache; runtime knobs include set_num_sms / set_tc_util / set_pdl and use_deterministic_algorithms.
**After:** unchanged
**Why:**
- Every statement is supported: runtime-only compilation (S4.4, S1.1), the low-CPU-overhead C++ JIT module (S3.12, PR #112) with an on-disk cache (S1.4), and the four knobs (S3.11). Since 2026-09 the module is the separate DeepJIT project (S3.12, S1.1); the title already names DeepJIT, so not saying so here is harmless.

### Impact / Measured Results
**Before:** Per-shape configuration and SM budget caps let GEMMs coexist with communication kernels; determinism switches serve reproducible training.
**After:** unchanged
**Why:**
- Supported: per-shape compile-time constants and deterministic config selection (S3.2-S3.5, S4.1); `set_num_sms` caps the SM count, a compile-time constant (S3.2, S3.6); `use_deterministic_algorithms` exists for reproducibility (S3.9, S5.1, S6.1).
- Omitted but not wrong: determinism has been opt-in (default False) since the 2026-09 release (S6.1). The row says "switches", which does not claim a default.

### Other properties
none

## 40 · Fused Sampler (decode post-processing fusion)

[deck](../decks/40-fused-sampler-decode-post-processing-fusion/) · [Notion row](https://app.notion.com/p/3070daaed63740e3be901cba2641e152)

### Core Mechanism
**Before:** Folds repetition penalty, temperature scaling, Top-K, Top-P, softmax, random sampling and penalty-mask updates into two kernels (temperature-only cases take an even lighter path); small top-k uses local heap/merge selection, and top-k reduction is merged with softmax statistics to cut vocabulary passes.
**After:** Folds repetition penalty, temperature scaling, Top-K, Top-P, softmax, random sampling and penalty-mask updates into two kernels (the temperature-only path is a single kernel); a block radix sort keeps a rolling top-K (bounded by max_topk 32 or 64) per block, and softmax statistics are gathered in the same pass to cut vocabulary passes.
**Why:**
- The temperature-only path is one kernel whose last-arriving block reduces the partials ([KT] header, l.84+), not just a "lighter path".
- Top-k selection is not a heap/merge: keeper threads hold a rolling top-K that each chunk merges into with `cub::BlockRadixSort` ([K] l.91-113, l.226-263; PR #51 "BlockRadixSort top-k path"). The README wording "local heap/merge" does not match the code.
- The candidate count is bounded by the compile-time `max_topk` (32 or 64) ([API], [H]).
- Online max/sum in K1 lets K2 form full-vocab probabilities without a second vocabulary pass ([K] l.265-288, l.329-579); kept, reworded.

### Impact / Measured Results
**Before:** Up to 8.5x over vLLM-style PyTorch / FlashInfer at vocab 120832 and batch 1-512 (largest win in small-batch RL rollout).
**After:** 4.6x to 8.5x over vLLM-style PyTorch and 2.1x to 5.2x over FlashInfer, at vocab 120832 (BF16 logits) and batch 1-512.
**Why:**
- 8.5x is the maximum over the vLLM-style PyTorch path only; over FlashInfer PR #51 reports 2.1x to 5.2x ([PR51] Benchmark line; [RM] Performance row).
- "Largest win in small-batch RL rollout" has no source: nothing mentions RL rollout or the batch size at which the maximum occurs ([RM] only says fragmented kernels hurt small-batch serving); dropped.
- BF16 logits from the benchmark config ([BEN]); the GPU is not stated in PR #51, so none is named.

### Other properties
none (the page body's LMSYS link is annotated "sampling benchmark results", but the blog has no sampler content, and the FlashInfer paper covers attention only; the body is out of scope)

## 41 · Mega Gate / Mega mHC fused kernels

[deck](../decks/41-mega-gate-mega-mhc-fused-kernels/) · [Notion row](https://app.notion.com/p/0bb7a1acbf924329aab698ccc8c5ffee)

### Core Mechanism
**Before:** Fuses router gating (Mega Gate) and mHC / HyperConnection residual mixing (Mega mHC) each into a single kernel, avoiding the swarm of small ops and intermediate tensors these new architectures introduce.
**After:** Fuses router gating (Mega Gate: BF16 gate GEMM, Sqrt(Softplus) scoring and top-k) and mHC residual mixing (Mega mHC: post-mix, 24-wide projection, Sinkhorn and RMSNorm, replacing a 4- to 7-kernel chain) each into one SM100 kernel, removing kernel boundaries and some stream re-reads; the new residual and split-K partials still pass through memory.
**Why:**
- "Avoiding ... intermediate tensors" overstates: Mega mHC still writes the new residual (an output), materialises the norm boundary and, in normal mode, re-reads the new residual (kept in L2 by EVICT_LAST); split-K partials go to a workspace, and Mega Gate passes scores through a workspace when experts or K are split ([MHC-K] lines 342-343, 570-655; [MHC-T] byte model; [GATE-K]).
- What the kernels fuse, from the code: the gate GEMM with sqrtsoftplus scoring, bias, top-k ([GATE-API], [GATE-K], [PR462]); post-mix, the 24-output TF32 projection, Sinkhorn and RMSNorm, replacing the 4-kernel (normal) or 7-kernel (shifted) baseline ([MHC-T] ref_mhc.py NORMAL_KERNELS / SHIFTED_KERNELS, [MHC-API]).
- Both kernels are SM100-only ([MHC-K], [GATE-K], tests filtered on arch 10).

### Impact / Measured Results
**Before:** Brings mHC's extra memory traffic into an acceptable range - a key enabler for shipping new residual structures to production.
**After:** Mega mHC is 45-85% faster than the multi-kernel TileLang mHC path and Mega Gate 30-75% faster than cuBLASLt GEMM + TileLang top-k (DeepGEMM, Sept 2026); mHC's 6.7% training overhead came earlier, from other fused kernels plus recomputation.
**Why:**
- The quantified "acceptable range" (6.7% overhead at n = 4) was reached by the mHC paper's TileLang kernels plus recomputation and DualPipe changes in training ([MHC] §1 p.4, §4.3; [V4] §3.4.2 p.20), before Mega mHC existed; crediting it to Mega mHC is a wrong attribution.
- DeepGEMM's only numbers for these kernels: Mega mHC 45-85% and Mega Gate 30-75% faster than the multi-kernel baselines ([PR432]; baselines from [MHC-T] and [GATE-T]).
- "Key enabler for shipping ... to production" has no source; dropped.

### Other properties
none changed. Source lists "DeepSeek V4": the V4 report describes fused mHC kernels but does not name Mega Gate or Mega mHC (those come from DeepGEMM PR #432); not plainly wrong, so left for the user to decide.

## 42 · Router / permutation / MLA-RoPE fused kernels

[deck](../decks/42-router-permutation-mla-rope-fused-kernels/) · [Notion row](https://app.notion.com/p/9d82ddbf31244093bc930a84ed679c2d)

### Core Mechanism
**Before:** --moe-router-fusion fuses routing score/topk/bias, layered with permutation, MLA RoPE and FP8-cast fusion, plus cuDNN fused attention / FlashAttention integration.
**After:** --moe-router-fusion fuses score function, (group) top-k and expert bias into one kernel, with separate fused aux-loss kernels; --moe-permute-fusion builds the row-id map once and fuses permute/unpermute with padding and FP8-scale permutation; MLA RoPE fusion applies YaRN RoPE to Q and KV in Triton kernels.
**Why:**
- cuDNN fused attention is a separate feature in the README list, not one of these glue-kernel fusions, and no source read mentions a FlashAttention integration; dropped ([RD]; discrepancy 1).
- Router: one kernel for score function + top-k (naive or group) + bias + scaling, plus aux-loss-score and aux-loss kernels under a separate flag ([MC] §4.3.4 p.37-38, [TE-R], [TE-1883], [MLM-CFG] `moe_router_aux_loss_fusion`).
- Permutation: row-id map built once in three passes and reused; padding and FP8 scales handled inside the permute kernel ([MC] §4.3.3 p.36-37, §5.4.1; [TE-P] `FUSION_PAD`, `PERMUTE_SCALE`; [TE-1921]).
- MLA RoPE: Triton Q (in place) and KV-split kernels with a cached YaRN cos/sin table; YaRN only ([MLM-ROPE], [MLM-CFG] `apply_rope_fusion`).

### Impact / Measured Results
**Before:** Cuts the launch and memory-traffic overhead of many small kernels; pairs with sync-free dropless MoE for higher compute efficiency.
**After:** Cuts the launch and memory-traffic overhead of many small kernels; only permute+pad fusion has published gains (1.1x to 1.6x permute, 1.7x to 3x unpermute kernels; +0.4% MFU, about 1 GB less peak memory in FP8 training of a 600B-class MoE).
**Why:**
- No isolated number is published for router fusion or MLA-RoPE fusion; the paper says only that the fusions "reduce kernel launch count" in the GB200 configuration ([MC] §9.2.2 p.75). The measured numbers are for permute+pad fusion ([TE-1921] kernel benchmarks and end-to-end, about 600B-parameter FP8 Megatron setup).
- "Pairs with sync-free dropless MoE for higher compute efficiency" is not stated for these kernels in any source read; the sync-free fused expert kernels of [BLOG-F] are GroupGEMM epilogues, a different fusion. Dropped.

### Other properties
none

## 43 · TileLang DSL + batch-invariant deterministic kernels

[deck](../decks/43-tilelang-dsl-batch-invariant-deterministic/) · [Notion row](https://app.notion.com/p/22c7e2d5f253481981afb6c6ee06e3f5)

### Core Mechanism
**Before:** Kernels written in the TileLang domain-specific language replace hundreds of fine-grained ATen ops; batch-invariant attention uses two kernels (full waves on a single SM, tail wave via distributed shared memory) to preserve accumulation order; the backward pass replaces atomicAdd with per-SM accumulation buckets and deterministic reduction.
**After:** Fused kernels written in the TileLang DSL replace most of the hundreds of fine-grained ATen ops; batch-invariant decoding attention forbids split-KV and uses two kernels (full waves on a single SM, tail wave spread over SMs via distributed shared memory, same accumulation order); DeepGEMM replaces cuBLAS and drops split-k in most GEMMs; deterministic backward: per-SM buffers plus deterministic summation for sparse-attention KV gradients, token-order pre-processing plus per-rank buffer isolation for MoE, separate split outputs plus deterministic reduction for the mHC GEMM.
**Why:**
- TileLang kernels replace "most" of the fine-grained ATen operators the architecture would need; the paper gives no exact count ([V4] §3.2 p.16).
- The dual-kernel strategy is for batch-invariant decoding attention and exists because split-KV is not batch-invariant ([V4] §3.3 p.18).
- Batch-invariant GEMMs come from replacing cuBLAS with DeepGEMM and abandoning split-k in most scenarios ([V4] §3.3 p.18-19).
- Per-SM accumulation buffers apply only to the sparse-attention backward; MoE backward and the mHC GEMM use different fixes ([V4] §3.3 p.19).

### Impact / Measured Results
**Before:** Outputs are independent of batch position and parallel strategy and consistent between training and inference, sharply reducing irreproducibility in RL and debugging.
**After:** A token's output is bitwise identical regardless of its position in the batch, aiming at bitwise alignment among pre-training, post-training and inference; V4 names debugging and loss-spike analysis as benefits but reports no measurement of the effect or of the overhead.
**Why:**
- [V4] defines batch invariance as independence from the token's position within a batch; it makes no claim about invariance across parallel strategies ([V4] §3.3 p.18).
- Bitwise alignment among pre-training, post-training and inference is the stated goal; debugging and stability analysis are the stated benefits ([V4] §3.3 p.18-19).
- "Sharply reducing irreproducibility" is not measured in [V4]; overhead is only described as "minimal" / "negligible" ([V4] §3.3 p.18). The 80-of-1000 figure on the slides is Thinking Machines' Qwen3/vLLM measurement ([TM]), not a DeepSeek result, so it is not used here.

### Other properties
none

## 44 · Ascend · W4A8/W8A8 hybrid quantization stack

[deck](../decks/44-ascend-w4a8-w8a8-hybrid-quantization-stack/) · [Notion row](https://app.notion.com/p/38d6d033f1b943b3b79615c26f4f0a2e)

### Core Mechanism
**Before:** QuaRot rotation preprocessing removes activation outliers, Flex SmoothQuant searches per-layer migration strength, SSZ (scale-shift-zero) handles weight quantization; W4A8 for MoE experts, W8A8 kept for sensitive layers.
**After:** In msModelSlim, QuaRot rotation suppresses activation outliers and Flex_AWQ_SSZ calibrates scaling (named but not defined in the GLM-5 report); MoE experts use W4A8 (INT4 weights, INT8 activations) and standard attention and MLP blocks W8A8, a structural split since experts hold about 97% of weights (derived).
**Why:**
- The report names Flex_AWQ_SSZ for scaling calibration; SmoothQuant does not appear in it ([GLM5] §5 p.21; lineage in [AWQ] Eq. 5, [SQ] §4).
- "SSZ (scale-shift-zero)" is an interpretation; the report never defines SSZ, so it is not presented as a separate weight-quantization step ([GLM5] §5 p.21).
- The precision split is by block type ("standard Attention and MLP blocks use W8A8 (INT8), while the MoE experts are compressed to W4A8 (INT4)"), not by layer sensitivity ([GLM5] §5 p.21).
- 97% = 724.8B routed-expert parameters of 744B, derived from [GLM5] Table 10 (sources.md derivation).

### Impact / Measured Results
**Before:** Halves weight memory at near-lossless accuracy, with large throughput gains on domestic chips.
**After:** No published accuracy, memory or throughput numbers, only 'without significant accuracy loss'; weights about half of all-INT8 (derived); the 50% long-sequence cost cut is for all Ascend co-optimizations.
**Why:**
- [GLM5] §5 gives no accuracy, memory or throughput figures for the quantized model; "near-lossless" and "large throughput gains" are unsupported.
- Hybrid weights 381.6 GB vs 744 GB all-INT8 = 1.95x, derived from Table 10; "halves" holds only against INT8 (BF16 is 3.9x larger).
- The only end-to-end number, 50% lower deployment cost in long-sequence scenarios, is credited to all Ascend co-optimizations together ([GLM5] §5 p.22).

### Other properties
none

## 45 · FP4 QAT (MXFP4 expert weights + indexer QK path)

[deck](../decks/45-fp4-qat-mxfp4-expert-weights-indexer-qk-path/) · [Notion row](https://app.notion.com/p/ad31c31051744ea2b53ea97c1417ac71)

### Core Mechanism
**Before:** Post-training MXFP4 QAT on MoE expert weights and the CSA indexer's QK path; stored as FP4 (E2M1) and dequantized to FP8 (E4M3) for compute (FP8's extra exponent bits make dequantization mathematically lossless), backward uses STE to update FP32 master weights, index scores drop from FP32 to BF16.
**After:** Post-training MXFP4 QAT on MoE expert weights and the CSA indexer's QK path; FP32 master weights are quantized to FP4 (E2M1, 1x32 sub-blocks) and dequantized to FP8 (E4M3, 128x128 blocks) so the FP8 training stack runs unmodified, lossless only while FP4 sub-block scale ratios inside each FP8 block stay under a threshold (verified empirically); gradients go straight to the FP32 master weights (STE); index scores drop from FP32 to BF16.
**Why:**
- Losslessness is conditional on the max/min ratio of FP4 sub-block scales within a 128x128 FP8 block staying under a threshold, which DeepSeek verified empirically for current weights; "mathematically lossless" overstates it ([V4] §5.2.1 p.33).
- Purpose of the FP4-to-FP8 path is to reuse the existing FP8 training framework unmodified; the backward pass computes gradients w.r.t. the FP8 weights and passes them to the FP32 master, equivalent to an STE ([V4] §5.2.1 p.33-34).

### Impact / Measured Results
**Before:** 2x faster top-k selector with KV recall held at 99.7%; deployment runs the FP4 weights directly, so training and serving behavior match.
**After:** Quantizing index scores from FP32 to BF16 within QAT makes the top-k selector 2x faster while recalling 99.7% of KV entries; inference and RL rollouts run native FP4 weights, so sampling matches online deployment.
**Why:**
- [V4] ties the 2x / 99.7% to the FP32-to-BF16 index-score quantization, not to FP4 QAT as a whole ([V4] §5.2.1 p.33).
- Native FP4 weights are stated for the inference and RL-rollout phases, so that sampling matches online deployment ([V4] §5.2.1 p.33-34); wording made precise.

### Other properties
none

## 46 · FP8 training (per-tensor / blockwise) and FP8 activation compression

[deck](../decks/46-fp8-training-and-fp8-activation-compression/) · [Notion row](https://app.notion.com/p/8406549e9fd645e49d2f2e90a838f1bd)

### Core Mechanism
**Before:** TE provides DelayedScaling / blockwise FP8 recipes with automatic autocast; MCore supports FP8 parameter all-gather and fine-grained recompute under FP8; K2 compresses activations in FP8-E4M3.
**After:** TE offers per-tensor (delayed or current scaling) and blockwise FP8 recipes under autocast; blockwise uses 1x128 tiles for activations and gradients, 128x128 blocks for weights; MCore casts FP32 master weights straight to FP8 for the parameter all-gather; K2 caches MoE up-projection and SwiGLU inputs in FP8-E4M3 but computes in BF16.
**Why:**
- TE also ships Float8CurrentScaling, and Megatron-Core does not recommend DelayedScaling for precision reasons ([TE-primer], [TE-cur], [MCore-MoE] Sec. 5.3.1).
- Blockwise granularity from [TE-blk] "Block size" and [DSV3] Sec. 3.3.2.
- FP8 parameter all-gather casts FP32 master shards directly to FP8, eliminating the BF16 copy ([MCore-MoE] Sec. 5.2.2).
- "Fine-grained recompute under FP8" is not in the sources read; dropped.
- K2 compresses activations for storage only and explicitly does not compute in FP8 ([K2] Sec. 2.4.3); the old wording under "FP8 training" implied FP8 GEMMs.

### Impact / Measured Results
**Before:** FP8 parameter all-gather cuts roughly 50% of that communication; activation memory halved - the default recipe for DeepSeek-V3-class models on H100.
**After:** FP8 parameter all-gather moves 1 instead of 2 bytes per parameter (by design); FP8 caching halves each eligible tensor, saving about 16 of 131 GB of DeepSeek-V3 activations per GPU; MCore recommends blockwise FP8 on Hopper, validated on DeepSeek-V3.
**Why:**
- The 50% all-gather saving is a byte count (1 vs 2 B/param), not a measured result ([MCore-MoE] Sec. 5.2.2).
- Only FP8-eligible linear-layer inputs are halved; for DeepSeek-V3 that is about 16 GB of 131 GB, about 12% ([MCore-MoE] Sec. 4.1.3, Table 3). K2 reports no memory number ([K2] Sec. 2.4.3).
- Megatron-Core recommends blockwise FP8 on Hopper and validated it on DeepSeek-V3 ([MCore-README] "FP8 Recipes"); DeepSeek-V3 itself trained on H800, so "default recipe on H100" was replaced ([DSV3]).

### Other properties
none

## 47 · MXFP8 micro-scaling training

[deck](../decks/47-mxfp8-micro-scaling-training/) · [Notion row](https://app.notion.com/p/02c9479032474c7e8e387fb2e6b7a964)

### Core Mechanism
**Before:** Every 32 contiguous values share one E8M0 (pure power-of-two) scale, with E4M3 for weights/activations/gradients; natively supported by Blackwell Tensor Cores, requires keeping both transposed and non-transposed copies.
**After:** Every 32 contiguous values share one E8M0 (pure power-of-two) scale, with E4M3 for weights, activations and gradients; Blackwell Tensor Cores apply the scales natively; each tensor keeps rowwise (1x32) and columnwise (32x1) copies, both quantized from high precision, so parameters are all-gathered in BF16.
**Why:**
- The two copies are not transposes of each other: 1x32 and 32x1 blocks are numerically different and each is quantized from the high-precision tensor; Blackwell needs no physical transpose ([TE-mx] "Handling transposes"; [MX] Sec. 3.2.1).
- Because the copies cannot be derived from each other, MXFP8 parameters are all-gathered in BF16 ([MCore-MoE] Sec. 5.2.2).

### Impact / Measured Results
**Before:** 1.28-1.37x over BF16 on B200 (up to 2x on GEMMs), loss curves indistinguishable from BF16, and covers more layers than per-tensor FP8.
**After:** Within 0.50% of BF16 validation perplexity (8B model, 15T tokens, MX simulated on Hopper); 1.22x BF16 tokens/s for DeepSeek-V3 and Qwen3-235B on 256 GB200 (derived) and 1.44x GEMM speedup on B300; quantizes every transformer block, unlike per-tensor FP8.
**Why:**
- "1.28-1.37x over BF16 on B200" appears in none of the sources read; the MXFP8 paper reports no measured speedup. Replaced by 4,020/3,298 and 6,212/5,100 tokens/s/GPU = 1.22x ([MCore-MoE] Table 11, derived; each precision with its own best parallel config) and the 5B-model GEMM speedup of 1.44x on B300 ([TE-speed]).
- "Up to 2x on GEMMs" is the peak Tensor Core math rate vs BF16 ([MX] Table 1), not a measurement; dropped.
- "Indistinguishable" is quantified in the paper as under 0.50% validation-perplexity difference, with MX numerics simulated on 3,072 Hopper GPUs ([MX] Sec. 3.1).
- More layers: MXFP8 covers all transformer blocks while the per-tensor FP8 baseline keeps the first and last in BF16 ([MX-v1] Sec. 3.1, [MX] Sec. 3.2.1); kept.

### Other properties
none (the discrepancy notes that Megatron-LM / MCore is missing from Source; that is an omission, not a wrong value, so Source was left unchanged and the question is left to the user)

## 48 · NVFP4 pretraining recipe

[deck](../decks/48-nvfp4-pretraining-recipe/) · [Notion row](https://app.notion.com/p/eae9504538664595866784566f5d14cd)

### Core Mechanism
**Before:** E2M1 with two-dimensional block scaling; stochastic rounding removes gradient-quantization bias, Random Hadamard transforms spread outliers, and sensitive layers stay high precision (bf16_with_nvfp4_mixed).
**After:** E2M1 values with an E4M3 scale per 16-value block plus an FP32 tensor scale; 16x16 2D scaling for weights only; stochastic rounding on gradients removes quantization bias; 16x16 random Hadamard transforms on Wgrad inputs only spread outliers; about 15% of linear layers, mostly the last blocks, stay BF16 (12B: first 2 and last 8 of 62 blocks).
**Why:**
- The NVFP4 format is 1x16 E4M3 block scales plus an FP32 tensor scale; 2D (16x16) scaling is a recipe choice for weights only ([NV4] Sec. 2, Sec. 4.3; [TE-nv4]).
- RHT is applied only to Wgrad inputs; Fprop/Dgrad transforms gave no benefit or hurt ([NV4] Sec. 4.2, App. E.4.1).
- SR is for gradients; SR on forward tensors is detrimental ([NV4] Sec. 4.4, App. E.3).
- `bf16_with_nvfp4_mixed` only switches GEMMs to NVFP4; which layers stay BF16 is set separately ([MB]). Replaced with the paper's rule and the 12B configuration ([NV4] Sec. 4, App. A.1).

### Impact / Measured Results
**Before:** <1% loss gap vs FP8 over 10T tokens on a 12B model (~1.5% late in training); a further 1.31-1.73x over FP8 on GB200/GB300.
**After:** Under 1% loss gap vs blockwise FP8 over 10T tokens on a 12B model (slightly above 1.5% after LR decay); NVFP4 GEMMs run 2.03x over BF16 on B300 (5B model), 1.41x over MXFP8 (derived).
**Why:**
- Loss gap kept ([NV4] Sec. 3, Fig. 2); the baseline is the blockwise 1x128 / 128x128 FP8 recipe ([NV4] App. A.1).
- "1.31-1.73x over FP8 on GB200/GB300" appears in none of the sources read; the paper reports no runtime. Replaced with the measured TE GEMM speedup on B300 and the derived 2.03 / 1.44 = 1.41x over MXFP8 ([TE-speed]).

### Other properties
none

## 49 · Precision-aware optimizer (BF16/FP8 optimizer states)

[deck](../decks/49-precision-aware-optimizer/) · [Notion row](https://app.notion.com/p/893ccbb87b354e7e8fedbc0a18ca7117)

### Core Mechanism
**Before:** --use-precision-aware-optimizer lets TE optimizers keep first/second moments and master weights in BF16/FP8, mixing precision as needed.
**After:** --use-precision-aware-optimizer lets TE FusedAdam store first/second moments in BF16 or FP8 (master weights FP32 or FP16) and upcast them to FP32 inside the kernel for the update; with BF16 params the FP32 master can be kept as a 16-bit remainder, still exact FP32.
**Why:**
- Master weights cannot be BF16 or FP8: `--main-params-dtype` accepts only fp32 or fp16 and FusedAdam supports FP32/FP16 masters; only the moments accept bf16/fp8 ([ARGS], [TEA]).
- Storage precision is decoupled from compute: states are cast to FP32 inside FusedAdam and the update runs in FP32 ([MC] §4.1.6; [ARGS] help text).
- The remainder option stores the 16 low bits of the FP32 master next to the BF16 weight, which reconstructs exact FP32 ([OC] `store_param_remainders`, [TEK]).

### Impact / Measured Results
**Before:** Large reduction in optimizer-state memory; standard for trillion-parameter MoEs under limited HBM.
**After:** BF16 moments cut optimizer-state memory about 50%: 10-12 GB of DeepSeek-V3's 32.1 GB of master weights and states per GPU, no performance overhead; used in DeepSeek-V3 training and MCore's recipes.
**Why:**
- "Large reduction" quantified: about 50% of optimizer-state memory, 10-12 GB of 32.1 GB for DeepSeek-V3 on 256 GPUs, with "no performance overhead" ([MC] §4.1.6 p.26-27, Table 3).
- No trillion-parameter run is documented in the sources read; the documented uses are MCore's DeepSeek-V3 (685B) final configs ([MC] Table 18) and DeepSeek-V3's own training with BF16 moments ([DSV3] §3.3.3).

### Other properties
none

## 50 · Distributed checkpointing and elastic-resource training

[deck](../decks/50-distributed-checkpointing-and-elastic-resource/) · [Notion row](https://app.notion.com/p/a065908961604600811834b0c1800744)

### Core Mechanism
**Before:** VeOmni uses Torch Distributed Checkpoint for parallelism-agnostic save/load; K2 designs variable parallelism (16-way PP + 16-way EP + ZeRO-1 DP) for clusters whose resources change over time.
**After:** VeOmni's open-source repo saves weights and optimizer state with Torch Distributed Checkpoint, each rank writing its own shards, and loads reshard into a new layout; K2 fixes a 256-GPU model-parallel unit (16-way PP x 16-way EP) and varies only ZeRO-1 DP, so it can train on any multiple of 32 nodes as resources change.
**Why:**
- DCP is what the VeOmni repo uses ([V-CKPT], [V-SRC]); the VeOmni paper describes ByteCheckpoint instead ([VO] §3.2.5), so the claim is scoped to the repo.
- DCP mechanism: per-rank shard files plus global metadata, load-time resharding ([PTD]).
- K2's parallelism is not "variable" as a whole: PP and EP stay fixed (256 GPUs = 32 nodes x 8 GPUs, derived) and only ZeRO-1 DP scales with node count; "any number of nodes that is a multiple of 32" ([K2] §2.4.1-2.4.2).

### Impact / Measured Results
**Before:** Changing parallelism or hardware needs no checkpoint rewrite; training continues through node-count fluctuation.
**After:** Weights and optimizer state reload under a new parallel layout with no checkpoint rewrite, but per-rank data cursors and RNG state do not reshard; K2 trains on any multiple of 32 nodes (resizing a running job is not described).
**Why:**
- "No checkpoint rewrite" holds for weights and optimizer state; a job whose new ranks lack cursor files resumes weights only ([V-GS]), and DCP gives no format guarantee across PyTorch versions ([PTD]).
- K2 describes training on any multiple of 32 nodes under dynamic resource availability, not a job that keeps running through node-count changes ([K2] §2.4.2).

### Other properties
none

## 51 · Dynamic batching / sequence packing

[deck](../decks/51-dynamic-batching-sequence-packing/) · [Notion row](https://app.notion.com/p/5cfd3adb77f74e0c8d09435a3af5d9b3)

### Core Mechanism
**Before:** Buffers samples and packs them to a target sequence length instead of uniform padding; the MCore counterpart is the THD packed format (e.g. GLM-5.2's 128K THD packed CP training).
**After:** Buffers samples and packs each micro-batch to a token budget instead of uniform padding, passing cu_seqlens so attention stays per sample; the MCore counterpart is the THD packed format; GLM-5 adds sequence reordering and variable-size CP groups for long sequences.
**Why:**
- The packing target is a token budget per micro-batch ([VO] §3.2.5; [V-BATCH] `get_samples`; [V-PACK]); boundaries are passed to varlen attention as `cu_seq_lens` ([V-PACK], [V-COLL]).
- THD is Megatron-Core's packed format ([MC] §6.4.1); kept.
- The GLM-5 report mentions no THD format, no "GLM-5.2" and no 128K THD packed CP run; it describes workload-aware sequence reordering and flexible CP groups of varying sizes for long sequences ([GLM5] §2.4.2). Replaced the example with what the report says.

### Impact / Measured Results
**Before:** Large gains in effective FLOP utilization for multimodal and variable-length data.
**After:** Megatron-Core reports 1.5-2x throughput and 40-60% less memory vs padding (RL thinking models); no VeOmni measurement isolates packing.
**Why:**
- The only quantified packing gains in the sources are Megatron-Core's, for RL with thinking models ([MC] §6.4.1 p.60); VeOmni publishes no packing-only measurement ([VO]).

### Other properties
none (Source keeps "GLM-5 series": the report does cover variable-length long-sequence CP scheduling, though not THD packing; left for the user to decide)

## 52 · Fine-grained selective recomputation

[deck](../decks/52-fine-grained-selective-recomputation/) · [Notion row](https://app.notion.com/p/2e31e16f18ce45edba1e3d9058756c08)

### Core Mechanism
**Before:** --recompute-granularity selective with --recompute-modules selects mla_up_proj / moe / moe_act / layernorm / norm per module, and works under FP8.
**After:** --recompute-granularity selective with --recompute-modules picks per module (layernorm, mla_up_proj, moe_act, mlp, moe, core_attn, shared_experts); the first three discard their outputs and rebuild them before backward, while moe re-runs experts and all-to-alls; FP8 works with TE 2.6+ and no delayed scaling.
**Why:**
- There is no `norm` module; the supported list is `core_attn, moe_act, layernorm, mla_up_proj, mlp, moe, shared_experts` (plus newer ones) ([CFG] `recompute_modules` docstring; [RD]).
- layernorm, mla_up_proj and moe_act use the output-discarding checkpoint (output storage freed, recomputed from a grad hook) ([RNG], [TL], [MLA], [EXP]; [MC] §4.1.4 p.23); `moe` uses standard checkpointing and replays the experts and their all-to-alls ([CFG]).
- FP8 support (MCore v0.14) has constraints: TE 2.6.0dev0 or newer for moe_act/layernorm, delayed scaling rejected ([CFG] `__post_init__`; [RD]).

### Impact / Measured Results
**Before:** Cuts activation memory for negligible extra compute, avoiding the 30%+ overhead of full-layer recomputation.
**After:** Saves 42.4 GB per GPU on DeepSeek-V3 (layernorm, mla_up_proj, moe_act; 256 GPUs) for under 5% extra compute, vs about 33% for full-layer recomputation.
**Why:**
- "Negligible" is quantified in the paper as "less than 5%" ([MC] §4.1.4 p.23); full recomputation is "~33%" (p.22).
- 42.4 GB = 30.4 (MLA up-proj) + 8.2 (LayerNorm) + 3.8 (SwiGLU), DeepSeek-V3, PP4 x VPP4 x EP64 ([MC] Table 4 p.24).

### Other properties
none

## 53 · Memory-efficient token permutation

[deck](../decks/53-memory-efficient-token-permutation/) · [Notion row](https://app.notion.com/p/3a91e54ff5c7400e9b2e65e0ba6479f5)

### Core Mechanism
**Before:** Algebraic reordering: moves the router-probability multiply out of unpermute and into the GroupedMLP activation function, avoiding an extra large buffer allocation.
**After:** Algebraic reordering: since fc2 is linear and bias-free, the router-probability multiply moves out of unpermute into each expert's activation function (fused weighted SwiGLU), so the router-weight gradient needs only the SwiGLU input already saved, not the expert output.
**Why:**
- The equivalence (Eq. 1 = Eq. 2) holds for bias-free experts; with Eq. 2 the probability gradient needs phi(z), recomputed from z, which SwiGLU saves anyway ([MC] §4.1.2 p.21; [SW]).
- Current code applies the probs inside the activation of every expert backend, not only GroupedMLP ([EXP]: TEGroupedMLP, SequentialMLP via `per_token_scale`).
- The saving is a tensor no longer retained for backward, not an avoided buffer allocation ([MC] §4.1.2).

### Impact / Measured Results
**Before:** Removes one full activation copy - a classic low-cost fix for the MoE memory wall.
**After:** Expert outputs are no longer retained for backward: about 26.3 GB per GPU on DeepSeek-V3 (about 20% of activations, derived), with no trade-off listed.
**Why:**
- What is removed is the expert output kept for the router-weight gradient (one hidden-size row per routed token copy per MoE layer), not a copy of all activations; measured at about 26.3 GB per GPU ([MC] §4.1.2 p.21); 26.3 / 131.0 GB = 20% is derived from [MC] Table 3.
- [MC] Table 6 lists no trade-off for the technique. (The code adds a small separate all-to-all for the probs, [TD]; not stated in the row.)

### Other properties
none (the discrepancy notes that the Reference, the MoE README, carries no numbers while all numbers come from arXiv 2603.07685; the README does describe the technique, so the Reference is not plainly wrong and was left unchanged for the user to decide)

## 54 · Pipeline-aware activation / optimizer offload

[deck](../decks/54-pipeline-aware-activation-optimizer-offload/) · [Notion row](https://app.notion.com/p/431ab65669934bd2af65da396bb96f7f)

### Core Mechanism
**Before:** --fine-grained-activation-offloading offloads activations to CPU per pipeline stage and --offload-optimizer-states offloads optimizer state; K2/K3 likewise combine CPU activation offload with selective recomputation to hold utilization.
**After:** --fine-grained-activation-offloading copies chosen modules' saved inputs to CPU on a side stream after forward and reloads them one layer early in backward, ordered by the 1F1B/VPP schedule (later PP ranks offload less); --offload-optimizer-states (dev branch, now --chunked-optimizer-state-offload) keeps optimizer states and master weights on CPU between steps; K2/K3 combine offload with recomputation and FP8 storage.
**Why:**
- Offload granularity is per module (each listed submodule's saved input), with one handler per micro-batch x VPP chunk; PP awareness is the ordering and the per-rank amount, not "per pipeline stage" ([MC] §4.1.5 p.24-25; [DOC]; [FGO]).
- `--offload-optimizer-states` is in the paper (Table 13) and on the dev branch as a deprecated alias of `--chunked-optimizer-state-offload`; it keeps exp_avg, exp_avg_sq and master weights on CPU between steps while the step stays on GPU ([MC] p.26; [ARG] dev; [OCFG] dev). Main's `--optimizer-cpu-offload` is a different mechanism (CPU step).
- K2 and K3 also compress activations (K2 FP8 storage; K3 block-wise FP8 and remote offload to other PP ranks) ([K2] §2.4.3; [K3] §5.2.2). "To hold utilization" is not stated as such; dropped.

### Impact / Measured Results
**Before:** Sustains stable trillion-scale MoE training on H800 clusters with 2TB of host memory per node.
**After:** Megatron-Core: 10-18% less memory for 1.6-2% throughput, or 15% more Qwen3-235B throughput via TP1/EP64; optimizer offload frees 15-20 GB per GPU; Kimi K2 offloads all remaining activations (no measurement).
**Why:**
- The H800 / 2 TB host RAM facts are Kimi K2's cluster ([K2] §2.4.1), but K2 does not attribute training stability to offload and reports no offload measurement ([K2] §2.4.3).
- Measured results come from Megatron-Core: 10-18% memory for 1.6-2% throughput ([MC] §4.1.5 p.25); Qwen3-235B 800 to 920 TF/s, +15% ([MC] Table 5); optimizer-state offload 15-20 GB per GPU on DeepSeek-V3 ([MC] §4.1.6 p.26-27).

### Other properties
none

## 55 · Loss-free balance routing + sigmoid gating

[deck](../decks/55-loss-free-balance-routing-sigmoid-gating/) · [Notion row](https://app.notion.com/p/efa722c88bed4a4cb1baa6d8d0a0d6fa)

### Core Mechanism
**Before:** A learnable per-expert bias with fixed update rate regulates load instead of an auxiliary loss; scoring uses sigmoid.
**After:** A per-expert bias, used only for top-K selection, is moved by a fixed sign step after each batch from token counts (no gradient) instead of a tuned auxiliary loss, though production keeps a 1e-4 sequence-wise loss; scoring uses sigmoid.
**Why:**
- The bias is not learned by back-propagation: it is updated by b_i += u·sign(e_i) from per-expert token counts ([LF] Algorithm 1 p.4; [V3] p.9) and stored as a non-gradient buffer in MCore ([CODE] `expert_bias`).
- The bias affects only which experts are picked; gate weights use the unbiased scores ([LF] Eq. 3; [V3] Eq. 16).
- DeepSeek-V3, GLM-4.5 and DeepSeek-V4 all keep a complementary sequence-wise balance loss with weight 0.0001 ([V3] p.23; [GLM] §2.4; [V4] p.25-26); the original paper uses none.

### Impact / Measured Results
**Before:** Now the default routing setup for frontier open MoEs; MCore ships this router and its fused kernels.
**After:** Adopted by DeepSeek-V3/V4 and GLM-4.5; MCore ships this router and its fused kernels as opt-in (defaults: no expert bias, softmax).
**Why:**
- In Megatron-Core it is not a default: `--moe-router-enable-expert-bias` defaults to False and `--moe-router-score-function` to softmax ([RD] Router Arguments table).
- "Frontier open MoEs" narrowed to the models checked in the sources ([V3], [V4], [GLM]); fused router from MCore v0.14 ([RD]; [MC] §4.3.4).

### Other properties
none. For the user to decide (not changed, since neither value is plainly wrong):
- Reference is GLM-4.5 (arXiv 2508.06471), which adopts the technique in one sentence plus hyper-parameters; the method and controlled experiments are in DeepSeek's arXiv 2408.15664 and DeepSeek-V3 §4.5.2.
- Source lists "DeepSeek V3.2", which the deck did not read; the deck relies on DeepSeek-V3 and DeepSeek-V4.

## 56 · Quantile Balancing

[deck](../decks/56-quantile-balancing/) · [Notion row](https://app.notion.com/p/b59b1fb31a654c95a2957772ad9e4d92)

### Core Mechanism
**Before:** Adjusts the routing distribution directly from quantile statistics instead of aux-loss-free bias updates (unstable at ~1k experts); needs no auxiliary loss.
**After:** Each step sets every expert's selection bias to minus the (1 - k/n)-quantile of its routing margins against each token's Top-(k+1) cutoff, estimated by histogram plus one all-reduce, replacing sign-step bias updates K3 calls not well behaved near 1,000 experts; mixture weights exclude the bias.
**Why:**
- QB sets the per-expert selection bias; the mixture weights p_ij exclude the bias, so the routing weights are unchanged and only which experts are selected changes ([K3] Eq. 13 p.8).
- Mechanism: Top-(k+1) cutoff per token, bias = minus the (1 - k/n)-quantile of margins, mean-centred, applied next step ([K3] Eq. 14 p.9, App. C); quantile from per-expert histograms with one integer all-reduce per layer per step ([K3] App. D p.45).
- "Unstable at ~1k experts": K3 only says nearly 10^3 experts "exceeds the regime in which existing auxiliary-loss-free bias updates remain well behaved" and publishes no measurement ([K3] p.6-7); wording softened to K3's claim.
- "Needs no auxiliary loss" is not stated in the sources read; dropped.

### Impact / Measured Results
**Before:** Keeps load balanced at ~10^3 experts, avoids expert under-training and directly improves EP training throughput.
**After:** K3 balances 896 experts per layer with no step size, reporting no measurable residual imbalance and histogram traffic under 1% of raw-margin exchange; no QB ablation or throughput number is published (EP micro-batch balance comes from MoonEP).
**Why:**
- 896 routed experts, 16 per token ([K3] §2.3, Table 1); "no measurable residual load imbalance" and "below 1%" of the cost of exchanging raw margins ([K3] App. D p.45).
- K3 gives no throughput number for QB and attributes per-micro-batch EP balance and its throughput to MoonEP ([K3] §5.2.1); "directly improves EP training throughput" removed.
- "Avoids expert under-training" is not stated as a QB result in the sources read; dropped.

### Other properties
none

## 57 · Stable LatentMoE

[deck](../decks/57-stable-latentmoe/) · [Notion row](https://app.notion.com/p/fafccf6521d44c569d9d5cfad419049f)

### Core Mechanism
**Before:** Experts down-project into a latent space (an MLA analogue), with SiTU-GLU to suppress activation blow-up and RMSNorm on routed experts for numerical stability.
**After:** A shared W-down projects each token from 7,168 to a 3,584 latent before dispatch and a shared W-up restores width after combine (from NVIDIA LatentMoE); RMSNorm on the weighted sum of routed outputs before W-up and SiTU-GLU (smoothly capped GLU) in every FFN suppress activation blow-up.
**Why:**
- The down-projection is one shared W-down applied before dispatch, not done by each expert ([K3] Eq. 11 p.7; [K3-cfg] `routed_expert_down_proj` 7168 to 3584, `routed_expert_up_proj`).
- K3 cites LatentMoE (arXiv 2601.18089) as the origin of the latent path and never calls it an MLA analogue; credited LatentMoE instead ([K3] §2.3; [LMoE] §3).
- RMSNorm is applied once to the aggregated routed latent u, not per routed expert ([K3] Eq. 11, §2.3.1 p.7; [K3-cfg] `routed_expert_norm`).
- SiTU-GLU described as a smooth cap on both GLU factors ([K3] §2.3.2, App. B Eq. 18-19).

### Impact / Measured Results
**Before:** Trains stably with 16 of 896 routed experts active per token (\~56x sparsity); the key enabler of K3's width scaling.
**After:** K3 (2.8T total, 104B active) trains with 16 of 896 routed experts active per token (sparsity 56) at K2's per-token all-to-all volume (derived); no isolated ablation of the RMSNorm or SiTU-GLU stabilizers is published.
**Why:**
- "Trains stably" and "key enabler" are inferences: K3 publishes no loss or activation curves isolating the stabilizers, and its 2.5x scaling-efficiency gain is credited jointly to KDA, AttnRes, Stable LatentMoE and recipes ([K3] §2.3.1, §3.2, Fig. 7).
- Sizes from [K3] abstract and Table 1; 896 / 16 = 56 exactly, so no "about".
- Equal all-to-all volume: 16 x 3,584 = 8 x 7,168 (derived, sources.md "2x all-to-all"; [LMoE] §2.2).

### Other properties
none (the page body lists DeepSeek-V2/MLA but not LatentMoE, the actual base design; body left untouched per brief)

## 58 · Ultra-sparse MoE (fine-grained routed experts + shared expert)

[deck](../decks/58-ultra-sparse-moe/) · [Notion row](https://app.notion.com/p/b2283eba9cf14a8fb2fa3f7b2ba4127f)

### Core Mechanism
**Before:** DeepSeekMoE-style fine-grained expert pool plus one shared expert; total and activated parameters fully decoupled, with sparsity chosen by scaling laws.
**After:** DeepSeekMoE-style pool of hundreds of narrow routed experts (6 or 8 active) plus one always-on shared expert; total parameters grow apart from per-token FLOPs, bounded by memory and EP all-to-all cost; only Kimi K2 publishes a sparsity scaling law, and stops at sparsity 48 for infra cost.
**Why:**
- "Fully decoupled" overstates: K2 stops at sparsity 48 because of "increased infrastructure complexity" ([K2] p.7), K2 needs about 6 TB for weights + grads and the smallest EP is 16 ([K2] §2.4.2), and GLM-5 cuts its layer count to 80 to limit EP communication ([GLM5] §2.1 p.4).
- "Sparsity chosen by scaling laws": only K2 publishes a sparsity law; V4 and GLM-5 give configs without a sweep ([K2] p.7 + Fig. 5; [V4] §4.2.1; [GLM5] Table 10).
- Pool shape from configs: 384 or 256 routed, top-6 (V4) or top-8 (K2, GLM-5), 1 shared ([V4] §4.2.1; [K2] Table 2; [GLM5] Table 10).

### Impact / Measured Results
**Before:** V4-Pro 1.6T/49B active, V4-Flash 284B/13B; K2 1.04T/32B; GLM-5 744B/40B.
**After:** unchanged
**Why:** all sizes match the sources ([V4] Abstract, §4.2.1; [GLM5] Table 10). K2's 32B is the abstract's rounding of Table 2's 32.6B ([K2] abstract, Table 2), so not an error.

### Other properties
none

## 59 · Attention Residuals (AttnRes / Block AttnRes)

[deck](../decks/59-attention-residuals-attnres-block-attnres/) · [Notion row](https://app.notion.com/p/8b86f64fd2d44d4bac732d2321f7722a)

### Core Mechanism
**Before:** Per-layer learnable pseudo-queries aggregate the embedding and all preceding layer outputs with softmax-attention weighting instead of equal-weight residual addition; the Block variant groups layers to cut memory from O(Ld) to O(Nd), with two-stage computation and cache-based pipelined communication.
**After:** unchanged
**Why:** every statement is supported: pseudo-query softmax over the embedding and preceding outputs ([AR] Eq. 1-4, §3); Block AttnRes memory O(Ld) to O(Nd) ([AR] §3.2); two-phase inference ([AR] Alg. 1, §4.2); cross-stage caching under pipeline parallelism ([AR] §4.1, Fig. 3).

### Impact / Measured Results
**Before:** Relieves PreNorm depth dilution - output magnitude and gradients stay uniform with depth; improves every task on Kimi Linear 48B / 1.4T tokens.
**After:** Relieves PreNorm depth dilution: output magnitudes stay bounded and gradients become substantially more uniform with depth; on Kimi Linear 48B / 1.4T tokens it matches or beats the baseline on all 15 benchmarks (MMLU-Pro tied at 52.2, GPQA-Diamond 36.9 to 44.4).
**Why:**
- Magnitudes are bounded with a periodic pattern within blocks, and gradients are "substantially more uniform", not uniform ([AR] §5.2, Fig. 5).
- "Improves every task" is contradicted by Table 3 (MMLU-Pro 52.2 vs 52.2); the paper's own text says "matches or outperforms the baseline on all benchmarks" ([AR] §5.2, Table 3 p.10).
- GPQA-Diamond added as the largest published gain, to keep a concrete number ([AR] Table 3; +7.5 stated in §5.2).

### Other properties
none (Source "Kimi K3" kept: K3 adopts Block AttnRes with 8 x 12-layer blocks, [K3] §2.2, Table 1; the measured results come from the AttnRes report, which the Reference already points to)

## 60 · Deep-and-narrow + more attention heads + partial RoPE

[deck](../decks/60-deep-and-narrow-more-attention-heads-partial/) · [Notion row](https://app.notion.com/p/ba32c8b828d748ea9f32a65e15598576)

### Core Mechanism
**Before:** Narrower than DeepSeek-V3/K2 (hidden dim, number of routed experts) but deeper (89 MoE layers); GQA with partial RoPE and 2.5x more heads (96 heads at 5120 hidden).
**After:** unchanged
**Why:** all statements match the sources: hidden 5120 vs 7168, 160 vs 256/384 routed experts, 3 dense + 89 MoE layers, GQA 96 Q / 8 KV heads with partial RoPE ([GLM45] §2.1, Table 1; [CFG-GLM45]). "2.5x more heads" is the paper's own published figure ([GLM45] §2.1 p.3); the derived 2.4x against a width-matched 40-head layout is a refinement, not a correction, so the published number is kept.

### Impact / Measured Results
**Before:** Deeper models reason better; more heads do not lower training loss yet keep improving MMLU/BBH.
**After:** GLM-4.5 authors report, without published ablation numbers, that deeper models reason better and that more heads do not lower training loss yet consistently improve MMLU/BBH.
**Why:**
- Both claims are the authors' qualitative findings; [GLM45] §2.1 p.2-3 publishes no numbers for either ablation, so they should not read as measured results.
- "Keep improving" changed to the paper's wording "consistently improves" ([GLM45] §2.1 p.3).
- Not added (length): at base-model level GLM-4.5-Base trails DeepSeek-V3-Base and Kimi-K2-Base on MMLU and BBH, confounded by 23T vs 14.8T / 15.5T tokens ([GLM45] Table 2, Abstract).

### Other properties
none

## 61 · mHC: manifold-constrained hyper-connections

[deck](../decks/61-mhc-manifold-constrained-hyper-connections/) · [Notion row](https://app.notion.com/p/99ccc6cd324244c0bf114f19b5440674)

### Core Mechanism
**Before:** Projects Hyper-Connections' residual mixing matrices onto the doubly-stochastic manifold (Birkhoff polytope) with Sinkhorn-Knopp, keeping spectral norm \<= 1 and restoring identity-mapping behavior; companion kernel optimizations cut the extra memory traffic.
**After:** unchanged
**Why:** all statements match the sources: H_res projected by Sinkhorn-Knopp onto the Birkhoff polytope, spectral norm at most 1, identity-mapping property restored ([mHC] Abstract, §4.1-4.2; [V4] §2.2); fused kernels cut residual I/O ([mHC] §4.3.1).

### Impact / Measured Results
**Before:** Eliminates HC's gradient explosion at 27B (compound gain at depth 64 goes from 1e16 to about 1); a single Sinkhorn iteration suffices; also adopted by GLM-5.3.
**After:** Eliminates HC's gradient explosion at 27B: peak composite residual gain drops from about 3000 (HC) to a maximum of about 1.6; Sinkhorn-Knopp uses 20 iterations (t_max, mHC and V4); about 6.7% extra training time at n = 4; adopted in DeepSeek-V4.
**Why:**
- No 1e16 value or depth-64 measurement exists; the 27B model (60 sublayers) shows HC peaking at about 3000 and mHC at a maximum of about 1.6 ([mHC] §3.1 Fig. 3(b), §5.4 Fig. 7(b)).
- "A single Sinkhorn iteration suffices" is contradicted: both papers use t_max = 20 ([mHC] §4.2 p.9; [V4] §2.2, §4.2.1); mHC-lite argues for replacing Sinkhorn, not one iteration ([lite] abstract).
- "Also adopted by GLM-5.3" dropped: the GLM-5 report never mentions mHC and no GLM-5.3 source was found; V4 adoption is documented ([V4] §2.2, §4.2.1).
- 6.7% overhead added as the published cost ([mHC] §1 p.4; [V4] §3.4.2).
- "Eliminates" kept: consistent with the HC loss surge near step 12k and stable mHC curves ([mHC] Fig. 2, Fig. 5).

### Other properties
none. Left for the user: Source still lists "GLM-5 series", but no source read shows GLM adopting mHC (GLM-5 report arXiv 2602.15763 has no mention; no GLM-5.3 source available). Not changed because absence in the sources read is not proof the value is wrong.

## 62 · MTP (multi-token prediction) layer

[deck](../decks/62-mtp-multi-token-prediction-layer/) · [Notion row](https://app.notion.com/p/354b71e013554721a039936a8fff8b2a)

### Core Mechanism
**Before:** One (MoE) MTP layer on top of the backbone provides multi-token supervision during training and serves as the speculative-decoding drafter at inference; MCore supports MTP + MLA + CP together.
**After:** One MTP module (a Transformer block, MoE in GLM-4.5) takes the backbone's final hidden state plus the next token's embedding, shares the embedding and output head, and predicts one token further; it adds a weighted extra loss in training and serves as the speculative-decoding drafter at inference; MCore supports MTP + MLA + CP together.
**Why:**
- The module is not just a layer stacked on the backbone: it projects RMSNorm(h) concatenated with RMSNorm(Emb(t_i+1)) through M_k, runs one Transformer block, and reuses the main embedding and output head ([V3] §2.2 Eq. 21-23, Fig. 3; [CODE] enorm/hnorm/eh_proj). The block is MoE in GLM-4.5 ([GLM] §2.1).
- Loss weighting: L_MTP scaled by lambda (0.3 then 0.1) and added to the main loss ([V3] Eq. 25, §4.2; [GLM] §2.4; [V4] §4.2.2).
- "MCore supports MTP + MLA + CP" kept: confirmed by [RD] v0.14.

### Impact / Measured Results
**Before:** Wins on both training efficiency and decode throughput; GLM-5.2's improved MTP raises accepted length by 20%; also raises per-step compute density in Ascend deployments.
**After:** Better quality at equal training tokens but extra training compute (DeepSeek-V3 ablation; adds 14B params to V3); 1.8x decode TPS as a speculative drafter with 85-90% second-token acceptance (DeepSeek-V3); GLM-5 accept length 2.76 vs DeepSeek-V3.2's 2.55 (4 steps, cross-model); raises compute density in Ascend deployments.
**Why:**
- "Training efficiency" overstates: MTP improves quality at equal tokens ([V3] §4.5.1, Table 4) but adds a Transformer block and an output-head pass; DeepSeek-V3 carries 14B MTP params on top of 671B ([MC] table note p.13).
- Decode throughput given its published numbers ([V3] §5.4.3).
- "GLM-5.2 ... +20% accepted length" appears in no source read; replaced by the GLM-5 report's published accept lengths, a cross-model comparison on a private prompt set, not an ablation ([GLM5] §2.1, Table 2).
- Ascend compute-density claim kept (qualitative, [GLM5] p.22).

### Other properties
none. Left for the user: the Reference (GLM-4.5, arXiv 2508.06471) does mention the MoE MTP layer, so it is not wrong, but the module design, ablation and acceptance numbers come from DeepSeek-V3 (arXiv 2412.19437).

## 63 · Anticipatory routing (asynchronous router parameters)

[deck](../decks/63-anticipatory-routing/) · [Notion row](https://app.notion.com/p/ecc924eb12db4b0fa39609cf42a53369)

### Core Mechanism
**Before:** Deliberately desynchronizes backbone and router updates: features use the newest parameters while routing decisions use cached older ones, breaking the "anomaly -\> routing drift -\> larger anomaly" feedback loop.
**After:** Features at step t use the current parameters, but routing indices come from the network's parameters Δt steps earlier: step t's data is fetched in advance and its indices computed and cached in one extra forward pass; meant to break the vicious cycle in which routing appears to amplify MoE outliers.
**Why:**
- What is old is the whole network θ_t−Δt (backbone layers feeding the router as well as the router), and what is cached is the routing indices, not router parameters ([V4] §4.2.3 p.26).
- The "anomaly to routing drift to larger anomaly" loop is Notion's elaboration; [V4] only says spikes are tied to MoE outliers, routing "appears to exacerbate" them, and the technique breaks "the vicious cycle induced by routing", with the principles not well understood ([V4] §4.2.3, §6).

### Impact / Measured Results
**Before:** With mHC and Muon, suppresses loss spikes and divergence in trillion-scale MoEs.
**After:** Turned on only after a detected loss spike (with a short rollback), at about 20% extra wall-clock while active; with SwiGLU clamping, averted V4's loss spikes without hurting performance.
**Why:**
- Companion technique in [V4] §4.2.3 is SwiGLU clamping; mHC and Muon are stability features elsewhere in the paper, not part of the spike mitigation ([V4] §2.2, abstract, §4.2.3).
- No divergence results are reported; "divergence" dropped.
- Spike-triggered operation and the 20% cost added: without them the row implies an always-on, free mechanism ([V4] §4.2.3 p.26: about 20% while active, "negligible" overall).

### Other properties
none

## 64 · Distributed engineering of Muon at scale

[deck](../decks/64-distributed-engineering-of-muon-at-scale/) · [Notion row](https://app.notion.com/p/9069ea0863a741658d052a19a186169b)

### Core Mechanism
**Before:** Muon needs full gradient matrices, which conflicts with ZeRO element-wise sharding: V4 uses knapsack-based hybrid ZeRO bucketing (\<10% padding memory), optimizes MoE params separately and syncs their gradients in BF16 with stochastic rounding; GLM-5 uses zero-redundancy communication so every DP rank need not all-gather full params.
**After:** Muon needs full gradient matrices, which conflicts with ZeRO element-wise sharding: V4 packs whole dense matrices into knapsack-balanced hybrid ZeRO buckets (under 10% padding memory), handles MoE params separately and syncs their gradients in BF16 with stochastic rounding; GLM-5 uses zero-redundancy communication so every DP rank need not all-gather full params.
**Why:**
- The under-10% padding figure holds for the dense buckets only ("each rank manages no more than five parameter matrices"); MoE padding is called "negligible" ([V4] §3.4.1 p.19-20). Wording now scopes it to dense matrices.
- Rest unchanged: MoE flattening and BF16 stochastic-rounding sync ([V4] §3.4.1 p.20); GLM-5's zero-redundant all-gather of owned shards ([GLM5] §2.4.1 p.9).

### Impact / Measured Results
**Before:** Halves MoE gradient-sync traffic; removes naive Muon's transient memory spikes and redundant communication.
**After:** V4: BF16 sync halves MoE gradient communication volume; GLM-5 reports, with no numbers, that its zero-redundant all-gather removes naive Muon's transient memory spikes and redundant communication.
**Why:**
- Halving is V4's result and applies to MoE gradients only ([V4] §3.4.1 p.20).
- The memory-spike / redundant-communication claim is GLM-5's statement about its own implementation, with no published number, not a V4 result ([GLM5] §2.4.1 p.9).

### Other properties
none

## 65 · MuonClip (Muon + QK-Clip)

[deck](../decks/65-muonclip-muon-qk-clip/) · [Notion row](https://app.notion.com/p/63cf5bee70464919acbac2e6ed917cdd)

### Core Mechanism
**Before:** Token-efficient Muon updates plus QK-Clip: after every step the Q/K projection weights are rescaled against threshold tau, killing attention-logit explosion at the source.
**After:** Token-efficient Muon updates plus QK-Clip: after each step, only heads whose max attention logit S_max exceeds tau (100 in K2) get their Q/K weights rescaled by tau / S_max, capping logits at tau from the next step.
**Why:**
- Per-head factor gamma_h = min(1, tau / S_max^h): heads under tau are untouched, so not every step rescales the weights ([K2] §2.1 p.3, Algorithm 1; App. D: per-head to avoid over-regularizing other heads).
- "Killing ... at the source" overstates: the clip is post-update, does not alter the current step, and caps logits at tau rather than preventing them from reaching it ([K2] §2.1 p.3, Fig. 2 right; tau = 100 p.4).

### Impact / Measured Results
**Before:** Zero loss spikes across K2's full 15.5T-token pretraining (1T MoE), with no downstream degradation from clipping.
**After:** Zero loss spikes across K2's full 15.5T-token pretraining (1T MoE); clipping acted on 12.7% of heads in the first 70K steps, then went inactive; no significant downstream degradation in a small ablation (3B-total MoE, tau 30).
**Why:**
- Zero loss spikes kept: matches [K2] abstract and Fig. 3.
- "No downstream degradation" rests on a small-scale ablation (0.5B active / 3B total, tau = 30) plus self-deactivation in K2; there is no full-scale ablation ([K2] App. D p.30, Fig. 12).
- 12.7% / 70K steps added to show how rarely the clip acts ([K2] App. D p.30).

### Other properties
none

## 66 · QK-Norm

[deck](../decks/66-qk-norm/) · [Notion row](https://app.notion.com/p/319387d19fc64c6d8079c56a5c3b7a69)

### Core Mechanism
**Before:** Normalizes Q/K so the dynamic range of attention logits stays in a stable band (the same problem QK-Clip attacks from the other side).
**After:** Per-head RMSNorm with learnable gains on Q and K (before RoPE) bounds attention logits at sqrt(d) times the gains, independent of Q/K weight norms; same problem QK-Clip attacks from the other side, but needs per-head keys, which MLA does not materialize.
**Why:**
- "Stable band" overstates: the bound is soft, sqrt(d) x max gain_q x max gain_k, set by 2 x 128 learnable gains per layer (analytic, sources.md "sqrt(128) = 11.3"; [HF-CODE] RMSNorm, norm before RoPE, scale 128^-0.5).
- Form described from the code: per-head RMSNorm with a 128-dim weight, keeping the 1/sqrt(d) scale ([HF-CODE]; [CFG-GLM45]), not the l2 + learnable-scalar form of the cited Henry et al. paper ([QKN] eq. 2-5).
- MLA limit added because the row presents QK-Clip as the alternative: QK-Norm is "not applicable" to MLA since keys are not materialized ([K2] §2.1 p.3; [DSV3] §2.1.1).

### Impact / Measured Results
**Before:** Together with Muon, supports stable convergence at larger batch sizes.
**After:** Stabilizes the attention-logit range in GLM-4.5 (no isolating measurement published); the larger-batch tolerance is credited to Muon, not QK-Norm.
**Why:**
- [GLM45] §2.4 credits Muon with "accelerate convergence and tolerate larger batch sizes"; QK-Norm is credited only with stabilizing "the range of attention logits" (§2.1 p.3).
- GLM-4.5 publishes no logit curves or ablation isolating QK-Norm.

### Other properties
none

## 67 · Asymmetric VPP / custom pipeline layout

[deck](../decks/67-asymmetric-vpp-custom-pipeline-layout/) · [Notion row](https://app.notion.com/p/0b667c3fda7240be99b87c3b35fdcfcd)

### Core Mechanism
**Before:** --pipeline-model-parallel-layout allows non-uniform assignment of layers, embeddings and MTP heads across PP stages (flexible asymmetric virtual pipelining).
**After:** unchanged
**Why:** matches the sources: the layout string places E, t, m and L per stage, stages split over PP x VPP ([DOC]; [CODE-L] `parse_str_to_list`, `__init__`; [RD] v0.13; [MC] §7.5).

### Impact / Measured Results
**Before:** Fixes the PP load imbalance caused by MTP/embedding layers and shrinks pipeline bubbles.
**After:** Lets users hand-balance stages around embedding, dense, MTP and loss layers (used for DeepSeek-V3 on H100 and GB200); no measured bubble or throughput gain from the layout itself is published.
**Why:**
- "Fixes" implies automatic balancing; the source describes manual, cost-aware placement written by the user (e.g. "Embed + 3 Dense about 2 MoE" on the first stage) ([MC] §7.5 p.66, Fig. 41; [DOC]).
- "Shrinks pipeline bubbles" has no measurement: the DeepSeek-V3 layouts appear only inside full-stack configurations ([MC] Table 17, §9.2.2 p.74-75); the deck's bubble numbers are analytic PTD-P estimates ([PTDP] §2.2.2).

### Other properties
none

## 68 · Context Parallelism / Dynamic CP

[deck](../decks/68-context-parallelism-dynamic-cp/) · [Notion row](https://app.notion.com/p/82d104d9bcc74fab931db687fed76e81)

### Core Mechanism
**Before:** Shards inputs and all activations along the sequence dimension (unlike SP, which only splits LayerNorm/Dropout), exchanging KV via ring/p2p; supports MHA/MQA/GQA, MLA and MTP, composes with TP/PP/DP, and now supports dynamic CP.
**After:** Shards inputs and all activations along the sequence dimension (unlike SP, which only splits LayerNorm/Dropout), exchanging KV via ring/p2p; supports MHA/MQA/GQA, MLA and MTP, composes with TP/PP/DP; Dynamic CP picks a power-of-2 CP size per sequence, not yet with MLA.
**Why:**
- Dynamic (hybrid) CP is not yet supported with MLA ([MLA] assert in `multi_latent_attention.py`), so "supports ... MLA ... and now supports dynamic CP" read together overstates; MLA/MTP support itself is for static CP since v0.14 ([RD]).
- Dynamic CP mechanism added in a few words: CP size = next power of 2 of seq_len / max_seqlen_per_rank, from pre-built power-of-2 groups ([HCP]; [MC] §6.4.2 p.62; [BLOG]).
- Rest kept: [CPDOC] overview (sequence sharding, KV ring exchange, SP comparison, TP x CP x PP x DP).

### Impact / Measured Results
**Before:** Beats full recomputation beyond 8K sequences by removing most recompute cost while balancing compute and communication.
**After:** Beats full recomputation at 16K and 32K on a 175B GPT (CP configs about 1.15x to 1.3x, read from the CP guide's chart) by avoiding the about 30% recompute overhead while keeping TP small enough to hide its communication.
**Why:**
- The only measurement covers 16K and 32K ([CPDOC] Fig. 2, 175B GPT); 8K is an illustrative example in the guide and the README's recommended threshold for enabling CP (Guideline 5), not a measured crossover.
- Speedups are approximate values read from the bar chart (TP8CP2 / TP16CP2 / TP8CP4: about 1.15 to 1.30), since the guide prints no numbers.
- "Removing most recompute cost" made concrete with the guide's own figure (about 30% overhead of full recompute) and its TP argument ([CPDOC] "Context Parallelism Benefits").

### Other properties
none

## 69 · KDA Context Parallelism (KCP)

[deck](../decks/69-kda-context-parallelism-kcp/) · [Notion row](https://app.notion.com/p/a5e78a14357748deb3032985bef70062)

### Core Mechanism
**Before:** Decomposes each sequence chunk into two locally computable terms - the cumulative transition applied to the entry state, and the state generated locally from zero - so only fixed-size recurrent states need an all-gather.
**After:** unchanged
**Why:** matches the sources: S_out = M S_in + S~, both terms computed locally, then an all-gather of fixed-size states and a prefix-scan merge ([K3] §5.1.2, Eq. 17; [FLA-CP] "Pre-Process Forward", `chunk_delta_h.py`).

### Impact / Measured Results
**Before:** Communication is independent of sequence length (softmax-attention CP must ship KV blocks that grow with length); compute scales linearly to 1M context.
**After:** Communication is independent of sequence length (softmax-attention CP must ship KV blocks that grow with length); K3 publishes no KCP timing for its linear-compute claim; FLA's open implementation reaches 86% fwd+bwd scaling efficiency vs 55% for head-split all-to-all CP (KDA, 32K, CP 4, H800).
**Why:**
- "Compute scales linearly to 1M context" reads as measured; K3 only states that KCP "achieves linear compute scaling" and that CP makes 1M-token training tractable, with no KCP measurement at any length ([K3] §5.1.2, §3.4).
- The only public timing is FLA PR #691 at 32K tokens, CP = 4, 64 heads, head dim 128, H800: 86.26% vs 55.12% fwd+bwd ([PR691] "Performance").
- Labeled "FLA's open implementation" because FLA's README says its CP code is a separate codebase from Moonshot's internal KCP ([FLA-CP] "Acknowledgments").

### Other properties
none (the page body describes the Kimi K3 repo as "architecture and training notes", but its README has nothing on KCP; body left untouched per brief)

## 70 · Megatron-FSDP / HSDP (with EP support)

[deck](../decks/70-megatron-fsdp-hsdp-with-ep-support/) · [Notion row](https://app.notion.com/p/bbdbb22301864a05bcbc8e4097a6b284)

### Core Mechanism
**Before:** Fully-sharded data parallel and hybrid sharding (--num-distributed-optimizer-instances), composable with EP since v0.15; complements the distributed optimizer (ZeRO-1 equivalent).
**After:** Fully-sharded data parallel and hybrid sharding (--num-distributed-optimizer-instances sets the outer DP size; --outer-dp-sharding-strategy picks HSDP or HFSDP), composable with EP since v0.15; complements the distributed optimizer (ZeRO-1 equivalent).
**Why:**
- The flag alone only sets the DP-outer size; with Megatron-FSDP the hybrid mode is chosen by `--outer-dp-sharding-strategy` (`no_shard` = HSDP, `optim` = HFSDP) ([FSDP] "Data-Parallel Sharding Strategies", "Understanding Hybrid-FSDP"; [ARGS] help texts).
- Rest kept: FSDP with EP in v0.15, HSDP for MoE in v0.13 ([RD] "What's New"); distributed optimizer is the ZeRO-1 point and is enabled automatically by Megatron-FSDP ([RD] feature list; [FSDP] feature table).

### Impact / Measured Results
**Before:** Model-state sharding and DP become interchangeable, simplifying memory planning for very large MoEs.
**After:** Per-GPU memory and collective volume scale with the expert-DP size, and FSDP + EP needs no PP stage balancing; no first-party benchmark (user report, Qwen3-30B-A3B on 32 H800: 23.5 vs 45.6 GB, 108 vs 138 TFLOPS against EP8 alone).
**Why:**
- "Interchangeable" is not a source claim; replaced by the source statements "Per-GPU memory and collective volume both scale with EDP size" and "PP-free flexibility" ([MC] §4.1.7 p.27-29).
- No first-party throughput benchmark exists ([BR] card marks performance claims `doc_only`); the only end-to-end measurement is a public user report, labeled as such, showing the memory gain costs throughput ([I2772] comment of 2026-01-14: 23,548 vs 45,552 MB max reserved, 108.0 vs 137.8 TFLOPS).

### Other properties
none

## 71 · MoE Parallel Folding

[deck](../decks/71-moe-parallel-folding/) · [Notion row](https://app.notion.com/p/ef4d75b8eee841a79ada63b79bff78d8)

### Core Mechanism
**Before:** Decouples the parallel mappings of attention and MoE layers so EP can be "folded" inside TP x CP groups, resolving the dense-sparse parallelism mismatch.
**After:** Decouples the parallel mappings of attention (TP x CP x DP) and MoE layers (ETP x EP x EDP), sharing only PP, so EP can be "folded" across any of TP, CP and DP, resolving the dense-sparse parallelism mismatch.
**Why:**
- EP folds across sub-groups that include DP, not only TP x CP: the 256-GPU example folds EP64 across TP4 x CP2 x DP8 ([MC] §3.3.3 p.17-18), and [PF] App. 6.1 validates EP folded "with all of TP, CP, DP".
- The two mappings and the single PP constraint added from [MC] §3.3.1 p.16 and [PF] §3.2 p.5; code asserts identical PP groups ([CODE-PS]).

### Impact / Measured Results
**Before:** Breaks the EP \<= DP limit: in a 256-GPU example EP rises from at most 8 to 64 (ETP=1, EDP=1).
**After:** unchanged
**Why:** matches [MC] §3.3.3 item 1 (p.17-18) and [RD] lines 340-342 exactly.

### Other properties
none. Left for the user: the Reference (arXiv 2603.07685) does describe folding, but the mechanism, dispatcher and all measured MFU results come from the companion paper arXiv 2504.14960 ([MC] p.16 defers to it).

