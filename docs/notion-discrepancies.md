# Notion rows vs. primary sources

While building each deck, the agent compared the Notion row (Core Mechanism, Impact / Measured Results) with the primary sources it read. The slides follow the sources; this page collects every difference found, per technique, so the database can be corrected. Source locations for each point are in the deck's `notes/sources.md`.

## 01 · CSA compressed sparse attention

[deck](../decks/01-csa-compressed-sparse-attention/) · [Notion row](https://app.notion.com/p/e36ba1f7295647b897468dc2b4d30590)

1. **Attribution of the 27% / 10% FLOPs result.** Notion lists it as CSA's impact. [V4] attributes the
   1M-token FLOPs (and KV) reduction to the whole design: hybrid CSA + HCA together with low-precision
   computation and storage (Abstract; §1 p.5; §2.3.4). No CSA-only ablation is published. The slides say
   "whole V4 stack" when quoting 27% / 10%.
2. **"1M → 250K".** [V4] states the compression factor (sequence length to 1/m with m = 4), not a "250K"
   figure. 1M/4 = 250K (decimal) or 262,144 (with 1M = 2^20). The slides use 1M = 2^20 → 262,144 and say
   "analytic".
3. **"learned overlapping-window weighted average".** Correct in spirit; precisely, each entry is a
   softmax-weighted sum (per channel) over 2m = 8 token projections (C^a of block i, C^b of block i−1)
   with learned positional biases ([V4] Eq. 11-12). The overlap is CSA-only; HCA does not overlap.
4. Notion omits that CSA's top-k (512/1024) is *smaller* than V3.2's DSA top-k (2048) ([V4] §2.3.4) and
   that each selected entry is a compressed block, not a token. Not a contradiction; added on the slides.

## 02 · DeepSeek Sparse Attention (DSA) + Lightning Indexer

[deck](../decks/02-deepseek-sparse-attention-lightning-indexer/) · [Notion row](https://app.notion.com/p/cf0641f4eac74fd9bd6c903a15b59ca3)

1. **"top-k (about 2048)"**: the source fixes k = 2,048 exactly ([V32] §2.1.1; [HF] index_topk). Slides say 2,048.
2. **"Large drop in long-context training … cost"**: DeepSeek quantifies only *inference* cost (Fig. 3); the training
   cost reduction is not measured in [V32]. GLM-5 asserts reduced training and inference cost qualitatively and gives
   "1.5–2×" for attention computation. Slides quantify serving cost only.
3. **"GLM-5 adopts it wholesale"**: GLM-5 changes the configuration (32 instead of 64 indexer heads, interleaved RoPE in
   the indexer per the transformers implementation, a 20B-token adaptation instead of 943.7B, deterministic top-k and a
   frozen indexer during RL). Slides say "adopts" and list the differences.
4. **Notion links NSA (arXiv 2502.11089)** as a paper for this row; NSA is a different, natively trained block-sparse
   method. It appears only as a failed-suitor mention ("native sparse pretraining"), not as part of DSA.
5. **Source-vs-source**: GLM-5's "roughly 1.5–2×" attention-compute reduction is much smaller than DeepSeek's measured
   3.6× (prefill) / 8.7× (decode) cost reduction at 128K; they measure different things (attention FLOPs vs end-to-end
   serving cost per token at a given position). Slides use DeepSeek's measured costs and attribute them.

## 03 · HCA highly-compressed attention + hybrid attention stacking

[deck](../decks/03-hca-highly-compressed-attention-hybrid-attention/) · [Notion row](https://app.notion.com/p/ceae2901ccd44561a14abacc7f6574bf)

1. **Name.** Notion calls it "highly-compressed attention"; [V4] names it **Heavily Compressed Attention
   (HCA)** (Abstract, §2.3.2). Slides use the paper's name.
2. **"every layer remains attention - no SSM or Gated-DeltaNet-style linear attention".** True of the
   released configs (every layer is SWA, CSA or HCA), but [V4] does not argue against SSMs or linear
   attention; it actually describes managing the SWA window and uncompressed tails "as a state-space
   model" for cache purposes (§3.5.1). Slides state only that every layer is softmax attention.
3. **Attribution of 10% / 7% KV.** Notion credits HCA alone; [V4] credits the hybrid CSA + HCA design
   together with low-precision storage (BF16 RoPE dims, FP8 elsewhere) (§1 p.5; §2.3.4). The slides say
   "whole V4 stack". Analytic model: CSA layers hold ~97% of the mixed cache, so HCA's contribution is to
   make half the layers nearly free, not to shrink the cache by itself.
4. **"interleaved with CSA".** Correct, with detail Notion omits: V4-Pro's first two layers are HCA and
   V4-Flash's first two are pure sliding-window ([V4] §4.2.1).
5. **"making 1M context routine".** The paper's wording is "routinely support one-million-token contexts"
   (Abstract). Retrieval quality is not flat to 1M: MRCR 8-needle drops from 0.92 at 128K to 0.59 at 1M
   (Fig. 9). The slides show this limit.

## 04 · IndexShare (cross-layer shared indexer)

[deck](../decks/04-indexshare-cross-layer-shared-indexer/) · [Notion row](https://app.notion.com/p/e18d1e0afe6f4d3d9c37bfdf815b0738)

1. **Name.** Notion calls the technique "IndexShare" and cites arXiv 2603.12201, whose title says **IndexCache**. The
   GLM-5.2 README and blog use **IndexShare** and link to the same arXiv paper. The GLM-5 technical report
   (2602.15763v2, the "GLM-5 series" source) mentions neither; IndexCache's paper only reports *preliminary* GLM-5
   experiments. Slides: "IndexShare (paper: IndexCache)".
2. **"Every four sparse-attention layers"** describes the GLM-5.2 production layout (uniform, trained). The paper's
   headline training-free results use a *searched, non-uniform* pattern, and the GLM-5.2 config keeps the first three
   layers all Full (21 indexers in 78 layers, not 78/4 = 19.5). Slides show both.
3. **"Improved MTP adds up to 20% more accepted length"**: the +20% (4.56 → 5.47) combines IndexShare + KVShare with
   rejection sampling and an end-to-end TV loss, measured on the GLM-5.1 backbone with 7 MTP steps; IndexShare +
   KVShare alone gives 4.56 → 5.10 (+12%, derived). Slides attribute it this way.
4. **"2.9× lower per-token FLOPs at 1M"** is a GLM-5.2 vs GLM-5.1 model-level FLOPs count from Z.ai's blog figure, not
   a measured speedup; the paper's measured speedups are 1.82× (prefill) / 1.48× (decode) at 200K on a 30B model.
   Slides keep the two separate.

## 05 · Kimi Delta Attention (KDA)

[deck](../decks/05-kimi-delta-attention-kda/) · [Notion row](https://app.notion.com/p/0a25e512d50542b08af4e090523e14b2)

1. **Notion "Impact" credits 2.5x to KDA "with AttnRes and Stable LatentMoE".** K3 (abstract, §2, §3.2) credits the ~2.5x
   scaling-efficiency gain to KDA, AttnRes, Stable LatentMoE **and refined data and training recipes** together; there is
   no KDA-only ablation at K3 scale. The slide states the joint attribution.
2. **Notion "three KDA layers alternate with one gated-MLA layer".** K3 also adds one extra Gated MLA layer at the end of
   the backbone (§2.1), so the count is 69 KDA + 24 MLA (Table 1), not an exact 3:1. Slides show both.
3. **Notion "State does not grow with length, so 1M context needs no RoPE interpolation".** K3 §3.4 attributes the
   no-interpolation property to NoPE: position comes from KDA's recurrent gating/decay, so there is no RoPE to rescale.
   The fixed state size is a memory property, not the reason. Slides use K3's reasoning.
4. **Internal inconsistency in Kimi Linear (batch-1 decode at 1M):** Fig. 7b annotates 2.2x, while §6.3 text says
   "Kimi Linear achieves a 2.3x speedup at a 1M token context". Slides use the figure value (2.2x) and the notes mention
   the text value. Kimi Linear's abstract rounds the Fig. 1b 6.3x to "up to 6x".
5. **Scope of measured numbers.** All speed and fair-comparison quality numbers come from the Kimi Linear 48B-A3B model
   (KDA before the K3 changes: unbounded decay, low-rank output gate). K3 publishes no KDA-specific speed numbers.
   The deck says so on the costs slide.

## 06 · Staged context extension + annealing / cooldown

[deck](../decks/06-staged-context-extension-annealing-cooldown/) · [Notion row](https://app.notion.com/p/ff95a3d4ac0745728425571a7344ee70)

1. **LR schedule.** Notion: "constant LR plus cosine decay and final annealing". That is K2's WSD schedule
   ([K2] §2.5). K3 uses cosine decay with a 1% linear warm-up ([K3] §3.3) and its scaling-law study
   favours cosine over WSD ([K3] §3.2); GLM-4.5 also uses cosine and reports WSD underfits ([GLM] §2.4).
   Slides present WSD as K2's choice and cosine as K3's and GLM-4.5's.
2. **"this is how K3 gets its 1M context" / "small tail of tokens".** K3 says the long-sequence work is a
   "small fraction of the overall training budget" but publishes no stage token budgets. The quantified
   tails on the slides come from K2 (60B at 32K) and GLM-4.5 (100B at 128K). K3's 1M also relies on NoPE
   + KDA (no positional rescaling, [K3] §3.4), not on staging alone.
3. **K2 "4k->32k->128k with YaRN".** Only 4K and 32K are trained stages; 128K is reached with YaRN on top
   of the 32K stage ([K2] §2.5). Slides draw the 128K step dotted, as an extension, not a training stage.
4. **"Measured results".** Notion's impact line has no measurement, and none of the three reports ablates
   staged vs. unstaged training. Slides label the cost comparison "analytic" and present end-to-end
   long-context scores as outcomes, not as an ablation.
5. **"cooldown / mid-training" for 256K→1M.** K3 calls its phase "cooldown"; GLM-4.5 calls its long stages
   "mid-training" and runs them within one cosine schedule. Slides use "low-LR tail" for both.
6. **K2 token total.** [K2] says 15.5T total and separately describes a 460B anneal + long-context stage
   "towards the end of pre-training"; whether the 460B is inside the 15.5T is not stated (see derived 2).

## 07 · Dynamic Decode Attention (dynamic task scheduling)

[deck](../decks/07-dynamic-decode-attention/) · [Notion row](https://app.notion.com/p/395fb76ae58944d0ae141c2a047478b6)

1. **Headline number**: Notion says "up to 2.88x" (README table). The newer LMSYS blog Table 1 (FP8 KV, H20) reports
   **2.95×** on 1×128K + 31×4K. Slides use 2.95× (with the table) and mention the README's 2.88× in notes.
2. **"builds a task graph ... greedily bin-packs"**: the source builds a flat *task map* (per-CTA bins of task
   descriptors), not a graph. The packing is a deterministic sequential fill — every CTA gets the same tile budget
   and bins are filled in (KV head, request) order, spilling a request's remaining tiles into the next bin — rather
   than a sort-based greedy bin-packing. README itself calls it "greedy bin-packing"; slides describe the actual fill.
3. **"sharply cutting CTA tail latency"**: qualitative; the measured effect is kernel latency (Table 1). End to end
   on H200 the attention-only gain is small (+2.0% / +3.0% output throughput, Table 6); the large TPOT gains on
   H20 (−15.1% to −48.8%) are for Attention + MoE together (Table 7). Slides separate the two.

## 08 · FlashKDA (chunkwise linear-attention kernel)

[deck](../decks/08-flashkda-chunkwise-linear-attention-kernel/) · [Notion row](https://app.notion.com/p/addf5d5fcfad445081c78ebac0e50d60)

1. **Notion "shared by training and prefill".** The K3 report (§5.1.1) says FlashKDA serves training and inference
   prefill, but the public FlashKDA v1 exposes only a forward kernel (`flash_kda.fwd`), and FLA dispatches it under
   `torch.inference_mode()` (README). No backward kernel is public. The deck states both.
2. **Notion body links the Kimi-Linear repo for "FlashKDA kernels".** FlashKDA lives in its own repository,
   github.com/MoonshotAI/FlashKDA (K3 ref. [14]); the Kimi-Linear README points to FLA's Triton KDA kernels
   (`fla/ops/kda`).
3. **"Overlaps intra-chunk computation with cross-chunk state propagation" (Notion, from K3 §5.1.1).** In the public
   v1 code the two kernels are launched one after the other on the same stream (fwd_launch.cu); the overlap comes from
   (a) hoisting all state-independent intra-chunk work into the token-parallel K1 and (b) warp specialization inside
   K2, where TMA load/store warps overlap memory traffic with the MMA warps' serial state chain. The deck describes it
   that way rather than as two concurrently running phases.
4. **Notion "Impact": "lets linear attention actually saturate the hardware on long training sequences".** No
   utilization or long-sequence training numbers are published; the only public measurements are forward-pass
   timings at T = 8K (H20, GB200). K3 itself only says FlashKDA "substantially outperforms the Triton reference".

## 09 · FP8 block-sparse prefill attention

[deck](../decks/09-fp8-block-sparse-prefill-attention/) · [Notion row](https://app.notion.com/p/6ee53460854a4c988e083e2ba08d1f88)

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

## 10 · FP8 MQA logits / sparse indexer scoring kernel

[deck](../decks/10-fp8-mqa-logits-sparse-indexer-scoring-kernel/) · [Notion row](https://app.notion.com/p/ac84a3f18a034bac9f7c33b63a8eb6e3)

1. **"E4M3 dot products"**: true on SM90. On SM100 the current API requires MXFP8 or MXFP4 inputs with packed UE8M0 block
   scales (block of 32), BF16 weights and BF16 logits ([API], PR #462); plain per-token-scaled E4M3 is SM90-only now.
2. **"Separate non-paged (prefill) and paged (decode) versions"**: correct, plus a third family since 2026-09: the Sparse
   Indexer (`fp8_fp4_sparse_mqa_logits`, `..._paged_sparse_...`) that scores only selected KV blocks for V4.1 (PR #432).
3. **Function names**: the Notion row and the vLLM blog use `fp8_mqa_logits` / `fp8_paged_mqa_logits`; since PR #462
   (2026-09-30) these are removed in favour of `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits`.
4. **Impact has no number**: DeepSeek publishes no benchmark table for these kernels. The only measured throughput used
   on slides is a community PR (#452, unmerged) on H200; slides label it as such.

## 11 · 1F1B / batch-level EP all-to-all overlap

[deck](../decks/11-1f1b-batch-level-ep-all-to-all-overlap/) · [Notion row](https://app.notion.com/p/0e6015aa2a8449ff917574f6139df4c2)

1. **Notion "buries EP communication in pipeline bubbles".** The primary source hides all-to-all behind the
   *compute of an adjacent micro-batch* in the 1F1B steady state (merged FWD-BWD), not in pipeline bubbles;
   the warmup's first forward and the flush's last backward stay exposed ([MC] p.33). Slides follow the paper.
2. **Notion "the default optimization for large-EP training".** Both flags default to `False` ([RD] lines
   563-564), and the paper's own best configurations turn it OFF where EP stays inside NVLink
   (DeepSeek-V3 on GB200; Qwen3-235B on GB300) ([MC] Table 17, App. B.2). Slides call it a per-platform switch.
3. **README vs paper on the pain number.** [RD] says EP all-to-all "can consume 30-40% of training time
   without optimization"; [MC] p.29 says 20–60% before optimization and p.32-33 says 30–40% remains *after*
   DeepEP. Slides use the paper's numbers and wording.
4. **Memory overhead.** [MC] p.33 says merged FWD-BWD has no additional (activation) memory overhead,
   while §9.1.4/§9.2.2 say the overlap needs extra buffers. Slides say "extra buffers" and attribute it.
5. **Table 17 vs Table 20 (H100 PP size).** Table 17 lists DeepSeek-V3 on H100 as TP2/PP8/EP64 (and its
   layout string has 32 virtual stages = PP8 × VPP4); App. B Table 20 lists "TP2 PP4". Slides use Table 17.

## 12 · Ascend · comm/compute fusion and PD-disaggregated scheduling

[deck](../decks/12-ascend-comm-compute-fusion-and-pd-disaggregated/) · [Notion row](https://app.notion.com/p/b1ba69567b7740fdbeead2013a32881c)

1. **"pipelines it with tiled matrix computation" (Notion Core Mechanism).** The GLM-5 report says only that
   FlashComm "splits AllReduce operations to hide communication latency behind computation" ([GLM5] §5 p.22).
   Tile-level fusion of the matmul with the collective is documented in the Ascend engine
   (`fuse_gemm_comms`, [VA-SP]) and in the Flux paper ([FLUX]), not in the GLM-5 report. The deck attributes
   it that way: the split is the report's, the tiling is the engine's and the literature's.
2. **"One domestic node matches a dual-GPU node" (Notion Impact).** The report writes "comparable to
   dual-GPU international clusters" and gives no baseline, benchmark, metric or configuration ([GLM5] §5
   p.22). The slide quotes the report and says the evidence is not published.
3. **Attribution of the 50%.** The Notion row attaches the 50% cost reduction to this technique; in the
   report it is the result of the whole Ascend adaptation stack (quantization + fused kernels + scheduling),
   with no per-pillar breakdown. The deck states this.
4. **No Ascend percentage exists.** Neither the report nor the upstream docs publish how much of a GLM-5 step
   the collectives take on Ascend. The pain number on slide 3 is therefore Flux's GPU-cluster measurement,
   labelled as such on the slide, and is used to show how the share grows when the fabric is weaker — not as
   a GLM-5 or Ascend result.

## 13 · Fused AllReduce + residual + RMSNorm

[deck](../decks/13-fused-allreduce-residual-rmsnorm/) · [Notion row](https://app.notion.com/p/2a55d26c7c7a4f858f5816d7ca24b473)

1. **Notion body cites the LMSYS HPC-Ops × SGLang blog as "vs NCCL / FlashInfer".** The blog covers only
   Attention, Router GEMM and MoE; it has no AllReduce + RMSNorm result. The 1.76× figure comes from the HPC-Ops
   README. Slides cite the README.
2. **Notion calls the LL mode "Lamport P2P".** In the code only the scatter (reduce) phase is P2P; the broadcast
   phase stores through the multicast address ([LL]). Slides show both.
3. **Notion: "fuses ... into an NVLink-native kernel".** HT is one kernel; LL is two kernels chained by PDL
   ([PR48], [LL]). Slides say "one op", "one kernel" (HT) and "two kernels" (LL).
4. Minor: the benchmark README command names `bench_allreduce_rmsnorm.py`; the script is
   `benchmark_fuse_allreduce_rmsnorm.py` (open PR #77). No effect on slides.

## 14 · Wave-based fine-grained communication/compute overlap

[deck](../decks/14-wave-based-fine-grained-communication-compute/) · [Notion row](https://app.notion.com/p/0a5b3fd612d24e81a538b34ef561d06c)

1. **"Fully hides MoE communication whenever per-layer comm time < compute time."** [V4] states that
   profiling shows communication < computation within a layer, so after fusion computation stays the
   bottleneck; the formal condition is C/B ≤ V_comp/V_comm. The pipeline's first dispatch and last combine
   (fill/drain) stay exposed, so "fully" is the steady-state view. Slides say "hidden in steady state".
2. **"C/B ≤ 2d (about 6144 FLOPs/Byte)".** The paper gives exactly 6144 and only for V4-Pro; here d is the
   expert intermediate dimension (3072), while §4.2.1 of the same paper uses d for the hidden size (7168).
   Slides write "d = expert width" and note the V4-Flash value (4096) as derived.
3. **"Up to 1.96x on RL rollout."** [V4] says up to 1.96× for latency-sensitive scenarios "such as RL rollouts
   and high-speed agent serving"; [PR316] shows 1.96× is the V4-Flash, 1-token-per-rank, EP8 kernel benchmark
   vs the non-fused path. It is a MoE-layer speedup, not an end-to-end RL speedup. Slides say so.
4. Notion does not mention the baseline; [V4] compares to "strong non-fused baselines" (DeepEP + DeepGEMM
   kernels, per [DG-SRC] test), not to Comet. The 1.42× vs 1.92× comparison with Comet is theoretical.
5. Notion lists Flux (arXiv 2406.06858) as a paper; [V4] cites Comet and FlashDMoE (Aimuyo et al., 2025) as
   inspirations, not Flux. Not used on slides.

## 15 · 0-SM PP / CP / Engram communication primitives

[deck](../decks/15-0-sm-pp-cp-engram-communication-primitives/) · [Notion row](https://app.notion.com/p/c3f064d3adbd46a983472f8d7992f155)

1. **"DP/TP all-gather and reduce-scatter implementations are included."** At V2 these were listed under
   *still on-going* ("All-gather updates and reduce-scatter implementations for DP & TP", [R2] l.37).
   V2.5 ships `BucketBuffer` all-gather / reduce-scatter / all-reduce for **CP and DP** (not TP) ([R25]
   l.14, l.35, l.293), and only all-gather is 0-SM: "reduce-scatter and all-reduce use GPU SMs" ([R25]
   l.328; code asserts "0-SM reduce-scatter is not implemented").
2. **"0 SMs" is about the payload path, not literally no kernels.** PP send/recv launch short
   one-warp-per-CTA TMA staging kernels (grid = all SMs by default) and the receiver spins on the arrival
   signal; Engram fetch launches issue warps that post GETs; hybrid/RDMA all-gather launches CTAs that post
   puts ([C-PP], [C-EN25], [C-AG25]). Only the pure-NVLink all-gather launches no kernel at all. The V2.5
   README no longer uses the "0 SM" label: "Engram (experimental, with RDMA)", "PP (experimental, with
   RDMA)", "Bucket collectives (experimental)" ([R25] l.33-35).
3. **"CP uses the Copy Engine."** True for the V2 NVLink-only all-gather and the V2.5 NVLink path; the
   V2.5 RDMA path uses the NIC and the hybrid path uses both ([C-AG25]). The CP all-gather was not
   publicly usable as a separate API until V2.5 ([I694]).
4. **Impact: "Communication no longer competes with compute for SMs."** Holds for these three
   primitives' transfers; EP dispatch/combine still need SMs ([R25] l.39: "zero-SM RDMA EP is not
   supported"). No source publishes bandwidth or end-to-end numbers for the PP / CP / Engram primitives.
5. Minor: the V2 `ElasticBuffer` docstring says PP and AGRS use NVLink ([C-AG2] `elastic.py` l.93-96)
   while the README says PP uses RDMA; the code issues GIN puts and V2.5 requires an RDMA-only PP group
   ([R25] l.390, l.101). Slides follow the README + V2.5 code.

## 16 · Elastic GPU/CPU hybrid memory buffer

[deck](../decks/16-elastic-gpu-cpu-hybrid-memory-buffer/) · [Notion row](https://app.notion.com/p/f8ca122ace3b41dba7d867f9c2731da3)

1. **"transparently supporting Engram and imbalanced EP"**: in V2 this was an *on-going* feature ([R2]
   l.35) and the V2 docstring says "currently GPU-only" ([EB2]). Host-backed storage shipped for Engram only
   (ElasticBuffer `num_cpu_bytes` in May 2026, `EngramBuffer(use_cpu_rdma_storage=True)` in V2.5). No
   released code backs EP buffers with host memory: `EPBuffer` passes zero storage bytes ([BUF]). It is
   also not "fully automatic": the caller sizes the storage and passes `use_cpu_rdma_storage`.
2. **"EP replay compresses intermediate buffers"**: listed as on-going in V2 ([R2] l.36); no replay
   mechanism appears in the V2.5 sources (grep for "replay" finds only cached-layout reuse in the README,
   l.281). The deck reports it as roadmap.
3. **Impact "worst-case load no longer dictates the memory budget"**: not yet true for EP in V2.5; the
   EP size formula still reserves room for N × T tokens ([EPSZ]); the deck shows this analytically.
4. The README describes one "contiguous virtual address space" mapping GPU and CPU memory; the code
   implements it, but registers the two parts as separate NCCL windows (one per storage shard): locally
   the GPU can address the whole range by pointer, while remote access to the storage goes through
   Gin-only (RDMA) windows, not NVLink ([CTX]).
5. Stale docstring: the ElasticBuffer docstring kept "currently GPU-only" after `num_cpu_bytes` landed
   ([EB-mid]).

## 17 · ElasticBuffer unified API + analytic SM/QP budgeting

[deck](../decks/17-elasticbuffer-unified-api-analytic-sm-qp/) · [Notion row](https://app.notion.com/p/288a723bba1048b2964f7c67a24d4a3a)

1. **"Communication SMs drop from 24 to 4-6 at equal or better performance in V3-style training"** — matches [R2]
   verbatim in substance. OK.
2. **"726/740 GB/s on SM100 NVLink EP8"** — matches [R2] row 4, but that row uses **64 SMs** ("Max perf");
   the minimum-SM row is 643/675 GB/s at 24 SMs. The Notion row omits the SM count; the slides show both rows.
3. **"handle caching reuses routing metadata"** — true for V2.0 ([R2] decode example: "reusing routing metadata
   across iterations when the gating decisions remain unchanged"). V2.5 ([R25]) narrows it: "A new routing decision
   needs a fresh dispatch; decoding alone does not make an old routing handle reusable"; the cached handle is for
   replaying the *same* routing (e.g., backward). Slides use the V2.5 wording.
4. **"ElasticBuffer"** — renamed/split into `EPBuffer` (+ Engram/PP/Bucket buffers) in V2.5 ([PR763]). Slides say
   "EPBuffer (ElasticBuffer in V2.0)".
5. **"with a new GEMM layout"** — the V2.5 README calls it the "expanded layout for grouped expert GEMMs"
   (`do_expand=True`, `expert_alignment`); slides use that name.
6. The **analytic model's constants changed between V2.0 and V2.5** (per-SM 200/50 → 180/45 GB/s, headroom ×1.25 → ×1.3,
   rounding 2 → 4, combine terms added); not a Notion discrepancy, but the worked example states which constants it uses.

## 18 · FP8 dispatch + BF16 combine low-precision communication

[deck](../decks/18-fp8-dispatch-bf16-combine-low-precision/) · [Notion row](https://app.notion.com/p/8f8e53cef45c40b58e4079637b461706)

1. **"About 3h bytes per token-expert" (Notion Impact).** V4 counts 3h (h FP8 + 2h BF16) and ignores scale
   factors and routing metadata. With DeepEP's FP32 scales the payload is 3h + h/32 = 3.03h (3.008h with UE8M0),
   plus top-k indices/weights. Slides say "3h (+3% scales)".
2. **"The backward dispatch is exactly the forward combine" (Notion Core Mechanism).** The source states the
   relation per operator: the backward of dispatch is a combine, and the backward of combine is a dispatch
   ([RD]). The backward-direction precisions follow the direction (token→expert FP8, expert→token BF16; [V3] §3.3.3).
   Slides state both halves of the mirror.
3. **"The Hybrid-EP backend further supports NVFP4" (Notion).** The main README lists "NVFP4 data type support"
   for the Hybrid-EP branch, but that branch's own doc lists only BF16 and FP8, and its token types are
   UINT8/UINT16 with an always-BF16 combine ([HE-DOC], [HE-CFG], fetched 2026-10-13). Slides say "NVFP4 listed for
   the Hybrid-EP branch" (limits slide) and do not claim it ships.
4. **Notion headline treats 6144 FLOPs/Byte as DeepEP's result.** It is a hardware-design balance point stated
   in the V4 report's "Observations and Proposals" (§3.1) for V4-Pro's dimensions (d = 3072), not a measured
   speedup. Slides label it as V4's analytic threshold.

## 19 · HybridEP + FlexDispatcher

[deck](../decks/19-hybridep-flexdispatcher/) · [Notion row](https://app.notion.com/p/3e7e3ebd211b43769e3a8acb1883c7f6)

1. **Notion "available by default in MCore v0.15".** The README lists HybridEP as *added* to the Flex
   dispatcher in v0.15 (2025/11), but it is not the default: `moe_token_dispatcher_type` defaults to
   `allgather`, `moe_flex_dispatcher_backend` defaults to `deepep` and is marked "[Experimental]"; HybridEP
   must be selected with `--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep`
   ([RD] line 16, 312; [MLM-CFG]). Slides say "added in v0.15, opt-in".
2. **Notion "abstracts the DeepEP and HybridEP backends".** On main the Flex dispatcher has four backends:
   deepep, deepepv2 (DeepEP v2 ElasticBuffer, deck 24), hybridep and ncclep (TransformerEngine NCCL EP)
   ([MLM-CFG]). Slides show all four.
3. **Blog text vs blog table.** The blog prose says "5.5% improvement in the BF16 scene" for Qwen3-235B, but
   its Table 1 gives 665 → 698 TFLOPS (1.05×, i.e. ~5.0%). Slides use the table values (and the table's
   "BF18" typo is read as BF16).

## 20 · MoonEP · dynamic redundant experts + online GPU planning + zero-copy static shapes

[deck](../decks/20-moonep-dynamic-redundant-experts-online-gpu/) · [Notion row](https://app.notion.com/p/dfc27fd2e9d443c9a0a0efd215bdc3da)

1. Notion: "every EP rank receives exactly **S x K / EP** tokens". [README] and [K3]: exactly **S × K** tokens per rank,
   with S = input tokens per rank. Slides use S × K.
2. Notion: "a fixed static S x K buffer". The real allocation is NvS = S·K plus per-group padding slack (≤ (P−1)·2·E/R
   rows, P = 128) rounded to P ([API]); 77,824 vs 65,536 slots at the benchmark shape. Slide 10 shows the padding.
3. Notion: "makes GroupedGEMM a fully static load". The per-GPU total is static, but per-expert segment sizes
   (`cu_seqlens`) still change every micro-batch; K3 adds a workload-aware expert-GEMM scheduler because per-expert
   counts inside a rank stay skewed ([K3] §5.2.1). Slide 14 says so.
4. Notion: "online planning overhead is about zero". The README says "negligible"; the README chart's planning segment
   is ≈ 40-50 µs (≈2% of dispatch), and the weight prefetch that planning triggers is ≈ 0.5-0.75 ms per dispatch
   (estimated), which the Notion row does not mention. Slide 12 shows both.
5. Notion: "CUDA Graph friendly". Not claimed by [README] or [K3] (K3: static shapes remove per-layer host sync and
   alleviate launch overhead). Not claimed on slides.
6. Notion: "with no host involvement" is correct per layer; the host still allocates the NVLink/VMM buffers once at
   `Buffer` construction ([API]).

## 21 · NCCL Gin backend (GPU-initiated RDMA)

[deck](../decks/21-nccl-gin-backend-gpu-initiated-rdma/) · [Notion row](https://app.notion.com/p/50c0563d46924e00830d20a62dd6cbde)

1. **"header-only"** — the V2 release README says "Header-only & lightweight" ([R2]); V2.5 still builds a host C++
   extension against NCCL and JIT-compiles kernels with DeepJIT ([R25]). "Header-only" describes NCCL's device API
   headers that the JIT kernels include, not the whole stack. Slides say "device API headers included by JIT kernels".
2. **"the whole library is JIT-compiled"** — true for GPU kernels in V2/V2.5; the host extension is compiled at install
   time in V2.5 ([R25] "Installation builds the host C++ extension"). Slides: "GPU kernels JIT-compiled".
3. **"put / put-with-signal"** — correct per [GIN] §III-A. DeepEP V2.5 additionally uses one-sided `get` (Engram) and
   `signal` / `flush` ([DEV2]); slides list all.
4. **"basis for 0-SM communication paths"** — the V2 release advertised 0-SM Engram/PP/CP ([R2]); V2.5 states
   "EP dispatch and combine require GPU SMs; zero-SM RDMA EP is not supported" ([R25] Notes). Gin removes the CPU
   and the proxy from the data path; the SMs still post the work. Slides do not claim 0-SM EP.
5. **"reusing existing NCCL communicators with symmetric memory window registration"** — correct ([HDL], [CTX25]);
   note the fallback: with older PyTorch (no `_comm_ptr`) DeepEP creates its own NCCL communicator.
6. **Performance evidence** — the only published GIN-vs-NVSHMEM numbers ([GIN] §V) are for **DeepEP v1.2.1 ported to
   GIN** on H100, not for DeepEP V2's rewritten kernels. Every result slide says so. The Notion row gives no numbers.
7. **Backend choice** — the GIN paper presents GDAKI and Proxy as interchangeable; DeepEP V2.5 requires GDAKI and
   asserts otherwise ([CTX25]). Slides say so on the backend slide.

## 22 · Network-side optimization: VL traffic isolation / adaptive routing / SM-free RDMA

[deck](../decks/22-network-side-optimization-vl-traffic-isolation/) · [Notion row](https://app.notion.com/p/c0f6478d3ece4c3c8cef0965065c4c0a)

1. **Notion: "adaptive routing is enabled" as a settled default.** DeepEP's advice changed three times: unsafe for
   normal kernels (Feb 2025), heavy-load only (Mar 2025 – Feb 2026), all loads (V2, Apr 2026). Fire-Flyer chose
   static routing for its storage-heavy network. Slides show the history.
2. **Notion: "VL isolates EP traffic from everything else".** V1 recommended three classes (normal kernels,
   low-latency kernels, other); only V2 reduced it to EP vs other. DeepEP only sets the SL; the VL mapping and
   arbitration weights are fabric configuration ([RD]).
3. **Notion: "returns RDMA-path SMs to compute" / SM-free RDMA.** Normal-SMFree (PR #347) is still an open PR
   against the `antgroup-opt` branch (not merged anywhere, 2026-10-13). Mainline DeepEP V2.5 states zero-SM RDMA EP
   is not supported; V2 instead cut SMs from 24 to 4–6. The PR's speedups are kernel-execution times with the
   transfer hidden under a GEMM, not end-to-end time.
4. **Notion body: Azure blog cited for "adaptive routing, traffic isolation".** The Azure post covers NUMA/HCA
   affinity and PCI atomics; it does not discuss VLs or AR. Not used for those claims.
5. **Notion: "colliding with DP/checkpoint traffic".** DeepSeek's own V3 fabric also puts storage on a separate
   RoCE plane ([ISCA] §5.1), i.e. physical isolation beyond VLs; Fire-Flyer's four classes were HFReduce, NCCL, 3FS
   storage and other. No public source quantifies the EP slowdown from such collisions; slides keep it qualitative.
6. **Sub-page: LL-SBO "+8–10%" vs PR #390.** The LMSYS blog reports +8–10% at BS 20–56; PR #390's bs-32 serving
   runs show +6.7% and +7.5% output throughput (6667 → 7111, 6671 → 7169 tok/s). Slides quote the blog's range
   and name the batch range.

## 23 · Ascend adaptation stack (GLM-5)

[deck](../decks/23-ascend-adaptation-stack-glm-5/) · [Notion row](https://app.notion.com/p/3eda454335e04ef09f6a0b6991ee5608)

1. **Model size in the deployment section.** [GLM5] §5 p.21 says "the 750B parameter GLM-5 model" while
   Table 10 p.36 gives 744B total parameters. The slides use 744B and never quote 750B as a fact.
2. **Notion: "A single domestic GLM-5 node matches a dual-GPU cluster".** The report's sentence is
   "GLM-5 on a single Chinese node achieves performance comparable to dual-GPU international clusters"
   ([GLM5] §5 p.22). The report names neither the GPU model, the node count behind "dual-GPU", the
   benchmark, the metric, nor the serving configuration, so the slide quotes the claim and labels it as
   the report's unqualified wording rather than restating it as a measured 2× result.
3. **Node naming.** [GLM5] §5 p.21 writes "a single Atlas 800T A3 machine"; the upstream deployment
   guide writes "Atlas 800 A3 (128GB × 8)" ([VA-GLM5] §3). The slides use the upstream spelling, since the
   node-memory figure comes from there.
4. **Notion property calls the row "domestic GPU/NPU ecosystems".** The report's own §1 heading phrase is
   "Chinese GPU ecosystems"; §5 is "Chinese Chip Infrastructure". Ascend parts are NPUs, Moore Threads and
   MetaX parts are GPUs. The slides say "domestic accelerators" and name Ascend Atlas where the report does.
5. **Training vs inference.** Neither the Notion row nor the report claims pre-training on domestic
   accelerators; the Notion wording "day-one full-stack engineering" could be read that way. The deck
   states explicitly that §5 covers deployment, and that the training hardware is not stated in the report.
6. **Press claim [GT] is about GLM-5.3-Flash, not GLM-5**, is a company statement relayed by a newspaper,
   and names no chip vendor. The deck cites it on the adoption slide with that attribution only.

## 24 · DeepEP v2 (EP communication library)

[deck](../decks/24-deepep-v2-ep-communication-library/) · [Notion row](https://app.notion.com/p/d471baa2bcdf452592855568ee99f726)

1. **"Several times lower SM usage than V1"** — consistent with [R2] ("several times fewer SM resources",
   "saving up to 4x SM count"). OK.
2. **"scale-up/out domains up to EP2048"** — matches the V2 release README [R2]; the current V2.5 README [R25] no longer
   states a number ("Larger scale-up & scale-out domain support"), and no published benchmark goes beyond EP32 (8×4). The deck therefore does not put EP2048 on a slide.
3. **"FP8 dispatch with IB virtual-lane isolation and adaptive routing"** — FP8 dispatch and VL isolation are documented;
   note the adaptive-routing advice changed between V1 and V2: V1 [R1] recommended adaptive routing only under heavy
   load and static routing under light load; V2 [R2]/[R25] recommends enabling adaptive routing "under all network load
   conditions". Slides use the V2 wording.
4. **Child list "elastic GPU/CPU buffers"** — in the V2 release this was listed under "Still on-going features"
   ([R2]); V2.5 ships multi-layer Engram storage on GPU **or** CPU ([R25] News), not the transparent hybrid
   GPU+CPU virtual address space. Slide 11 says "RDMA gets from GPU or CPU tables" and the notes say the hybrid buffer was listed as on-going in V2.
5. **Child list "0-SM PP/CP/Engram"** — "0 SM" wording is from the V2 release ([R2]); the V2.5 README drops the 0-SM
   label: PP send/recv uses TMA copies on SMs around a Gin put, Engram uses RDMA gets, and only the bucket all-gather is
   copy-engine driven (`num_sms=0`) while reduce-scatter/all-reduce use SMs ([R25] Bucket collectives). Slide 11 says
   "all-gather on copy engines, reduce-scatter on SMs" and the notes attribute the 0-SM label to the V2 release.
6. Notion names the API **ElasticBuffer**; V2.5 renamed/split it into `EPBuffer`, `EngramBuffer`, `PPBuffer`,
   `BucketBuffer` ([R25] News, [PR763]). Slides use `EPBuffer` and mention the rename.

## 25 · DeepGEMM (unified tensor-core kernel library)

[deck](../decks/25-deepgemm-unified-tensor-core-kernel-library/) · [Notion row](https://app.notion.com/p/c0133413563f4ee390f9e4fb55027780)

1. **Notion Core Mechanism** lists "GEMM, fused MoE, indexer scoring, HyperConnection" — accurate but
   predates the 2026-09 release: the current library also ships Mega Gate (fused routing GEMM),
   a sparse indexer, einsum and epilogue classes, and the JIT has been split out into the separate
   DeepJIT repository (S1.4, S3.4, S6.1). Slides follow the primary source.
2. **Notion Impact** ("matches or beats expert-tuned libraries") reproduces the README's qualitative
   claim (S1.2) and carries no number. The slides use the measured numbers of S2.4, S1.3 and S3.3 and
   state their hardware and baseline, because the published speedup baseline is DeepSeek's own
   CUTLASS 3.6-based implementation, not a vendor library (S2.3).
3. **"Very few kernels"** (Notion) is true per kernel family but the library is no longer one kernel:
   19 kernel implementation files, 9,593 lines of device code (S6.2). Slides say "a few hundred lines
   per kernel family" and mark the count as derived.
4. No numeric conflict was found between the Notion row and the repository.

## 26 · HPC-Ops (Tencent Hunyuan inference operator library)

[deck](../decks/26-hpc-ops/) · [Notion row](https://app.notion.com/p/85f1abe2180f4b86bf1e9ea6d23705ce)

1. **"upstreamed into vLLM / SGLang"** — the LMSYS blog documents upstream integration into **SGLang** main
   (Attention, Router GEMM, MoE). For vLLM the README only says the API is "designed for seamless integration"
   with vLLM and SGLang, and vLLM appears as a *benchmark baseline*; no vLLM upstreaming is documented. Slides say
   "integrated into SGLang main"; vLLM is not claimed.
2. **"about 30% QPM ... about 17% for DeepSeek-class models"** — not in any README version (checked all six
   README commits). The figures come from MarkTechPost's release article ("about 30 percent ... Tencent-HY
   models ... about 17 percent ... DeepSeek models on mainstream inference cards"). Kept on the adoption slide,
   labelled as Tencent-reported service-level numbers with the secondary source named.
3. **"15.1-48.8% lower end-to-end TPOT for Hy3-FP8 on 8x H20"** — matches [L] Table 7 for batch 4–64 (batch 1
   is only −3.3%); slides state the batch range.
4. **Scope** — Notion lists "normalization" as a hot path; in the source normalization appears as
   RMSNorm/RoPE/act-quant utilities plus the fused AllReduce+RMSNorm; no standalone normalization speedup is
   published. Not given its own slide.
5. **Dynamic decode headline** (affects child deck 07): README says "up to 2.88x" vs static split-k; the newer
   LMSYS Table 1 reports 2.95× (1×128K + 31×4K, FP8). Slide 11 uses the README table (2.88×) for consistency
   with the other README maxima; slides 3/5 use the LMSYS table (2.95×) and say so.

## 27 · MoonEP (perfectly balanced EP communication library)

[deck](../decks/27-moonep/) · [Notion row](https://app.notion.com/p/65b197b767bb4d5c9dc17bbce5c0ca13)

1. Notion (deck 20 row, inherited here): "every EP rank receives exactly S x K / EP tokens". Primary sources: each rank
   receives exactly **S × K** tokens, with S = input tokens *per rank* ([README] property 1, [K3] §5.2.1). Slides use S × K.
2. Notion: "communication buffer drops from a worst-case S x K x R to a fixed S x K" matches [K3], but the actual
   allocation is NvS = S·K plus per-expert-group padding slack (≤ 127·2·E/R slots) rounded to 128 ([API]): 77,824 slots
   instead of 65,536 at the benchmark config. Slide 10 plots the real NvS and says "S·K + padding".
3. Notion: "increasingly outperforms DeepEP v2 as routing imbalance grows". True for combine and for the totals at high
   imbalance, but at low imbalance MoonEP's dispatch (with planning + prefetch) is slower (2,598 vs 2,296 µs at maxvio
   0.2) and e2e iteration time is 1.9% worse at maxvio 0.2 ([CHART-COMM], [CHART-E2E]). Slides show both sides.
4. K3 says "S is the sequence length" in §5.2.1; README/API define S as input tokens per rank. Same thing when one
   sequence per rank; slides use the README definition.
5. Notion (deck 20 row) says static shapes are "CUDA Graph friendly"; neither README nor K3 claims CUDA Graph support
   (K3 says static shapes remove the per-layer host sync and "alleviate" launch overhead). Not claimed on slides.

## 28 · VeOmni (model-centric omni-modal training framework)

[deck](../decks/28-veomni/) · [Notion row](https://app.notion.com/p/9b9145984b6a433abf69f32d2c06db67)

1. **Notion: "More than 2800 tokens/s/GPU for a 30B omni-modal MoE on 128 GPUs with 160K context."**
   The paper states two separate facts: throughput "over 2,800 tokens/sec/GPU" and the ability to
   "scale to 160K context lengths". Fig. 6(a) shows the highest throughput at the shortest context
   (16K, SP1/EP1) and the 160K point only with larger SP/EP at much lower throughput; the text-only
   Qwen3-30B-A3B table shows the same pattern (2,853 tokens/s/GPU at 16K vs 1,075 at 128K, Table 4).
   Slides keep the two claims separate.
2. **Notion: "FSDP1/2".** The paper integrates FSDP1 and FSDP2 (§3.2.1), but the current repo says
   FSDP1 "has since been deprecated upstream and removed from VeOmni"; FSDP2 only ([DOC-EP]).
   Slides say FSDP2 (HSDP mode).
3. **Notion: "DCP checkpointing".** The paper (Aug 2025) uses ByteCheckpoint (§3.2.5); the open-source
   repo uses PyTorch Distributed Checkpoint ([README], [DOC-CKPT]). Slides follow the repo and deck 50.
4. **Notion: "adding a new modality takes very little code".** The paper says "minimal code change"
   without a number; the repo guide lists `__init__.py` + 4 more files for an omni-modal MoE and
   per-model patches for SP and fused MoE ([DOC-NEW]). Slides state the file count as the cost.
5. **README news calls the AAAI 2026 paper "OmniScale"** while the arXiv title is VeOmni; slides use VeOmni.

## 29 · FP8 grouped GEMM (per-tensor / block-wise scaling)

[deck](../decks/29-fp8-grouped-gemm/) · [Notion row](https://app.notion.com/p/4b75d7ab1bfc40239d6b9566c5d6fc14)

1. **Notion: "accepting cu_num_tokens input".** The API arguments are `seqlens` and `cu_seqlens`; the README
   example names the tensors `num_tokens_per_group` / `cu_num_tokens_per_group`. Same concept; slides use the API
   names.
2. **Notion body: LMSYS blog as the "prefill/decode comparison vs DeepGEMM".** The LMSYS blog has no grouped-GEMM
   vs DeepGEMM comparison (its MoE tables compare the fused MoE against SGLang / vLLM). The 1.1× / 1.88× figures
   come from the HPC-Ops README (current and January 2026 versions). Slides cite the README.
3. **Notion: "vs DeepGEMM: 1.1x on prefill and 1.88x on decode".** The README states these as maxima ("Up to",
   "Maximum observed speedup"); slides say "up to".
4. Notion lists MegaBlocks and the FP8-formats paper as references; neither is about this kernel and neither is
   used for numbers.

## 30 · FP8/FP4/BF16 dense GEMM + UE8M0 scaling layout

[deck](../decks/30-fp8-fp4-bf16-dense-gemm-ue8m0-scaling-layout/) · [Notion row](https://app.notion.com/p/eedb72bc30e14aa8ba49b28b4344d0b9)

1. **Notion Core Mechanism** says "SM90 supports only the NT layout with FP32 scale factors, while
   SM100 supports all layouts with 4x UE8M0 packed into an int" — matches S1.1/S1.2 exactly. It omits
   that SM90 has two kernels (1d1d when B is also per-token scaled, 1d2d for 128x128 block scales,
   S3.1) and that SM100 accepts gran_k 32 as well as 128 (S2.1). Slides state both.
2. **Notion Impact** "Peaked at 1550 TFLOPS on H800" matches the README news (S1.4). The README states
   this as a peak across shapes; the per-shape table (S6) tops out at 1426 TFLOPS, and PR #74 reports
   1470 TF for the same shape after the large-block-tile change (S7). The slides therefore quote 1426
   TFLOPS for the measured shape and 1550 TFLOPS as the published peak, with dates.
3. The Notion row's title includes BF16, but BF16 dense GEMM usually runs through **cuBLASLt** inside
   DeepGEMM unless determinism is requested or the SM budget is capped (S3.4) — stated on the slides.
4. No numeric conflict was found between the Notion row and the repository.

## 31 · Fused FP8 MoE low-latency pipeline

[deck](../decks/31-fused-fp8-moe-low-latency-pipeline/) · [Notion row](https://app.notion.com/p/4d0436aff16240398bbe1c41ab56d5ad)

1. **Notion: "Fuses ... into a single pipeline".** It is five PDL-chained *stages* with HBM intermediates, not one
   kernel: the count/index stage is itself 2 kernels (≤ 128 tokens: `count_and_build_kernel` + task-map build) or
   3 kernels (> 128 tokens), so 6–7 kernels in total ([CPF], [CNT]). The README says "one pipelined execution
   path". The gain is not fewer kernels but no gather copy, no per-pair global atomics, overlapped launches and
   higher occupancy. Slides say "five stages, chained with PDL".
2. **Notion: "Up to 1.6x (TP) and 1.5x (EP) over vLLM CUTLASS, Triton and SGLang".** PR #49 states these as
   speedups relative to the *median* of the compared backends (TP8 1.5–1.6×, EP8 1.2–1.5×). Against the *best*
   baseline per batch (LMSYS Tables 4–5, Hy3) the maxima are 1.18× (TP8) and 1.60× (EP8), means 1.08× and 1.21×.
   Slides use the best-baseline numbers.
3. **The low-latency (cp.async, gather-free, no warp specialization) path is only taken when the per-rank
   intermediate is ≤ 512** ([EN]). In the EP8 benchmark mode (intermediate 1536) the TMA warp-specialized path with
   a gather runs. Notion attributes all results to the low-latency mechanisms; slides label the EP8 curve "TMA path".
4. **Benchmark README says "Operator: SM90 blockwise FP8 FusedMoE"**, but its HPC-Ops provider calls the per-tensor
   `hpc.fuse_moe` with per-expert scales ([BB]). Slides describe the per-tensor path.
5. Notion body cites MegaBlocks and FlashInfer papers; neither is used for numbers.

## 32 · K-axis grouped GEMM (MoE weight backward)

[deck](../decks/32-k-axis-grouped-gemm-moe-weight-backward/) · [Notion row](https://app.notion.com/p/56c25e6b3cbe4425b251e2b59530b802)

1. **Notion "purpose-built ... with a matching K-grouped UE8M0 packing kernel"** matches [RD] l.162, but the UE8M0
   packing applies to SM100 only; SM90 uses FP32 scales ([LAYOUT] l.103-109). Slides say so.
2. **Notion "puts the MoE backward pass on the FP8 fine-grained-scaling path ... unlike inference-only operator
   libraries".** True for weight gradients; the comparison with "inference-only libraries" is not in the primary
   sources and is not repeated on the slides.
3. **Notion names only `k_grouped_fp8_gemm_tn_contiguous`.** That API is SM100-only ([API] l.361-370); Hopper uses
   `k_grouped_fp8_gemm_nt_contiguous` (since Nov 2025), and the May 2025 Hopper version was a host loop
   ([V1-API]). Slides show both.
4. **The only published per-kernel numbers ([PR95]) measure the May 2025 per-expert loop** (kernel time, launch gaps
   excluded, [V1-TEST] l.294), on an unnamed Hopper GPU; there are no published numbers for the single-kernel
   versions beyond "~20% faster MoE backward GEMMs" ([PR432]). Slides label the chart accordingly.

## 33 · M-axis grouped GEMM (contiguous layout)

[deck](../decks/33-m-axis-grouped-gemm-contiguous-layout/) · [Notion row](https://app.notion.com/p/3e1e63852a934fad8b5d53773913bce5)

1. **Notion "without padding waste".** The layout does pad: every expert segment is rounded up to the M
   alignment (128 rows by default, 256 on SM100), with −1 / zero rows ([RD] l.78; [HEUR]). What it avoids is
   capacity padding (padding every expert to a fixed capacity). Slides say "≤ one tile of padding per expert".
2. **Notion Core Mechanism "unlike CUTLASS generic grouped GEMM"** matches [RD] l.78. No discrepancy, but note
   the baseline in the performance table is DeepSeek's internal CUTLASS 3.6-based implementation, not a public
   CUTLASS grouped GEMM ([RD-0325] l.10); slides say "tuned CUTLASS 3.6 baseline".
3. **API names drift.** The 2025 API was `m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(lhs, rhs, out, m_indices)`
   ([OLD-API]); main exposes `m_grouped_fp8_fp4_gemm_nt_contiguous` with aliases `m_grouped_fp8_gemm_{nt,nn}_contiguous`
   and a `grouped_layout` that is either per-row indices or per-expert prefix sums ([API]). Slides use the current alias.
4. **Performance numbers are from March 2025 (H800).** The README dropped its performance tables in April 2025
   (PR #86); newer releases claim "much faster MoE GEMM" ([PR304]) without published grouped-GEMM tables, so the
   slides show the last published table and say so.

## 34 · Masked grouped GEMM (CUDA Graph friendly)

[deck](../decks/34-masked-grouped-gemm-cuda-graph-friendly/) · [Notion row](https://app.notion.com/p/d8af0edc72e64994bbb502dc9a3bfcea)

1. **Notion "removes the per-MoE-layer host sync".** Correct in spirit, but the sync is removed by the pair
   (low-latency dispatch + masked GEMM); the GEMM alone only removes the need for host-side shapes. Normal dispatch
   still blocks the CPU ([EP1] l.173). Slides credit the pair.
2. **Notion "it consumes DeepEP low-latency kernel output directly".** True for DeepEP V1's `low_latency_dispatch`
   ([EP1-BUF]); DeepEP V2.5 removed the V1 API ([EP2] l.19) and its decode path (`do_cpu_sync=False`) emits an
   expanded 2D layout with GPU-side prefix sums, which DeepGEMM consumes through the contiguous kernel's psum layout
   with `expected_m_for_psum_layout` ([EP2] l.163; [TEST] l.203-205). Slides present V1 masked as the original
   pairing and the psum layout as its successor.
3. **"rollout".** The primary sources describe inference decoding only; RL rollout is decoding, so it applies, but
   no source measures it. Slides do not claim rollout numbers.
4. **No measured host-sync cost.** None of the sources reports the latency saved by removing the sync; the 58 on
   the pain slide is a derived count and is labelled as such.
5. **Performance numbers are from March 2025 (H800)**, the last published masked table ([RD-0325]).

## 35 · Mega MoE fused MegaKernel

[deck](../decks/35-mega-moe-fused-megakernel/) · [Notion row](https://app.notion.com/p/77562a16a085453aa50e5744e3382f7e)

1. **Notion "eliminates … intermediate buffer write-backs".** Not all of them: the post-SwiGLU FP8 activation
   still goes through a global-memory ring buffer between linear 1 and linear 2 ([K] L1 epilogue TMA-stores
   into `l2_token_buffer`; [TEST]'s HBM model counts "L1 output write" and "L2 acts read"). What disappears
   is the BF16 [n, 2I] linear-1 output and its re-read by a separate SwiGLU kernel, the local BF16 linear-2
   output re-read by the combine kernel, and the kernel boundaries. Slides say "fewer and smaller
   round trips" and give the analytic 10·I → 2·I.
2. **Notion "single-stream overlap is friendlier to numerical reproducibility than dual-stream or micro-batch
   splitting".** No primary source makes this comparison. What the sources show: the fused output is bitwise
   equal to the non-fused path in the test, and the combine sums in a fixed order. Slides state only that.
3. **Notion "linear1 (FP8 x FP4 / FP8 x FP8)".** At first release only FP8×FP4 was supported ([PR304]);
   FP8×FP8 (`mma_type='fp8xfp8'`) and BF16×BF16 (`bf16_mega_moe`, [PR364]) were added later ([RD], [API]).
   Slides say FP8×FP4 or FP8×FP8 (current README).
4. **Paper vs PR on the speedup range.** [V4] gives 1.50–1.73× for general inference and up to 1.96× for
   latency-sensitive cases; [PR316]'s eight points span 1.50–1.96× (V4-Pro at 1 token/rank is 1.61×).
   Slides plot the PR table and quote "1.50–1.96×".
5. **"Waves".** [V4] Fig. 5 describes splitting experts into waves; the released kernel implements this as a
   persistent tile scheduler (linear-1 warmup waves, then interleaved L1/L2 tiles gated by counters) rather
   than explicit expert groups ([SCHED]). Slides describe the code; deck 14 covers the wave model.

## 36 · Route GEMM (BF16 x FP32 via dual-BF16 split)

[deck](../decks/36-route-gemm-bf16-x-fp32-via-dual-bf16-split/) · [Notion row](https://app.notion.com/p/1ca14bb9a0ce4d439cc9a0686b60a02d)

1. **Notion "Up to 3.22x over cuBLAS FP32/TF32"** — the 3.22× is over FP32 cuBLAS only (M=512); over TF32 the
   maximum is 1.78× ([LM] Table 2; [RM] lists both baselines but one number). Slides say "vs FP32 cuBLAS".
2. **Notion / README "keeping FP32 accuracy"** — the split keeps ~16 of FP32's 24 significand bits; the measured
   max abs error vs FP32 cuBLAS is 0.00177, not zero ([LM]). Slides say "close to FP32".
3. **"truncation" vs rounding of w_high** — the LMSYS blog says W_high is formed "by direct truncation"; the
   HPC-Ops API docstring, test, benchmark and SGLang integration all use `w.to(torch.bfloat16)`, which rounds
   to nearest-even. Both give an error ≤ 2^-16·|w| (simulation above). Slides follow the code (round to BF16).
4. **Notion body links** — "FP8 formats for deep learning" and the Megatron-Core MoE paper do not discuss this
   kernel; they were not used.
5. Notion "shared input gathering" — the kernel shares the *X tile load* (one TMA load per K step consumed by
   both WGMMAs); there is no gather. Slides say "loads each x tile once".

## 37 · Ascend · super-operator fusion (MLAPO / Sparse FA / MoE Mega-Fusion)

[deck](../decks/37-ascend-super-operator-fusion/) · [Notion row](https://app.notion.com/p/5926e0ea48474805988cb4d1419bac18)

1. **Notion: "MoE routing + permute + GroupedGEMM become a Mega-Fusion".** The GLM-5 report names only
   three fusion kernels — Lightning Indexer, Sparse Flash Attention and MLAPO ([GLM5] §5 p.21-22) — and
   says nothing about a MoE Mega-Fusion. The Ascend software stack does fuse grouped matmul + SwiGLU +
   quantization ([VA-MOE]), and Pangu Ultra MoE fuses GroupedMatMul with gradient accumulation and fuses
   permute/unpermute for training ([PANGU] §4.5), but neither is attributed to GLM-5. The deck presents
   the three kernels the report describes and flags the MoE fusion as stack-level, not GLM-5-reported.
   (GPU-side Mega MoE is deck 35.)
2. **Notion: "Sharply reduces kernel-launch and intermediate memory traffic".** The report's stated
   benefits are "overlap computation with memory access" (indexer) and "parallel processing between Vector
   and Cube units" (MLAPO); it publishes **no** kernel-launch counts, no memory-traffic numbers and no
   per-kernel speedups. The deck's traffic number is analytic and labelled as such.
3. **Notion: "the key to domestic hardware matching GPUs".** The only end-to-end claim in the report is
   that "GLM-5 on a single Chinese node achieves performance comparable to dual-GPU international
   clusters, while reducing deployment costs in long-sequence scenarios by 50%", and it attributes this to
   all the hardware-level co-optimizations together (quantization + fusion + engine scheduling), not to
   fusion alone ([GLM5] §5 p.22). The deck keeps that attribution.
4. **744B vs 750B.** §2.1 and Table 10 give 744B total parameters; §5 says "the 750B parameter GLM-5
   model". The deck uses 744B and mentions the single-node target without repeating 750B.
5. **80 layers vs Table 10.** §2.1 says the layer count is reduced to 80; Table 10 lists 3 dense + 75 MoE
   + 1 MTP = 79, and vLLM-Ascend's GLM-5 deployment guide partitions "78 layers ... as 41/37" across two
   pipeline ranks. The derived kernel count uses 80 and is stated as "≈1,000", which is insensitive to the
   difference (13 × 78 = 1,014).
6. **"dual-GPU international clusters" is undefined** in the report: no GPU model, node count, batch size,
   sequence length or latency target is given, so the deck reports the claim as the report words it and
   does not turn it into a speedup number.
7. **Hardware scope.** The report says "Ascend Atlas series" / "Atlas 800T A3". The vLLM-Ascend kernels
   used as corroboration declare support for Atlas A2/A3 and, for some newer kernels, Ascend 950 only
   ([VA-IDX], [VA-SFA] product tables). The deck notes that the kernels are tied to specific NPU
   generations rather than implying one portable implementation.

## 38 · CUDA Graphs (full-iteration / per-layer / TE graphed callables)

[deck](../decks/38-cuda-graphs/) · [Notion row](https://app.notion.com/p/7d97c06866b54afdb590597a7dc39756)

1. **Notion "transformer_engine (… FP8- and PP-schedule-aware)" as if only TE handles FP8/PP.** [BG] states
   that the MCore `local` CudaGraphManager also handles FP8 (deferred amax, weight caching) and PP (recorded
   order) automatically; the TE path differs by capturing ahead of time from an explicit `_order` and by
   needing explicit helper calls. Slides describe all three.
2. **Notion "Removes kernel-launch overhead".** The primary docs scope it more precisely: full_iteration
   excludes the optimizer step; per-layer graphs leave everything between graphs (and dropless expert compute)
   eager ([DOC]; [MC] p.41).
3. **Background doc vs current code on memory pools.** [BG] (written for core_v0.14.0) says the default with
   PP uses a separate pool per microbatch with graph reuse (e.g. 160 graphs for 80 layers); the paper and the
   current code create one graph pair per layer per microbatch, all in one global mempool
   (`CudaGraphManager.global_mempool`; `reuse_cudagraphs` only when PP = 1), i.e. L×M×2 graphs
   ([MC] p.42; [CG]). Slides follow the paper/code.

## 39 · DeepJIT runtime compilation + PDL + determinism switches

[deck](../decks/39-deepjit-runtime-compilation-pdl-determinism/) · [Notion row](https://app.notion.com/p/1ebac796b72b428f9c32afc18e7b5fd4)

1. **Notion Core Mechanism** says "a low-CPU-overhead C++ JIT module with on-disk cache". Since the
   2026-09 release that module is a **separate project, DeepJIT** (S3.12, S1.1), header-only, with
   both CUDA and Ascend backends and a cache that can be shared across users and nodes (S1.5). The
   slides name DeepJIT as the external component.
2. **Notion Impact** "determinism switches serve reproducible training" — correct, but the primary
   source adds the direction of the default: before the 2026-09 release *all* kernels were
   deterministic and batch-invariant; now determinism is **opt-in** (S6.1). The slides state this.
3. The Notion row lists TileLang as a paper reference; TileLang is a separate DSL used by DeepSeek-V4
   for other kernels (V4 §3.2) and is not part of DeepJIT. Its determinism work is mentioned only as
   context (S5.4), and the sibling deck 43 covers TileLang.
4. No numeric conflict was found between the Notion row and the repositories.

## 40 · Fused Sampler (decode post-processing fusion)

[deck](../decks/40-fused-sampler-decode-post-processing-fusion/) · [Notion row](https://app.notion.com/p/3070daaed63740e3be901cba2641e152)

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

## 41 · Mega Gate / Mega mHC fused kernels

[deck](../decks/41-mega-gate-mega-mhc-fused-kernels/) · [Notion row](https://app.notion.com/p/0bb7a1acbf924329aab698ccc8c5ffee)

1. **Notion "brings mHC's extra memory traffic into an acceptable range".** The quantified "acceptable range"
   in the primary sources (6.7% overhead) was achieved by the mHC paper's TileLang kernels plus recomputation
   and DualPipe changes in training ([MHC] §4.3, [V4] §3.4.2), before Mega mHC existed (2026-09). DeepGEMM
   reports Mega mHC only as 45–85% faster than that multi-kernel path ([PR432]). Slides keep the two apart.
2. **Notion "avoiding … intermediate tensors".** Mega mHC still writes the new residual (it is an output) and,
   per the test's own byte model, materialises the norm boundary and re-reads the new residual in normal
   mode (kept in L2 by an EVICT_LAST hint); Mega Gate still passes scores through a workspace when experts
   or K are split across CTAs. What disappears are the kernel boundaries and the re-read of the residual
   for the 24-wide GEMM. Slides say "one pass over the stream", not "no intermediates".
3. **Notion source "DeepSeek V4".** The V4 report describes fused mHC kernels for training and inference
   but does not name Mega mHC or Mega Gate; those names and numbers come from DeepGEMM PR #432.
4. **Sinkhorn iterations.** V4 uses t_max = 20 ([V4], [MHC]); the DeepGEMM test uses 10 as a parameter
   (`num_sinkhorn_iters` is an argument). Slides quote 20 for V4.
5. **H_post scale.** [V4]/[MHC] define H_post = 2σ(·); the DeepGEMM API takes `hc_post_scale` as an argument
   (the test uses 1.25). Slides say "σ × scale".

## 42 · Router / permutation / MLA-RoPE fused kernels

[deck](../decks/42-router-permutation-mla-rope-fused-kernels/) · [Notion row](https://app.notion.com/p/9d82ddbf31244093bc930a84ed679c2d)

1. **Notion "fuses routing score/topk/bias, layered with permutation, MLA RoPE and FP8-cast fusion, plus cuDNN
   fused attention / FlashAttention integration".** The primary sources describe router, aux-loss,
   permutation(+padding), MLA-RoPE and quantization fusions; fused attention (cuDNN) is listed separately in
   the README feature list. The deck covers the fused glue kernels and mentions attention only as context.
2. **Notion "Impact: cuts launch and memory-traffic overhead".** No isolated end-to-end number is published
   for router fusion or MLA-RoPE fusion; measured numbers exist only for permute/pad fusion ([TE-1921]) and for
   the fused grouped-GEMM epilogues ([BLOG-F]). The slides say so.
3. **Paper Fig. 21 vs TE kernels.** The paper describes "three sections, each fused into a single kernel"; TE
   also has backward kernels for each and a quantile-balancing variant; the slides count forward kernels.

## 43 · TileLang DSL + batch-invariant deterministic kernels

[deck](../decks/43-tilelang-dsl-batch-invariant-deterministic/) · [Notion row](https://app.notion.com/p/22c7e2d5f253481981afb6c6ee06e3f5)

1. **"Outputs are independent of batch position and parallel strategy."** [V4] defines batch invariance as
   independence from the token's position within a batch and claims bitwise alignment across pre-training,
   post-training and inference pipelines. It does not claim invariance across parallel strategies (TP/EP/PP
   layouts). Slides do not claim it.
2. **"The backward pass replaces atomicAdd with per-SM accumulation buckets."** In [V4] this is the sparse-attention
   backward only; MoE backward uses token-order pre-processing plus buffer isolation, and the mHC GEMM uses
   separate split outputs plus a deterministic reduction. Slides show all three.
3. **"Sharply reducing irreproducibility in RL and debugging."** [V4] reports no measurement of the effect (nor of
   the overhead: "minimal", "negligible"). The 80-of-1000 number on slide 3 is Thinking Machines' measurement on
   Qwen3-235B in vLLM, used to show the size of the problem, not a DeepSeek result.
4. **"Full waves on a single SM, tail wave via distributed shared memory"** matches [V4]; [V4] places this in
   batch-invariant *decoding* attention.

## 44 · Ascend · W4A8/W8A8 hybrid quantization stack

[deck](../decks/44-ascend-w4a8-w8a8-hybrid-quantization-stack/) · [Notion row](https://app.notion.com/p/38d6d033f1b943b3b79615c26f4f0a2e)

1. **Notion: "Flex SmoothQuant searches per-layer migration strength".** The report names
   **Flex_AWQ_SSZ**, not SmoothQuant; SmoothQuant appears nowhere in the GLM-5 report. The deck presents
   AWQ-style scaling search as the named algorithm and cites SmoothQuant only as the lineage of the
   migrate-the-difficulty idea (including its α = 0.75 for GLM-130B, which is a *SmoothQuant* result about
   an older GLM model, not a GLM-5 number).
2. **Notion: "SSZ (scale-shift-zero) handles weight quantization".** The report writes "Flex_AWQ_SSZ"
   as one name for scaling calibration and never defines SSZ. The deck says the name, flags that it is
   undefined in the report, and points at the per-channel scale/offset/scale_bias actually present in
   msModelSlim W4A8 checkpoints.
3. **Notion: "W8A8 kept for sensitive layers".** The report's wording is structural, not sensitivity-
   based: "standard Attention and MLP blocks use W8A8 (INT8), while the MoE experts are compressed to
   W4A8 (INT4)". The deck uses the report's wording and gives the memory reason (experts are 97% of the
   weights, derived).
4. **Notion: "Halves weight memory at near-lossless accuracy, with large throughput gains".**
   The report gives **no** accuracy numbers, **no** memory numbers and **no** throughput numbers for the
   quantized model; it says W4A8 reduces the memory footprint "without significant accuracy loss", and
   the only end-to-end number is the 50% long-sequence deployment-cost claim for the whole co-optimization
   set. Against INT8 the hybrid plan is a 1.95× reduction (derived), i.e. "halves" is right for
   experts-vs-INT8 and understates the gap to BF16 (3.9×). The deck labels all memory numbers derived and
   says plainly that accuracy numbers are not published.
5. **"Fit onto a single Atlas 800T A3 machine" (report) vs the deployment guide.** The report motivates
   W4A8 by fitting the model on one machine; vLLM-Ascend lists *both* `GLM-5-w4a8` and `GLM-5-w8a8` as
   deployable on one Atlas 800 A3 node, and recommends w4a8 for long contexts because it leaves more KV
   cache. The deck follows the deployment guide on this point and presents the memory as KV-cache
   headroom rather than "it would not fit at all".
6. **744B vs 750B / layer count.** Table 10 gives 744B parameters and 3 + 75 + 1 layers; §5 says "the
   750B parameter GLM-5 model"; the deployment guide partitions "78 layers". The deck uses Table 10 for
   every derived number.
7. **QuaRot's published numbers are not GLM-5's.** The 0.47 perplexity / 99% zero-shot / 3.33× prefill
   figures are LLaMA2-70B on GPUs from the QuaRot paper, used on the accuracy slide as evidence about the
   *method*, labelled as such. No public accuracy number for GLM-5-w4a8 was found.

## 45 · FP4 QAT (MXFP4 expert weights + indexer QK path)

[deck](../decks/45-fp4-qat-mxfp4-expert-weights-indexer-qk-path/) · [Notion row](https://app.notion.com/p/ad31c31051744ea2b53ea97c1417ac71)

1. **"mathematically lossless".** [V4] makes it conditional: FP4→FP8 dequantization is lossless only
   while the ratio between the largest and smallest FP4 sub-block (1×32) scale inside each 128×128 FP8
   block stays below a threshold, which DeepSeek "empirically verif[ied]" holds for the current weights
   (§5.2.1). The slides say "exact when scale ratios fit".
2. **"backward uses STE".** The paper's wording: gradients are computed w.r.t. the same FP8 weights and
   propagated directly to the FP32 master weights, "equivalent to" an STE through the quantizer. Same
   meaning; slides say so.
3. **"2x faster top-k selector with KV recall held at 99.7%".** [V4] attributes the 2× and 99.7% to
   quantizing the index scores FP32 → BF16 within the QAT process (sentence order in §5.2.1), alongside
   the FP4 QK path; Notion reads as if the whole FP4 QAT produced it. The slide says "after QAT: FP4 QK,
   BF16 scores" and cites §5.2.1.
4. **"deployment runs the FP4 weights directly".** [V4] states this for the inference and RL-rollout
   phases (native FP4 weights instead of simulated quantization) so sampling matches online deployment.
   Consistent.
5. Notion omits that the FP4 weights give **no FLOP gain on current hardware** (FP4×FP8 peak = FP8×FP8,
   [V4] §1); the benefit today is memory and memory traffic. Added on the limits slide.

## 46 · FP8 training (per-tensor / blockwise) and FP8 activation compression

[deck](../decks/46-fp8-training-and-fp8-activation-compression/) · [Notion row](https://app.notion.com/p/8406549e9fd645e49d2f2e90a838f1bd)

1. **"activation memory halved"** — The sources halve only the FP8-eligible tensors (linear-layer inputs saved
   for backward). For DeepSeek-V3 the whole activation budget drops by ~16 GB of 131 GB, i.e. ~12%
   ([MCore-MoE] Sec. 4.1.3). Kimi K2 reports no memory number at all, only "no measurable loss increase"
   ([K2] Sec. 2.4.3). Slides say "50% per cached tensor, ~12% of DeepSeek-V3's activations".
2. **"FP8 parameter all-gather cuts roughly 50% of that communication"** — Correct as a byte count (1 vs 2
   bytes/param) for per-tensor and blockwise FP8 ([MCore-MoE] Sec. 5.2.2), but it is a design property, not a
   measured result, and it does not hold for MXFP8, which all-gathers in BF16 because both quantization
   directions would be needed.
3. **"the default recipe for DeepSeek-V3-class models on H100"** — Megatron-Core recommends blockwise FP8 on
   Hopper and validated it on DeepSeek-V3 ([MCore-README]); DeepSeek-V3 itself was trained on H800, not H100
   ([DSV3]). Slides say "Hopper (H100/H800)".
4. **"K2 compresses activations in FP8-E4M3"** — True, but storage only: K2 explicitly does *not* compute in
   FP8 because of risks seen in preliminary studies ([K2] Sec. 2.4.3). The Notion row's framing under "FP8
   training" could suggest K2 trains with FP8 GEMMs; it does not.
5. **"TE provides DelayedScaling / blockwise FP8 recipes"** — TE also provides Float8CurrentScaling; Megatron-Core
   does not recommend DelayedScaling for precision reasons ([MCore-MoE] Sec. 5.3.1). Not a contradiction, an
   omission.

## 47 · MXFP8 micro-scaling training

[deck](../decks/47-mxfp8-micro-scaling-training/) · [Notion row](https://app.notion.com/p/02c9479032474c7e8e387fb2e6b7a964)

1. **"1.28-1.37x over BF16 on B200"** — Not found in the MXFP8 paper (v1 or v2), the TE docs, the Megatron-Core
   report or README, or the NVIDIA scaling blog. The paper reports no measured speedup at all (its accuracy runs
   were simulated on Hopper). Measured numbers available: end-to-end 1.22x tokens/s on GB200 for DeepSeek-V3
   and Qwen3-235B (derived from [MCore-MoE] Table 11) and GEMM-level 1.44x on B300 ([TE-speed]). Slides use
   these.
2. **"up to 2x on GEMMs"** — The 2x is the peak Tensor Core math rate of MXFP8 vs BF16 ([MX] Table 1), not a
   measured GEMM speedup. Slides call it "peak math rate".
3. **"loss curves indistinguishable from BF16"** — The paper quantifies "< 0.50% validation-perplexity difference"
   (8B, 15T tokens) and "within 0.1%" final loss (16B MoE, 1T tokens), with MXFP8 numerics simulated on Hopper.
4. **"requires keeping both transposed and non-transposed copies"** — Correct, but they are not transposes of each
   other: rowwise (1x32) and columnwise (32x1) copies are quantized separately from high precision; Blackwell
   needs no physical transpose ([TE-mx]).
5. **"covers more layers than per-tensor FP8"** — Consistent: MXFP8 quantizes all transformer blocks, whereas the
   per-tensor FP8 baseline keeps the first and last blocks in BF16 ([MX-v1], [MX] Sec. 3.2.1). Embedding, output
   layer and attention BMMs remain BF16 in MXFP8 too.
6. Notion lists only TransformerEngine as source; Megatron-Core is where MXFP8 is the Blackwell default and where
   the end-to-end numbers come from.

## 48 · NVFP4 pretraining recipe

[deck](../decks/48-nvfp4-pretraining-recipe/) · [Notion row](https://app.notion.com/p/eae9504538664595866784566f5d14cd)

1. **"a further 1.31-1.73x over FP8 on GB200/GB300"** — Not found in the NVFP4 paper (which states it is about
   algorithms, not runtime), the TE docs, the Megatron-Core report or the three NVIDIA blogs read. What the sources
   give: peak FP4 math 2x (GB200) / 3x (GB300) the FP8 rate ([NV4] Table 1, Sec. 2); measured GEMM speedup on B300
   NVFP4 2.03x vs BF16, i.e. 1.41x over MXFP8 and 1.26x over FP8 delayed scaling (derived from [TE-speed]); MLPerf
   405B at 1.9x vs a prior FP8 round on older GPUs ([NV-3ways]). Slides use the TE GEMM numbers.
2. **"<1% loss gap vs FP8 over 10T tokens (~1.5% late in training)"** — Consistent with [NV4] Sec. 3 ("below 1%"
   in the stable phase, "slightly above 1.5%" at the end of decay).
3. **"E2M1 with two-dimensional block scaling"** — The NVFP4 *format* uses 1x16 E4M3 block scales plus an FP32
   tensor scale; 2D (16x16) scaling is a recipe choice applied only to weights ([NV4] Sec. 4.3; [TE-nv4]).
4. **"Random Hadamard transforms spread outliers"** — Applied only to Wgrad inputs (16x16); Fprop/Dgrad transforms
   gave no benefit or hurt in ablations ([NV4] Sec. 4.2, App. E.4.1).
5. **"sensitive layers stay high precision (bf16_with_nvfp4_mixed)"** — In Megatron-Bridge, `bf16_with_nvfp4_mixed`
   only switches GEMMs to NVFP4 (fp4 e2m1, FP4 param gather); BF16 layers are set separately
   (`first_last_layers_bf16`, `num_layers_at_end_in_bf16`; the Nemotron-3 Super/Ultra variants keep the last 14/16
   layers) ([MB]). The paper's 12B kept the first 2 and last 8 blocks (16% of linear layers) ([NV4] App. A.1).
6. Notion omits that the FP8 baseline itself is the blockwise DeepSeek-style recipe with BF16 first/last blocks
   ([NV4] App. A.1), and that evidence is a 12B dense hybrid model; MoE is listed as future work ([NV4] Sec. 6).

## 49 · Precision-aware optimizer (BF16/FP8 optimizer states)

[deck](../decks/49-precision-aware-optimizer/) · [Notion row](https://app.notion.com/p/893ccbb87b354e7e8fedbc0a18ca7117)

1. **Notion "keep first/second moments and master weights in BF16/FP8".** Master weights cannot be BF16 or FP8 in
   this optimizer: `--main-params-dtype` accepts only fp32 or fp16, and FusedAdam supports FP32/FP16 master
   weights ([ARGS], [TEA]). Only the moments accept bf16/fp8. With BF16 parameters the FP32 master can be stored
   as a 16-bit remainder ([OC], [TEK]), which is exact FP32, not reduced precision. Slides follow the code.
2. **Notion "standard for trillion-parameter MoEs under limited HBM".** The sources show it in MCore's DeepSeek-V3
   recipes (685B) on H100 and GB200 ([MC] Table 18) and in DeepSeek-V3's own training ([DSV3]); no
   trillion-parameter run is documented in the sources read. Slides say "DeepSeek-V3 recipes".
3. **Notion "Large reduction in optimizer-state memory".** Quantified in [MC] as ~50% of optimizer-state memory,
   10–12 GB of 32.1 GB for DeepSeek-V3 (BF16 moments). Slides use these numbers.
4. **Inside [MC]:** §4.1.6 calls the 32.1 GB "16% of the total" (32.1 / 199.5 = 16.1% ✓); the "~50% reduction"
   refers to the optimizer-state part (moments 8 → 4 B); measured on the whole 32.1 GB budget it is 10–12 GB
   (31–37%). Slides quote "10–12 GB of 32.1 GB".

## 50 · Distributed checkpointing and elastic-resource training

[deck](../decks/50-distributed-checkpointing-and-elastic-resource/) · [Notion row](https://app.notion.com/p/a065908961604600811834b0c1800744)

1. **Notion: "training continues through node-count fluctuation."** K2 says its strategy "allows Kimi K2
   to be trained on any number of nodes that is a multiple of 32" under "dynamic resource availability"
   and reuses one configuration for small and large runs; it does not describe changing node count
   inside one uninterrupted job, nor its checkpoint format. Slides say "restart on any multiple of
   32 nodes", not live elasticity.
2. **Notion: VeOmni "uses Torch Distributed Checkpoint".** True for the open-source repo ([V-CKPT],
   README); the VeOmni paper (§3.2.5) describes ByteCheckpoint instead. Slides follow the repo.
3. **Notion: "Changing parallelism or hardware needs no checkpoint rewrite."** Holds for weights and
   optimizer state (DCP load-time resharding). Per-rank data cursors and RNG state do not reshard: a job
   whose new ranks lack cursor files resumes weights only ([V-GS]); DCP gives no backward-compatibility
   guarantee across PyTorch versions ([PTD]). Slides list these as limits.
4. **K2 checkpoint-engine timing.** The K2 paper says "less than 30 seconds"; the checkpoint-engine
   README says "about 20s" and its table shows 14.36-16.04 s for K2-Instruct FP8 on H20 inference
   clusters. Different hardware/precision; slides use the README table for the chart and quote the paper's
   bound in the notes.

## 51 · Dynamic batching / sequence packing

[deck](../decks/51-dynamic-batching-sequence-packing/) · [Notion row](https://app.notion.com/p/5cfd3adb77f74e0c8d09435a3af5d9b3)

1. **Notion: "the MCore counterpart is the THD packed format (e.g. GLM-5.2's 128K THD packed CP
   training)".** The GLM-5 report (arXiv 2602.15763) does not mention THD, packing format, or a
   "GLM-5.2"; it describes workload-aware sequence reordering and variable-size CP groups for long
   sequences, and a 4K → 200K context schedule. Slides cite GLM-5 only for what the report says; THD is
   attributed to Megatron-Core ([MC] §6.4.1).
2. **Notion: "Large gains in effective FLOP utilization."** No VeOmni measurement isolates packing.
   The quantified gains come from Megatron-Core (1.5–2× throughput, 40–60% memory for RL thinking models;
   Dynamic-CP 1.25–1.48× TFLOPS/GPU on Llama 13B). Slides attribute each number to its source.
3. **Notion: "Buffers samples and packs them to a target sequence length".** Matches [VO] §3.2.5 and
   [V-BATCH]; note the target is a token budget per micro-batch, and with `dyn_bsz` off VeOmni still
   packs (fixed sample count) ([V-PACK]).

## 52 · Fine-grained selective recomputation

[deck](../decks/52-fine-grained-selective-recomputation/) · [Notion row](https://app.notion.com/p/2e31e16f18ce45edba1e3d9058756c08)

1. **Module list.** Notion lists "mla_up_proj / moe / moe_act / layernorm / norm". There is no `norm`
   module; the supported list is `core_attn, moe_act, layernorm, mla_up_proj, mlp, moe, shared_experts`
   (plus newer `gdn_norm_out, gdp_in_proj, gdp_qkv, mhc, shortcut_pre_mlp_layernorm, residual_stream`)
   ([CFG], [RD]). Also, `moe` is not fine-grained: it uses standard checkpointing and re-runs the experts and
   their all-to-alls. Slides use the code's list and mark `moe` as the expensive one.
2. **"Negligible extra compute".** The paper says "less than 5%" ([MC] p.23); slides use "< 5%, not zero".
   The "30%+" full-recompute figure matches the paper's "~33%" (p.22).
3. **"Works under FP8".** True since MCore v0.14, with constraints: TE ≥ 2.6.0dev0 for moe_act/layernorm and
   no delayed scaling ([CFG]). Slides state the constraints.
4. **Table 17 vs App. B.2 (DeepSeek-V3 recompute set).** Table 17 / §9.2.2 list H100 = `mlp, mla_up_proj,
   moe_act, layernorm`, GB200 = `mlp`; App. B.2 (Table 11 runs) lists "None on GB300; mlp on GB200; up_proj,
   mlp on H100". Slides use Table 17 (the case study the paper discusses).
5. **Parameter count.** Megatron-Core says 685B (includes MTP); the DeepSeek-V3 report says 671B. Slides use
   685B (as in the source of the memory numbers).
6. **"89.8 GB globally"** ([MC] §6.2) is not stated per GPU; slides say "in total" and do not compare it
   numerically with per-GPU figures.

## 53 · Memory-efficient token permutation

[deck](../decks/53-memory-efficient-token-permutation/) · [Notion row](https://app.notion.com/p/3a91e54ff5c7400e9b2e65e0ba6479f5)

1. **"Removes one full activation copy."** The source removes the expert output kept for the router-weight
   gradient: one hidden-size row per routed token copy per MoE layer, about 26.3 GB per GPU on DeepSeek-V3
   (≈ 20% of activations, derived) — not a copy of all activations. Slides use the measured number.
2. **"Into the GroupedMLP activation function."** Current code applies the probs inside the activation of every
   expert backend (TEGroupedMLP, and SequentialMLP via `per_token_scale`), not only GroupedMLP ([EXP]). The
   README wording (v0.13) matches Notion.
3. **"Avoiding an extra large buffer allocation."** The paper frames the saving as tensors no longer *retained*
   for backward (peak activation memory), not an allocation; slides use the paper's framing.
4. **"Low-cost" vs "zero overhead".** The paper and Table 6 list no trade-off; the code shows one small cost
   the paper does not mention: probs need their own all-to-all (one scalar per routed row). Slides mention it
   as a limit, with the "negligible" judgment marked analytic.
5. **Reference.** Notion's reference (the MoE README) has no numbers; all numbers come from arXiv 2603.07685,
   so the summary QR points to the paper.

## 54 · Pipeline-aware activation / optimizer offload

[deck](../decks/54-pipeline-aware-activation-optimizer-offload/) · [Notion row](https://app.notion.com/p/431ab65669934bd2af65da396bb96f7f)

1. **"Offloads activations to CPU per pipeline stage."** The mechanism is per *module* (each listed submodule's saved
   input), tracked per (micro-batch, VPP chunk) handler; PP awareness is in the ordering and the per-rank amount
   ([MC] p.25, [DOC]). Slides describe the module granularity.
2. **`--offload-optimizer-states`.** This flag appears in the paper (Table 13) and on the Megatron-LM dev branch, where
   it is now a deprecated alias of `--chunked-optimizer-state-offload`; main and the MoE README document
   `--optimizer-cpu-offload`, which is a different mechanism (the step runs on the CPU) ([ARG], [OCFG], [RD]).
   Slides show both and label which is which.
3. **"Sustains stable trillion-scale MoE training on H800 clusters with 2TB of host memory per node."** The hardware
   facts are Kimi K2's ([K2] p.7) and K2 does offload all remaining activations, but the report does not attribute
   training *stability* to offload (its stability claim is about MuonClip / zero loss spikes) and gives no offload
   measurement; K3's report does not name its GPUs. The measured numbers on the slides come from Megatron-Core
   (Table 5, §4.1.6, §6.2).
4. **"K2/K3 combine CPU activation offload with selective recomputation."** Correct, and both also compress
   activations (K2: FP8 storage; K3: block-wise FP8 + remote offload to other PP ranks) ([K2], [K3]).
5. **Table 5 hardware** is not stated; the DeepSeek-V3 row uses MXFP8 (a Blackwell format) and 169 GB exceeds an
   80 GB H100, so it is a Blackwell-class run. Slides do not name the GPU for Table 5.

## 55 · Loss-free balance routing + sigmoid gating

[deck](../decks/55-loss-free-balance-routing-sigmoid-gating/) · [Notion row](https://app.notion.com/p/efa722c88bed4a4cb1baa6d8d0a0d6fa)

1. **"A learnable per-expert bias."** The bias is not learned by back-propagation: [LF] Algorithm 1 updates it with a fixed ±u sign step from token counts, and MCore stores it as a non-gradient buffer ([CODE]). ([MC] §7.1 also uses the word "learnable"; the slides say "feedback-updated".)
2. **"Instead of an auxiliary loss."** Every production recipe found keeps a tiny complementary sequence-wise balance loss (weight 0.0001: [V3] p.23, [GLM] §2.4, [V4] p.25-26). The original paper uses no aux loss at all.
3. **"Scoring uses sigmoid."** Sigmoid is the scoring function in [LF]'s main runs, [V3] and GLM-4.5, but the method also works with softmax (with a proportional update, [LF] App. C) and DeepSeek-V4 replaced Sigmoid with Sqrt(Softplus) ([V4] §2.1).
4. **"Fixed update rate."** Correct for the rule, but both DeepSeek-V3 (last 500B tokens) and GLM-4.5 (after 15T tokens) set the rate to 0.0 late in training, i.e. the bias is frozen ([V3] p.23; [GLM] §2.4).
5. **"Now the default routing setup."** True for the open frontier models checked (DeepSeek-V3/V4, GLM-4.5; Kimi K3 keeps bias routing but replaces the update rule), but in Megatron-Core it is opt-in: `--moe-router-enable-expert-bias` defaults to False and `--moe-router-score-function` defaults to softmax ([RD]).
6. **Reference.** The Notion reference (GLM-4.5, arXiv 2508.06471) only adopts the technique (one sentence in §2.1 and its hyper-parameters in §2.4); the method and all controlled experiments come from DeepSeek's arXiv 2408.15664 and DeepSeek-V3 §4.5.2. The deck cites both.
7. **Source "DeepSeek V3.2".** DeepSeek-V3.2 was not read; the deck uses DeepSeek-V3 (origin of the production recipe) and DeepSeek-V4.

## 56 · Quantile Balancing

[deck](../decks/56-quantile-balancing/) · [Notion row](https://app.notion.com/p/b59b1fb31a654c95a2957772ad9e4d92)

1. **"unstable at ~1k experts"**: K3 says balancing nearly 10³ experts "exceeds the regime in which existing
   auxiliary-loss-free bias updates remain well behaved" (p.6-7) but publishes no measurement of the
   instability. The deck uses a third-party measurement ([BIP]) for the trend and says so.
2. **"directly improves EP training throughput"**: K3 gives no throughput number for QB. Imbalance is said to
   slow EP training (p.8, citing Step 3.5 Flash); K3 attributes per-micro-batch EP balance and its throughput
   to MoonEP (§5.2.1). The deck does not claim a throughput gain for QB.
3. **"Adjusts the routing distribution directly from quantile statistics"**: QB sets the per-expert selection
   bias; the mixture weights p_ij exclude the bias (Eq. 13), so the routing *weights* are unchanged; only
   which experts are selected changes.
4. **Bin count**: K3 §2.3.3 says "a few hundred bins per expert"; App. D quotes the error bound for B = 1000.
   The deck uses B = 1000 only as the error-bound case and as an upper bound in the analytic chart.
5. **Source-internal ([BIP] Table 1)**: the "Routed experts" and "Activated routed experts" rows are swapped
   (4/8 vs 16/64); the text and table captions give 16 experts with k = 4 and 64 experts with k = 8. The deck
   uses the captions.

## 57 · Stable LatentMoE

[deck](../decks/57-stable-latentmoe/) · [Notion row](https://app.notion.com/p/fafccf6521d44c569d9d5cfad419049f)

1. **"RMSNorm on routed experts"**: K3 applies one RMSNorm to the *aggregated* routed latent u (after the
   weighted sum over the selected experts, before W↑), not inside or per routed expert (Eq. 11; [K3-cfg]
   `routed_expert_norm` applied to the combined output). Slides follow K3.
2. **"an MLA analogue"**: K3 cites LatentMoE (NVIDIA, arXiv 2601.18089) as the origin of the latent routed path
   and does not describe it as an MLA analogue; the analogy (compress to a latent, expand later) is Notion's.
   The deck credits LatentMoE and mentions MLA only as an analogy in the notes.
3. **"Trains stably"**: K3 states the stabilizers suppress activation explosion but publishes no loss or
   activation curves isolating them; "trains stably" is an inference from the completed 2.8T run.
4. **Notion body** lists DeepSeek-V2/MLA as a paper but not LatentMoE, which is the actual base design.

## 58 · Ultra-sparse MoE (fine-grained routed experts + shared expert)

[deck](../decks/58-ultra-sparse-moe/) · [Notion row](https://app.notion.com/p/b2283eba9cf14a8fb2fa3f7b2ba4127f)

1. **Kimi K2 activated parameters.** Notion says "K2 1.04T/32B"; the K2 abstract says 32B, Table 2 says 32.6B.
   Slides use 32.6B where a ratio is computed and "32B" nowhere else.
2. **"Total and activated parameters fully decoupled".** Notion's wording; the sources show the two are decoupled
   only up to infrastructure limits: K2 explicitly stops at sparsity 48 because of infrastructure complexity, and
   GLM-5 reduces its layer count to cut EP communication. Slides present sparsity as a dial bounded by infra cost.
3. **"Sparsity chosen by scaling laws".** Only Kimi K2 publishes a sparsity scaling law. DeepSeek-V4 and GLM-5 state
   their configurations without a published sparsity sweep. Slides attribute the law to K2 only.
4. **Not every model uses "one shared expert + fine-grained routed" identically**: V4 also hash-routes the first 3 MoE
   layers and K2 keeps 1 dense layer (GLM-5 keeps 3 dense layers). Mentioned in speaker notes.

## 59 · Attention Residuals (AttnRes / Block AttnRes)

[deck](../decks/59-attention-residuals-attnres-block-attnres/) · [Notion row](https://app.notion.com/p/8b86f64fd2d44d4bac732d2321f7722a)

1. **"improves every task on Kimi Linear 48B / 1.4T tokens"**: Table 3 shows MMLU-Pro unchanged (52.2 vs 52.2);
   the paper's text says "matches or outperforms the baseline on all benchmarks" (§5.2) while its abstract says
   "improves ... across all evaluated tasks". The deck says "matches or beats on all 15".
2. **"output magnitude and gradients stay uniform with depth"**: [AR] says output magnitudes stay *bounded* with
   a periodic pattern within blocks and gradients are "substantially more uniform" (Fig. 5), not uniform.
3. **Notion source is Kimi K3, reference is arXiv 2603.15031**: the measured results all come from the AttnRes
   report (Kimi Linear 48B and the scaling runs); K3 adopts Block AttnRes (8 × 12-layer blocks) but publishes
   no AttnRes ablation of its own. The deck labels results with their model.

## 60 · Deep-and-narrow + more attention heads + partial RoPE

[deck](../decks/60-deep-and-narrow-more-attention-heads-partial/) · [Notion row](https://app.notion.com/p/ba32c8b828d748ea9f32a65e15598576)

1. **"2.5× more heads".** Notion and [GLM45] say 2.5×; the paper does not name the baseline. The width-matched baseline
   5120 / 128 = 40 heads gives 96 / 40 = 2.4×. Slides say "2.4× a width-matched layout (paper: 2.5×)".
2. **Kimi K2 activated parameters.** [GLM45] Table 1 lists 32B; [K2] Table 2 lists 32.6B. Slides use 32B (the GLM table, so
   the three models are counted the same way) and the notes mention 32.6B.
3. **GLM-5 layer count.** [GLM5] §2.1 text says it "reduces its layer count to 80"; [GLM5] Table 10 gives 3 dense + 75 MoE
   (78) and [CFG-GLM5] num_hidden_layers = 78. Slides use 78.
4. **Notion "deeper models reason better; more heads … keep improving MMLU/BBH"** matches [GLM45], but the paper publishes no
   numbers for either ablation, and at the *base-model* level GLM-4.5-Base scores slightly *below* DeepSeek-V3-Base and
   Kimi-K2-Base on MMLU and BBH ([GLM45] Table 2). Slides present the claims as the authors' qualitative findings and show
   the base-model numbers as they are.
5. **Confounders.** GLM-4.5 was pre-trained on 23T tokens vs 14.8T (DSV3) and 15.5T (K2), with different data and Muon; any
   model-level comparison mixes shape with data and optimizer. Stated on the results and limits slides.

## 61 · mHC: manifold-constrained hyper-connections

[deck](../decks/61-mhc-manifold-constrained-hyper-connections/) · [Notion row](https://app.notion.com/p/99ccc6cd324244c0bf114f19b5440674)

1. **"Compound gain at depth 64 goes from 1e16 to about 1"** (Notion). [mHC] reports a peak composite gain of about
   **3000** for HC and a maximum of about **1.6** for mHC in the 27B model (30 layers = 60 sublayers); no 1e16 value or
   depth-64 measurement appears in the paper. Slides use ~3000 → ≤ 1.6.
2. **"A single Sinkhorn iteration suffices"** (Notion). Both [mHC] and [V4] use **t_max = 20**; [mHC] §5.4 notes that
   the finite iteration count already makes the backward gain deviate slightly from 1. The follow-up mHC-lite [lite]
   argues for *replacing* Sinkhorn with explicit convex combinations of permutation matrices, not for one iteration.
   Slides use 20; the toy figure shows column sums after 1 iteration only as an illustration of convergence.
3. **"Also adopted by GLM-5.3"** (Notion). The cached GLM-5 report (arXiv 2602.15763) has no mention of mHC or
   hyper-connections; no GLM-5.3 source was available. The slides do not claim GLM adoption.
4. **"Eliminates HC's gradient explosion at 27B"** — consistent with [mHC] Fig. 2/5 (HC loss surge near step 12k,
   gradient-norm spikes; mHC stable).

## 62 · MTP (multi-token prediction) layer

[deck](../decks/62-mtp-multi-token-prediction-layer/) · [Notion row](https://app.notion.com/p/354b71e013554721a039936a8fff8b2a)

1. **"GLM-5.2's improved MTP raises accepted length by 20%."** Not found in any source read. The GLM-5 report (arXiv 2602.15763, Table 2) gives accept length 2.76 for GLM-5 vs 2.55 for DeepSeek-V3.2 at 4 speculative steps on a private prompt set, i.e. +8% (derived), and it is a cross-model comparison, not an ablation. No GLM-5.2 source was available; the 20% figure is not used on slides.
2. **"Wins on ... training efficiency."** The primary sources show better quality at equal training tokens and identical inference cost ([V3] Table 4; "may improve data efficiency"), but MTP *adds* training compute and memory (an extra Transformer block, extra output-head pass; 14B extra parameters for DeepSeek-V3). The slides say "quality per token", not "efficiency".
3. **"One (MoE) MTP layer on top of the backbone."** Correct for GLM-4.5 (an MoE layer); DeepSeek's MTP module is "a Transformer block" whose block type follows the model; it shares the main model's embedding and output head and is fed the next token's embedding, so it is not just a layer stacked on the backbone.
4. **"MCore supports MTP + MLA + CP together."** Confirmed by the README (v0.14: CP support for MTP and MLA); additionally MCore limits MTP to ≤ 1 layer when EP all-to-all overlap is enabled ([CODE]).
5. **Reference.** The Notion reference (GLM-4.5) states only that an MoE MTP layer is added and gives λ; the module design, ablation and acceptance numbers come from DeepSeek-V3 (arXiv 2412.19437).

## 63 · Anticipatory routing (asynchronous router parameters)

[deck](../decks/63-anticipatory-routing/) · [Notion row](https://app.notion.com/p/ecc924eb12db4b0fa39609cf42a53369)

1. **"Asynchronous router parameters" / "routing decisions use cached older ones".** [V4] computes routing indices
   with the historical *network* parameters θ_{t−Δt}, i.e. the backbone layers that produce the router input as
   well as the router itself, in a forward pass over step t's data run at step t−Δt; what is cached is the indices,
   not router parameters. Slides follow the paper.
2. **"Breaking the anomaly → routing drift → larger anomaly feedback loop."** [V4] says spikes are tied to MoE
   outliers, routing "appears to exacerbate" them, and the technique breaks "the vicious cycle induced by routing";
   it does not describe the loop's mechanism and states the principles are insufficiently understood. Slides draw
   the loop as a hypothesis.
3. **"With mHC and Muon, suppresses loss spikes and divergence."** In [V4] §4.2.3 the companion technique is SwiGLU
   clamping; mHC and Muon are stability-related elsewhere in the paper but not tied to spike mitigation. No
   divergence results are reported. Slides pair it with SwiGLU clamping and mention mHC/Muon as separate decks.
4. Notion omits the key operating detail: the mode is spike-triggered (with a short rollback) and costs ≈ 20%
   wall-clock while active; always-on use is not what DeepSeek does.

## 64 · Distributed engineering of Muon at scale

[deck](../decks/64-distributed-engineering-of-muon-at-scale/) · [Notion row](https://app.notion.com/p/9069ea0863a741658d052a19a186169b)

1. **"Halves MoE gradient-sync traffic"** (Notion) matches [V4] ("halving the communication volume") — consistent, but it
   applies only to MoE parameters; dense parameters are not stated to use BF16.
2. **"Knapsack-based hybrid ZeRO bucketing (<10% padding memory)"** — [V4] gives "< 10%" for the dense buckets only
   ("in our setup, where each rank manages no more than five parameter matrices"); MoE padding is "negligible".
3. **"Removes naive Muon's transient memory spikes"** is GLM-5's claim about its own zero-redundant all-gather ([GLM5]
   §2.4.1), not a DeepSeek-V4 result; no number is published. Slides attribute it to GLM-5.
4. **"Every DP rank need not all-gather full params"** — GLM-5's text says naive implementations all-gather full
   *parameters*; Moonlight's Distributed Muon gathers *gradients/momentum* for the matrix. Slides keep the two separate.

## 65 · MuonClip (Muon + QK-Clip)

[deck](../decks/65-muonclip-muon-qk-clip/) · [Notion row](https://app.notion.com/p/63cf5bee70464919acbac2e6ed917cdd)

1. **"after every step the Q/K projection weights are rescaled against threshold tau".** QK-Clip runs after
   every step but rescales only heads whose max logit exceeds τ (γ_h = min(1, τ/S_max^h)); in K2 it acted on
   12.7% of heads during the first 70K steps and was inactive afterwards ([K2] §2.1, App. D). Slides say
   "only heads above τ".
2. **"killing attention-logit explosion at the source".** QK-Clip bounds the weights' contribution after the
   fact: the step in which S_max is measured runs unclipped, and the clip acts from the next step
   ([K2] §2.1). The logits are capped at τ, not prevented from rising to τ (Fig. 2 right). Slides say "caps".
3. **"no downstream degradation from clipping".** The evidence is a small-scale ablation (0.5B active / 3B
   total MoE, τ = 30: negligible loss change, no statistically significant downstream degradation) plus the
   fact that QK-Clip self-deactivates in K2; there is no full-scale K2 ablation ([K2] App. D). Slides
   attribute it accordingly.
4. **"Zero loss spikes across K2's full 15.5T-token pretraining"** matches [K2] (abstract, Fig. 3). No
   discrepancy.

## 66 · QK-Norm

[deck](../decks/66-qk-norm/) · [Notion row](https://app.notion.com/p/319387d19fc64c6d8079c56a5c3b7a69)

1. **Notion "Impact: together with Muon, supports stable convergence at larger batch sizes."** [GLM45] §2.4 credits *Muon* with "accelerate convergence and tolerate larger batch sizes"; it credits QK-Norm only with stabilizing "the range of attention logits" (§2.1). GLM-4.5 publishes no measurement isolating QK-Norm (no logit curves, no ablation). Slides attribute the batch-size claim to Muon and present QK-Norm's effect with the analytic bound plus measurements from other labs.
2. **Notion "keeps logits in a stable band".** The bound is soft: it scales with the learned gains (max|g_q|·max|g_k|), so it is "bounded by construction at a level set by 2 × 128 learnable gains per layer", not a fixed band. Slides say "√d × gains".
3. **Implementation vs cited paper.** [GLM45] cites Henry et al. 2020 ([QKN]), which uses ℓ2 normalization plus one learnable scalar instead of 1/√d. GLM-4.5's code uses per-head RMSNorm with a 128-dim learnable weight and keeps the 1/√d scale ([HF-CODE]), i.e. the Dehghani/OLMo-2 form. Slides describe the code.
4. **"The same problem QK-Clip attacks from the other side"** (Notion) is accurate, with one limit the row omits: QK-Norm needs materialized per-head keys, which MLA does not have ([K2] §2.1). That is why K2 (MLA) uses QK-Clip and GLM-5 (MLA) dropped QK-Norm.

## 67 · Asymmetric VPP / custom pipeline layout

[deck](../decks/67-asymmetric-vpp-custom-pipeline-layout/) · [Notion row](https://app.notion.com/p/0b667c3fda7240be99b87c3b35fdcfcd)

1. **Notion: "shrinks pipeline bubbles".** No primary source reports a measured bubble or throughput change from the layout
   itself; the Megatron report gives the DeepSeek-V3 layouts only inside full-stack configurations. The slides use
   the PTD-P bubble formula as an *analytic* estimate and say so.
2. **Notion: "Fixes the PP load imbalance caused by MTP/embedding layers".** The source describes cost-aware manual
   placement ("practitioners can design layouts"); there is no automatic balancer. Slides say the user writes the balance.
3. **README vs docs vs paper on the example string**: [DOC] uses `Et*3|(tt|)*29,m|L` (PP16, VPP2); the paper's H100 layout
   is `Et*3|(tt|)*29m|L` (PP8, VPP4). Both parse to the same 32 stages; the PP size decides the VPP size. Not a conflict, but
   the same string means different rank tables under different PP.
4. **[MC] Table 17 vs Table 20 (H100 PP)**: Table 17 says PP8 (consistent with the 32-stage layout and VPP4); App. B Table 20
   says "TP2 PP4". Slides use PP8.
5. Notion lists the PTD-P and Megatron-LM papers; neither covers asymmetric layouts (they predate it); PTD-P is used only for
   the interleaved-schedule bubble model.

## 68 · Context Parallelism / Dynamic CP

[deck](../decks/68-context-parallelism-dynamic-cp/) · [Notion row](https://app.notion.com/p/82d104d9bcc74fab931db687fed76e81)

1. **Notion "beats full recomputation beyond 8K sequences".** The CP guide's measurement (Fig. 2) covers 16K and
   32K only; 8K appears in the guide as an illustrative example (8K split into two 4K halves) and in the MoE
   README as the *recommended threshold* for enabling CP (Guideline 5: sequence length ≥ 8K). No source measures
   CP vs recompute at 8K. Slides say "16K and 32K" for the measurement and cite the 8K guideline separately
   (notes only).
2. **Notion "supports MHA/MQA/GQA, MLA and MTP".** The CP guide lists MHA, MQA, GQA; CP for MLA and MTP arrived
   in MCore v0.14 ([RD]). However, Dynamic (hybrid) CP is not yet supported with MLA ([MLA] assert). Slides
   state the Dynamic CP restriction.
3. **"Dynamic CP" naming.** The blog and [MC] call it Dynamic-CP; on Megatron-LM main it is
   `--hybrid-context-parallel` (+ `--max-seqlen-per-dp-cp-rank`, `--sequence-packing-scheduler`); the dev branch
   renamed it `--dynamic-context-parallel` ([PR2000]). Slides use "Dynamic CP" and give flags only in notes.
4. **Inside [MC]:** §3.2.2 says each GPU processes "a contiguous chunk" and that activation memory "scales
   quadratically" with sequence length; the code uses zigzag (non-contiguous) chunks ([UTILS]), and [CPDOC] and
   [MC] §6.4.2 / [BLOG] say activation memory grows (about) linearly. Slides follow the code and [CPDOC].
5. **Dynamic CP gain:** [BLOG] says "over 35%" end-to-end at multi-thousand-GPU scale; [MC] says "35–60%" in
   highly imbalanced real-world scenarios. Both are quoted with their own wording.

## 69 · KDA Context Parallelism (KCP)

[deck](../decks/69-kda-context-parallelism-kcp/) · [Notion row](https://app.notion.com/p/a5e78a14357748deb3032985bef70062)

1. **Notion "compute scales linearly to 1M context".** K3 §5.1.2 states that KCP "achieves linear compute scaling"
   and §3.4 that CP makes million-token training tractable, but no KCP measurement at 1M (or any other length) is
   published. The only public timing is FLA's 32K, CP = 4 benchmark (PR #691). The deck shows the 32K measurement and
   labels the 1M comparison as analytic.
2. **Which implementation the numbers describe.** K3 footnote 2 says "The KDA implementation is available in FLA PR
   #691"; FLA's own README says its CP code "was independently contributed to FLA and is a separate codebase from the
   internal Moonshot implementation" (where it is called KCP, "Kimi Context Parallel"). The measured numbers are
   therefore for the open FLA implementation, not necessarily Moonshot's internal one. Stated on slides 10 and 12.
3. **Name.** K3 expands KCP as "KDA Context Parallelism" (as Notion does); FLA's README calls it "Kimi Context
   Parallel". Same technique.
4. **Notion body: "Kimi K3 repo - architecture and training notes".** The Kimi-K3 README contains a model summary and
   usage notes but nothing about KCP; all mechanism detail comes from the K3 report and FLA.
5. **Precision.** Neither Notion nor K3 mention precision; FLA's PR calls the outputs and gradients "lossy" relative to
   all-to-all CP and its README requires the M chain in FP32. Mentioned in the costs slide notes.

## 70 · Megatron-FSDP / HSDP (with EP support)

[deck](../decks/70-megatron-fsdp-hsdp-with-ep-support/) · [Notion row](https://app.notion.com/p/bbdbb22301864a05bcbc8e4097a6b284)

1. **Notion "complements the distributed optimizer (ZeRO-1 equivalent)".** Correct, and in the code Megatron-FSDP
   *uses* the distributed optimizer (`--use-distributed-optimizer` is set automatically, [FSDP] feature table);
   its `optim` strategy is the ZeRO-1 point of the same ladder. Slides present them as one ladder.
2. **Notion "hybrid sharding (--num-distributed-optimizer-instances)".** That flag sets the DP-outer size. For the
   classic distributed optimizer it gives HSDP-style partial optimizer sharding (MCore v0.13 "Hybrid Shard
   Data-Parallel support for MoE"); for Megatron-FSDP it is combined with `--outer-dp-sharding-strategy`
   (`no_shard` = HSDP, `optim` = HFSDP). Slides name both flags.
3. **Notion "model-state sharding and DP become interchangeable".** Not a phrase from the sources; the closest
   source statements are [MC] §4.1.7 "Per-GPU memory and collective volume both scale with EDP size" and
   "PP-free flexibility". Slides phrase it as "DP size becomes the memory knob" and label it as our summary.
4. **No first-party throughput benchmark for Megatron-FSDP (+EP) was found** ([BR] card marks performance
   claims `doc_only`). The only end-to-end measurement used is a public user report ([I2772]); slides label it
   as such. [MC] contributes the ~10% and SM-count numbers.
5. **Parameter count.** [MC] uses 685B for DeepSeek-V3 (with MTP); [EX] says 671B. Slides use [MC]'s 685B
   together with its Table 3.

## 71 · MoE Parallel Folding

[deck](../decks/71-moe-parallel-folding/) · [Notion row](https://app.notion.com/p/ef4d75b8eee841a79ada63b79bff78d8)

1. **Notion: "EP can be folded inside TP x CP groups".** Primary sources say EP folds across arbitrary
   sub-groups of the attention mapping, including DP: in the 256-GPU example EP64 folds across
   TP×CP×DP ([MC] §3.3.3 p.17), and [PF] App. 6.1 validates EP folded "with all of TP, CP, DP". Slides say TP×CP×DP.
2. **Notion lists only arXiv 2603.07685.** The mechanism, the dispatcher and all measured MFU results come
   from the companion paper arXiv 2504.14960 ([MC] defers implementation details to it, p.16). Slides cite both.
3. **[PF] Table 5 (context scaling)** prints identical numbers for the Qwen2-57B and Mixtral-8x22B rows,
   while the text gives Qwen 38.7% → 35.9% and Mixtral 47.6% → 42.9%. Not used on slides.
4. **[PF] Table 3, Llama3-8x70B "MCore w/ Folding" row** has an ambiguous column alignment (only four
   numbers for five columns). Its config is not used on slides; only its MFU (Table 1) is.
5. Notion's numbers (EP 8 → 64, ETP=1, EDP=1 at 256 GPUs) match [MC] §3.3.3 exactly.

