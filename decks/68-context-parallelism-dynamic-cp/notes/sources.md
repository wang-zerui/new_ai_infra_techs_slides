# Sources — deck 68: Context Parallelism / Dynamic CP

## Sources read

| Tag | Source | Where |
|---|---|---|
| [CPDOC] | Megatron-LM `docs/user-guide/features/context_parallel.md` (cached copy) and its two figures `docs/images/context_parallel/CP_overview.png`, `CP_results.png` (downloaded 2026-10-13). https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/context_parallel.md | "Context Parallel Overview", "Context Parallelism Benefits", "Enabling Context Parallelism"; Fig. 1 (TP2CP2), Fig. 2 (speedup vs full recompute) |
| [MC] | NVIDIA, "Scalable Training of Mixture-of-Experts Models with Megatron Core", arXiv 2603.07685v2. https://arxiv.org/abs/2603.07685 | §3.2.2 (p.~14), §6 intro + §6.1 + Fig. 34 (p.57), §6.2 (p.58), §6.3 + Fig. 35 (p.58-59), §6.4 (p.59), §6.4.1 (p.60), §6.4.2 + Fig. 36-38 (p.60-63) |
| [BLOG] | K. Li, T. Ma, P. Mannan, S. Yang, G. Wu, C. Wang, "Speeding Up Variable-Length Training with Dynamic Context Parallelism and NVIDIA Megatron Core", NVIDIA Technical Blog, 28 Jan 2026. https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/ | all sections; Table 1 |
| [RD] | Megatron-LM `megatron/core/transformer/moe/README.md` (cached copy) | "What's New" v0.14 (CP for MTP and MLA); memory table (Step 1); Guideline 5 (CP for sequence length ≥ 8K) |
| [ARGS] | Megatron-LM `megatron/training/arguments.py` (main, fetched 2026-10-13) | `--cp-comm-type` (p2p, a2a, allgather, a2a+p2p; per-layer list), `--hierarchical-context-parallel-sizes` assert for a2a+p2p, `seq_length % (2 * context_parallel_size) == 0` assert, `--hybrid-context-parallel` + `--max-seqlen-per-dp-cp-rank` + asserts (no PP, no CUDA graph, no Megatron-FSDP, per-token loss) |
| [UTILS] | Megatron-LM `megatron/core/utils.py` `_get_batch_on_this_cp_rank_per_sequence_balancing` docstring | zigzag: sequence split into 2×cp_size chunks; CP=2: (chunk_0, chunk_3) → GPU0, (chunk_1, chunk_2) → GPU1 |
| [TE] | TransformerEngine `transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py` (main, fetched 2026-10-13) | `AttnFuncWithCPAndKVP2P` docstring + forward comment table (cp_size=4: diagonal / lower-triangle / upper-triangle tiles), `cp_p2p_fwd_prepare_qkv` sections; `AttnFuncWithCPAndKVAllGather` docstring (exposed KV all-gather, Llama 3 §3.3.2); `AttnFuncWithCPAndQKVOA2A` docstring (Ulysses-style A2A on QKVO) |
| [HCP] | Megatron-LM `megatron/core/pipeline_parallel/hybrid_cp_schedule.py` (main) | `BalancedCPScheduler.get_total_workload` (S²/cp_size), `gpus_needed` (next power of 2 of seq_len / max_seq_len_per_rank), groups for every power of 2 up to the DP×CP group |
| [PR2054] | Megatron-LM PR #2054 "[Dev] Hybrid Data x Context Parallelism Feature" (merged 2025-12-04, dev) | body: problem statement, limitations, LLaMa3-8B 128K example |
| [PR2304] | Megatron-LM PR #2304 "hybrid-cp part 2 for main branch" | known limitations |
| [PR2000] | Megatron-LM PR #2000 "[Dev] feat: Dynamic CP (part 2)" (merged 2026-04-07, dev) | `--dynamic-context-parallel`, PP/VPP + MTP/Mamba support, known limitations |
| [MLA] | Megatron-LM `megatron/core/transformer/multi_latent_attention.py` (main) | assert: hybrid_context_parallel not supported with MLA yet |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 16K–64K+ tokens | sequence lengths of reasoning / RL-based training | [MC] §6 intro p.57 ("sequences of 16K, 64K, or even longer") |
| 69% of FLOPs at 64K; 10–15% at short lengths | SDPA share of total FLOPs (DeepSeek-V3-class MoE profile) | [MC] §6.1 p.57 |
| "grows about linearly" | activation memory vs sequence length | [CPDOC] "Context Parallelism Benefits" |
| ~30% | overhead of full activation recomputation | [CPDOC] "Context Parallelism Benefits" ("about 30 percent with full recomputation") |
| ≈0.85× | TP32CP1 speed relative to full recompute (TP8CP1), 175B GPT, BF16 + MHA, seq 32K | [CPDOC] Fig. 2, value read from the bar chart (pixel-measured: bar top 471 px, 0.0 at 928 px, 1.4 at 174 px → 0.849) |
| ≈1.13× / ≈1.25× | TP16CP1 / TP8CP2 vs full recompute, BF16 + MHA, 16K | [CPDOC] Fig. 2 (pixel-measured: 1.129, 1.248) |
| ≈1.15× / ≈1.20× | TP16CP2 / TP8CP4 vs full recompute, BF16 + MHA, 32K | [CPDOC] Fig. 2 (pixel-measured: 1.147, 1.198) |
| ≈1.28× / ≈1.30× | TP8CP2 (16K) / TP8CP4 (32K) vs full recompute, FP8 + GQA | [CPDOC] Fig. 2 (pixel-measured: 1.277, 1.298) |
| 175B GPT | model in the CP speedup figure | [CPDOC] Fig. 2 caption |
| TP2CP2 on 4 GPUs; GPU0↔GPU2 and GPU1↔GPU3 exchange KV | overview example | [CPDOC] Fig. 1 + text |
| world size = TP × CP × PP × DP | composition | [CPDOC] "Enabling Context Parallelism" |
| MCore ≥ 0.5.0, TE ≥ 1.1 | requirements | [CPDOC] "Enabling Context Parallelism" |
| 2 × CP chunks; rank r holds chunks r and 2CP−1−r; CP=2: (0,3) and (1,2) | zigzag assignment | [UTILS] docstring; rank r ↔ chunk 2CP−1−r generalization is the same reshape/index_select pattern (and matches [TE]'s lower/upper half logic) |
| seq length divisible by 2 × CP | constraint | [ARGS] assert |
| 2 / 6 / 10 / 14 vs 8 / 8 / 8 / 8 block units | attention work per GPU at CP=4, contiguous vs zigzag (analytic, derived below) | derived |
| CP steps; diagonal / lower / upper tiles | ring schedule per GPU and step | [TE] forward comment table (cp_size = 4) |
| p2p (default), allgather, a2a, a2a+p2p; per-layer list | CP comm types | [ARGS] `--cp-comm-type` |
| a2a inside node + p2p across nodes as starting point | hierarchical CP guideline | [MC] §6.3 p.59 |
| CP sizes 1 … dp×cp, powers of two | pre-built CP groups for Dynamic CP | [MC] §6.4.2 p.62; [BLOG] "Building multiple context parallel groups per rank"; [HCP] |
| workload ≈ S² / cp; cp = next power of 2 ≥ S / max_seqlen_per_rank | scheduler cost model in code | [HCP] |
| micro-batch count swept from PP×1 to a small multiple of PP | solver grid search | [MC] p.62; [BLOG] "Solver runtime" |
| Llama 13B, GBS 2048, PP=8, CP=8, full recompute, 10 iterations (first discarded) | Dynamic CP benchmark setup | [BLOG] "Benchmark results" |
| 195.88 → 289.32 TFLOPS/GPU (1.48×), GitHub | Dynamic CP vs only packing | [BLOG] Table 1; 289.32 / 195.88 = 1.477 |
| 139.17 → 174.39 TFLOPS/GPU (1.25×), CommonCrawl | Dynamic CP vs only packing | [BLOG] Table 1; 174.39 / 139.17 = 1.253 |
| > 35% end-to-end | Dynamic CP in a multi-thousand-GPU industrial environment | [BLOG] "Benchmark results" |
| 35–60% end-to-end | Dynamic CP in real-world highly imbalanced scenarios (e.g. multimodal) | [MC] §6.4.2 p.62-63 |
| power-of-2 groups; no CUDA Graphs; FlashAttention preferred (cuDNN FusedAttention recompiles per shape); FSDP + PP unsupported | Dynamic CP known limitations | [PR2304], [PR2000] |
| MLA not yet supported with hybrid (dynamic) CP | limitation | [MLA] assert |
| per-token loss required; main-branch `--hybrid-context-parallel` asserts no PP | limitation | [ARGS] |

### Derived numbers (analytic)

Causal attention over a sequence split into 2·CP = 8 equal chunks (CP = 4). Count work in units of one
chunk×chunk block; a diagonal block is causal, so it counts ½. Query chunk c needs c full blocks plus ½.

- Contiguous split (GPU r holds chunks 2r, 2r+1): work(r) = (2r + ½) + (2r + 1 + ½) = 4r + 2 → **2, 6, 10, 14**.
  The last GPU does 7× the first GPU's work; every GPU waits for GPU3.
- Zigzag (GPU r holds chunks r and 7 − r): work(r) = (r + ½) + (7 − r + ½) = **8** for every r. Total 32 = 8²/2.

Ratios on the Dynamic CP chart are computed from [BLOG] Table 1 as shown above.

Packed micro-batches (slide 11, panel a and notes): attention work grows with the square of each sequence's
length, so one sequence of length 4L costs (4L)² = 16 L² while four sequences of length L cost 4 L²: the
single long sequence carries 4× the attention work of an equal-token pack of four short ones (analytic; the
triangles in fig_dcp_problem are drawn with areas in that ratio). Same reasoning as [MC] §6.4.2 / Fig. 37 and
[PR2054] ("a single long sequence of 1k has 2x more compute than a packed sequence made of 2x512").

## Mechanism facts (no numbers) used on slides

- CP partitions network inputs and all activations along the sequence; modules other than attention run
  unchanged because they have no inter-token operations; attention needs KV of all tokens → all-gather of KV
  in forward, reduce-scatter of dKV in backward, implemented as P2P in a ring; each GPU stores only its KV
  chunk in forward and re-gathers KV in backward; GQA/MQA reduce KV volume; KV exchange is between a GPU
  and its counterparts in other TP groups ([CPDOC]).
- CP vs Ring Attention: uses open-source and cuDNN flash-attention kernels and avoids extra work from the
  lower-triangle causal mask while keeping load balanced ([CPDOC]).
- More TP fixes OOM but makes Linear compute too short to hide TP communication ([CPDOC]).
- Megatron sequence parallelism splits only Dropout and LayerNorm activations ([CPDOC]).
- Ring step i on GPU r (causal): i = 0 → diagonal (own Q vs own KV, causal); 1 ≤ i ≤ r → all local Q vs first
  half of the received KV; i > r → second half of local Q vs all received KV. Next step's KV send/recv overlaps
  the current step's compute ([TE]). Every off-diagonal step is half a block's work on every rank (derived from
  the same chunk arithmetic). Partial outputs of the steps are combined using each step's softmax
  log-sum-exp ([TE] forward, softmax_lse correction).
- All-gather CP: KV all-gather is exposed ([TE]). A2A CP: sequence-sharded → head-sharded before SDPA and back
  after (Ulysses) ([TE], [MC] §6.3). Hierarchical CP: heads in low-level groups, sequence in high-level groups
  ([TE]); a2a+p2p requires `--hierarchical-context-parallel-sizes` ([ARGS]).
- CP shards activations 1/N, weights not at all, optimizer states 1/N only with the distributed optimizer
  ([RD] memory table).
- Static CP size is chosen for the longest packed sample; shorter samples are still sharded, causing
  unnecessary CP communication; with many short sub-sequences the per-shard compute may not hide CP
  collectives, especially across inter-node links ("CP inefficiency"); equal-length packed samples carry
  different attention work (quadratic), causing DP imbalance and amplified PP bubbles ([MC] §6.4.2, [BLOG]).
- Resizing CP changes only how token slices are partitioned and which CP group attention uses; no parameter
  redistribution or optimizer-state migration ([MC] p.60-61). Resizing TP/PP would need weight redistribution
  or pipeline restructuring ([BLOG]).
- Dynamic CP plumbing: data_iterator_wrapper reschedules/packs the global batch, picks cp_size, returns the
  effective num_microbatches; metadata (num_microbatches, max_seqlen, cu_seqlens) broadcast to PP stages;
  PackedSeqParams carries cp_size and cp_group; per-token loss ([MC] p.62, [BLOG]).
- Scheduler: cost model → solver (heuristic packing + CP size, alternating workload and memory objectives) →
  simulator (pick the plan with minimum time under peak-memory limit); solver runs asynchronously in the data
  sampler ([BLOG], [MC] p.62).

## Discrepancies

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
