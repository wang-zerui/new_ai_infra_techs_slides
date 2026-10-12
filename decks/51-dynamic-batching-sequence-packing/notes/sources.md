# Sources — deck 51: Dynamic batching / sequence packing

## Sources read

| Tag | Source | Where |
|---|---|---|
| [VO] | VeOmni, arXiv 2508.02317v3. https://arxiv.org/abs/2508.02317 | §3.2.5 "Dynamic Batching" bullet (p.7); App. B.1 Listing 1 (`OmniDataCollatorWithPacking`) |
| [V-PACK] | VeOmni `docs/usage/data_packing_and_dyn_bsz.md` (repo main @ 57e5f6d, 2026-10-12) | "Background of VeOmni data packing", "Background of dyn_bsz", "Usage", "Padding packed inputs" |
| [V-BATCH] | VeOmni `veomni/data/dynamic_batching.py` | `DynBszBuffer.get_samples`, `TextBatchingStrategy` (default `buffer_size=500`, batch-size warm-up), `DynamicBatchSizeDataLoader`; "TODO: add state dict for buffer to resume training" |
| [V-COLL] | VeOmni `veomni/data/data_collator.py` | `PackingCollator.__call__` (first label of each later sample → IGNORE_INDEX; `cu_seq_lens` from `position_ids`), `SequenceParallelCollator` (SP padding, slicing after FA kwargs) |
| [V-ULY] | VeOmni `docs/key_features/ulysses.md` | all-to-all re-sharding; `reduce_sequence_parallel_loss` rescales by valid tokens |
| [MC] | NVIDIA, "Scalable Training of Mixture-of-Experts Models with Megatron Core", arXiv 2603.07685v2 (cached) | §6.4 intro (p.59), §6.4.1 Packed Sequence Support (p.60), §6.4.2 Dynamic-CP (p.60-63), Fig. 36-38 |
| [NVB] | NVIDIA Technical Blog, "Speeding Up Variable-Length Training with Dynamic Context Parallelism and NVIDIA Megatron Core" (Jan 28, 2026), linked from the Notion row; saved to `$SCRATCH/research/51/dl/nv_dyncp.txt` | summary bullets; Table 1 |
| [MCP] | Megatron-LM `docs/user-guide/features/context_parallel.md` (cached `megatron-context_parallel.md`) | CP overview (sequence split, KV all-gather in a ring) |
| [GLM5] | GLM-5 Team, "GLM-5: from Vibe Coding to Agentic Engineering", arXiv 2602.15763 (cached) | Figure 5 description (context 4K → 200K in mid-training); §2.4.2 "Efficient long-sequence training" (p.10) |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 4K, 8K, 32K; ~60% | a batch with these lengths padded to 32K wastes "~60% of computation and memory" | [MC] §6.4 p.59 |
| 54% of token slots, 64% of attention FLOPs | analytic check of that example: pad slots = 1 − 44/96 = 0.54; attention ∝ L²: 1 − (4²+8²+32²)/(3·32²) = 1 − 1104/3072 = 0.64 (derived, notes only) | derived |
| 4K → 200K | GLM-5 mid-training context extension | [GLM5] Fig. 5 description |
| hundreds to tens of thousands of tokens | RL thinking-model sequence lengths | [MC] §6.4.1 |
| S1..S4 = [A B C D], [E F], [G H I], [J]; position_ids [0 1 2 3 0 1 0 1 2 0]; cu_seqlens [0 4 6 9 10] | packing worked example (cu_seqlens follow from position_ids; the doc's own examples use the same rule) | [V-PACK] "dyn_bsz = True" example; cu_seqlens rule from [V-PACK] "Padding packed inputs" and [V-COLL] |
| Rank0 [A B C D E F] (6 tokens) vs Rank1 [G H I J] (4 tokens) | fixed-sample-count imbalance example | [V-PACK] "Background of dyn_bsz" |
| 500 | default minimum number of samples in the dyn_bsz buffer | [V-BATCH] `TextBatchingStrategy(buffer_size=500)` |
| 1.5 | default `dyn_bsz_physical_overflow_ratio` (physical-token cap = ceil(micro_batch_size × max_seq_len × ratio) in `effective` count mode) | [V-PACK] |
| 40–60% less memory; 1.5–2× throughput | packed sequences vs padding, RL with thinking models | [MC] §6.4.1 p.60 |
| 35–60% | Dynamic-CP end-to-end improvement with highly imbalanced length distributions (e.g. multimodal) | [MC] §6.4.2 p.62-63 |
| 195.88 → 289.32 TFLOPS/GPU (1.48×); 139.17 → 174.39 (1.25×) | Llama 13B, GBS 2048, PP 8, CP 8, full recompute; packing only vs Dynamic-CP; GitHub and CommonCrawl data | [NVB] Table 1 |
| > 35% | Dynamic-CP end-to-end gain in a multi-thousand-GPU industrial environment | [NVB] |
| CP sizes 1 … dp×cp, powers of two | pre-built CP groups | [MC] §6.4.2; [NVB] |
| P = 4, 10 tokens padded to 12 | ❸ figure example (P chosen for the figure; padding rule: pad to a multiple of the SP size) | rule from [V-COLL] `sp_padding`; numbers illustrative |

Chart labels round Table 1 to integers (196, 289, 139, 174).
Derived: 1.48 = 289.32 / 195.88 and 1.25 = 174.39 / 139.17 are printed in [NVB] itself.

## Mechanism facts (no numbers) used on slides

- VeOmni packing: flatten real tokens of the micro-batch into one sequence, record boundaries with
  `cu_seq_lens_q/k`, run FlashAttention (or FlexAttention) and FFN on real tokens only ([V-PACK]).
  `cu_seq_lens` and `max_length` are derived from `position_ids` restarting at 0 ([V-COLL]). The first
  label of every packed sample after the first is set to IGNORE_INDEX so no token is trained to predict
  the next sample ([V-COLL]).
- dyn_bsz ON: number of samples varies to hit a token budget (recommended for pretraining); OFF: fixed
  sample count, still packed without padding (recommended for SFT) ([V-PACK]).
- Count modes: `total` (attention_mask.sum) or `effective` (labels != IGNORE_INDEX) with a physical cap
  ([V-PACK]; [V-BATCH] `get_length_fn`, `physical_token_cap`).
- Selection: scan the buffer in order, take each sample that still fits the remaining budget (the first
  sample is always taken), leave the rest for the next micro-batch; optional token-budget warm-up
  ([V-BATCH] `get_samples`, `get_cur_token_micro_bsz`).
- `pad_to_length`: pad the packed stream to a fixed length (labels IGNORE_INDEX, cu_seqlens recomputed);
  required with per-block torch.compile and useful for CUDA graphs ([V-PACK]).
- SP: the packed stream is padded to a multiple of the SP size and sliced per rank; FA kwargs are
  computed on the full stream before slicing; the loss is rescaled by valid tokens ([V-COLL], [V-ULY]).
- Megatron THD: attention tensors [total_tokens, num_heads, head_dim]; cumulative sequence lengths go to
  Transformer Engine's fused attention and RoPE; with CP padding requirements both padded and unpadded
  cu_seqlens are kept ([MC] §6.4.1).
- Packing alone leaves DP imbalance: equal token counts, unequal attention work (quadratic in
  sub-sequence length); Dynamic-CP picks CP size per micro-batch from pre-built groups, reschedules packs
  across DP ranks, loss per valid token ([MC] §6.4.2; [NVB]).
- GLM-5 long-sequence training: workload-aware sequence reordering, dynamic redistribution of attention
  computation, flexible partitioning of DP ranks into CP groups of varying sizes ([GLM5] §2.4.2).

## Discrepancies

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
