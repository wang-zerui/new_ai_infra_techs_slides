# Story — deck 52: Fine-grained selective recomputation

Family: decks 52 / 53 / 54 attack the MoE activation memory wall three ways (recompute, reorder, offload)
and share two charts: the DeepSeek-V3 per-GPU memory stack (Table 3) and the "memory toolbox" savings
ledger (each deck highlights its own bar). Cross-references: 46 (FP8 activations), 49 (BF16 optimizer
moments), 70 (Megatron-FSDP), 11 (EP-overlap schedule, which forbids `moe` recompute).

- **Where it lives** (prosperity): Megatron-Core MoE (v0.13 fine-grained recompute, v0.14 FP8 support);
  DeepSeek-V3 685B (37B active) needs 199.5 GB per GPU in BF16 on PP4 × VPP4 × EP64 / 256 GPUs, of which
  131 GB are activations. Kimi K2 uses the same recipe in its own stack.
- **Pain number** (betrayal): full-layer recomputation costs ~33% extra compute, and in MoE it also
  replays the EP dispatch/combine all-to-alls.
- **Failed suitors**:
  1. Full-layer checkpointing: ~33% compute + all-to-all replay.
  2. Standard checkpointing of one cheap op: frees nothing, because the next GEMM saves the op's output as
     its own input (the output is pinned by the consumer).
  3. Classic selective recompute of core attention (Korthikanti et al. 2022): fused attention already
     avoids storing the score matrix, so `core_attn` is moot (Megatron warns), and at long context it is
     the most expensive thing to recompute.
- **Villain**: memory-to-compute mismatch inside a layer — many saved tensors are cheap functions of
  other saved tensors (norm outputs of norm inputs; Q/K/V of 31× narrower MLA latents; SwiGLU output of the
  fc1 output), and the consumer GEMM pins the output.
- **Turn / insight**: saved outputs are cheap to re-derive from inputs that are kept anyway → drop the
  output, keep the input, rebuild just before backward (analytic: MLA Q/K/V = 65,536 values per token vs
  2,112 latent values, 31×).
- **Mechanisms**:
  ❶ Output-discarding checkpoint (CheckpointWithoutOutput): free the output storage after the consumer used
     it; a grad hook on the consumer's output recomputes and re-points the storage (answers: consumer pins
     the output).
  ❷ Module menu chosen by memory/compute ratio: layernorm, mla_up_proj, moe_act (output-discarding);
     core_attn, mlp, moe, shared_experts (standard) (answers: which activations are worth it).
  ❸ Exact replay: RNG states + FP8 recipe + TE recompute phase; zero-copy storage swap (answers: rebuilt
     tensor must equal the forward value, also under FP8).
- **Release**: 42.4 GB per GPU saved (MLA up-proj 30.4, LayerNorm 8.2, SwiGLU 3.8) for < 5% compute vs
  ~33% for full recompute; largest single activation saver in the toolbox; at 64K context, non-attention
  recompute saves 89.8 GB while core-attention recompute costs +18% compute for 9 GB.
- **Limits**: < 5% is not zero; `moe` recompute replays all-to-all and is rejected by the EP-overlap
  schedule; module prerequisites (grouped GEMM, MLA); FP8 needs TE ≥ 2.6 and no delayed scaling;
  shared_experts recompute excludes shared-expert overlap; set is per-platform (H100: 4 modules, GB200: 1).

## Slide plan (15 logical slides, 16 PDF pages, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Activations are two-thirds of MoE memory | prosperity | chart mem_stack (family) |
| 3 | Full recompute costs a third more compute | pain | `%% big: ~33%` |
| 4 | Checkpointing is too coarse or frees nothing | failed suitors / villain | fig_conventional |
| 5 | Insight: saved outputs are cheap to re-derive | turn | fig_pairs (analytic widths) |
| 6 | Three modules drop their outputs in every layer | overview | fig_overview |
| 7 | ❶ Free the output, rebuild it from a grad hook | mechanism (build ×2) | fig_discard_1/2 |
| 8 | ❷ Drop what is big to store, cheap to redo | mechanism | fig_menu |
| 9 | ❸ Replay with the forward's RNG and FP8 recipe | mechanism | fig_fp8 |
| 10 | 42.4 GB saved per GPU for under 5% compute | release | chart recompute_savings |
| 11 | The largest single activation saver in the toolbox | result (family) | chart toolbox |
| 12 | At 64K, recompute everything but attention | result | two-col text |
| 13 | H100 recomputes four modules, GB200 only one | result / adoption | fig_platform |
| 14 | The cost: compute, prerequisites, and conflicts | limits | two-col text |
| 15 | Summary | | QR arxiv.org/abs/2603.07685 |
