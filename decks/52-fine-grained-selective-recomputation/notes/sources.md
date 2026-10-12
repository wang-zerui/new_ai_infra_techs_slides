# Sources — deck 52: Fine-grained selective recomputation

## Sources read

| Tag | Source | Where |
|---|---|---|
| [MC] | NVIDIA, "Scalable Training of Mixture-of-Experts Models with Megatron Core", arXiv 2603.07685v2. https://arxiv.org/abs/2603.07685 (cached text) | §1.2 (p.6), Table 3 + §4.1.1 (p.20), §4.1.2 (p.21), §4.1.3 (p.22), §4.1.4 (p.22-23), Table 4 (p.24), §4.1.5-4.1.6 (p.23-27), §6.2 (p.58), Table 13 (p.72), Table 17 + §9.2.2 (p.74-75), §11 (p.78), App. B.2 (p.87) |
| [RD] | Megatron-LM `megatron/core/transformer/moe/README.md` (cached copy) | What's New v0.13 / v0.14; "Memory Optimization" table; "Fine-grained Recomputation" module list |
| [CFG] | `megatron/core/transformer/transformer_config.py` (main, fetched 2026-10-13 via `gh api`) | `recompute_granularity`, `recompute_modules` docstrings (l.604-660); `__post_init__` validation (l.2380-2490) |
| [RNG] | `megatron/core/tensor_parallel/random.py` (main) | `CheckpointWithoutOutputFunction`, `CheckpointWithoutOutput.checkpoint / _recompute / _discard_outputs / discard_output_and_register_recompute`, `_get_share_storage` |
| [EXP] | `megatron/core/transformer/moe/experts.py` (main) | `TEGroupedMLP.forward`: moe_act checkpoint around `bias_act_func`, hook on fc2 output |
| [MLA] | `megatron/core/transformer/multi_latent_attention.py` (main) | `qkv_up_checkpoint` around `qkv_up_proj_and_rope_apply`, hook on `core_attn_out`; Q/K/V assembly |
| [TL] | `megatron/core/transformer/transformer_layer.py` (main) | `input_layernorm_checkpoint` (hook on attention output), `pre_mlp_norm_checkpoint` |
| [DSV3] | DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv 2412.19437 (cached) | §4.2 Model Hyper-Parameters (p.22) |
| [K2] | Kimi Team, "Kimi K2: Open Agentic Intelligence", arXiv 2507.20534 (cached) | §2.4.1 (p.7), §2.4.3 Activation Reduction (p.8) |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 685B total / 37B active | DeepSeek-V3 parameters (MCore counts MTP; DeepSeek report says 671B) | [MC] §1.2 p.6, Table 3 text p.20 |
| 199.5 GB; 36.4 / 32.1 / 131.0 GB | per-GPU memory (weights+grads / main weights+optimizer / activations), DeepSeek-V3 BF16, PP4 × VPP4 × EP64, 256 GPUs | [MC] Table 3 p.20 |
| 80 GB | H100 capacity | [MC] §4.1.1 p.20 |
| 66% (derived) | 131.0 / 199.5 = 0.657 | derived from Table 3 |
| ~33% | extra compute of full-layer recomputation | [MC] §4.1.4 p.22 ("∼33% computational overhead") |
| < 5% | extra compute of fine-grained recomputation | [MC] §4.1.4 p.23 ("less than 5% additional compute overhead"); drawn as an upper-bound bar at 5 |
| 42.4 GB = 30.4 (MLA up-proj) + 8.2 (LayerNorm) + 3.8 (SwiGLU) | memory saved per GPU by fine-grained recompute, same DeepSeek-V3 config | [MC] Table 4 p.24 |
| 31× (analytic) | per-token values: Q 128×(128+64) + K 128×(128+64) + V 128×128 = 65,536 vs latents 1,536 + 512 + 64 = 2,112; 65,536/2,112 = 31.0 | derived from [DSV3] §4.2 (128 heads, d_h 128, d_c 512, d'_c 1536, d_h^R 64) and [MLA] (query = cat(q_no_pe, q_pe); key = cat(k_no_pe, k_pe expanded to all heads); value split from kv) |
| 7,168 / 7,168 (fig_pairs) | per-token values of a norm input and output (hidden 7168) | [DSV3] §4.2 |
| 32,768 / 16,384 (fig_pairs) | per-token fc1 output 8 × 4,096 (gated, 2 × 2,048) and SwiGLU output 8 × 2,048 over top-8 routed copies | [DSV3] §4.2 (expert intermediate 2048, 8 routed experts) ; derived |
| 26.3 GB | memory-efficient permutation savings (toolbox chart) | [MC] §4.1.2 p.21 |
| ~16 GB | FP8 activation storage savings (toolbox chart) | [MC] §4.1.3 p.22 |
| 10–12 GB | BF16 optimizer moments savings (toolbox chart) | [MC] §4.1.6 p.26 |
| 15–20 GB; 0.1–0.2 s/iter | optimizer-state offload savings and overhead, DeepSeek-V3 (GB200 C2C context) | [MC] §4.1.6 p.26-27 |
| 199.5 GB → under 80 GB | all memory optimizations together, DeepSeek-V3 | [MC] §11 p.78 |
| 64K; 72%; +18%; −16%; 9 GB; 89.8 GB | DeepSeek-V3 at 64K: SDPA up to 72% of compute; core-attention recompute ≈ +18% compute, ≈ 16% performance loss, 9 GB saved; non-SDPA recompute 89.8 GB "globally" | [MC] §6.2 p.58 |
| 1,024 H100, 368 TFLOPS/GPU; 256 GB200, 1,048 TFLOPS/GPU; recompute sets | DeepSeek-V3 final configs: H100 `mlp, mla_up_proj, moe_act, layernorm`; GB200 `mlp` | [MC] Table 17 p.74; §9.2.2 p.75 |
| 80 GB vs 192 GB | H100 vs GB200 memory | [MC] §9.2.2 p.74 |
| Qwen3-235B: `moe_act, layernorm` on GB200 and H100 | | [MC] App. B.2 p.87 |
| Kimi K2: LayerNorm, SwiGLU, MLA up-projections, MoE down-projections | | [K2] §2.4.3 p.8 |
| v0.13 (2025/07), v0.14 (2025/09) | releases: fine-grained recompute; FP8 support for it | [RD] What's New |
| TE ≥ 2.6 | FP8 + moe_act/layernorm output-discarding recompute needs transformer-engine ≥ 2.6.0dev0; delayed scaling rejected | [CFG] `__post_init__` |

## Mechanism facts (no numbers) used on slides

- Output-discarding checkpoint ([RNG] `CheckpointWithoutOutput`): forward runs `run_function` under `no_grad`,
  saves tensor and non-tensor args into the autograd ctx plus CPU/CUDA/tracker RNG states; after the consumer
  has used the output, `discard_output_and_register_recompute(hook_tensor)` resizes the output's untyped
  storage to 0 (tensor metadata kept) and registers `_recompute` as a grad hook on `hook_tensor`. `_recompute`
  restores RNG states, reruns the function under `enable_grad` (and FP8 autocast with the saved recipe +
  TE `activation_recompute_forward(recompute_phase=True)` when FP8), then makes the original output's
  StorageImpl point at the recomputed data (zero-copy; all views see it; no version bump). [MC] §4.1.4 p.23
  describes the same idea as "output-discarding recomputation".
- Hook placement: layernorm → input-layernorm output, hook on the attention output; pre-MLP layernorm,
  hook on the MLP output ([TL]); mla_up_proj → Q/K/V from `qkv_up_proj_and_rope_apply(q_compressed,
  kv_compressed, k_pos_emb, rotary_pos_emb)`, hook on `core_attn_out` ([MLA]); moe_act → SwiGLU (with probs)
  output, hook on the fc2 output ([EXP]).
- `core_attn`, `mlp`, `moe`, `shared_experts` use standard checkpointing ([CFG] docstring; [RD]).
- Constraints ([CFG]): selective mode applies to all layers, `recompute_num_layers` must be None; `moe_act`
  needs `moe_grouped_gemm`; `mla_up_proj` needs MLA; `core_attn` warns (fused attention); `shared_experts`
  excludes `moe_shared_expert_overlap`; FP8 rules above; `--moe-layer-recompute` is deprecated → `moe`.
  The 1F1B EP-overlap schedule asserts no full recompute and no `moe` recompute ([CFG] l.3638-3649).

## Discrepancies

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
