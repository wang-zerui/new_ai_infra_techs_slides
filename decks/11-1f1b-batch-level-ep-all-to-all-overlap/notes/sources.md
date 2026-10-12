# Sources — deck 11: 1F1B / batch-level EP all-to-all overlap

## Sources read

| Tag | Source | Where |
|---|---|---|
| [MC] | NVIDIA, "Scalable Training of Mixture-of-Experts Models with Megatron Core", arXiv 2603.07685v2 (10 Mar 2026). https://arxiv.org/abs/2603.07685 | §1.2 (p.6), §4.2 intro (p.29-30), §4.2.1 (p.30), §4.2.3 + Fig. 16-19 (p.32-35), §8.1 (p.68), §9.1.4 (p.73), §9.2 + Tables 17-18 (p.73-76), App. B.2 (p.87) |
| [RD] | Megatron-LM `megatron/core/transformer/moe/README.md` (cached copy, "What's New" v0.14, Communication Optimization, Feature table) | lines 20, 251, 429-438, 563-564 |
| [CODE-PLAN] | `megatron/core/models/common/model_chunk_schedule_plan.py` (main, fetched 2026-10-13) | `TransformerLayerSchedulePlan.run` docstring + body; `TransformerModelChunkSchedulePlan.run` docstring |
| [CODE-1F1B] | `megatron/core/pipeline_parallel/combined_1f1b.py` | `combined_1f1b_schedule_for_no_pipelining` docstring |
| [CODE-SCHED] | `megatron/core/pipeline_parallel/schedules.py` | `get_pp_rank_microbatches` (extra warmup micro-batch when overlap is on) |
| [CODE-CFG] | `megatron/core/transformer/transformer_config.py` `__post_init__` asserts (lines ~3619-3718); `megatron/core/model_parallel_config.py` docstrings of `overlap_moe_expert_parallel_comm`, `delay_wgrad_compute`, `ep_overlap_early_attn_memory_release` | |
| [DSV3] | DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv 2412.19437 | §3.2.1 (comp:comm ≈ 1:1; DualPipe), Table 2 (DualPipe 2× parameters) |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 685B total / 37B active | DeepSeek-V3 parameter counts | [MC] §1.2 p.6 |
| 256 experts, top-8 routing | DeepSeek-V3 MoE config | [MC] §9.2 p.73 ("memory pressure from 256 experts, high all-to-all volume from top-8 routing") |
| 58 MoE layers; 2 all-to-alls per layer; 116 per forward pass; backward doubles it | dispatch/combine count | [MC] §4.2 p.29 and §4.2.1 p.30 |
| EP64 spans 8 nodes on H100 (NVL8) | topology | [MC] §9.2.2 p.75 |
| 20–60% | EP all-to-all share of training time before optimization | [MC] §4.2 p.29 |
| ~20% within NVLink (DeepSeek-V3 EP64 on GB200 NVL72); 40–60% cross-node (EP64 on H100) | same | [MC] §4.2 p.29 |
| 30–40% | EP all-to-all share of iteration time that remains after DeepEP, DeepSeek-V3 EP64 (H100) | [MC] §4.2.3 p.32 ("can still account for 30–40%") and p.33 ("from 30–40% (after DeepEP)") |
| "about half of layer time" | cross-node EP comm share of layer time for fine-grained MoE | [MC] §4.2.3 p.32 |
| ≈ 1:1 | DeepSeek-V3 computation-to-communication ratio with cross-node EP | [DSV3] §3.2.1 |
| 2× parameters | DualPipe keeps two copies of model parameters | [DSV3] §3.2.1 and Table 2 |
| 2× peak activation memory; forward ≈ ½ backward compute | merged FWD-FWD trade-offs | [MC] §4.2.3 p.32 |
| < 5% | EP communication share after 1F1B FWD-BWD overlap + W/D split, DeepSeek-V3 on H100 | [MC] §4.2.3 p.33 and p.34 |
| 93% | overlap ratio for expert communication latency (overlap + optimized dispatchers), DeepSeek-V3 | [MC] §4.2.3 p.34 |
| < 10% | all-to-all contribution to training time with all communication-wall techniques combined | [MC] §4.2.4 p.34 |
| 20 SMs per GPU, ~20% GEMM-efficiency overhead | DeepEP SM carve-out in the DeepSeek-V3 benchmark | [MC] §4.2.3 p.34 |
| 1,024 H100, TP2/PP8/EP64, VPP4, 368 TFLOPS/GPU, EP all-to-all overlap enabled | final DeepSeek-V3 H100 configuration | [MC] Table 17 p.74 |
| 256 GB200, TP1/PP4/EP64, VPP4, 1,048 TFLOPS/GPU, overlap not used | final DeepSeek-V3 GB200 configuration | [MC] Table 17 p.74 |
| ON/OFF per platform: DeepSeek-V3 ON for GB300 and H100, OFF for GB200; Qwen3-235B OFF for GB300, ON for GB200 and H100 | "1F1B overlap" setting in the Table 11 benchmark runs | [MC] App. B.2 p.87 |
| MCore v0.14 (2025/09) | release that added the flags | [RD] line 20 |

Derived numbers: none. The chart `charts/a2a_share.py` plots the three published ranges above
(40–60% cross-node before optimization, 30–40% after DeepEP, <5% after overlap); the "<5%" bar is drawn
from 0 to 5 and labelled "<5%" (an upper bound, not a measured point).

## Mechanism facts (no numbers) used on slides

- Per-layer pairing: forward layer i of micro-batch n+1 runs together with backward layer L−1−i of micro-batch n ([CODE-PLAN] `TransformerModelChunkSchedulePlan.run` docstring: "forward_layer[0] + backward_layer[3]" ...).
- Per-layer order, two CUDA streams ([CODE-PLAN] `TransformerLayerSchedulePlan.run` docstring):
  comm stream: combine_bwd | dispatch_fwd → dispatch_bwd | combine_fwd;
  comp stream: pre_dispatch_fwd (attention, norm, router) | mlp_bwd → mlp_bwd_dw → mlp_fwd | pre_dispatch_bwd.
  The attention weight-gradient (`pre_dispatch_computation.backward_dw`) is delayed; for the last layer it overlaps the PP send/recv.
- W/D split (`--delay-wgrad-compute`): backward MLP split into data-gradient (feeds B/dispatch) and weight-gradient (independent) ([MC] §4.2.3 p.33; [CODE-CFG] docstring "Delay the weight gradient computation to improve batch-level communication overlapping").
- No-pipelining schedule: phase 0 = forward of micro-batch 1 alone; phase k = backward of k + forward of k+1; last phase = backward of the last micro-batch alone ([CODE-1F1B] docstring). Matches [MC] p.33 "first FWD and last BWD remain on the end-to-end critical path".
- Interleaved PP: one extra warmup micro-batch when overlap is on, "to ensure the forward and backward computations are independent in all 1f1b steps" ([CODE-SCHED]; [MC] p.33).
- Prerequisites asserted in code ([CODE-CFG]): EP > 1; `alltoall` or `flex` dispatcher; VPP must be set when PP > 1; no full recompute, no recompute_method/num_layers, `moe` not in recompute modules; BF16 or FP16 model; MTP ≤ 1 layer; `--moe-shared-expert-overlap` must be off; torch ≥ 2.6; `--delay-wgrad-compute` requires the overlap flag. [RD] adds CUDA_DEVICE_MAX_CONNECTIONS > 1.
- Memory: "No additional memory overhead (activations from the forward pass are reused)" for merged FWD-BWD ([MC] p.33), but "EP communication overlap requires extra buffers" ([MC] §9.1.4 p.73) and on H100 the overlap "is only feasible because FP8 freed enough memory for the extra buffers" ([MC] §9.2.2 p.75). [CODE-CFG] `ep_overlap_early_attn_memory_release` docstring: the overlap can raise peak memory when the overlapped forward module allocates more than the backward module frees.

## Discrepancies

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
