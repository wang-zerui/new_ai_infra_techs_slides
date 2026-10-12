# Sources — deck 49: Precision-aware optimizer (BF16/FP8 optimizer states)

## Sources read

| Tag | Source | Where |
|---|---|---|
| [MC] | NVIDIA, "Scalable Training of Mixture-of-Experts Models with Megatron Core", arXiv 2603.07685v2. https://arxiv.org/abs/2603.07685 | §3.1.1 Llama-405B memory table (p.12); §4.1.1 + Table 3 (p.20); §4.1.6 "Precision-Aware Optimizer" and "State Offloading" (p.25-27); Table 6 (p.29); §5.2.1 (p.49); Table 13 (p.72); Table 18 (p.74); §10 conclusion (p.78) |
| [RD] | Megatron-LM `megatron/core/transformer/moe/README.md` (cached copy) | "What's New" MCore v0.13: BF16 optimizer states via `--use-precision-aware-optimizer` |
| [ARGS] | Megatron-LM `megatron/training/arguments.py` (main, fetched 2026-10-13) | `--use-precision-aware-optimizer`, `--main-grads-dtype {fp32,bf16}`, `--main-params-dtype {fp32,fp16}`, `--exp-avg-dtype` / `--exp-avg-sq-dtype {fp32,fp16,bf16,fp8}` (help: storage dtype, "does not affect the precision in the kernel computation"); optimizer CPU offload asserts `--use-precision-aware-optimizer` |
| [OC] | Megatron-LM `megatron/core/optimizer/optimizer_config.py` and `optimizer/__init__.py` (main) | `store_param_remainders: bool = True` (docstring: store the 16-bit FP32 remainders, excluding the 16 bits shared with BF16 params); disabled when TE < 2.1; precision-aware optimizer asserts `optimizer == 'adam'` and TE FusedAdam; FusedAdam built with `master_weights=True`, `use_decoupled_grad=True`, `master_weight_dtype=main_params_dtype`, `store_param_remainders` |
| [TEA] | TransformerEngine `transformer_engine/pytorch/optimizers/fused_adam.py` (main) | docstring (dtype options; FP32 scalar scaling factor for non-FP32 states; `store_param_remainders`); `_apply_scale` (scale = absmax / range; FP16 range = max/2, FP8 E4M3 range = 448; "Only scaling with one scaling factor per tensor"); `fuse_unscale` when both moments are BF16; capturable-mode restrictions |
| [TEK] | TransformerEngine `transformer_engine/common/multi_tensor/adam.cu` (main) | `AdamFunctorMasterParamRemainder`: reconstruct FP32 from BF16 high half + int16 low half ("Undo rounding"), FP32 Adam math, split back with round-to-nearest |
| [DSV3] | DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv 2412.19437 (cached) | §3.3 intro and §3.3.3 "Low-Precision Optimizer States" (p.18): BF16 first/second moments "without incurring observable performance degradation"; master weights and gradients kept in FP32 |
| [FSDP] | Megatron-LM `docs/user-guide/features/megatron_fsdp.md` | "Mixed-Precision & Quantization" table: precision-aware optimizer with `--exp-avg-dtype`, `--exp-avg-sq-dtype`, both support fp8 |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| ~810 GB, ~1,620 GB, ~4,860 GB, ~5,575 GB | Llama-405B, BF16 + Adam: model parameters, gradients, optimizer states, activations (8K sequence) | [MC] §3.1.1 p.12 |
| 6× | 4,860 / 810 | derived |
| 12 B = 4 + 4 + 4 | FP32 master weight + FP32 first and second moments per parameter | [MC] §4.1.6 p.26 (moments "8 bytes per parameter" in FP32); FP32 master from [DSV3] §3.3.3 and [MC] §4.1.6 ("main parameters ... remain in FP32") |
| 18 B | BF16 weight 2 + FP32 grad 4 + 12 | consistent with Megatron-LM dist_optimizer.md (deck 70 sources) |
| 32.1 GB | DeepSeek-V3 main weights + optimizer states per GPU (BF16, PP4 × VPP4 × EP64, 256 GPUs) | [MC] Table 3 p.20 |
| 16%, 199.5 GB | share of total and total per-GPU memory | [MC] §4.1.6 p.25 ("representing 16% of the total memory footprint"); Table 3 |
| 80 GB | H100 capacity | [MC] §4.1.1 p.20 |
| 8 → 4 (BF16) → 2 (FP8) bytes | moment storage per parameter | [MC] §4.1.6 p.26 |
| ~50%, 10–12 GB of 32.1 GB | optimizer-state memory reduction with BF16 moments, DeepSeek-V3 | [MC] §4.1.6 p.26 |
| 6 + 12/d → 6 + 8/d | bytes per parameter per DP rank with the distributed optimizer, BF16 moments | [MC] §4.1.6 p.26 |
| 15–20 GB, 47–62%, 0.1–0.2 s per iteration | optimizer state offloading on GB200 for DeepSeek-V3 | [MC] §4.1.6 p.26-27 |
| "no performance overhead" | precision-aware optimizer (FP32 cast inside the fused Adam kernel) | [MC] §4.1.6 p.27 |
| 199.5 GB → < 80 GB | combined memory techniques for DeepSeek-V3 (incl. BF16 moments) | [MC] §10 p.78 |
| 368 TFLOPS/GPU (1,024 H100), 1,048 TFLOPS/GPU (256 GB200) | final DeepSeek-V3 configurations, both listing "Low-precision optimizer states" | [MC] Table 17 + Table 18 p.74 |
| 448 | FP8 E4M3 range used for the per-tensor scale | [TEA] `dtype_to_range_map` |
| v0.13 | MCore release adding BF16 optimizer states | [RD] |
| TE ≥ 2.1 | needed for `store_param_remainders` | [OC] |
| 16 + 16 bits | BF16 = high half of an FP32 word; remainder = int16 low half | [OC] docstring; [TEK] |
| 2⁻⁸ of \|w\| | half the BF16 spacing relative to the weight (BF16 has 8 significant bits incl. the implicit one) | derived from the BF16 format (1 sign, 8 exponent, 7 stored mantissa bits); spacing between neighbours in [2^e, 2^(e+1)) is 2^(e−7), half of it is 2^(e−8) ≤ 2⁻⁸·\|w\| |

### Derived numbers (analytic chart `charts/bytes_stack.py`)

Per parameter, no sharding (d = 1). BF16 compute weight 2 B and FP32 main gradient 4 B are kept fixed (the
`--main-grads-dtype bf16` option is mentioned in notes only).

| Configuration | weight | grad | master | m | v | optimizer state (master + m + v) | total |
|---|---|---|---|---|---|---|---|
| FP32 Adam (default) | 2 | 4 | 4 | 4 | 4 | 12 | 18 |
| BF16 moments, FP32 master ([MC] "typical") | 2 | 4 | 4 | 2 | 2 | 8 | 14 |
| BF16 moments + remainder ([OC] default with TE ≥ 2.1) | 2 | 4 | 2 | 2 | 2 | 6 | 12 |
| FP8 moments + remainder | 2 | 4 | 2 | 1 | 1 | 4 | 10 |

The remainder stores exactly the 16 bits of the FP32 master weight not present in the BF16 weight ([OC], [TEK]),
so the master stays bit-exact FP32 at 2 extra bytes. Per-tensor FP32 scales (FP8/FP16 states) are one value per
tensor and ignored.

DeepSeek-V3 chart `charts/dsv3_state.py`: remaining main-weight + optimizer-state memory per GPU =
32.1 GB − savings: BF16 moments 32.1 − (10 to 12) = 20.1–22.1 GB; state offloading 32.1 − (15 to 20) =
12.1–17.1 GB. Both are derived from [MC] §4.1.6 ranges.

## Mechanism facts (no numbers) used on slides

- Precision-aware optimizer decouples storage precision from compute precision; moments stored in BF16 or FP8
  are cast to FP32 inside TE's FusedAdam, the update is computed in FP32 ([MC] §4.1.6; [ARGS] help text).
- Four configurable precisions: main grads, main params, first moment, second moment ([MC] §4.1.6; [ARGS]).
- BF16 moments: scaling is skipped and the cast is fused into the Adam kernel (`fuse_unscale`); FP16/FP8 moments
  get one FP32 scale per tensor = absmax / range, unscaled into temporary FP32 tensors before the kernel and
  re-scaled after ([TEA]).
- FP16 master weights also get a per-tensor FP32 scale; FusedAdam supports FP32/FP16 master only ([TEA]).
- Remainder kernel: if the stored remainder is negative, the BF16 weight was rounded up, so subtract one before
  reassembling; after the FP32 update, split and round the BF16 half to nearest ([TEK]).
- Capturable (CUDA-graph) mode requires FP32 master weights, both moments FP32 or both BF16, and no remainders
  ([TEA]).
- Precision-aware optimizer only with `--optimizer adam` and TE FusedAdam ([OC]).
- With FP8 primary weights under non-delayed recipes and FP32 main params, MCore manages the master weights
  itself (comment in [OC] `__init__.py`), so remainders do not apply there.
- Optimizer CPU offload must be used with `--use-precision-aware-optimizer` ([ARGS]); BF16 states shrink what is
  offloaded and the transfer time ([MC] §4.1.6 p.27).
- Megatron-FSDP supports the precision-aware optimizer with the same dtype flags, including fp8 ([FSDP]).
- DeepSeek-V3 keeps master weights and gradients in FP32 "to ensure numerical stability" ([DSV3] §3.3.3).

## Discrepancies

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
