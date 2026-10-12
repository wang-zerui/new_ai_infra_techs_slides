# Sources and number trace — deck 45 (FP4 QAT: MXFP4 expert weights + indexer QK path)

## Sources read

| Tag | Source | Where |
|---|---|---|
| [V4] | DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence", arXiv:2606.19348v1 (26 Apr 2026). https://arxiv.org/abs/2606.19348 (cached `$CACHE/arxiv-2606.19348.txt`) | Abstract; §1 p.4-5 (FP4 routed experts, FP4×FP8 peak FLOPs remark); §2.3.4 p.13 (indexer in FP4); §4.2.1 p.24-25 (shapes); §5.2.1 FP4 QAT p.33-34; Table 1 (V3.2 671B total) |
| [V3.2] | DeepSeek-AI, DeepSeek-V3.2, arXiv:2512.02556 (cached) | §2.1 (lightning indexer "can be implemented in FP8") |
| [MX] | Rouhani et al., "Microscaling Data Formats for Deep Learning", arXiv:2310.10537 (downloaded 2026-10-13 to `$SCRATCH/research/45/mx/`) | §2 Fig. 1 (block = shared scale X + k elements); Table 1 (MXFP4: block 32, E8M0 8-bit scale, FP4 E2M1 4-bit elements); §3 (MX conversion and transposition do not commute) |
| [cfg] | `config.json` of huggingface.co/deepseek-ai/DeepSeek-V4-Pro / -Flash (2026-10-13) | `expert_dtype: fp4`; `quantization_config`: fp8 e4m3, `weight_block_size` [128, 128], `scale_fmt` ue8m0 |
| [code] | V4 reference inference code `inference/model.py` (linked from [V4] footnote 1) | `fp4_block_size = 32`; FP4 weight = `float4_e2m1fn_x2` with one `float8_e8m0fnu` scale per 32 inputs; `Indexer.forward`: Hadamard rotation (`rotate_activation`) then `fp4_act_quant` on q and on compressed keys |
| [HF] | Hugging Face API tree listings of deepseek-ai/DeepSeek-V4-Pro, DeepSeek-V4-Flash, DeepSeek-V3.2 (2026-10-13; saved in `$SCRATCH/research/45/*-tree.json`) | Sum of `*.safetensors` sizes: V4-Pro 864.7 GB (64 files), V4-Flash 159.6 GB (46), V3.2 689.5 GB (163) |
| [deck01] | Deck 01 `notes/analytic.py` | Per-query multiply-adds of a CSA layer at 2²⁰ tokens |

## Numbers on slides

| Slide | Number | Source |
|---|---|---|
| 1, 3 | V4-Pro 1.6T total parameters (49B activated); V4-Flash 284B (13B) | [V4] Abstract |
| 2 | FP4 applied to (1) MoE expert weights and (2) the QK path of CSA's indexer, during post-training QAT | [V4] §1 p.4; §5.2.1 |
| 2 | Other weights FP8 E4M3 with 128 × 128 blocks; expert weights FP4 | [cfg] `quantization_config`, `expert_dtype` |
| 3 | **~97%**: routed-expert parameters = 61 × 384 × 3 × 7168 × 3072 = 1.547T of V4-Pro's 1.6T (96.7%); Flash 277.0B of 284B (97.5%) — **analytic** | `notes/analytic.py`; shapes from [V4] §4.2.1 (61 layers, all MoE; 384 routed experts; expert hidden 3072; d = 7168; SwiGLU = 3 matrices) |
| 3 | ~1.55 TB of routed-expert weights at 1 byte (FP8) — analytic | Same; 1 B per FP8 weight |
| 4 | V3.2's indexer "can be implemented in FP8"; index scores FP32 before V4's QAT | [V3.2] §2.1; [V4] §5.2.1 ("quantize the index scores from FP32 to BF16") |
| 5 | E2M1 = 1 sign, 2 exponent, 1 mantissa bit; E4M3 = 1, 4, 3; "2 additional exponent bits" | [V4] §5.2.1; [MX] Table 1 |
| 5, 7 | FP4 sub-blocks are 1 × 32 tiles; FP8 quantization blocks are 128 × 128 tiles; dequantization is lossless as long as the max/min ratio of FP4 sub-block scales inside an FP8 block stays under a threshold, verified empirically for current weights | [V4] §5.2.1 p.33 |
| 5 | MXFP4 = one E8M0 scale per 32 FP4 elements → 4 + 8/32 = 4.25 bits = 0.53 B per weight; FP8 = 1 B → 1.88× | [MX] Table 1; derived |
| 6-7 | QAT path: FP32 master → FP4 → FP8 → existing FP8 training framework unmodified; backward: gradients w.r.t. the FP8 weights go directly to the FP32 master (= STE); avoids re-quantizing transposed weights | [V4] §5.2.1 p.33-34 |
| 7 | MX conversion and transposition do not commute (why skipping transposed re-quantization matters) | [MX] §3 |
| 8 | Indexer QK activations cached, loaded and multiplied in FP4; index scores FP32 → BF16; Hadamard rotation before FP4 quantization (reference code); 64 heads × 128 dims | [V4] §5.2.1, §2.3.4, §4.2.1; [code] `Indexer.forward` |
| 8 | Indexer = 14× core attention per CSA layer at 2²⁰ tokens (2.15 G vs 0.151 G multiply-adds, V4-Pro) — analytic | [deck01] / `notes/analytic.py` |
| 9 | Inference and RL rollout use native FP4 weights (not simulated quantization): sampling matches online deployment; less kernel memory loading; lower memory; indexer QK path handled the same way; QAT covers teacher and reference models | [V4] §5.2.1 p.33-34 |
| 9 | Native FP4 kernels load about half the expert bytes of FP8 (0.53 B vs 1 B per weight) | Derived from [MX] Table 1 |
| 7 | MXFP4 = 4.25 bits per weight vs 8 bits in FP8 | [MX] Table 1 (4-bit element + 8-bit scale per 32) |
| 10 | **2×** faster top-k selector; **99.7%** recall of KV entries | [V4] §5.2.1 p.33 |
| 11 | Released checkpoints: V4-Pro 864.7 GB for 1.6T params = 0.54 B/param; V4-Flash 159.6 GB / 284B = 0.56; V3.2 689.5 GB / 671B = 1.03; V4-Pro has 2.38× V3.2's parameters in 1.25× the bytes — derived | [HF] sizes; parameter counts [V4] Abstract and Table 1 (rounded values, so ratios are approximate) |
| 12 | FP4 × FP8 peak FLOPs equal FP8 × FP8 on existing hardware; could be 1/3 more efficient on future hardware | [V4] §1 p.5 |
| 12 | QAT is a post-training stage; no FP4-vs-FP8 accuracy ablation is reported | [V4] §1 p.4, §5.2.1 (absence checked) |
| 13 | Summary repeats 97% (analytic), 2× / 99.7%, 0.54 B/param | above |

## Discrepancies (Notion row vs primary source; the primary source wins on the slides)

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
