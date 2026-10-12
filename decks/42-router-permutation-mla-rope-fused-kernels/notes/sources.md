# Sources — deck 42: Router / permutation / MLA-RoPE fused kernels

Downloads are in `$SCRATCH/research/42/` (fetched 2026-10-13 with `gh api` / `curl`).

## Sources read

| Tag | Source | Where |
|---|---|---|
| [MC] | NVIDIA, "Scalable Training of Mixture-of-Experts Models with Megatron Core", arXiv 2603.07685v2 (cached txt) | §4.1.2 p.21 (memory-efficient permutation, fused backward); §4.2.1 p.30 (58 MoE layers); §4.3.1 p.35 (~128 tokens/expert, many small kernels); §4.3.2-4.3.4 p.36-38 (Grouped GEMM, permutation fusion pipeline Fig. 20, router + aux-loss fusion Fig. 21); §5.4.1-5.4.2 p.54-55 (padding fused into permutation, grouped quantization); Table 15-16 p.73; Table 17-18 p.74; §9.2.2 p.75 |
| [DSV3] | DeepSeek-V3 Technical Report, arXiv 2412.19437 (cached txt) | §2.1.2 (sigmoid affinity, top-K normalization); §3.2.2 p.13 (≤ 4 nodes); §4.2 p.22 (256 routed experts, 8 activated, MLA: 128 heads, d_h 128, d_R 64) |
| [MLM-U] | Megatron-LM `megatron/core/transformer/moe/moe_utils.py` (main, fetched 2026-10-13) | `topk_routing_with_score_function` (unfused path), `group_limited_topk`, `permute`, `unpermute` (FP32 promotion comment) |
| [MLM-R] | Megatron-LM `megatron/core/transformer/moe/router.py` | `fused=self.config.moe_router_fusion` (routing) and `moe_router_aux_loss_fusion` (aux loss) call sites; quantile balancing asserts router fusion off |
| [MLM-ROPE] | Megatron-LM `megatron/core/fusions/fused_mla_yarn_rope_apply.py`; `megatron/core/transformer/multi_latent_attention.py` | Triton kernels `_mla_rope_fwd_inplace_kernel` (Q, in place on the trailing emb_dim) and `_mla_rope_fwd_kv_split_kernel` (split kv, RoPE k_pos_emb, write key/value); unfused branch: `torch.split` ×2, `apply_rotary_pos_emb` ×2, `torch.cat` ×2, `expand`, `.contiguous()` ×3; fused branch uses `get_cached_cos_sin` |
| [MLM-CFG] | Megatron-LM `megatron/core/transformer/transformer_config.py` | `moe_router_fusion` ("TransformerEngine 2.7.0 and above"), `moe_router_aux_loss_fusion`, `moe_permute_fusion`, `apply_rope_fusion` (MLA: "only supports training … experimental"; "only works with YARN RoPE"); batch-invariant MoE asserts unfused permute |
| [TE-R] | TransformerEngine `transformer_engine/pytorch/router.py`; `common/fused_router/fused_topk_with_score_function.cu` | `fused_topk_with_score_function` (softmax/sigmoid/sqrtsoftplus, group top-k, expert bias, scaling, routing map / bitmap / dense indices); kernel comment "Each warp is responsible for one token, and has own shared memory buffer"; `fused_compute_score_for_moe_aux_loss`, `fused_moe_aux_loss` |
| [TE-P] | TransformerEngine `transformer_engine/common/triton/permutation.py`; `pytorch/permutation.py` | `_row_id_map_pass_1/2/3_kernel`, `_permute_kernel` (PERMUTE_PROBS, PERMUTE_SCALE, FUSION_PAD), `_unpermute_kernel`; `moe_permute_with_probs`, `moe_permute_and_pad_with_probs`, `moe_unpermute`, `moe_sort_chunks_by_index` |
| [TE-1883] | TransformerEngine PR #1883 "Kernels fusions for the MoE router" (merged 2025-07-10) | fuses topk + softmax/sigmoid, aux-loss scores, aux loss; forward and backward |
| [TE-1921] | TransformerEngine PR #1921 "Fuse permute+pad and unpermute+unpad ops for FP8 optimization" (merged 2025-12-27) | Results: 1.1x~1.6x fused permute-and-pad, 1.7x~3x fused unpermute-and-unpad (test_permutation.py); +0.4% MFU, ~1 GB peak memory reduction in a ~600B-parameter FP8 Megatron setup |
| [MLM-6967] | Megatron-LM PR #6967 (merged 2026-09-15) | fused aux loss not bitwise deterministic: 7/7 (aux_loss), 5/7 (seq_aux_loss), 5/7 (global_aux_loss) mismatches vs 0/7 unfused; `--deterministic-mode` rejects only the aux-loss fusion |
| [BLOG-F] | R. Garg, M. Nicely, "Boosting MoE Training Throughput with Advanced Fusion Kernels", NVIDIA Technical Blog. https://developer.nvidia.com/blog/boosting-moe-training-throughput-with-advanced-fusion-kernels/ (`blog_fusion.txt`) | GroupGEMM + activation + quantize/transpose kernels (CuTe DSL); up to 1.3x forward, 2.1x backward vs unfused; up to 8% end-to-end DeepSeek-V3, up to 93% GPT-OSS pre-training; available in cuDNN FE ≥ 1.23, TE ≥ 2.15, MCore 26.04-alpha.rc2 |
| [RD] | Megatron-LM MoE README (cached) | v0.14 "Router fusion kernels for MoE models (--moe-router-fusion)"; perf flags `--moe-router-fusion --moe-permute-fusion` |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 256 routed experts, top-8, ≤ 4 of 8 node groups; sigmoid scores + bias | DeepSeek-V3 routing | [DSV3] §2.1.2, §3.2.2 p.13, §4.2 p.22 |
| 128 heads; 128-dim nope + 64-dim RoPE per head (192 for q/k); v 128 | DeepSeek-V3 MLA dims | [DSV3] §4.2 p.22 |
| 58 MoE layers | DeepSeek-V3 | [MC] §4.2.1 p.30 |
| ~128 tokens per expert | DeepSeek-V3 expert GEMM M | [MC] §4.3.1 p.35 |
| **22** ops | **derived**: PyTorch ops launched by the unfused forward of `topk_routing_with_score_function` for sigmoid + expert bias + group-limited top-k (8 groups, 4 kept) + top-8 + normalization, FP32 router logits, no scaling factor. Count: sigmoid (1), bias add (2), per-group topk (3), group sum (4), group topk (5), zeros_like (6), scatter_ (7), expand→reshape copy (8), .bool() (9), logical not (10), masked_fill (11), topk-8 (12), gather (13), sum (14), +eps (15), div (16), zeros_like (17), scatter probs (18), zeros_like (19), .int() (20), scatter map (21), .bool() (22). Views are not counted; a top-k scaling factor adds one more multiply; internal multi-kernel ops (e.g. topk) are counted once, so 22 is a lower bound on kernel launches | [MLM-U] |
| ≈ 1,276 | **derived**: 22 × 58 MoE layers, router launches per forward pass | [MLM-U], [MC] p.30 |
| 1 kernel (+1 aux-loss scores, +1 aux loss) | fused router forward (also fused backward) | [TE-R], [TE-1883], [MC] §4.3.4 p.37-38 |
| 3 passes | row-id map construction kernels, built once per forward and reused | [TE-P]; [MC] §4.3.3 p.37 |
| 16 / 32 / 128 | token alignment: 16 per-tensor/blockwise FP8, 32 MXFP8/NVFP4, 128 for grouped quantization + CUDA graphs | [MC] §5.4.1 p.54-55 |
| 1.1–1.6× | fused permute+pad vs separate ops (kernel benchmark) | [TE-1921] |
| 1.7–3× | fused unpermute+unpad vs separate ops | [TE-1921] |
| +0.4% MFU, ~1 GB peak memory | end-to-end FP8 training, ~600B-parameter MoE, Megatron | [TE-1921] |
| up to 1.3× fwd, up to 2.1× bwd | fused GroupGEMM + activation + quantize kernels vs unfused (microbenchmarks) | [BLOG-F] |
| up to 8% (DeepSeek-V3), up to 93% (GPT-OSS) | end-to-end from the sync-free fused expert kernels | [BLOG-F] |
| 1,048 TFLOPS/GPU, 256 GB200 | DeepSeek-V3 final configuration that lists kernel fusions (router, permute, MLA RoPE) with CUDA graphs | [MC] Table 17-18 p.74, §9.2.2 p.75 |
| TE ≥ 2.7 | router fusion requirement | [MLM-CFG] |
| 7/7 mismatches | fused aux loss bitwise non-determinism (aux_loss type) | [MLM-6967] |

## Mechanism facts (no numbers) used on slides

- Router fusion decomposes the non-GEMM router into three fused kernels: score computation incl. top-k and
  softmax/sigmoid (naive or group top-k, scaling), scores for aux loss, aux loss ([MC] §4.3.4 p.37-38).
  TE kernel: one warp per token with its own shared-memory buffer; outputs probs and routing map (byte map,
  bit map or dense top-k indices) ([TE-R]).
- Permutation fusion: preprocessing builds an offset map (row-id map) once in the forward pass and other
  components reuse it; permute moves tokens (and probs in memory-efficient mode) to expert-contiguous rows;
  unpermute sums the copies of a token, weighted by probs unless memory-efficient permutation already applied
  them, accumulating in FP32 ([MC] §4.3.3 p.36-37; [TE-P]).
- Unfused permute uses transpose + argsort + gather/index_select; unfused unpermute uses masked_select of probs,
  multiply and scatter_add, and "may promote permuted_tokens to higher precision (fp32/fp64) if probs is in
  higher precision … Use --moe-permute-fusion flag to avoid this extra memory allocation" ([MLM-U]).
- Padding fused into permutation avoids one global-memory read/write pass and is the default when available;
  routing-map padding is the alternative ([MC] §5.4.1). FP8 scales are permuted alongside tokens in the same
  kernel (`PERMUTE_SCALE`) ([TE-P]).
- HybridEP can fuse permute/unpermute into its dispatch/combine kernel (`moe_permute_fusion_into_hybridep`,
  Megatron PRs #4073/#4089) (deck 19).
- MLA RoPE fusion: Q kernel applies RoPE in place to the trailing emb_dim (64) of each head; KV kernel splits kv
  into key/value, applies RoPE to the shared k_pos_emb and concatenates it onto every head's key; cos/sin come
  from a cached YaRN table (`get_cached_cos_sin`) ([MLM-ROPE]; Megatron commit 23e6471 "MLA RoPE fusion and
  YARN embedding cache").
- Paper on GB200: "Kernel fusions (router fusion, permute fusion, MLA RoPE fusion) reduce kernel launch count"
  ([MC] §9.2.2 p.75).

## Discrepancies

1. **Notion "fuses routing score/topk/bias, layered with permutation, MLA RoPE and FP8-cast fusion, plus cuDNN
   fused attention / FlashAttention integration".** The primary sources describe router, aux-loss,
   permutation(+padding), MLA-RoPE and quantization fusions; fused attention (cuDNN) is listed separately in
   the README feature list. The deck covers the fused glue kernels and mentions attention only as context.
2. **Notion "Impact: cuts launch and memory-traffic overhead".** No isolated end-to-end number is published
   for router fusion or MLA-RoPE fusion; measured numbers exist only for permute/pad fusion ([TE-1921]) and for
   the fused grouped-GEMM epilogues ([BLOG-F]). The slides say so.
3. **Paper Fig. 21 vs TE kernels.** The paper describes "three sections, each fused into a single kernel"; TE
   also has backward kernels for each and a quantile-balancing variant; the slides count forward kernels.
