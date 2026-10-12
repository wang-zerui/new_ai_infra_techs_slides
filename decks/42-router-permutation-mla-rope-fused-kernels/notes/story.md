# Story — deck 42: Router / permutation / MLA-RoPE fused kernels

- **Where it lives** (prosperity): Megatron-Core MoE + Transformer Engine. DeepSeek-V3-class layers: MLA (128
  heads, 128-dim nope + 64-dim RoPE slice), sigmoid router with expert bias and node-limited group top-k
  (256 experts, top-8, at most 4 of 8 groups), token permutation around EP all-to-all. Flags
  `--moe-router-fusion`, `--moe-permute-fusion`, `--apply-rope-fusion` (MLA + YaRN). The paper's GB200
  DeepSeek-V3 configuration lists "kernel fusions (router fusion, permute fusion, MLA RoPE fusion)" next to
  CUDA graphs because CPU overhead dominates there.
- **Pain number** (betrayal): routing one microbatch in eager PyTorch takes **22** separate ops for a
  DeepSeek-V3-style router (sigmoid, bias add, group top-k, top-k, gather, normalize, two scatters…),
  **derived** by counting the unfused forward path in `moe_utils.topk_routing_with_score_function`; ×58 MoE
  layers ≈ 1,276 launches per forward pass (derived). Each op reads and writes a [tokens × 256] tensor.
- **Failed suitors**:
  1. CUDA graphs (deck 38): remove launch cost, not the HBM round trips; dropless glue may stay eager.
  2. Op-by-op kernels (unfused PyTorch / TE elementwise): every op still re-streams the tensor; unfused
     unpermute even promotes tokens to FP32 when the router runs in FP32 (code comment).
- **Villain**: glue written as a chain of tensor-wide ops: per-token work is tiny (256 scores, a 64-dim RoPE
  slice) but every step pays a launch and a full HBM pass.
- **Turn / insight**: the glue is per-token and fits on chip. One warp can hold a token's 256 scores in
  shared memory; one row-id map serves permute, unpermute and their backward; RoPE touches only 64 of 192
  dims per head. So run each chain as one kernel: read once, compute on chip, write once.
- **Mechanisms**:
  ❶ Router fusion (challenge: top-k + score + bias + grouping chain): TE `fused_topk_with_score_function`,
     one warp per token, scores in shared memory, group top-k, normalize, scale, probs + routing map in one
     kernel; aux-loss scores and aux loss in one kernel each; all three fused in backward too.
  ❷ Permutation fusion (challenge: sort/gather/scatter + FP32 promotion): row-id map built once (3 Triton
     passes), permute kernel moves tokens + probs (+FP8 scales, + alignment padding), unpermute merges with
     probs in FP32 in one kernel; HybridEP can fold it into dispatch/combine (deck 19).
  ❸ MLA RoPE fusion (challenge: split/rope/cat/expand/contiguous): Q kernel rotates the 64-dim slice in
     place with cached YaRN cos/sin; KV kernel splits kv, rotates k_pe once per token and writes key and value
     for all heads.
  ❹ Low-precision glue: padding/FP8 scales inside permutation, grouped quantization, activation + quantize in
     the grouped-GEMM epilogue (CuTe DSL).
- **Release**: kernel level, fused permute+pad 1.1–1.6× and unpermute+unpad 1.7–3× (TE PR #1921); fused
  grouped-GEMM epilogues up to 1.3× fwd / 2.1× bwd (NVIDIA blog). End to end: +0.4% MFU and ~1 GB less peak
  memory (~600B FP8 MoE, permute+pad); up to 8% DeepSeek-V3 and 93% GPT-OSS (sync-free fused expert kernels);
  part of the 1,048 TFLOPS/GPU GB200 DeepSeek-V3 stack. Router: 22 → 1 kernel per layer forward (derived).
- **Limits**: no isolated end-to-end number published for router or MLA-RoPE fusion; TE ≥ 2.7 for router
  fusion; fused aux loss is not bitwise deterministic (7/7 mismatches) and `--deterministic-mode` rejects it;
  batch-invariant MoE needs the unfused permute; MLA RoPE fusion is training-only, YaRN-only, marked
  experimental; quantile-balancing routing does not support router fusion; Triton dependency.

## Slide plan (15 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Between the GEMMs, MoE layers run on glue | prosperity / hook | fig_glue_1 (SVG) |
| 3 | Routing one microbatch takes 22 eager ops | pain | `%% big: 22` |
| 4 | Every glue op re-streams the tensor through HBM | villain | fig_hbm (SVG) |
| 5 | Graphs and per-op kernels only go halfway | failed suitors | two-col text |
| 6 | Insight: glue is per-token and fits on chip | turn | text + boxed sentence |
| 7 | Three flags turn glue stacks into kernels | overview | fig_glue_2 (SVG, same canvas) |
| 8 | ❶ One warp routes one token | mechanism | fig_router (SVG) |
| 9 | ❷ One row-id map drives permute and unpermute | mechanism | fig_perm (SVG) |
| 10 | ❸ RoPE touches only the 64-dim slice | mechanism | fig_rope (SVG) |
| 11 | ❹ Padding and quantization ride along | mechanism | text |
| 12 | Fused permutation kernels run 1.1–3× faster | release | chart fusion_speedup |
| 13 | Fusions add 0.4–8% on DeepSeek-class training | release | bullets + callout |
| 14 | Fusion trades determinism and flexibility | limits | two-col text |
| 15 | Summary | | QR |
