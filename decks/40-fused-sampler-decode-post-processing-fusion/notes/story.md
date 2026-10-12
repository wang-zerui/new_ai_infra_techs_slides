# Story — deck 40: Fused Sampler (decode post-processing fusion)

- **Where it lives** (prosperity): the last step of every decode iteration in LLM serving (HPC-Ops, Tencent
  Hunyuan; tuned for H20/SM90). The LM head emits B × V logits (V = 120,832 in the HPC-Ops benchmark) and the
  sampler turns each row into one token: repetition penalty → temperature → top-k → softmax → top-p →
  random draw → penalty-state update. It runs once per generated token, and small batches make its fixed
  costs visible (README).
- **Pain number** (betrayal): a vLLM-style PyTorch sampler issues **~20** (21 counted) separate vocabulary-wide tensor ops
  per decode step (counted from the HPC-Ops benchmark's full-sampling baseline), each reading and/or writing a
  B × 120,832 tensor in HBM.
- **Failed suitors**:
  1. Stage-at-a-time PyTorch (vLLM-style): one op per stage, full-vocab sort for top-k/top-p, two softmaxes,
     and a presence mask rebuilt every step from host token lists (copy + scatter).
  2. FlashInfer's fused top-k/top-p sampling kernel fuses the tail, but penalty, temperature and the mask
     rebuild still run as separate passes in front of it (benchmark's FlashInfer path).
- **Villain**: stage-at-a-time execution materializes a full B × V tensor after every stage, although with
  top-k ≤ 64 everything after the first selection only concerns ≤ 64 tokens per row; penalty state lives as
  token lists, not as GPU-resident state.
- **Turn / insight**: with a bounded top-k, only one pass needs the whole vocabulary: penalty and temperature
  are elementwise, the softmax normalizer is a mergeable reduction (online max + sum), and Gumbel-max turns the
  random draw into an argmax — so the pipeline collapses to one read of the logits plus a tiny tail over the
  N × K candidates.
- **Mechanisms**:
  ❶ K1 single vocabulary pass (answers: repeated full-vocab reads): N = clamp(SMs/B, 8, Nmax) blocks per row,
     penalty bit test and ×1/T in registers, online max/sum, rolling top-K held by keeper threads and merged
     by a block radix sort per chunk.
  ❷ K2 candidate tail with Gumbel-max (answers: sort + cumsum + softmax + multinomial over V): one block per
     row merges N × K candidates, softmax over survivors, top-p by warp prefix scan, argmax(log p + g).
  ❸ GPU-resident penalty bitmask (answers: per-step mask rebuild from host token lists): 1 bit per token,
     read by K1, sampled token's bit set by K2 with atomicOr. Temperature-only requests take a one-kernel
     fast path.
- **Release**: 4.6–8.5× faster than the vLLM-style PyTorch sampler and 2.1–5.2× faster than the FlashInfer
  path (batch 1–512, BF16 logits, V = 120,832); README headline "up to 8.5×". Bit-exact vs a PyTorch reference
  when the Gumbel noise is injected.
- **Limits**: bounded to the top-32/64 candidates (top-k = 0 does not mean full-vocab sampling); V is a
  compile-time constant (only 120,832 built); repetition penalty only (a bitmask cannot count frequencies);
  temperature path uses a per-device workspace (one stream per device); draft-token mask only on that path;
  GPU for the published numbers not stated.

## Slide plan (13 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Every decode step ends in a sampler | prosperity | fig_pipeline (SVG) |
| 3 | A PyTorch sampler runs ~20 vocabulary-wide ops | pain | `%% big: ~20` |
| 4 | Each stage re-reads the full vocabulary | villain / failed suitors | fig_baseline (SVG) |
| 5 | Insight: after top-k, at most 64 tokens matter | turn | bullets + boxed sentence (build) |
| 6 | Two kernels: one vocabulary pass, one tiny tail | overview | fig_overview (SVG) |
| 7 | ❶ K1 reads each logit exactly once | mechanism | fig_k1 (SVG) |
| 8 | ❷ K2 samples by Gumbel-max over survivors | mechanism | fig_k2 (SVG) |
| 9 | ❸ The penalty mask never leaves the GPU | mechanism | fig_mask (SVG) |
| 10 | Runs 4.6–8.5× faster than a vLLM-style sampler | result | chart speedup (range bars) |
| 11 | Same noise in, bit-identical token out | result (fidelity) | fig_exact (SVG) |
| 12 | The cost: top-64 bound, one vocabulary size | limits | two-col |
| 13 | Summary | | QR |
