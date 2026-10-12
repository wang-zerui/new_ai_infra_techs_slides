# Story: FP8 block-sparse prefill attention (HPC-Ops)

- **Where it lives** (prosperity): long-prompt prefill (64K–128K contexts: documents, RAG, agents) in Tencent's
  HPC-Ops on H20, paired with Stem (Tencent, ICML 2026), which decides which 128-token KV blocks each query block
  needs.
- **Pain number** (betrayal): dense prefill attention grows quadratically: 24 ms at 16K → **1540 ms** at 128K for
  Llama-3.1-8B on H20 (64× for 8× the length) [S Fig. 1].
- **Failed suitors**: (1) dense FP8 kernels (HPC-Dense FP8, FA3 FP8): faster per tile, but still every causal tile;
  (2) BF16 block-sparse kernels (MIT Block-Sparse-Attention, FlashPrefill-BSA): skip tiles but run the kept ones in
  BF16; (3) mask generation in Triton is slow and can drift from the FP32 mask.
- **Villain**: a dense loop visits every causal KV tile regardless of relevance, and the existing sparse loops gave
  up FP8 — skipping and low precision were never combined in one kernel.
- **Turn / insight**: most tiles can go (Stem keeps 25–31% of causal blocks within ≈0.5 point of dense on
  LongBench), and the mask for a query tile is known before its loop starts, so the kernel can turn it into a work
  list: skipped tiles then cost neither bytes nor FLOPs.
- **Mechanisms** (numbered in data-flow order):
  ❶ on-GPU Stem mask at the kernel's 128-token granularity (pool → OAM GEMM → position-aware top-k, always keeping
  initial, window and diagonal blocks) ↔ mask cost and correctness;
  ❷ active-tile list: the producer warp compacts the mask row into an SMEM list (ballot + popc) while Q loads, then
  TMA-loads only the kept KV tiles; consumers walk the same list ↔ "skipping must be free";
  ❸ FP8 math path with fine-grained scales: FP8 wgmma for Q·Kᵀ and P·V, per-token Q and per-tensor/per-token K
  scales, P × 256 before its FP8 cast, a fresh FP32 accumulator per tile ↔ FP8 accuracy.
- **Release**: latency tracks density — 101% of dense at 100% density, 11% at 10% (64K × 4, H20); up to 3.16× over
  BF16 block-sparse and dense FP8 kernels (README); mask generation 7.2–44× faster than Triton with ≥ 99.9% mask
  match.
- **Limits**: needs a good mask (accuracy is the mask method's); SM90/H20, head_dim 128, page size dividing 128;
  per-Q-head masks (no sharing of loads across a GQA group); an all-skipped Q tile yields NaN, so the diagonal must
  stay; end-to-end accuracy with the FP8 kernel not published (Stem's accuracy numbers used BF16 MIT kernel).

## Slide plan (14 logical slides)

| # | Headline | Device |
|---|---|---|
| 1 | Title | |
| 2 | Long prompts make prefill attention quadratic | bullets |
| 3 | A 128K prompt spends 1.5 s in dense attention | big 1540 ms |
| 4 | Today: compute every tile, or lose FP8 | fig_before |
| 5 | Insight: most tiles can go, if skipping is free | bullets + boxed (build) |
| 6 | A block mask steers one persistent FP8 kernel | fig_overview |
| 7 | ❶ Stem builds the mask on the GPU | fig_stem |
| 8 | ❷ The mask becomes a compact tile list | fig_list |
| 9 | ❸ Kept tiles run end to end in FP8 | fig_fp8 |
| 10 | Latency tracks density: 11% at 10% | chart_density |
| 11 | Up to 3.16× over BF16 sparse and dense FP8 | big 3.16× (ours) |
| 12 | Stem masks build 7.2–44× faster than Triton | chart_stem + bullets |
| 13 | Costs: a good mask, SM90, head_dim 128 | bullets |
| 14 | Summary | bullets + QR |
