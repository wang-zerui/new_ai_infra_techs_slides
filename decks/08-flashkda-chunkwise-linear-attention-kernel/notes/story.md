# Story: FlashKDA (chunkwise KDA kernel)

Deck 08 of the KDA chain: 05 model-level attention -> **08 FlashKDA kernel** -> 69 KDA Context Parallelism.
Deck 05 owns the math (state update, chunkwise form, bounded decay); this deck owns how one GPU executes it;
deck 69 owns the cross-GPU split. Recap the math in one line only, then cite deck 05.

- **Where it lives** (prosperity): Kimi K3's 69 KDA layers; FlashKDA is the CUTLASS chunkwise kernel K3 uses for
  training and prefill, auto-dispatched from flash-linear-attention's `chunk_kda` (K3 §5.1.1; FlashKDA README).
- **Pain number** (betrayal): the Triton reference `chunk_kda` forward takes **4.84 ms** for one 8K-token sequence
  with 96 heads (D = 128) on H20, about 1.5x the scalar-gated GDN kernel (3.20 ms) (FlashKDA BENCHMARK_H20.md).
  KDA's richer per-channel gate looks expensive.
- **Failed suitors**:
  1. Naive chunkwise execution: intra-chunk work and the serial state hand-off alternate; SMs idle during the
     hand-off (K3 §5.1.1).
  2. One fused kernel: the token-parallel work is throttled by the recurrence's low parallelism (one CTA per
     sequence x head); splitting gave at least 15% (FlashKDA deep dive §2).
  3. 64-token chunks (FLA default): need secondary sub-chunking / log-space rescaling and a 64 x 64 inverse
     (deep dive §1; K3 §2.1.1).
- **Villain**: two kinds of parallelism forced into one schedule. Per chunk, almost everything (decayed q/k,
  the 16 x 16 inverse, the masked q k matrix) depends only on the chunk's own tokens; only a few small matmuls
  need the incoming state. A schedule that interleaves them runs everything at the recurrence's parallelism.
- **Turn / insight**: only the d_k x d_v state crosses chunk boundaries, so all state-independent work can be
  prepared for every chunk up front, leaving a thin serial chain.
- **Mechanisms**:
  1. ❶ Split by parallelism: K1 token-parallel "prepare" (grid tiles x heads), K2 head-parallel "recurrence"
     (grid sequences x heads); each tuned independently (>= 15% over the fused prototype).
  2. ❷ Warp-specialized recurrence: 1 TMA load warp (3-stage pipeline), 4 MMA warps carrying the state,
     1 store warp (2-stage) so loads and stores overlap the serial chain; state stays on chip in BF16 with FP32 FMA.
  3. ❸ 16-token chunks: K3's bounded decay (g >= -5, deck 05) keeps exp(cumsum g) in BF16 range, the 16 x 16
     inverse is exact forward substitution (two 8 x 8 FP32 blocks + two BF16 HMMA merges), SM80 MMA only.
  Plus: long prefill splits one sequence across the SMs of a GPU (segment transitions composed exactly), the
  intra-device twin of KCP (deck 69).
- **Release**: same currency: **2.62 ms vs 4.84 ms (1.85x) on H20, 1.01 vs 2.33 ms (2.31x) on GB200**, up to
  3.27x on variable-length batches, and faster than the scalar-gated GDN Triton kernel (1.22x H20, 1.27x GB200).
- **Limits**: SM90+, CUDA 12.9+, K = V = 128 only, lower bound in [-5, 0] required; public v1 is a forward kernel
  called under inference mode (K3 says training also uses FlashKDA); H = 64 fixed-length on GB200 is 0.96x of GDN;
  decoding needs a separate fused kernel (K3 §5.4.2); only forward benchmarks at T = 8K are public.

## Slide plan (final, 14 logical slides incl. title)

| # | Headline |
|---|---|
| 1 | Title |
| 2 | Every KDA layer in K3 runs through FlashKDA |
| 3 | The reference KDA kernel trails simpler GDN |
| 4 | Serial state hand-offs leave SMs idle |
| 5 | Insight: only the state has to wait |
| 6 | FlashKDA: prepare in parallel, then recur |
| 7 | ❶ Split the kernel along its parallelism |
| 8 | ❷ Warps overlap memory with the state chain |
| 9 | ❸ 16-token chunks make the inverse cheap |
| 10 | Long prefill also splits one sequence across SMs |
| 11 | 1.85-2.31× faster than the Triton kernel |
| 12 | Variable-length batches gain up to 3.27× |
| 13 | Costs: Hopper+, fixed head size, forward only |
| 14 | Summary |
