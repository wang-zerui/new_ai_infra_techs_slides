# Story — deck 18: FP8 dispatch + BF16 combine low-precision communication

- **Where it lives** (prosperity): DeepSeek-V3 training (671B, 256 experts top-8, h = 7168, 2048 H800,
  IB 50 GB/s vs NVLink 160 GB/s) and DeepSeek-V4-Pro (1.6T, 384 experts top-6, d = 3072); the default precision
  pair in every DeepEP benchmark (V1, V2) and in the EPBuffer API.
- **Pain number** (betrayal): cross-node EP gives DeepSeek-V3 training a computation-to-communication ratio of
  **≈ 1:1** ([V3] §3.2.1).
- **Failed suitors**:
  1. BF16 both ways: 2h bytes each direction = 4h per token-expert per pass; the receiver then casts the dispatch
     payload to FP8 for its GEMM anyway — half the bytes on the wire carry bits nobody uses.
  2. Lower precision both ways (FP8 / LogFMT combine): combine sums k expert outputs into the residual stream,
     "accuracy requirements" keep it BF16; LogFMT-10 matches BF16 but its fused codec costs 50–100% ([ISCA] §3.2).
- **Villain**: communication precision chosen per tensor ("activations are BF16") instead of per consumer —
  the wire carries the producer's dtype, not the receiver's.
- **Turn / insight**: the dispatch payload's only consumer is an FP8 GEMM (up-projection Fprop forward,
  down-projection Dgrad backward), so quantizing before the wire loses nothing the receiver would have kept;
  the combine payload is summed, so it stays BF16.
- **Mechanisms**:
  ❶ Quantize-then-send: per-token 1×128 E4M3 tiles with inline FP32/UE8M0 scales, laid out for the grouped GEMM
     (answers: dispatch bytes).
  ❷ BF16 combine with FP32 accumulation and a two-level (NVLink, then IB) sum (answers: precision of the sum).
  ❸ Backward mirror: d(dispatch) = combine, d(combine) = dispatch, same handle; precision follows direction
     (answers: what to do in backward).
- **Release**: 3h instead of 4h bytes per token-expert pass (−24% with FP32 scales); measured Hybrid-EP FP8
  dispatch takes 54–55% of BF16 dispatch time at 16 SMs on 16–64 H100s (derived from GB/s); V4's balance point
  C/B ≤ 2d = 6144 FLOPs/Byte (4608 if BF16 both ways, derived).
- **Limits**: a quantize kernel before every dispatch; scales +3% (FP32) of h; combine still carries 2/3 of the
  bytes; FP8 combine/LogFMT unresolved (codec overhead); NVFP4 only listed for an experimental branch; needs an
  FP8 GEMM consumer (Hopper+) and an FP8 training recipe (V3: < 0.25% loss gap for the whole FP8 framework).

## Slide plan (14 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Every MoE token crosses the network twice per layer | prosperity | fig_layer (SVG) |
| 3 | Cross-node EP makes V3 training communication-bound | pain | `%% big: ≈ 1:1` |
| 4 | BF16 dispatch ships bits the receiver discards | villain / suitor 1 | fig_before (SVG) |
| 5 | Insight: dispatch feeds an FP8 GEMM anyway | turn | bullets + box |
| 6 | Precision follows the direction of traffic | overview | fig_overview (SVG) |
| 7 | ❶ Quantize per 1×128 tile, ship scales inline | mechanism | fig_message (SVG) |
| 8 | ❷ Combine stays BF16 and sums in FP32 | mechanism | fig_combine (SVG) |
| 9 | ❸ Backward reuses both kernels, mirrored | mechanism | fig_mirror (SVG) |
| 10 | FP8 dispatch moves 3h instead of 4h bytes | release (analytic) | chart bytes_per_pair |
| 11 | FP8 dispatch takes ~55% of BF16 time | release (measured) | chart dispatch_time |
| 12 | V4 hides EP traffic below 6144 FLOPs/Byte | release (balance point) | `%% big: 6144` |
| 13 | The cost: a cast, scales, and a BF16 combine | limits | two-col |
| 14 | Summary | | QR |
