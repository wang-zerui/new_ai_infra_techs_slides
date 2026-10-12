# Sources — deck 18: FP8 dispatch + BF16 combine low-precision communication

## Sources read

| Tag | Source | Where |
|---|---|---|
| [V3] | DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv 2412.19437 (cached `arxiv-2412.19437.txt`) | §3.1 (2048 H800), §3.2.1 (comp:comm ≈ 1:1, p.12), §3.2.2 (IB/NVLink path, 160 vs 50 GB/s, ≤ 4 nodes, 20 SMs, p.13-14), §3.3 (FP8 framework, < 0.25% loss error, p.15), §3.3.2 (1×128 activation tiles, p.16), §3.3.3 "Low-Precision Communication" (p.18), §3.5.1 (20 of 132 SMs, p.20), §4.2 (61 layers, h = 7168, 256 routed experts, top-8, p.22) |
| [V4] | DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence", arXiv 2606.19348 (cached) | §3.1 "Observations and Proposals" (C/B ≤ 2d = 6144 FLOPs/Byte, p.16); §4.2.1 model setups (V4-Pro: h = 7168, 384 routed experts, expert intermediate dim 3072, top-6; p.25) |
| [ISCA] | Zhao et al., "Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures", ISCA '25 Industry Track, arXiv 2505.09343 (downloaded to `$SCRATCH/research/18/`) | §2.3.2 (decode comm-time worked example, p.5), §3.2 "LogFMT: Communication Compression" (p.6), §3.2.1 limitations (p.7) |
| [RD-V1] | DeepEP README at commit 567632dd59 (2026-02-03, last V1 README) | "Performance" tables (normal and low-latency kernels, H800 + CX7, FP8 dispatching and BF16 combining); example code `dispatch_backward` / `combine_backward` comments |
| [RD-V2] | DeepEP README at commit b306af06af (2026-04-29, EPv2 release) | "Performance" table (8K tokens, h 7168, top-8, FP8 dispatching, BF16 combining) |
| [RD] | DeepEP README, current main (cached `gh-deepep-README.md`, V2.5) | intro ("including FP8 dispatch"); `dispatch_forward` docstring ("Supports both BF16 and FP8 (x as a tuple of [data, scale_factors])"); `dispatch_backward` / `combine_backward` docstrings; "Combine inputs must be BF16"; Experimental branches list (Hybrid-EP: "NVFP4 data type support") |
| [EP-PY] | `deep_ep/buffers/ep.py` (main, fetched 2026-10-13) | `EPBuffer.dispatch` docstring (FP8 mode = tuple of `float8_e4m3fn` data + scale factors; `use_tma_aligned_col_major_sf`); `EPBuffer.combine` docstring (inputs and outputs `torch.bfloat16`) |
| [EP-CPP] | `csrc/buffers/ep.hpp` (main) | line ~351 "SF must be FP32 or packed UE8M0x4"; buffer sizing `use_fp8_dispatch` elem size |
| [TOKEN] | `deep_ep/include/deep_ep/layout/ep/token.cuh` (main) | `TokenLayout`: hidden bytes + SF bytes + metadata (top-k idx int32, top-k weight float, source index) per token message |
| [RED] | `deep_ep/include/deep_ep/impls/ep/combine_utils.cuh` (main) | `combine_reduce`: FP32 (`float2`) accumulation, one `__float22bfloat162_rn` rounding; BF16 add bypass only for ≤ 2 contributions and no bias |
| [MATH] | `deep_ep/utils/math.py` (main) | `per_token_cast_to_fp8`: per-token 1×128 groups, E4M3, scale = amax/448 |
| [HE-DOC] | DeepEP `hybrid-ep` branch, `docs/README_Hybrid-EP.md` (first version 7febc6e256, 2026-02-09; current 17cfb817bc) | "Supported Data Types: BF16, FP8"; H100 table (4096 tokens, h 7168, top-8, 8 local experts, ranks 16/32/64, SM 4/8/16, "Kernel Only" Dispatch (FP8) / Dispatch (BF16) / Combine GB/s) |
| [HE-TEST] | `tests/test_hybrid_ep.py` on `hybrid-ep` at a48493600c (2026-01-20, the version current when [HE-DOC] was written) and at HEAD | `fp8_factor = (1 + 4 / 128) / 2` applied to FP8 dispatch bytes when computing GB/s |
| [HE-CFG] | `csrc/hybrid_ep/config.cuh` (`hybrid-ep` branch) | token types UINT8 (FP8) / UINT16 (BF16); comment "combine kernel (always uint16_t, 2 bytes per token)" |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 671B / 37B active, 61 layers, h = 7168, 256 routed experts, top-8 | DeepSeek-V3 config | [V3] abstract; §4.2 p.22 |
| 2048 H800 | V3 training cluster | [V3] §3.1 p.11 |
| 160 GB/s NVLink vs 50 GB/s IB (3.2×) | per-GPU bandwidths in V3's cluster | [V3] §3.2.2 p.13 |
| ≤ 4 nodes per token | node-limited routing | [V3] §3.2.2 p.13 |
| ≈ 1:1 | computation-to-communication ratio caused by cross-node EP in V3 training (pain) | [V3] §3.2.1 p.12 |
| 1×128 tiles (per token per 128 channels) | activation quantization granularity | [V3] §3.3.2 p.16; [MATH] |
| powers of 2 | dispatch-activation scale factors | [V3] §3.3.3 p.18 |
| FP32 or UE8M0 scales | DeepEP accepted SF formats | [EP-CPP] |
| 50% | dispatch volume cut vs BF16 | [ISCA] §3.2 p.6 ("reducing communication volume by 50% compared to BF16") |
| 6hd FLOPs, 3h bytes, C/B ≤ 2d = 6144 FLOPs/Byte, 6.1 TFLOP/s per GB/s | V4 balance point | [V4] §3.1 p.16 |
| V4-Pro h = 7168, d = 3072, 384 experts, top-6 | V4-Pro config | [V4] §4.2.1 p.25 |
| 50–100% | LogFMT encode/decode overhead when fused with all-to-all | [ISCA] §3.2.1 p.7 |
| LogFMT-8 > E4M3/E5M2; LogFMT-10 ≈ BF16 combine | LogFMT accuracy finding | [ISCA] §3.2 p.6 |
| < 0.25% | relative loss error of V3's FP8 training vs BF16 (whole FP8 framework, not comm alone) | [V3] §3.3 p.15 |
| 20 SMs (of 132) | V3 communication SM budget | [V3] §3.2.2 p.14, §3.5.1 p.20 |
| H100 Hybrid-EP kernel-only GB/s, 16 SMs: EP16 FP8 68.48 / BF16 71.71; EP32 55.35 / 57.69; EP64 46.52 / 49.97 | chart input | [HE-DOC] H100 table |

### Derived numbers (derivations)

- **Bytes per token-expert (h = 7168).** BF16 payload = 2h = 14,336 B. FP8 payload = h = 7,168 B plus scales:
  FP32 per 128 channels = 4·h/128 = 224 B (= h/32, +3.1% of h); packed UE8M0 = h/128 = 56 B (+0.8% of h).
  FP8 + FP32 scales = 7,392 B = 0.516 of BF16, identical to [HE-TEST] `fp8_factor = (1 + 4/128)/2`.
  Top-k index/weight metadata ([TOKEN]) is excluded on both sides.
- **Per pass (dispatch + combine).** BF16 both ways = 4h = 28,672 B. FP8 dispatch + BF16 combine = 3h + h/32 =
  3.03h = 21,728 B (FP32 scales) or 3.008h (UE8M0). Ratio 0.758 ⇒ "−24%" (−25% with V4's 3h approximation).
  Chart `charts/bytes_per_pair.py` plots these in units of h.
- **Combine share.** After the change combine carries 2h of 3.03h = 66% ("2/3") of the bytes per pass.
  Dispatch alone: 7,392 / 14,336 = 0.516, i.e. −48% ("about half"; [ISCA] says 50% ignoring scales).
- **Per training step.** Forward and backward each move one token→expert and one expert→token transfer per
  token-expert pair ([RD] backward helpers), so BF16 = 8h, ours = 6.06h; same 0.758 ratio.
- **BF16-both balance point.** With V4's own formula, V_comp/V_comm = 6hd / 4h = 1.5d = 4,608 FLOPs/Byte for
  d = 3072, i.e. 4.6 TFLOP/s of compute hidden per GB/s (vs 2d = 6,144 and 6.1 TFLOP/s for FP8/BF16).
  6hd = 132.1 MFLOP per token-expert for V4-Pro.
- **Hybrid-EP FP8 vs BF16 dispatch time (chart `charts/dispatch_time.py`).** [HE-TEST] reports FP8 GB/s with
  FP8 bytes = BF16 bytes × 0.516, so time_FP8 / time_BF16 = 0.516 × GB/s_BF16 / GB/s_FP8.
  Kernel-only, 16 SMs: EP16 0.516·71.71/68.48 = 0.540; EP32 0.516·57.69/55.35 = 0.537; EP64 0.516·49.97/46.52 = 0.554.
  Across 4/8/16 SMs the kernel-only ratio spans 0.537–0.704 (SM4: 0.673, 0.704, 0.660; SM8: 0.639, 0.548, 0.559).
  Assumption: the [HE-DOC] table was produced by [HE-TEST] (the doc and that test version are contemporaneous).
- **ISCA decode example (notes only).** (1 + 2) B × 32 tokens × 9 experts × 7,000 / 50 GB/s = 120.96 µs per layer
  ([ISCA] §2.3.2), TPOT bound 61 × 2 × 120.96 µs = 14.76 ms. With BF16 dispatch (2 + 2 B) the same formula gives
  161.28 µs and 19.68 ms (derived).

## Mechanism facts (no numbers) used on slides

- Dispatch quantizes the activation before the MoE up-projection to FP8 and dispatches it; compatible with FP8
  Fprop of the up-projection ([V3] §3.3.3).
- The activation gradient before the MoE down-projection is treated the same way (FP8 on the backward dispatch);
  forward and backward combine stay BF16 "to preserve training precision" ([V3] §3.3.3).
- "The backward pass of MoE dispatch is actually a combine"; "The backward pass of MoE combine is actually a
  dispatch" ([RD], [RD-V1] example code). Backward helpers reuse the forward `handle`; cached dispatch replays the
  saved layout without another CPU sync ([RD]).
- `combine_backward` accepts `Union[Tensor, Tuple[Tensor, Tensor]]` (BF16 or FP8 data + scales); `dispatch_backward`
  calls `combine` with BF16 gradients ([RD]).
- FP8 dispatch input is a tuple `(float8_e4m3fn data [tokens, hidden], scale factors)`; receiver can lay out scale
  factors TMA-aligned column-major for grouped GEMMs ([EP-PY]); SF must be FP32 or packed UE8M0×4 ([EP-CPP]).
- Combine inputs must be BF16 ([RD]); combine sums expert outputs with FP32 accumulation and one final BF16
  rounding ([RED]).
- Hierarchical path (V3/DeepEP normal kernels): dispatch sends one IB copy per target node to the same-index GPU,
  then forwards over NVLink; combine does "NVLink-to-IB forwarding and accumulation" then "IB receiving and
  accumulation" ([V3] §3.2.2).
- LogFMT tried and not adopted; FP8, E5M6 and mixed FP8-BF16 combine "actively testing" ([ISCA] §3.2).
- Hybrid-EP combine kernel is "always uint16_t, 2 bytes per token" ([HE-CFG]).

## Discrepancies

1. **"About 3h bytes per token-expert" (Notion Impact).** V4 counts 3h (h FP8 + 2h BF16) and ignores scale
   factors and routing metadata. With DeepEP's FP32 scales the payload is 3h + h/32 = 3.03h (3.008h with UE8M0),
   plus top-k indices/weights. Slides say "3h (+3% scales)".
2. **"The backward dispatch is exactly the forward combine" (Notion Core Mechanism).** The source states the
   relation per operator: the backward of dispatch is a combine, and the backward of combine is a dispatch
   ([RD]). The backward-direction precisions follow the direction (token→expert FP8, expert→token BF16; [V3] §3.3.3).
   Slides state both halves of the mirror.
3. **"The Hybrid-EP backend further supports NVFP4" (Notion).** The main README lists "NVFP4 data type support"
   for the Hybrid-EP branch, but that branch's own doc lists only BF16 and FP8, and its token types are
   UINT8/UINT16 with an always-BF16 combine ([HE-DOC], [HE-CFG], fetched 2026-10-13). Slides say "NVFP4 listed for
   the Hybrid-EP branch" (limits slide) and do not claim it ships.
4. **Notion headline treats 6144 FLOPs/Byte as DeepEP's result.** It is a hardware-design balance point stated
   in the V4 report's "Observations and Proposals" (§3.1) for V4-Pro's dimensions (d = 3072), not a measured
   speedup. Slides label it as V4's analytic threshold.
