# Sources: deck 08, FlashKDA

## Sources read

| Key | Source | Where | What was used |
|---|---|---|---|
| K3 | Kimi Team, "Kimi K3: Open Frontier Intelligence", arXiv 2607.24653 (https://arxiv.org/abs/2607.24653); cached text `$CACHE/arxiv-2607.24653.txt` | §2.1.1 (chunkwise form Eq. 3-4, lower-bounded decay Eq. 5); Table 1; §5.1 intro; §5.1.1 (FlashKDA, intra-device CP); §5.4.2 (KDA decoding); ref. [14] | Role of FlashKDA, the alternating-phase problem, intra-device SM-level CP, separate decode kernel |
| README | https://github.com/MoonshotAI/FlashKDA README (via `gh api`, saved in `$SCRATCH/research/08/`) | Requirements; FLA backend; Kernel API | SM90+, CUDA 12.9+, K = V = 128, lower_bound in [-5, 0], cu_seqlens varlen, bf16/fp32 state, forward API, `FLA_FLASH_KDA=0` opt-out, inference_mode dispatch |
| BENCH-H20 | FlashKDA `BENCHMARK_H20.md` (generated 2026-04-22) | tables "T=8192, H=96, D=128" and "T=8192, H=64, D=128" | forward times and speed-ups on H20 |
| BENCH-GB200 | FlashKDA `BENCHMARK_GB200.md` (generated 2026-05-26) | same two tables | forward times and speed-ups on GB200 |
| BLOG | FlashKDA `docs/20260420-flashkda-v1-deep-dive.md` | §1 chunk size; §2 kernel fusion strategy; §3 numerical precision; §4 other optimizations | CHUNK = 16 vs FLA 64, K1/K2 split and >= 15%, bf16 state with fp32 FMA, 8x8 forward substitution + HMMA merge, ex2/tanh tricks, __launch_bounds__(256, 8), MOVM_T |
| SRC | FlashKDA `csrc/smxx/fwd_launch.cu`, `fwd_kernel1.cuh`, `fwd_kernel2.cuh` (fetched 2026-10-13) | launch config; kernel-2 warp roles and phase comments | grid_k1(total_tiles, H) with 256 threads; grid_k2(N, H) with 192 threads = 4 MMA warps + 1 load + 1 store; kInputStages = 3, kOutputStages = 2; per-chunk phases (k@S, q@S; u = (v - kS) beta; U = INV@u; out += Mqk@U; S = S*g_total + k_restored^T U); workspace tensors k_decayed, q_decayed, k_restored, g_total, INV, Mqk |
| PR852 | fla-org/flash-linear-attention PR #852 "[KDA] Support FLASHKDA backend" (merged 2026-04-21) | metadata only | FLA integration exists |
| KL | Kimi Linear, arXiv 2510.26692 | §6.2, §6.3 | FLA chunk size 64, secondary chunking background |

## Numbers on slides

| Slide | Number | Source location | Note |
|---|---|---|---|
| 2 | 69 KDA layers | K3 Table 1 | |
| 3 | 4.84 ms (4.8388) Triton chunk_kda; 3.20 ms (3.1985) Triton GDN | BENCH-H20, T=8192, H=96, D=128, row "Fixed" | |
| 3 | about 1.5x | derived: 4.8388 / 3.1985 = 1.51 | slide says "derived" |
| 6 | 16x16 inverse; load warp 3 stages; 4 MMA warps; store warp 2 stages | BLOG §1, §3; SRC fwd_launch.cu (kInputStages = 3, kOutputStages = 2, kK2Threads = 32*2 + 128), fwd_kernel2.cuh (kComputeThreads = 128, warp roles) | |
| 7 | 512 tiles of 16 tokens; 49,152 CTAs (K1); 96 CTAs (K2) | derived from SRC grid definitions for the benchmark shape: 8192 / 16 = 512; 512 x 96 = 49,152; N x H = 1 x 96 = 96 | counts, labelled as an example shape |
| 7 | at least 15% end-to-end from splitting into two kernels | BLOG §2 | |
| 8 | 3 stages in flight; per-chunk chain k S, q S, U = INV beta (v - k S), o = q S + Mqk U, S <- gamma S + k^T U | SRC fwd_kernel2.cuh phases 1-6 ("s_acc = s_acc * g_total + k_restored_t @ U") | gamma = exp(g_total), the chunk's cumulative decay |
| 9 | 64-token chunks (FLA) vs 16-token (FlashKDA); 16-token sub-chunks; 64x64 inverse | BLOG §1; K3 §2.1.1 ("secondary 16-token tiles") | |
| 9 | g >= -5 keeps exp(cumsum g) in BF16 over 16 tokens | BLOG §1; K3 §2.1.1 Eq. 5 | |
| 9 | two 8x8 FP32 forward substitutions, two BF16 HMMA merge GEMMs | BLOG §3 | off-diagonal block = -C^-1 B A^-1 is standard block-triangular algebra |
| 9 (notes) | 8 K1 blocks per SM | BLOG §4 (`__launch_bounds__(256, 8)`) | |
| 11 | H20: 2.6220 vs 4.8388 ms (1.85x); GB200: 1.0087 vs 2.3271 ms (2.31x) | BENCH-H20 / BENCH-GB200, H=96, Fixed | |
| 11 | GDN 3.1985 ms (H20), 1.2792 ms (GB200); FlashKDA vs GDN 1.22x (H20), 1.27x (GB200) | same rows, column "Speedup vs gdn" | |
| 12 | H20: 1.85x, 2.06x, 2.29x; GB200: 2.31x, 2.71x, 3.27x | BENCH tables H=96: Fixed; Varlen [1300, 547, 2048, 963, 271, 3063]; Varlen 1024 x 8 | |
| 12 | Triton time ~flat: H20 4.8388 / 4.8291 / 4.6723 ms; GB200 2.3271 / 2.3340 / 2.3105 ms | same tables | |
| 12 | 64 heads on GB200: 1.70x to 3.21x | BENCH-GB200, H=64 table | H20 H=64: 1.95x, 1.91x, 2.31x (notes only) |
| 13 | SM90+, CUDA 12.9+, K = V = 128, lower bound in [-5, 0] | README Requirements and Kernel API | |
| 13 | 0.96x of GDN (H=64, fixed, GB200) | BENCH-GB200, H=64, Fixed, "Speedup vs gdn" | |
| 14 | 4.84 ms, 1.5x, 1.85x, 3.27x | as above | |

## Discrepancies

1. **Notion "shared by training and prefill".** The K3 report (§5.1.1) says FlashKDA serves training and inference
   prefill, but the public FlashKDA v1 exposes only a forward kernel (`flash_kda.fwd`), and FLA dispatches it under
   `torch.inference_mode()` (README). No backward kernel is public. The deck states both.
2. **Notion body links the Kimi-Linear repo for "FlashKDA kernels".** FlashKDA lives in its own repository,
   github.com/MoonshotAI/FlashKDA (K3 ref. [14]); the Kimi-Linear README points to FLA's Triton KDA kernels
   (`fla/ops/kda`).
3. **"Overlaps intra-chunk computation with cross-chunk state propagation" (Notion, from K3 §5.1.1).** In the public
   v1 code the two kernels are launched one after the other on the same stream (fwd_launch.cu); the overlap comes from
   (a) hoisting all state-independent intra-chunk work into the token-parallel K1 and (b) warp specialization inside
   K2, where TMA load/store warps overlap memory traffic with the MMA warps' serial state chain. The deck describes it
   that way rather than as two concurrently running phases.
4. **Notion "Impact": "lets linear attention actually saturate the hardware on long training sequences".** No
   utilization or long-sequence training numbers are published; the only public measurements are forward-pass
   timings at T = 8K (H20, GB200). K3 itself only says FlashKDA "substantially outperforms the Triton reference".
