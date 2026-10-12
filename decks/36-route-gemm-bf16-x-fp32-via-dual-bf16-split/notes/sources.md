# Sources — deck 36: Route GEMM (BF16 x FP32 via dual-BF16 split)

Repository state read: Tencent/hpc-ops `main` at HEAD 2a2e265 (2026-09-07). Local copies in
`$SCRATCH/research/36/` (src/, lmsys.txt, sglang_pr_*.md, hpcops_prs.md).

## Sources read

| Tag | Source | Where |
|---|---|---|
| [RM] | HPC-Ops README (cached `research/cache/gh-hpc-ops-README.md`), https://github.com/Tencent/hpc-ops | "Updates / June 2026 / Route GEMM" write-up; Operator Catalog row "GEMM BF16 x FP32"; Performance table row "BF16 x FP32 GEMM"; Quick Start requirements |
| [PR47] | hpc-ops PR #47 "Feature/gemmbf16xfp32" (merged 2026-06-09), https://github.com/Tencent/hpc-ops/pull/47 | PR description |
| [API] | `hpc/gemm.py` (`gemm_bf16xfp32` docstring, `get_gemm_bf16xfp32_workspace`) | docstring: scale = 1/256, w_high = w_fp32.to(bf16), w_low = ((w_fp32 − w_high)/scale).to(bf16) |
| [KER] | `src/gemm/sm90/gemm.cu` (kernel `gemm_bf16xfp32_kernel`, `launch_gemm_bf16xfp32_kernel`, `gemm_bf16xfp32_async`) | lines 87–407 (kernel), 411–475 (launch), 488–554 (config dispatch) |
| [ENT] | `src/gemm/entry.cc` (`select_config`, `normalized_m`, `select_split_k_by_work`, `select_tile16_wgn`, input checks) | lines 26–84, 92–104 |
| [TST] | `tests/test_gemm_bf16xfp32.py` | SM90 skip, shape sweep, tolerance rtol 0.08 / atol 0.01 |
| [BEN] | `benchmark/route_gemm/README.md` + `benchmark_gemm_bf16xfp32.py` | default sweep N=192, K=4096; providers; FP32/TF32 baselines time `torch.matmul(x.float(), w.t())` (line 127–129) |
| [LM] | LMSYS blog "HPC-Ops × SGLang" (Tencent Hunyuan AI Infra and SGLang team, 2026-08-07), https://www.lmsys.org/blog/2026-08-07-hpc-ops-sglang/ | Highlights; "Router GEMM: balancing routing precision and throughput"; Table 2; Table 3; Table 10; Table 11; fidelity paragraph |
| [SG1] | SGLang PR #30247 "Optimize LongCat-Flash router GEMM with the HPC-Ops bf16xfp32 kernel" (merged 2026-07-21) | description: H200 kernel table, GSM8K A/B, bench_one_batch_server, min-M guard; diff of `longcat_flash.py` (router shapes and min-M) |
| [SG2] | SGLang PR #31943 "Reject online weight updates while the HPC-Ops router GEMM split cache is active" (merged 2026-07-24) | description |
| [CL] | hpc-ops `CLAUDE.md` line 346 | "hpc::gemm_bf16xfp32 does not run on sm103: implemented for 90" |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| K=6144, N=768 (512 routed + 256 zero experts) | LongCat-Flash-Chat router shape | [SG1] diff `longcat_flash.py` comment + test config |
| K=3072, N=384 (256 routed + 128 zero experts) | LongCat-Flash-Lite router shape | [SG1] same |
| 403.728 µs / 205.760 µs | FP32 cuBLAS / TF32 cuBLAS latency, M=4096, K=4096, N=192, H20 | [LM] Table 2, row M=4096 |
| **1.96×** | FP32-cuBLAS time ÷ TF32-cuBLAS time at M=4096 | **derived**: 403.728 / 205.760 = 1.962 ([LM] Table 2) |
| 24 / 11 / 8 significant bits | significand precision of FP32 / TF32 / BF16 (23 / 10 / 7 stored mantissa bits + implicit 1) | IEEE-754 binary32 and the BF16/TF32 format definitions (standard; analytic) |
| 1/256 = 2^-8 | fixed scale for w_lo | [RM] Route GEMM write-up; [API] docstring; [PR47] |
| w ≈ w_hi + w_lo/256 | decomposition | [LM] Router GEMM section; [API] |
| ~16 kept bits; error ≤ 2^-16·|w| | analytic bound of the hi/lo split: |w − w_hi| ≤ 2^-8|w| and the BF16 rounding of 256(w − w_hi) loses ≤ 2^-8 of that, so ≤ 2^-16|w| (2^-17 with round-to-nearest for both halves). Verified by simulation (2M N(0,1) weights): max rel. error BF16 3.89e-3 (≤2^-8), TF32 4.88e-4 (≤2^-11), split 7.6e-6 (RNE hi) / 1.52e-5 (truncated hi), both ≤ 2^-16 = 1.53e-5 | **analytic / derived**; script `notes/split_sim.py` (run: `uv run --with numpy python notes/split_sim.py`) |
| Worked example w = 0.1234567 → w_hi = 0.1235352, 256·(w − w_hi) = −0.02008 → w_lo = −0.02014, w_hi + w_lo/256 = 0.1234565 (error 2.2e-7); TF32(w) = 0.1234741 (error 1.7e-5) | concrete split of one value with round-to-nearest BF16 (as `torch.Tensor.to(torch.bfloat16)` in [API]) | **derived** by `split_sim.py` (numpy bit emulation) |
| bits_kept chart: 8 / 11 / 16 / 24 | significant bits of w kept by BF16 / TF32 / hi+lo split / FP32 | **analytic** (rows above) |
| 1.30–3.22× vs FP32 cuBLAS; 1.25–1.78× vs TF32 cuBLAS | speedup range over the measured M values, K=4096, N=192, H20 | [LM] Router GEMM operator-benchmark paragraph + Table 2 |
| Table 2 latencies (µs): M=1: 11.200 / 14.576 / 14.048; M=16: 11.744 / 23.808 / 18.752; M=48: 12.144 / 31.008 / 20.064; M=96: 13.904 / 31.760 / 24.720; M=208: 17.088 / 39.280 / 28.928; M=512: 26.992 / 86.976 / 44.736; M=1024: 50.640 / 110.480 / 68.544; M=2048: 76.688 / 198.576 / 100.800; M=4096: 141.120 / 403.728 / 205.760 (HPC-Ops / FP32 cuBLAS / TF32 cuBLAS) | latency chart | [LM] Table 2 |
| 3.22× at M=512 | peak speedup vs FP32 (also the README "up to 3.22×") | [LM] Table 2 row M=512; [RM] Performance table |
| 0.00177 / 0.06464 | max abs error vs FP32 cuBLAS reference: HPC-Ops / TF32 cuBLAS (same sweep) | [LM] Highlights + operator-benchmark paragraph |
| **36×** | TF32 error ÷ HPC-Ops error | **derived**: 0.06464 / 0.00177 = 36.5 |
| 1.06–2.83× (Chat), 1.09–2.46× (Lite) | LongCat router-shape kernel speedup over SGLang default within SGLang dispatch ranges, H20 | [LM] Table 3 |
| +5.5–6.1% (bs 4–64), +0.5% (bs 1) | LongCat-Flash-Lite-FP8 input throughput with HPC-Ops Router GEMM only, 8×H20, 1,024-token input / 128-token output; per batch 0.5, 6.1, 5.6, 5.7, 5.5, 5.7% for bs 1, 4, 8, 16, 32, 64 | [LM] Table 10 |
| 1.56–4.31× | H200 kernel speedup on LongCat router shapes vs current FP32 path | [SG1] Performance table; [LM] Table 11 |
| +2.8–5.4% | H200 LongCat-Flash-Lite input throughput (bs 1–64, prefill) | [SG1] bench_one_batch_server; [LM] Table 11 |
| GSM8K 0.798 → 0.802 | LongCat-Flash-Lite-FP8 on H200 TP1, main vs PR | [SG1] Model-level validation |
| min-M 64 (Chat) / 128 (Lite) | SGLang dispatch guard below which the cuBLAS FP32 path is kept (benchmarked on H200) | [SG1] diff `_LONGCAT_FLASH_ROUTER_HPC_GEMM_MIN_M` + description |
| split-K 8 / 4 / 2 / 1 for normalized M ≤ 64 / ≤ 144 / ≤ 304 / larger (16-row tiles); 64-row tiles above normalized M 624 | kernel config picker | [ENT] lines 34–84 |
| WGMMA atom 64×16×16 (16-token tiles) / 64×64×16; tile N = 64 per warpgroup; K tile 128 (16-row) / 64 (64-row); 3–5 pipeline stages | kernel config | [KER] lines 448–450, 509–546 |
| producer warpgroup 24 regs / math warpgroups 168 regs | warp specialization | [KER] lines 166, 226 |
| N % 64 == 0; BF16 inputs; SM90 | input checks / arch | [ENT] lines 96–104, 142–146; [TST]; [CL] |

## Mechanism facts (no numbers) used on slides

- Two accumulators `tYr_low`, `tYr_high` in registers; per K step: wait X tile, WGMMA with W_low tile → acc_low,
  then WGMMA with W_high tile → acc_high, both reading the same shared-memory X tile; epilogue
  `y = acc_low * scale + acc_high`, convert, one TMA store ([KER] lines 242–364).
- The MMA is computed with the weight as operand A and the activations as operand B (`partition_A(sW)`,
  `partition_B(sX)`, output tile (kTileN, kTileM)), i.e. Yᵀ = W·Xᵀ: "swap-AB" so a 16-token tile maps to the
  16-wide side of a 64×16 WGMMA ([KER] lines 236–243, 448–450).
- Split-K partials go to an FP32 workspace; per-tile `split_flag` counters (atomicAdd) gate an in-kernel
  reduction pass; counters reset to zero afterwards; persistent grid = min(SM count, tiles) ([KER] lines
  340–405, 468–469; [API] workspace; [TST] asserts flags are zero after the call).
- SGLang caches (w_high, w_low, split flag) on the FP32 weight at load and reuses them across requests and
  CUDA-graph replays; shape-aware dispatch keeps the default path below measured crossovers ([LM]; [SG1]).
- Online weight updates (`update_weights_from_disk/tensor/distributed/ipc`) are rejected while the split
  cache is enabled ([SG2]).
- Baselines: the FP32 and TF32 cuBLAS providers time `torch.matmul(x.float(), w.t())`, i.e. including the
  BF16→FP32 cast of x ([BEN] lines 126–129), matching SGLang's default `x.float() @ w.t()` path ([SG1]).
- The README calls the result "FP32-level accuracy" ([RM]); the blog says "precision close to a full FP32
  GEMM" ([LM]). The slides use "close to FP32" and give the analytic 16-bit bound.

## Discrepancies

1. **Notion "Up to 3.22x over cuBLAS FP32/TF32"** — the 3.22× is over FP32 cuBLAS only (M=512); over TF32 the
   maximum is 1.78× ([LM] Table 2; [RM] lists both baselines but one number). Slides say "vs FP32 cuBLAS".
2. **Notion / README "keeping FP32 accuracy"** — the split keeps ~16 of FP32's 24 significand bits; the measured
   max abs error vs FP32 cuBLAS is 0.00177, not zero ([LM]). Slides say "close to FP32".
3. **"truncation" vs rounding of w_high** — the LMSYS blog says W_high is formed "by direct truncation"; the
   HPC-Ops API docstring, test, benchmark and SGLang integration all use `w.to(torch.bfloat16)`, which rounds
   to nearest-even. Both give an error ≤ 2^-16·|w| (simulation above). Slides follow the code (round to BF16).
4. **Notion body links** — "FP8 formats for deep learning" and the Megatron-Core MoE paper do not discuss this
   kernel; they were not used.
5. Notion "shared input gathering" — the kernel shares the *X tile load* (one TMA load per K step consumed by
   both WGMMAs); there is no gather. Slides say "loads each x tile once".
