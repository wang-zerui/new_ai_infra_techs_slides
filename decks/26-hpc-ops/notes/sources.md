# Sources for deck 26 (HPC-Ops container)

Downloaded copies live in `$SCRATCH/research/26/` (README cache in `$SCRATCH/research/cache/gh-hpc-ops-README.md`).

## Sources read

| Tag | Source | What was used |
|---|---|---|
| [R] | HPC-Ops README, main branch (identical to cached copy), https://github.com/Tencent/hpc-ops | "Why HPC-Ops?", "Updates / June 2026" (six operator write-ups), "Operator Catalog", "Performance" table, "Quick Start", "Roadmap" |
| [R-jan] | README at commit bea9bbc2cb (2026-01-26), the release-time version | "Maximum observed speedup per operator" table (prefill/decode columns) |
| [CL] | Repository development guide `CLAUDE.md` (main) | execution path `hpc/<op>.py -> torch.ops.hpc -> src/<op>/entry.cc -> HPC_ARCH_DISPATCH -> src/<op>/sm<arch>/*.cu`; one `_C_sm<arch>.abi3.so` per architecture |
| [MK] | `Makefile` (main) | `KNOWN_ARCHS=90 100 103` |
| [SRC] | `src/attention/decode/sm90/assign_task.cu`, `hpc/attention.py`, `src/attention/entry.cc`, `src/attention/decode/sched_task_info.h` | decode task map: 64-token tiles, per-CTA tile budget, 4 CTAs per SM for single-token decode, min work 512 tokens |
| [L] | LMSYS blog "HPC-Ops × SGLang" (Tencent Hunyuan AI Infra and the SGLang Team, 2026-08-07), https://www.lmsys.org/blog/2026-08-07-hpc-ops-sglang/ | Highlights; design sections; Tables 1, 2, 4, 5, 7, 8, 11; integration table (upstream PRs) |
| [MTP] | MarkTechPost, 2026-01-27, "Tencent Hunyuan Releases HPC-Ops" | service-level QPM gains (secondary source quoting Tencent's release) |

## Numbers on slides

| Slide | Number | Source location |
|---|---|---|
| 2, 13 | SM90 / H20 target; "powering large-scale production inference in Tencent" | [R] badges + "Why HPC-Ops?" |
| 2 | open-sourced 2026-01 (repo created 2026-01-20), MIT license | GitHub API `created_at`; [R] license badge |
| 3 | **2.95×**: static split-KV 0.186 ms vs dynamic 0.063 ms on 1×128K + 31×4K, FP8 KV decode, H20 | [L] Table 1, last row ("Dynamic vs. static 2.95×") |
| 3 | uniform 64×0.5K batch: 0.013 ms both, 1.00× | [L] Table 1, first row |
| 5 | insight: 1.00× (uniform) vs 2.95× (skewed) with identical attention math | [L] Table 1 rows 1 and 6 |
| 7 | 64-token KV tiles; per-CTA budget = total tiles / #persistent CTAs; combine by log-sum-exp | [L] "Scheduling around live KV work"; [SRC] `assign_task.cu` (`kTileN=64`, `num_tile_per_cta`) |
| 7 | block-sparse prefill: precomputed block mask, masked KV tiles skipped, FP8 | [R] "Sparse Attention" update; `hpc/attention.py` `attention_with_kvcache_blocksparse_prefill_fp8` docstring |
| 8 | MoE stages fused: routing, Gate-Up, act-quant, Down, top-k weighted reduce; gather removed; PDL | [R] "Fused MoE" update; [L] "A fused, latency-oriented MoE pipeline" |
| 9 | AllReduce + residual + RMSNorm fused; multicast (high-throughput) / Lamport P2P (low-latency) | [R] "Fused AllReduce + RMSNorm" update |
| 9 | sampler: penalty, temperature, top-k, top-p, softmax, sampling, mask update → two kernels | [R] "Sampler" update |
| 10 | W ≈ W_hi + W_lo/256; two BF16 Tensor-Core GEMMs in one kernel; FP32 accumulators | [R] "Route GEMM" update; [L] "A precision-aware BF16 formulation" |
| 10 | max abs error 0.00177 (HPC-Ops) vs 0.06464 (TF32 cuBLAS), FP32 cuBLAS as reference | [L] Highlights + "Router GEMM" paragraph |
| 11 | per-operator max observed speedups: sampler 8.5×, BF16×FP32 GEMM 3.22×, sparse attention FP8 3.16×, dynamic decode 2.88×, attention BF16 decode 2.22×, attention FP8 decode 2.0×, GroupGEMM FP8 decode 1.88×, AllReduce+RMSNorm 1.76×, fused FP8 MoE 1.6× (TP) | [R] "Performance" table, "Representative result" column |
| 12 | Hy3-FP8 TPOT, 8×H20, TP8, FP8 KV, 8K in / 4K out, SGLang default vs HPC-Ops (Attention + MoE): bs1 7.56→7.31 ms, bs4 11.10→9.42, bs8 14.29→10.76, bs16 22.90→13.09, bs32 35.33→18.09, bs64 40.70→23.81 | [L] Table 7 |
| 12 | TPOT −15.1% to −48.8% at batch 4–64; −3.3% at batch 1 | [L] Table 7 "Improvement" column |
| 13 | ~30% QPM gain for Tencent-HY (Hunyuan) models, ~17% for DeepSeek models, in Tencent production | [MTP] paragraph 2 and key-takeaway 3 (secondary; not in the repo README) |
| 13 | SGLang upstream: Attention (#30540, #32304), Router GEMM (#30247, #31943), MoE (#30541); in `lmsysorg/sglang:dev` images | [L] integration table + "Install" |
| 13 | H200 validation: Hy3-FP8 attention output throughput +3.7–5.9% vs FlashAttention | [L] Table 11 |
| 14 | MoE 0.98× at TP1/EP8 batch 512; H200 Hy3 MoE serving −4.2% to +6.3% | [L] Table 5 row 512; Table 11 |
| 14 | AllReduce+RMSNorm hidden sizes 4096/5120/7168 only | `benchmark/fuse_allreduce_rmsorm/README.md` (TORCH_CHECK quoted there) |
| 14 | head_dim 128 kernels (`*_dim128*` file names) | repo tree `src/attention/*/sm90/*dim128*` |
| 14 | Blackwell: build knows SM 90/100/103, only `src/gemm/sm100/gated_mla_gemm` exists for sm100; sm103 falls back to sm100 only; attention dispatch pairs list 90 | [MK]; `CMakeLists.txt` `HPC_ARCH_SOURCE_FALLBACK_103 100`; `src/attention/entry.cc` `HPC_ARCH_DISPATCH(... 90, ...)` |
| 14 (notes) | Router GEMM: below measured crossover points the FP32 path is faster, so SGLang dispatches by shape | [L] "A precision-aware BF16 formulation", last paragraph |
| 7 (notes) | 4 persistent CTAs per SM for one-token decode (3,3,2,2 for MTP 1-4) | `src/attention/decode/sched_task_info.h` `kCtaPerSmMap {9: {4,3,3,2,2}}` |
| 9 (notes) | sampler K1 = one vocab pass (penalty + temperature + online max/sum + per-block top-k), K2 = merge + softmax/top-p + Gumbel-max sample + penalty bit | `src/sampler/fused_sampler.cu` header comment |
| 4 (notes) | vocab 120,832 in the sampler benchmark | `benchmark/sampler/README.md` |
| 12 (notes) | TTFT −3.3% to −9.0% at batch 1–16 (8K input) | [L] Table 8 |
| 15 | summary repeats 2.95×, −15.1–48.8% TPOT, ~30%/~17% QPM | as above |

Derived numbers: none beyond the ratios already printed in [L] (2.95× is the blog's own ratio, 0.186/0.063).

## Discrepancies (Notion row vs primary sources)

1. **"upstreamed into vLLM / SGLang"** — the LMSYS blog documents upstream integration into **SGLang** main
   (Attention, Router GEMM, MoE). For vLLM the README only says the API is "designed for seamless integration"
   with vLLM and SGLang, and vLLM appears as a *benchmark baseline*; no vLLM upstreaming is documented. Slides say
   "integrated into SGLang main"; vLLM is not claimed.
2. **"about 30% QPM ... about 17% for DeepSeek-class models"** — not in any README version (checked all six
   README commits). The figures come from MarkTechPost's release article ("about 30 percent ... Tencent-HY
   models ... about 17 percent ... DeepSeek models on mainstream inference cards"). Kept on the adoption slide,
   labelled as Tencent-reported service-level numbers with the secondary source named.
3. **"15.1-48.8% lower end-to-end TPOT for Hy3-FP8 on 8x H20"** — matches [L] Table 7 for batch 4–64 (batch 1
   is only −3.3%); slides state the batch range.
4. **Scope** — Notion lists "normalization" as a hot path; in the source normalization appears as
   RMSNorm/RoPE/act-quant utilities plus the fused AllReduce+RMSNorm; no standalone normalization speedup is
   published. Not given its own slide.
5. **Dynamic decode headline** (affects child deck 07): README says "up to 2.88x" vs static split-k; the newer
   LMSYS Table 1 reports 2.95× (1×128K + 31×4K, FP8). Slide 11 uses the README table (2.88×) for consistency
   with the other README maxima; slides 3/5 use the LMSYS table (2.95×) and say so.
