# Sources — deck 14: wave-based fine-grained communication/compute overlap

## Sources read

| Tag | Source | Where |
|---|---|---|
| [V4] | DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence", arXiv 2606.19348v1 (26 Apr 2026). https://arxiv.org/abs/2606.19348 (cached text `arxiv-2606.19348.txt`) | Abstract p.1; §1 p.4 (infra list); §2.1 p.7 (DeepSeekMoE, routing); §3.1 p.15-16 incl. Fig. 5 (the technique); §4.2.1 p.24-25 (model configs) |
| [V3] | DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv 2412.19437 (cached) | §3.2.1 p.12-13 (comp:comm ≈ 1:1, DualPipe); Table 2 p.13 (DualPipe 2× parameters); §3.2.2 p.14 (20 SMs for communication); §3.4.1 p.19 and §3.4.2 p.20 (two-micro-batch overlap in prefill/decoding) |
| [PR304] | DeepGEMM PR #304 "[Public release 26/04] Introducing Mega MoE ...", merged 2026-04-17, merge commit 7f2a703. https://github.com/deepseek-ai/DeepGEMM/pull/304 (fetched with `gh api`) | PR body |
| [PR316] | DeepGEMM PR #316 "Add various optimizations and Mega MoE benchmarks", merged 2026-04-24. https://github.com/deepseek-ai/DeepGEMM/pull/316 | benchmark tables (EP8) |
| [DG-README] | DeepGEMM README (cached `gh-deepgemm-README.md`) | News; "Mega MoE" section; utilities |
| [DG-SRC] | DeepGEMM source at commit 7f2a703 (the Mega MoE release): `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`, `csrc/jit_kernels/heuristics/mega_moe.hpp`, `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`, `tests/test_mega_moe.py` (downloaded to `$SCRATCH/research/14/src_v0416/`) | see "Mechanism facts" |
| [COMET] | Zhang et al., "Comet: Fine-grained computation-communication overlapping for mixture-of-experts", arXiv 2502.19811 | only as characterized in [V4] Fig. 5 caption (not read in full) |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 1.6T total / 49B activated | DeepSeek-V4-Pro | [V4] abstract p.1; §4.2.1 p.25 |
| 284B total / 13B activated | DeepSeek-V4-Flash | [V4] abstract p.1; §4.2.1 p.24 |
| 384 routed experts, top-6, expert intermediate dim 3072, hidden 7168 | V4-Pro MoE config | [V4] §4.2.1 p.25; same values in [PR316] |
| 256 routed experts, top-6, expert intermediate dim 2048, hidden 4096 | V4-Flash MoE config | [V4] §4.2.1 p.24; same values in [PR316] |
| 32 experts per rank | V4-Flash under EP8: 256 / 8 (derived, simple division) | [V4] §4.2.1 + [PR316] "EP8" |
| ≈ 1 : 1 | DeepSeek-V3 computation-to-communication ratio with cross-node EP | [V3] §3.2.1 p.12 ("inefficient computation-to-communication ratio of approximately 1:1") |
| 2× parameters | DualPipe keeps two copies of the model parameters | [V3] Table 2 and p.13 text |
| 20 of 132 SMs | SMs V3 dedicated to all-to-all communication on H800 | [V3] §3.2.2 p.14 and §3.5.1 p.20 (only in speaker notes) |
| 1.42× | theoretical speedup of Comet-style per-stage overlap, V4-Flash configuration | [V4] Fig. 5(b) p.15 |
| 1.92× | theoretical speedup of the wave scheme, V4-Flash configuration | [V4] Fig. 5(c) p.15 |
| 1.50–1.73× | measured speedup over strong non-fused baselines, general inference workloads (NVIDIA GPUs and Ascend NPUs) | [V4] §3.1 p.16 |
| up to 1.96× | measured speedup for latency-sensitive cases (RL rollout, high-speed agent serving) | [V4] §3.1 p.16 |
| Flash: 1.96× / 1.73× / 1.56× / 1.62× at 1 / 512 / 8192 / 32768 tokens per rank | Mega MoE vs legacy (non-fused) path, EP8, averaged over 8 ranks | [PR316] V4-Flash table |
| Pro: 1.61× / 1.54× / 1.50× / 1.54× at 1 / 512 / 8192 / 32768 tokens per rank | same, V4-Pro | [PR316] V4-Pro table |
| 6hd FLOPs, 3h bytes per token-expert pair | SwiGLU gate+up+down FLOPs; FP8 dispatch (h bytes) + BF16 combine (2h bytes) | [V4] §3.1 p.16 |
| C/B ≤ V_comp/V_comm; C/B ≤ 2d = 6144 FLOPs/Byte | condition for full hiding; value for V4-Pro (d = 3072 = expert intermediate dim) | [V4] §3.1 p.16 |
| 6.1 TFLOP/s per GB/s | each GB/s of interconnect hides 6.1 TFLOP/s of compute (V4-Pro) | [V4] §3.1 p.16 |
| 4096 FLOPs/Byte | balance point for V4-Flash: 2d with d = 2048 (**derived**: [V4] formula applied to the Flash config in §4.2.1) | derived |
| 4hd : 2hd (L1 : L2 FLOPs) and h : 2h (dispatch : combine bytes) | split of the paper's 6hd and 3h: Linear-1 = gate + up projections (2 × 2hd), Linear-2 = down projection (2hd); FP8 = 1 byte/elem, BF16 = 2 bytes/elem (**derived** from [V4] §3.1 p.16) | derived |
| ≤ 32 experts per wave; ×2 imbalance factor | wave-size heuristic | [DG-SRC] heuristics `get_num_experts_per_wave_for_mega_moe` |
| 128 / 128 / 256 threads | dispatch / TMA+MMA / epilogue threads per CTA | [DG-SRC] heuristics (`num_dispatch_threads = 128`, `num_non_epilogue_threads = 128`, `num_epilogue_threads = 256`) |
| PyTorch ≥ 2.9; FP8×FP4 only at release | Mega MoE prerequisites (April release) | [PR304]; README now also lists FP8×FP8 [DG-README] |

## Derived and analytic content (stated as such on slides)

1. **Balance-point chart (`charts/balance.py`, analytic).** Under perfect overlap, MoE-layer time is
   max(V_comp/C, V_comm/B); serially it is V_comp/C + V_comm/B. Normalising by the compute time V_comp/C and
   writing x = B / (C·V_comm/V_comp) (interconnect bandwidth relative to the balance point of [V4] §3.1),
   overlapped time = max(1, 1/x) and serial time = 1 + 1/x. No measured data are plotted. The knee x = 1 is
   B = C/6144 for V4-Pro ([V4] §3.1).
2. **Schematic timelines (`figures/gen_overlap.py`).** Stage durations are schematic, not measurements. They
   are chosen to be consistent with three published facts: (i) total communication < total computation in a
   V4 MoE layer ([V4] §3.1 "Communication Latency Can Be Hidden"); (ii) L1 has 2× the FLOPs of L2 and the
   combine moves 2× the bytes of the dispatch (derived split above); (iii) the paper's theoretical ratios
   serial : per-stage : ideal waves = 1.92 : 1.92/1.42 : 1. Solving (iii) with stage times D, L1, A, L2, C under
   serial = D+L1+A+L2+C, per-stage = max(D,L1)+A+max(L2,C), ideal waves = L1+A+L2 gives (units, compute = 100):
   D ≈ 30.7, L1 ≈ 52.2, A ≈ 21.7, L2 ≈ 26.1, C ≈ 61.3. The waves panel draws 4 waves, so pipeline fill (first
   wave's dispatch) and drain (last wave's combine) remain visible; it is therefore longer than the ideal.
   This model is our reconstruction, not the paper's; slides call the figure "schematic".
3. **"Small dispatch pairs with big L1, big combine with small L2"** (slide 5 callout): follows from the
   derived 4hd:2hd and h:2h split above; it is our explanation of why per-stage overlap leaves slack, not a
   sentence from the paper.

## Mechanism facts (no numbers) used on slides

- Four stages per MoE layer: Dispatch and Combine (communication-bound), Linear-1 and Linear-2
  (computation-bound); Fig. 5 also shows "SwiGLU + FP8 cast" between the GEMMs ([V4] §3.1, Fig. 5).
- Waves: each wave holds a small portion of the experts; computation of a wave starts as soon as all its
  experts' communication has completed, without waiting for other experts; in steady state the current
  wave's computation, the next wave's token transfer and the result sending of completed experts run
  concurrently ([V4] §3.1 p.15). Benefits long-tail small batches such as RL rollout ([V4] p.16).
- Comet overlaps Dispatch with Linear-1 and Linear-2 with Combine, separately ([V4] Fig. 5 caption).
- Fused into a single pipelined kernel ([V4] §3.1 p.15); open-sourced as Mega MoE in DeepGEMM ([V4] p.16,
  footnote 2 → PR #304).
- Pull-based dispatch: each GPU reads activations from remote GPUs, avoiding the high notification latency of
  fine-grained push ([V4] §3.1 p.16 "Communication Primitives"). In code: dispatch warps TMA-load tokens from
  peers' symmetric buffers over NVLink ([DG-SRC] impl, "Pull token data and SF from remote ranks into local
  L1 buffer").
- Scheduler state machine (release version): for each wave, hand out all Linear-1 tiles of the wave's experts,
  then all Linear-2 tiles of the same experts, then move to the next wave's Linear-1 ([DG-SRC] scheduler
  `get_next_block`).
- Wave size: the smallest expert count whose Linear-1 tiles, after dividing by an imbalance factor of 2,
  can keep all SMs busy; capped at 32 and rounded to a divisor of experts-per-rank so all waves are equal
  ([DG-SRC] heuristics).
- Epilogues: Linear-1 epilogue applies SwiGLU (with clamping) and casts to FP8 with UE8M0 scale factors into a
  local buffer; Linear-2 epilogue writes BF16 output directly into the source rank's combine buffer via NVLink;
  after an NVLink barrier, each token's top-k contributions are reduced and written out ([DG-SRC] impl
  comments).
- Legacy baseline in the benchmark: DeepEP ElasticBuffer FP8 dispatch → DeepGEMM grouped FP8×FP4 GEMM →
  TileLang SwiGLU+FP8 cast → grouped GEMM → DeepEP combine; the test asserts the fused output is bit-identical
  to this baseline (`torch.equal`) ([DG-SRC] tests/test_mega_moe.py).
- Requirements: multi-process launch with symmetric memory ([DG-README]); the release kernel is SM100-only
  (file `sm100_fp8_fp4_mega_moe.cuh`, test comment "TODO: skip the test for SM90") ([DG-SRC]); "still under
  development" ([PR304]).
- Proposals/limits from [V4] §3.1 p.16: full overlap hinges on the compute-communication ratio, not
  bandwidth alone; extreme fusion loads compute, memory and network at once, so power throttling becomes a
  limiter; SwiGLU's exponential/division can stall the GEMM pipeline (they propose a cheaper activation).
- Validated on NVIDIA GPUs and Huawei Ascend NPUs ([V4] p.16).
- Batch-level overlap in V3: DualPipe pairs forward and backward chunks of different micro-batches in
  training ([V3] §3.2.1); prefill and decoding overlap two micro-batches with similar workloads ([V3] §3.4).

## Discrepancies (Notion row vs primary sources)

1. **"Fully hides MoE communication whenever per-layer comm time < compute time."** [V4] states that
   profiling shows communication < computation within a layer, so after fusion computation stays the
   bottleneck; the formal condition is C/B ≤ V_comp/V_comm. The pipeline's first dispatch and last combine
   (fill/drain) stay exposed, so "fully" is the steady-state view. Slides say "hidden in steady state".
2. **"C/B ≤ 2d (about 6144 FLOPs/Byte)".** The paper gives exactly 6144 and only for V4-Pro; here d is the
   expert intermediate dimension (3072), while §4.2.1 of the same paper uses d for the hidden size (7168).
   Slides write "d = expert width" and note the V4-Flash value (4096) as derived.
3. **"Up to 1.96x on RL rollout."** [V4] says up to 1.96× for latency-sensitive scenarios "such as RL rollouts
   and high-speed agent serving"; [PR316] shows 1.96× is the V4-Flash, 1-token-per-rank, EP8 kernel benchmark
   vs the non-fused path. It is a MoE-layer speedup, not an end-to-end RL speedup. Slides say so.
4. Notion does not mention the baseline; [V4] compares to "strong non-fused baselines" (DeepEP + DeepGEMM
   kernels, per [DG-SRC] test), not to Comet. The 1.42× vs 1.92× comparison with Comet is theoretical.
5. Notion lists Flux (arXiv 2406.06858) as a paper; [V4] cites Comet and FlashDMoE (Aimuyo et al., 2025) as
   inspirations, not Flux. Not used on slides.
