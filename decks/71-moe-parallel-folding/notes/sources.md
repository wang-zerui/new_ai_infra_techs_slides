# Sources — deck 71: MoE Parallel Folding

## Sources read

| Tag | Source | Where |
|---|---|---|
| [MC] | NVIDIA, "Scalable Training of Mixture-of-Experts Models with Megatron Core", arXiv 2603.07685v2 (Notion reference). https://arxiv.org/abs/2603.07685 | §1.2 (p.6), §2.2.1 Table 1 + §2.2.2 (p.10-11), §3.1-3.3 incl. Table 2, Fig. 5-6 (p.12-18), §9.1.2 example (p.72), Table 17 + §9.2.2 (p.74-75), §9.2.3 (p.76) |
| [PF] | D. Liu, Z. Yan, X. Yao et al., "MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core", arXiv 2504.14960v3 (cited as [26] by [MC] and by the MoE README). https://arxiv.org/abs/2504.14960 | Abstract; §3.2 (p.5); §3.3 + Fig. 2 (p.5-6); §4.1 (p.7); Table 1 (p.8); §4.3-4.4 (p.9-10); Table 2 (p.11); App. 6.1-6.4, Listing 1, Tables 3-5 (p.15-20) |
| [RD] | Megatron-LM `megatron/core/transformer/moe/README.md` (cached), "MoE Parallel Folding" section and Guideline 4 | lines 325-355, 524 |
| [CODE-PS] | `megatron/core/parallel_state.py` (main, fetched 2026-10-13): `initialize_model_parallel` default `order="tp-cp-ep-dp-pp"`; separate `decoder_rank_generator` (tp, cp, dp, pp; ep=1) and `expert_decoder_rank_generator` (tp=ETP, ep, dp=EDP, pp; cp=1); assert that both generators produce identical PP groups | lines ~638, 891-941 |
| [CODE-ARG] | `megatron/training/arguments.py` line ~509 and `megatron/core/model_parallel_config.py` `expert_tensor_parallel_size` docstring: ETP defaults to TP when not set | |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 685B total / 37B active, 18:1 | DeepSeek-V3 parameter asymmetry | [MC] §3.2.1 table (p.13) |
| TP4 · CP2 · DP8 · PP4 = 256 GPUs; traditional max EP = 8; folded ETP1 · EP64 · EDP1 | the worked example | [MC] §3.3.3 item 1 (p.17-18); [RD] lines 340-342 |
| 8× more expert parallelism | 64 / 8 | [MC] §3.3.3 (p.17); [RD] line 342 |
| CP8 + EP8: ≥ 64 GPUs traditional vs 8 GPUs folded | minimum GPU count | [MC] §3.2.4 Challenge 1 (p.16) and §3.3.3 item 2 (p.18); [RD] lines 344-346 |
| 5–10× lower bandwidth cross-node than NVLink | why EP crossing nodes hurts | [MC] §3.2.4 Challenge 3 (p.16) |
| > 70% of MoE-layer latency is communication when ETP×EP > 8 (inter-node), fine-grained model | ablation | [PF] §4.4 finding (3) (p.9-10) |
| MFU, MCore (coupled) → MCore w/ Folding, best config, GBS 256, BF16, H100: Mixtral-8x22B 128 GPUs 46.3% → 49.3%; Llama3-8x70B 256 GPUs 38.8% → 41.6%; Qwen2-57B-A14B 64 GPUs 35.3% → 39.0%; Mixtral-8x22B-G8T8 128 GPUs 17.1% → 28.8% | headline results (chart `charts/mfu.py`) | [PF] Table 1 (p.8) and Table 3 (p.18) |
| 17.1% | best coupled-mapping MFU for fine-grained Mixtral-8x22B-G8T8 on 128 H100 (pain number) | [PF] Table 1 (p.8) |
| Mixtral-8x22B-G8T8 configs: MCore CP1 TP2 EP8 PP8 (ETP = TP = 2) vs Folding CP1 TP4 EP8 PP8 ETP1 | why it gains | [PF] Table 3 (p.18) |
| Strong scaling, GBS 1024, Mixtral-8x22B-G8T8: MCore 19.8 / 18.4 / 16.3 / 13.4 % vs Folding 30.0 / 29.3 / 26.7 / 25.5 % at 128 / 256 / 512 / 1024 GPUs | chart `charts/scaling.py` | [PF] Table 4 (p.18-19); the "18,4%" typo in the PDF is read as 18.4% |
| FP8 Mixtral-8x22B, 128 H100: 575.1 → 631.7 TFLOPS (1.10×); BF16 458.3 → 487.7 (1.06×) | folding still pays under FP8 | [PF] Table 2 (p.11) |
| Accuracy: Mixtral 8x7B, 40B tokens, TP2 CP2 PP2 EP8 ETP1, loss curves match MCore v0.9 | correctness | [PF] App. 6.1 (p.15) |
| DeepSeek-V3: 1,024 H100 TP2/PP8/EP64; 256 GB200 TP1/PP4/EP64; expert TP always 1; 368 / 1,048 TFLOPS/GPU | production use of folding | [MC] Table 17 (p.74) |
| EP64 → each GPU holds exactly 4 of 256 experts; no local token permutation | why EP64 | [MC] §9.2.2 (p.74) |
| 8-GPU worked example: attention TP2·CP2·DP2, MoE ETP1·EP8·EDP1 | ❶ figure | [PF] App. 6.2 (p.16) uses exactly this mapping ("TP2CP2DP2" / "TP1EP8DP1") |

Derived (labelled "derived" on slides):
- Mixtral-8x22B-G8T8 MFU ratio 28.8 / 17.1 = **1.68×** ([PF] Table 1).
- MFU point gains on the other three models: Mixtral-8x22B 49.3 − 46.3 = 3.0; Llama3-8x70B 41.6 − 38.8 = 2.8;
  Qwen2-57B-A14B 39.0 − 35.3 = 3.7 → "+2.8 to +3.7 points" ([PF] Table 1).
- "ETP2 × EP8 = 16 GPUs spans two 8-GPU nodes" for the G8T8 baseline (TP2, so ETP = TP = 2, EP8; [PF] Table 3 and
  the ablation's "ETP×EP > 8 needs inter-node communication", §4.4). Speaker notes only.
- Per-GPU expert weight in the 8-GPU example: traditional ETP2·EP2 → each GPU holds ½ of 4 experts = 2 experts' worth; folded EP8 → 1 expert (from the mapping, assuming 8 experts as drawn). Used only in speaker notes.

## Mechanism facts used on slides

- Attention groups over TP × CP × DP × PP; MoE groups over ETP × EP × EDP × PP; "the sole constraint" is a consistent PP ([MC] §3.3.1 p.16; [PF] §3.2 p.5). Code asserts identical PP groups ([CODE-PS]).
- Rank order `tp-cp-ep-dp-pp` (TP innermost, PP outermost) for both generators ([CODE-PS]); so within a PP stage, attention factors the ranks as TP·CP·DP and MoE as ETP·EP·EDP; EDP = world / (ETP·EP·PP).
- The attention→MoE transition "requires only a reshape" that flattens (sub)sequences into a batch of tokens, no explicit communication ([PF] App. 6.2 p.16); the dispatcher workflow is the same regardless of the attention mapping because MoE computes tokens independently ([PF] §3.3 p.5).
- Dispatcher with ETP>1: permute → All-to-All-V over EP → AllGather-V over ETP → expert GEMMs → ReduceScatter-V over ETP → All-to-All-V → unpermute ([PF] §3.3, Fig. 2, p.6).
- Process groups per component: router tp/cp/tp_cp; dispatcher ep/tp_ep; experts ep/expt_tp/expt_dp ([MC] Table 1 p.11).
- Gradients: expert params marked allreduce=False; dense grads reduce over dp_cp, expert grads over expt_dp, scaled by edp_size/dp_size; ChainedOptimizer; distributed-optimizer states for experts sharded over EDP ([MC] §2.2.2 p.11, §3.3.2 p.17).
- ETP defaults to TP unless `--expert-tensor-parallel-size` is set; README recommends 1 for fine-grained MoE ([CODE-ARG]; [RD] line 524).
- Token dropping under folding uses sub-sequence dropping (decision on local logits) — no adverse convergence effect observed ([PF] §3.3 p.6).

## Discrepancies

1. **Notion: "EP can be folded inside TP x CP groups".** Primary sources say EP folds across arbitrary
   sub-groups of the attention mapping, including DP: in the 256-GPU example EP64 folds across
   TP×CP×DP ([MC] §3.3.3 p.17), and [PF] App. 6.1 validates EP folded "with all of TP, CP, DP". Slides say TP×CP×DP.
2. **Notion lists only arXiv 2603.07685.** The mechanism, the dispatcher and all measured MFU results come
   from the companion paper arXiv 2504.14960 ([MC] defers implementation details to it, p.16). Slides cite both.
3. **[PF] Table 5 (context scaling)** prints identical numbers for the Qwen2-57B and Mixtral-8x22B rows,
   while the text gives Qwen 38.7% → 35.9% and Mixtral 47.6% → 42.9%. Not used on slides.
4. **[PF] Table 3, Llama3-8x70B "MCore w/ Folding" row** has an ambiguous column alignment (only four
   numbers for five columns). Its config is not used on slides; only its MFU (Table 1) is.
5. Notion's numbers (EP 8 → 64, ETP=1, EDP=1 at 256 GPUs) match [MC] §3.3.3 exactly.
