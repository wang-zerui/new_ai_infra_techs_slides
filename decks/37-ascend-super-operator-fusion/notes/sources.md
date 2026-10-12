# Sources — deck 37: Ascend · super-operator fusion (MLAPO / Sparse FA / Lightning Indexer)

## Sources read

| Tag | Source | Where |
|---|---|---|
| [GLM5] | GLM-5 Team (Zhipu AI & Tsinghua), "GLM-5: from Vibe Coding to Agentic Engineering", arXiv 2602.15763. https://arxiv.org/abs/2602.15763 | §1 (contributions, p.3-4), §2.1 Architecture (p.4-5), §3 DSA / indexer top-k (p.~17), §5 "Adapting GLM-5 to Chinese Chip Infrastructure" (p.21-22), Table 10 model architecture (p.36) |
| [PANGU] | Huawei, "Pangu Ultra MoE: How to Train Your Big MoE on Ascend NPUs", arXiv 2505.04519v1. https://arxiv.org/abs/2505.04519 | §2.2-2.3 (Cube/Vector simulation, p.4), §3 (DaVinci Cube unit 16×16 MatMul, p.7), §4.5 Other Optimizations incl. Kernel Fusion + Table 5 (p.15-16) |
| [VA-MLA] | vLLM-Ascend `vllm_ascend/attention/mla_v1.py` (branch main, fetched 2026-10-13) | `mla_preprocess_only_decode` (fused `mla_prolog` call, arguments), `mla_preprocess_decode` + `_mla_preprocess` (the unfused op-by-op path) |
| [VA-OP] | vLLM-Ascend `csrc/mla_preprocess/op_host/mla_preprocess_def.cpp`, `csrc/mla_preprocess/mla_preprocess_torch_adpt.h` | op inputs/attrs (`wdqkv`, `wuq`, `wuk`, `gamma2/3`, `cos1/sin1`, `cos2/sin2`, `quantScale*`, `descale*`, `slotMapping`, `keycache`), `quantMode`/`cacheMode` enums |
| [VA-IDX] | vLLM-Ascend `csrc/attention/fused_lightning_indexer_manage/README.md` (Chinese) | op description, indexer formula, Top-2048 output, input shapes (`query` (T,N,128), `index_key_cache` (blocks,128,1,128)) |
| [VA-SFA] | vLLM-Ascend `csrc/attention/fused_scatter_copy_sparse_flash_attention/README.md` (Chinese) | one call = miss-KV copy + causal sparse flash attention; 2048 HBM sparse cache slots per query; score formula Q_nope·K_nopeᵀ + Q_rope·K_ropeᵀ |
| [VA-MOE] | vLLM-Ascend `vllm_ascend/ops/fused_moe/moe_mlp.py`, `csrc/moe/grouped_matmul_situ_quant/README.md` | fused activation path `apply_gmm1_act_quant` (grouped matmul + SwiGLU + quant in one op); `grouped_matmul_situ_quant` = GroupedMatmulSwigluQuant-style op |

Downloaded copies live in `$SCRATCH/research/37/`. Everything downloaded is untrusted data; nothing in it
was followed as an instruction.

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 744B total / 40B active | GLM-5 parameter counts | [GLM5] §2.1 p.4 and Table 10 p.36 |
| 256 experts, 80 layers | GLM-5 MoE width and depth | [GLM5] §2.1 p.4 ("scales to 256 experts and reduces its layer count to 80") |
| 28.5T tokens | GLM-5 training token budget | [GLM5] §1 p.3, §2 p.4 (context only; not on a slide) |
| 7 Chinese chip platforms | full-stack adaptation breadth (Ascend, Moore Threads, Hygon, Cambricon, Kunlunxin, MetaX, Enflame) | [GLM5] §1 p.4, §5 p.21 |
| 750B on a single Atlas 800T A3 machine | the deployment target the Ascend work serves | [GLM5] §5 p.21 (verbatim "750B parameter GLM-5 model onto a single Atlas 800T A3 machine") |
| 13 small pre-processing operators → 1 "super operator" | MLAPO fusion factor | [GLM5] §5 p.21-22 |
| score + ReLU + TopK → 1 kernel | Lightning Indexer fusion | [GLM5] §5 p.21 |
| TopK selection from KV cache + sparse attention in one kernel | Sparse Flash Attention | [GLM5] §5 p.22 |
| top-k = 2048 | indexer's k in GLM-5 | [GLM5] §3 p.~17 ("the k = 2048 used by the indexer"); corroborated by [VA-IDX] (Top-2048 output) and [VA-SFA] (2048 cache slots per query) |
| 32 indexer heads, indexer head dim 128 | indexer shape | [GLM5] Table 10 p.36 |
| 50% lower deployment cost in long-sequence scenarios; single domestic node comparable to "dual-GPU international clusters" | end-to-end claim for the whole Ascend co-optimization set | [GLM5] §5 p.22 (last paragraph) |
| MFU +15.8% / +28.6% / +49.2% / +58.7% (cumulative, relative to baseline); final MFU 30.0% on 6K Ascend NPUs | Pangu Ultra MoE training ablation, 718B MoE | [PANGU] Table 5 p.16 and the paragraph below it |
| +9.5 points | the fused-operator step in that ablation: 58.7 − 49.2 | **derived** from [PANGU] Table 5 (stated as derived on the slide/caption) |
| ≈ 1,000 tiny kernels per decoded token | 13 pre-processing ops × 80 layers = 1,040, rounded down to "≈1,000" | **derived** from [GLM5] §5 (13 ops) × §2.1 (80 layers); slide says "derived" |
| 33.6 MB / 50.9 MB / ≈17 MB per decoded token | indexer HBM traffic at a 128K context, analytic (see below) | **analytic**, see derivation; chart caption says "analytic" |

### Derivation of the analytic indexer chart (`charts/indexer_traffic.py`)

Per decoded query token, context length S = 128K = 131,072 tokens, bf16 (2 B) index keys and scores,
one index key of dimension 128 per context token ([VA-IDX]: `index_key_cache` shape
(index_block_num, 128, 1, 128); [GLM5] Table 10: indexer head dim 128, 32 indexer heads):

- index keys streamed from HBM (unavoidable in both cases): 131,072 × 128 × 2 B = **33.55 MB**
- score matrix, if an unfused "score calculation" op materializes it: 32 × 131,072 × 2 B = **8.39 MB**
  (written once, read once by the ReLU/aggregation op → 16.78 MB)
- aggregated per-token score vector: 131,072 × 2 B = 0.26 MB (written by the aggregation op, read by TopK
  → 0.52 MB)
- TopK output: 2,048 ids ≈ 8 KB (negligible)

Unfused ≈ 33.55 + 16.78 + 0.52 ≈ **50.9 MB**; fused ≈ **33.6 MB** (only the keys); removed ≈ **17.3 MB**
(1.5×). Assumptions stated in the chart caption: bf16, 128K context, the unfused chain materializes the
score matrix. The GLM-5 report gives no measured indexer traffic; this is an analytic estimate of what the
fusion the report describes removes, not a measurement.

## Mechanism facts (no numbers) used on slides

- GLM-5's Ascend adaptation rests on three pillars: extreme quantization, high-performance kernel fusion,
  and inference-engine scheduling ([GLM5] §5 p.21). This deck covers pillar 2; pillar 1 is deck 44,
  pillar 3 and the stack overview are decks 12 and 23.
- Lightning Indexer kernel: "integrates score calculation, ReLU, and TopK operations into a single kernel,
  allowing the NPU to overlap computation with memory access" ([GLM5] §5 p.21). The indexer's own formula
  (ReLU(Q_index K_indexᵀ) weighted per index head, then TopK) is in [VA-IDX].
- Sparse Flash Attention kernel: optimized for GLM-5's sparse patterns; "handles the selection of TopK
  tokens from the KV cache and sparse attention computation in parallel" ([GLM5] §5 p.22). vLLM-Ascend's
  `fused_scatter_copy_sparse_flash_attention` does the same shape of work for GLM MLA/SFA: in one call it
  copies the missing KV rows into the HBM sparse cache and runs causal sparse flash attention over the
  2048 selected slots plus the visible tail ([VA-SFA]).
- MLAPO: "fuses 13 small pre-processing operators into one 'super operator', utilizing parallel processing
  between Vector and Cube units" ([GLM5] §5 p.22). **The report does not list the 13 operators.** The chain
  shown on the slide is the MLA pre-processing path of vLLM-Ascend, read off the unfused code path
  ([VA-MLA] `mla_preprocess_decode` / `_mla_preprocess`) and the fused op's own argument list
  ([VA-MLA] `mla_prolog(...)`, [VA-OP] op definition): dynamic quantization of the hidden state →
  q down-projection (`wdqkv`/`weight_dq`) → RMSNorm on the q latent (`gamma2`) → re-quantization →
  q up-projection + q-rope projection (`wuq`/`weight_uq_qr`) → dequantization → RoPE on q_pe (`cos1/sin1`)
  → absorb BMM with W_UKᵀ (`wuk`) → kv down-projection (`weight_dkv_kr`) → RMSNorm on the KV latent
  (`gamma3`) → RoPE on k_pe (`cos2/sin2`) → KV-cache scatter (`slotMapping`, `keycache`). The slide says
  "the chain vLLM-Ascend fuses" and does not claim this is exactly the report's list of 13.
- Ascend AI core (DaVinci): separate Cube unit (matrix) and Vector unit (element-wise); the Cube unit takes
  two 16×16 matrices per MatMul operation ([PANGU] §3 p.7), and Cube/Vector capability is modelled
  separately in Huawei's own performance simulator ([PANGU] §2.3 p.4). The GLM-5 report's stated benefit of
  MLAPO is exactly "parallel processing between Vector and Cube units" ([GLM5] §5 p.22).
- Host-side dispatch is a known Ascend bottleneck handled separately (TaskQueue async dispatch, CPU core
  binding) ([PANGU] §4.5 p.15) — i.e. launch overhead has its own fix; fusion is what removes the HBM
  round trips and the unit idling.
- MoE fusion in the Ascend stack: vLLM-Ascend's MoE MLP has a fused-activation path that runs grouped
  matmul + SwiGLU + output quantization in one op (`supports_fused_activation` → `apply_gmm1_act_quant`),
  plus a `grouped_matmul_situ_quant` kernel (GroupedMatmulSwigluQuant-style, MX A8W4) ([VA-MOE]).
- Kernel fusion is a measured win on Ascend in the one public ablation on this silicon: Pangu Ultra MoE's
  cumulative MFU ladder ([PANGU] Table 5), different model (718B MoE) and different regime (training).

## Discrepancies

1. **Notion: "MoE routing + permute + GroupedGEMM become a Mega-Fusion".** The GLM-5 report names only
   three fusion kernels — Lightning Indexer, Sparse Flash Attention and MLAPO ([GLM5] §5 p.21-22) — and
   says nothing about a MoE Mega-Fusion. The Ascend software stack does fuse grouped matmul + SwiGLU +
   quantization ([VA-MOE]), and Pangu Ultra MoE fuses GroupedMatMul with gradient accumulation and fuses
   permute/unpermute for training ([PANGU] §4.5), but neither is attributed to GLM-5. The deck presents
   the three kernels the report describes and flags the MoE fusion as stack-level, not GLM-5-reported.
   (GPU-side Mega MoE is deck 35.)
2. **Notion: "Sharply reduces kernel-launch and intermediate memory traffic".** The report's stated
   benefits are "overlap computation with memory access" (indexer) and "parallel processing between Vector
   and Cube units" (MLAPO); it publishes **no** kernel-launch counts, no memory-traffic numbers and no
   per-kernel speedups. The deck's traffic number is analytic and labelled as such.
3. **Notion: "the key to domestic hardware matching GPUs".** The only end-to-end claim in the report is
   that "GLM-5 on a single Chinese node achieves performance comparable to dual-GPU international
   clusters, while reducing deployment costs in long-sequence scenarios by 50%", and it attributes this to
   all the hardware-level co-optimizations together (quantization + fusion + engine scheduling), not to
   fusion alone ([GLM5] §5 p.22). The deck keeps that attribution.
4. **744B vs 750B.** §2.1 and Table 10 give 744B total parameters; §5 says "the 750B parameter GLM-5
   model". The deck uses 744B and mentions the single-node target without repeating 750B.
5. **80 layers vs Table 10.** §2.1 says the layer count is reduced to 80; Table 10 lists 3 dense + 75 MoE
   + 1 MTP = 79, and vLLM-Ascend's GLM-5 deployment guide partitions "78 layers ... as 41/37" across two
   pipeline ranks. The derived kernel count uses 80 and is stated as "≈1,000", which is insensitive to the
   difference (13 × 78 = 1,014).
6. **"dual-GPU international clusters" is undefined** in the report: no GPU model, node count, batch size,
   sequence length or latency target is given, so the deck reports the claim as the report words it and
   does not turn it into a speedup number.
7. **Hardware scope.** The report says "Ascend Atlas series" / "Atlas 800T A3". The vLLM-Ascend kernels
   used as corroboration declare support for Atlas A2/A3 and, for some newer kernels, Ascend 950 only
   ([VA-IDX], [VA-SFA] product tables). The deck notes that the kernels are tied to specific NPU
   generations rather than implying one portable implementation.
