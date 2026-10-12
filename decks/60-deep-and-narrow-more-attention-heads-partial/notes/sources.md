# Sources — deck 60: Deep-and-narrow + more attention heads + partial RoPE (GLM-4.5)

## Sources read

| Tag | Source | Where |
|---|---|---|
| [GLM45] | GLM-4.5 Team, "GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models", arXiv 2508.06471v1 (8 Aug 2025). https://arxiv.org/abs/2508.06471 | Abstract (p.1); §1 + Fig. 1 (p.1-2); §2.1 Architecture + Table 1 (p.2-3); §2.3 (p.5); §2.4 Hyper-Parameters (p.5); §4.1 Table 2 (p.14); Table 5 (p.16) |
| [BLOG] | GLM-4.5 blog, https://z.ai/blog/glm-4.5 (text read from the page's JS bundle, fetched 2026-10-13) | "Architecture" paragraph: same claims as [GLM45] §2.1, no additional numbers |
| [DSV3] | DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv 2412.19437 | §2.1.1 MLA, cached vectors c_KV and k_R (p.8); §4.1 data 14.8T (p.4 abstract/intro); §4.2 Model Hyper-Parameters (p.22) |
| [K2] | Kimi Team, "Kimi K2: Open Agentic Intelligence", arXiv 2507.20534 | §2.2 15.5T tokens (p.1, p.6); §2.3 Model Architecture + Table 2 (p.6); "Sparsity Scaling Law" and "Number of Attention Heads" + Fig. 5-6 (p.7) |
| [GLM5] | GLM-5 Team, "GLM-5: from Vibe Coding to Agentic Engineering", arXiv 2602.15763 | §2.1 Architecture (p.4-5); Table 10 (p.36) |
| [CFG-GLM45] | `config.json` of `zai-org/GLM-4.5` on Hugging Face (fetched 2026-10-13) | num_hidden_layers 92, first_k_dense_replace 3, hidden_size 5120, num_attention_heads 96, num_key_value_heads 8, head_dim 128, partial_rotary_factor 0.5, n_routed_experts 160, use_qk_norm true, rope_theta 1e6 |
| [CFG-AIR] | `config.json` of `zai-org/GLM-4.5-Air` | 46 layers, hidden 4096, 96 heads / 8 KV, partial_rotary_factor 0.5, use_qk_norm false |
| [CFG-DSV3] | `config.json` of `deepseek-ai/DeepSeek-V3` | 61 layers, hidden 7168, 128 heads, qk_nope_head_dim 128, qk_rope_head_dim 64, kv_lora_rank 512, 256 routed experts |
| [CFG-K2] | `config.json` of `moonshotai/Kimi-K2-Base` | 61 layers, first_k_dense_replace 1, hidden 7168, 64 heads, qk_nope 128 + qk_rope 64, kv_lora_rank 512, 384 routed experts |
| [CFG-GLM5] | `config.json` of `zai-org/GLM-5` | num_hidden_layers 78, hidden 6144, 64 heads, MLA (kv_lora_rank 512, qk_nope 192 + qk_rope 64), 256 routed experts, no use_qk_norm key |
| [HF-CODE] | Hugging Face transformers, `src/transformers/models/glm4_moe/modeling_glm4_moe.py` (main, fetched 2026-10-13) | `Glm4MoeRotaryEmbedding.compute_default_rope_parameters` (rotary dim = head_dim × partial_rotary_factor); `apply_rotary_pos_emb` (first rotary_dim dims rotated, rest passed through); `Glm4MoeAttention` (q_proj 96×128, k/v_proj 8×128, per-head q_norm/k_norm before RoPE, scaling head_dim^-0.5) |
| [NEOX] | Black et al., "GPT-NeoX-20B", arXiv 2204.06745 | §2.1.1 (p.2-3): rotary on the first 25% of dims, "best balance of performance and computational efficiency" |
| [PROPE] | Barbero et al., "Round and Round We Go! What makes Rotary Positional Encodings useful?", ICLR 2025, arXiv 2410.06205 | Abstract; §6.1 + Tables 1-2 (p.9-10): lowest RoPE frequencies act as (non-robust) semantic channels; p-RoPE keeps a distance-agnostic channel |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 355B total / 32B active | GLM-4.5 parameters (incl. MTP, excl. embeddings/output) | [GLM45] Abstract, Table 1 |
| 671B / 37B | DeepSeek-V3 total / active | [GLM45] Table 1; [DSV3] §4.2 p.22 |
| 1043B / 32B | Kimi K2 total / active (K2's own Table 2 says 1.04T / 32.6B) | [GLM45] Table 1; [K2] Table 2 |
| 92 layers = 3 dense + 89 MoE (+1 MTP) | GLM-4.5 depth | [GLM45] Table 1; [CFG-GLM45] num_hidden_layers 92 |
| 61 = 3 dense + 58 MoE | DeepSeek-V3 depth | [GLM45] Table 1; [DSV3] §4.2 p.22 |
| 61 = 1 dense + 60 MoE | Kimi K2 depth | [GLM45] Table 1; [K2] Table 2 |
| 5120 vs 7168 | hidden dim GLM-4.5 vs DeepSeek-V3/K2 | [GLM45] Table 1 |
| 160 / 256 / 384 routed experts, 8 active, 1 shared | expert counts | [GLM45] Table 1 |
| 96 Q heads / 8 KV heads, head dim 128 | GLM-4.5 GQA | [GLM45] Table 1; [CFG-GLM45] |
| 128 heads (DSV3), 64 heads (K2), QK head dim 192 = 128 + 64 RoPE | MLA heads | [GLM45] Table 1; [DSV3] §4.2; [CFG-DSV3]/[CFG-K2] qk_nope 128 + qk_rope 64 |
| QK-Norm yes (GLM-4.5) / no (Air, DSV3, K2) | | [GLM45] Table 1 |
| "2.5× more heads (96 heads at 5120 hidden)" | paper's phrasing | [GLM45] §2.1 p.3 |
| 40 heads; 2.4× | **derived**: a width-matched layout has 5120 / 128 = 40 heads; 96 / 40 = 2.4 (the paper rounds to 2.5×) | from [GLM45] Table 1 |
| 12,288 query width; 1,024 KV width | **derived**: 96 × 128 and 8 × 128 | [CFG-GLM45] |
| 12 Q heads per KV head | **derived**: 96 / 8 | [GLM45] Table 1 |
| 64 of 128 dims rotated | partial_rotary_factor 0.5 × head_dim 128 | [CFG-GLM45]; [HF-CODE] |
| 64 RoPE dims of 192 in DSV3/K2 heads | decoupled RoPE dim d_h^R = 64, nope 128 | [DSV3] §4.2 p.22; [CFG-DSV3] |
| RoPE base 10,000 → 1,000,000 at 32K | GLM-4.5 long-context extension | [GLM45] §2.4 p.5 |
| +31 layers, 1.5× | **derived**: 92 − 61, 92 / 61 = 1.51 | [GLM45] Table 1 |
| 29% narrower | **derived**: 1 − 5120/7168 = 0.286 | [GLM45] Table 1 |
| 2.9×; 34% / 53% | **derived**: 1043 / 355 = 2.94; 355 / 1043 = 0.34; 355 / 671 = 0.53 | [GLM45] Table 1 |
| 0.5–1.2% | validation-loss reduction from doubling attention heads (iso-token, several FLOP budgets) | [K2] p.7 "Number of Attention Heads", Fig. 6 |
| +83% | inference FLOPs at 128K context when K2 goes 64 → 128 heads with 384 experts | [K2] p.7 |
| 1.69× | FLOPs saved by sparsity 48 vs sparsity 8 at the same validation loss 1.5 | [K2] p.7 "Sparsity Scaling Law" |
| MMLU 87.2 / 87.8 / 86.1; BBH 88.4 / 88.7 / 86.2 | DeepSeek-V3-Base / Kimi-K2-Base / GLM-4.5-Base (GLM internal evaluation framework) | [GLM45] Table 2 p.14 |
| 1.7 and 2.5 points | **derived**: K2-Base minus GLM-4.5-Base on MMLU (87.8 − 86.1) and BBH (88.7 − 86.2) | [GLM45] Table 2 |
| 3rd of 12 models | GLM-4.5 average rank on the 12 ARC benchmarks (post-trained) | [GLM45] §1 p.2, Fig. 1 |
| SWE-bench Verified 64.2 (GLM-4.5) vs 65.4 (Kimi K2) | post-trained | [GLM45] Table 5 p.16 |
| 23T vs 14.8T vs 15.5T tokens | pre-training tokens GLM-4.5 / DSV3 / K2 | [GLM45] Abstract; [DSV3] Abstract, §4.2; [K2] §2.2 |
| 368 KiB vs 68.6 KiB per token; 5.4× | **analytic** BF16 KV cache per token: GLM-4.5 = 92 layers × 2 (K,V) × 8 KV heads × 128 × 2 B = 376,832 B = 368 KiB; DSV3/K2 MLA = 61 layers × (512 latent + 64 RoPE key) × 2 B = 70,272 B = 68.6 KiB; ratio 5.36 | [CFG-GLM45], [CFG-DSV3], [CFG-K2]; MLA caches only c_KV and k_R ([DSV3] §2.1.1 p.8) |
| 178 vs 116 (vs 120) | **analytic** EP all-to-alls per forward pass = 2 (dispatch + combine) × MoE layers: 2 × 89, 2 × 58, 2 × 60; 178 / 116 = 1.53 → "53% more" | [GLM45] Table 1 |
| +10% | **analytic** dispatched elements per token per forward pass: 89 × 8 × 5120 = 3.65M vs 58 × 8 × 7168 = 3.33M (ratio 1.10) | [GLM45] Table 1 |
| GLM-5: 744B / 40B, 3 dense + 75 MoE = 78 layers, hidden 6144, 64 MLA heads with 256 QK dims (192 nope + 64 RoPE), 256 experts | successor shape | [GLM5] §2.1 p.4, Table 10 p.36; [CFG-GLM5] |
| "reduces its layer count … to minimize expert parallelism communication overhead" | GLM-5's stated reason for fewer layers | [GLM5] §2.1 p.4 |
| MLA "superior GPU memory savings" | GLM-5's reason for MLA over GQA | [GLM5] §2.1 p.4 |
| 25% rotary dims (GPT-NeoX) | prior partial-RoPE practice | [NEOX] §2.1.1 |

## Mechanism facts (no numbers) used on slides

- Width reduced = "hidden dimension and number of routed experts"; height increased = "number of layers";
  reason given: "deeper models exhibited better reasoning capacity" ([GLM45] §2.1 p.2). No ablation numbers published.
- More heads: "does not improve training loss compared to models with fewer heads, it consistently improves
  performance on reasoning benchmarks such as MMLU and BBH" ([GLM45] §2.1 p.3). No ablation numbers published.
- GQA + partial RoPE; QK-Norm "to stabilize the range of attention logits" ([GLM45] §2.1 p.3). QK-Norm deep dive: deck 66.
- Order in the attention block ([HF-CODE] `Glm4MoeAttention.forward`): q/k/v projections → per-head RMSNorm on q and k
  (QK-Norm) → partial RoPE on the first 64 dims of q and k → scaled dot-product attention (scale 128^-0.5) with each KV
  head shared by 12 query heads → o_proj back to 5120.
- Partial RoPE: the dot product splits into a rotated part that depends on the relative position (n − m) and an
  unrotated part that depends on content only (follows from [HF-CODE] `apply_rotary_pos_emb`). Rationale for keeping
  position-free dims is from the literature, not from GLM-4.5: [PROPE] §6.1 (lowest frequencies used as semantic
  channels; truncating them keeps a distance-agnostic channel); [NEOX] §2.1.1 (25% rotary dims).
- MLA analogue: DeepSeek-V3/K2 heads already split each QK head into a 128-dim "nope" part and a 64-dim decoupled RoPE part
  ([DSV3] §2.1.1, §4.2).
- K2 cut heads to 64 "to reduce computational overhead during inference" and because doubling heads gave only modest loss
  gains ([K2] §2.3 p.6-7). DeepSeek-V3 set heads to roughly twice the layer count for memory-bandwidth / compute
  efficiency ([K2] p.7; [GLM5] p.5: chosen "according to the roofline of H800").
- K2's case against QK-Norm on MLA (keys not materialized at inference) is in deck 66 / deck 65, not here.

## Discrepancies

1. **"2.5× more heads".** Notion and [GLM45] say 2.5×; the paper does not name the baseline. The width-matched baseline
   5120 / 128 = 40 heads gives 96 / 40 = 2.4×. Slides say "2.4× a width-matched layout (paper: 2.5×)".
2. **Kimi K2 activated parameters.** [GLM45] Table 1 lists 32B; [K2] Table 2 lists 32.6B. Slides use 32B (the GLM table, so
   the three models are counted the same way) and the notes mention 32.6B.
3. **GLM-5 layer count.** [GLM5] §2.1 text says it "reduces its layer count to 80"; [GLM5] Table 10 gives 3 dense + 75 MoE
   (78) and [CFG-GLM5] num_hidden_layers = 78. Slides use 78.
4. **Notion "deeper models reason better; more heads … keep improving MMLU/BBH"** matches [GLM45], but the paper publishes no
   numbers for either ablation, and at the *base-model* level GLM-4.5-Base scores slightly *below* DeepSeek-V3-Base and
   Kimi-K2-Base on MMLU and BBH ([GLM45] Table 2). Slides present the claims as the authors' qualitative findings and show
   the base-model numbers as they are.
5. **Confounders.** GLM-4.5 was pre-trained on 23T tokens vs 14.8T (DSV3) and 15.5T (K2), with different data and Muon; any
   model-level comparison mixes shape with data and optimizer. Stated on the results and limits slides.
