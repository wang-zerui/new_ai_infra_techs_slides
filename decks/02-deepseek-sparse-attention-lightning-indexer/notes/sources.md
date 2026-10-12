# Sources — deck 02: DeepSeek Sparse Attention (DSA) + lightning indexer

## Sources read

| Tag | Source | Where |
|---|---|---|
| [V32] | DeepSeek-AI, "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models", arXiv 2512.02556v1 (2 Dec 2025). https://arxiv.org/abs/2512.02556 (cached text + PDF) | §1 (p.2), §2.1 + Eq. 1-2 + Fig. 2 (p.3-4), §2.1.1 Eq. 3-4 (p.4), §2.2 (p.5), §2.3 + Fig. 3 (p.5-6), §3 (p.5), App. A Fig. 7 (p.19-20) |
| [V3] | DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv 2412.19437 | Abstract; §4.2 hyper-parameters (61 layers, 128 heads, d_h 128, d_c 512, d'_c 1536, d_h^R 64) |
| [HF] | HuggingFace `deepseek-ai/DeepSeek-V3.2-Exp`: `config.json`, `inference/model.py`, `inference/kernel.py` (fetched 2026-10-13) | config: index_n_heads 64, index_head_dim 128, index_topk 2048, num_hidden_layers 61, num_attention_heads 128; model.py `Indexer` (wq_b from the query latent, wk + LayerNorm, weights_proj, Hadamard `rotate_activation`, FP8 `act_quant` block 128, `k_cache` E4M3 + `k_scale_cache` FP32, `fp8_index`, topk), `MLA.forward` (MHA-mode prefill branch vs MQA-mode decode branch, index mask added to scores); kernel.py `fp8_index_kernel` |
| [VLLM] | vLLM blog, "DeepSeek-V3.2-Exp in vLLM: Fine-Grained Sparse Attention in Action", 2025-09-29. https://vllm.ai/blog/2025-09-29-deepseek-v3-2 | "New Cache Entry and Quantization Scheme" (656 B MLA FP8 cache entry; separate indexer K cache, per-block value/scale layout, block size 64); "New Computation with Masking" (q (h,d), context (n,d), logits, top-2048, ks/ke for prefill batching); "Fusion pass" (FlashMLA sparse kernel, DeepGEMM indexer kernels) |
| [DSNEWS] | DeepSeek API docs, "Introducing DeepSeek-V3.2-Exp", 2025-09-29. https://api-docs.deepseek.com/news/news250929 | "API prices cut by 50%+" |
| [GLM5] | GLM-5 Team, "GLM-5: from Vibe Coding to Agentic Engineering", arXiv 2602.15763v2 (24 Feb 2026) | §1 (p.3: adopts DSA; 744B; 28.5T tokens), §2.1.1 + Table 3 (p.5-6), Table 4-5 (p.6-7), Table 6 (p.7-8), "DSA RL insights" (p.21 of text, §RL), Ascend fused Lightning Indexer kernel (p.21-22), App. A Table 10 (p.36) |
| [DG] | deepseek-ai/DeepGEMM `tests/test_attention.py` and README (commit 057ca59, 2026-09-30) | non-paged `fp8_fp4_mqa_logits` (q [seq, H, D], kv [seq_kv, D] + per-token scale, weights [seq, H], ks/ke, output compressed to [seq, max_seqlen_k]); paged `fp8_fp4_paged_mqa_logits` (q [batch, next_n, H, D], kv_cache [blocks, 64, 1, D+4], block_table, context_lens, schedule metadata). README: prefill = non-paged, decode = paged; first released 2025-09-28 (PR #200) as `fp8_mqa_logits` / `fp8_paged_mqa_logits` |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 671B total / 37B active, MLA, 61 layers, 128 heads | V3 backbone shared by V3.1-Terminus and V3.2 | [V3] abstract, §4.2; [V32] §2.1 "exactly the same architecture … only architectural modification … DSA" |
| 128K | context length of V3.1-Terminus base and of the DSA training sequences | [V32] §2.1.1 |
| 744B (40B active) | GLM-5 size | [GLM5] §1 p.3, Table 10 |
| ≈ $0.07 and ≈ $2.15 per M tokens; **31×** | V3.1-Terminus decode cost at token position 0 and 128K; ratio 2.1543/0.0702 = 30.7 | **Derived**: [V32] Fig. 3(b), read from the PDF vector data (`notes/digitize_fig3.py`, output `notes/digitize_fig3.out`); H800, $2/GPU-hour |
| ≈ $0.25 per M tokens; **8.7×** lower | V3.2 decode cost at 128K (0.2488); 2.1537/0.2488 = 8.66 | Derived from [V32] Fig. 3(b), same script |
| ≈ $0.67 → ≈ $0.19; **3.6×** lower | prefill cost at 128K, V3.1-Terminus 0.6658 vs V3.2 0.1871 = 3.56 | Derived from [V32] Fig. 3(a), same script |
| ≈ 5K (decode) / ≈ 10K (prefill) tokens | position below which V3.2's plotted cost is above V3.1-Terminus's (crossover 4.8K / 10.4K) | Derived from [V32] Fig. 3, same script |
| 50%+ | API price cut at the V3.2-Exp launch | [DSNEWS] |
| 64 heads × 128 dims; k = 2,048 | indexer heads/dim, selected tokens per query | [HF] config.json; [V32] §2.1.1 ("select 2048 key-value tokens") |
| 576-d latent (512 + 64), 128 query heads | MLA cache entry and heads in MQA mode | [V3] §4.2 (d_c 512, d_h^R 64, 128 heads); [V32] §2.1 + App. A |
| 8,192 vs 139,264 MACs per key; **17×** | indexer QK (64 × 128) vs MQA-mode attention QK + PV (128 × (576 + 512)) per cached key | **Analytic**: `notes/analytic.py` (output `notes/analytic.out`) |
| 40,960 MACs per key; 5× | MHA-mode (dense prefill) attention per key, 128 × (192 + 128) — used in notes only | Analytic, same script |
| 1.6% | 2,048 / 131,072 keys attended at 128K | Analytic, same script |
| 79% | indexer share of DSA's per-query MACs at 128K (1.074 G of 1.359 G) | Analytic, same script |
| 132 B per token per layer; +20% | indexer K cache = 128 E4M3 bytes + one FP32 scale; vs 656 B MLA FP8 entry | [HF] model.py (`k_cache` E4M3 [.., 128], `k_scale_cache` FP32 [.., 128/128]); [DG] tests (`head_dim + 4` bytes per token); [VLLM] (656 B); ratio analytic |
| 1,000 steps × 16 × 128K = 2.1B tokens; lr 1e-3 | dense warm-up | [V32] §2.1.1 |
| 15,000 steps × 480 × 128K = 943.7B tokens; lr 7.3e-6 | sparse training | [V32] §2.1.1 |
| +4 points on AA-LCR | V3.2-Exp vs V3.1-Terminus, reasoning mode | [V32] §2.2 |
| RULER 97.44 … 79.21 / 97.51 … 71.35 / 96.77 … 78.86 (4K…128K) | GLM-4.7-Flash, + DSA warm-up only, + DSA (150B tokens) | [GLM5] Table 6 |
| −7.9 / −0.35 at 128K | 79.21 − 71.35 = 7.86 (warm-up only); 79.21 − 78.86 = 0.35 (full DSA) | Derived from [GLM5] Table 6; 0.35 stated in text p.8 |
| 150B tokens | GLM-4.7-Flash DSA joint training | [GLM5] p.7 |
| −30.35 / −5.69 / −11.28 / −8.25 RULER@128K | GLM-9B (1:1 efficient:full layers, 190B tokens): SWA interleave / SWA searched pattern / GDN / SimpleGDN vs full attention | [GLM5] Table 5 |
| 20B tokens; 47× | GLM-5 sparse-adaptation tokens; 943.7 / 20 = 47.2 | [GLM5] §2.1.1 p.6; ratio derived |
| 1,000 steps × 14 × 202,752 tokens | GLM-5 warm-up (2.84B tokens, derived) | [GLM5] §2.1.1 p.6 |
| 32 indexer heads × 128 | GLM-5 indexer | [GLM5] Table 10 |
| MLA vs DSA (GLM-5 base): MQ-NIAH 100.0/100.0, MV-NIAH 95.5/97.0, SQuAD 79.7/86.0, HotpotQA 66.3/63.0 (128K) | notes only | [GLM5] Table 3 |
| "1.5–2×" attention computation reduction, "half the GPU cost" at 128K | GLM-5's own statement — notes only | [GLM5] p.6 |

## Mechanism facts used on slides (no numbers)

- Index score I(t,s) = Σ_j w(t,j) · ReLU(q(t,j) · k(s)); q and w derived from the query token, k from the preceding token;
  ReLU chosen for throughput; "small number of heads and can be implemented in FP8" ([V32] Eq. 1).
- u_t = Attn(h_t, {c_s | I(t,s) ∈ Top-k(I(t,:))}) ([V32] Eq. 2).
- DSA instantiated on the MQA mode of MLA because "each key-value entry must be shared across multiple queries" at the
  kernel level ([V32] §2.1). Appendix A: V3.1-Terminus used MHA mode for training and prefill, MQA mode for decode.
- Indexer queries come from the MLA query latent (`wq_b(qr)`), the key from the hidden state (`wk`, LayerNorm), the
  head weights from the hidden state (`weights_proj`), all quantised to FP8 after a Hadamard rotation ([HF] model.py).
  The indexer keeps its own key cache next to the MLA cache ([HF]; [VLLM]).
- Warm-up: main model frozen, dense attention, target = attention summed over heads and L1-normalised; loss KL(p ‖
  softmax(I)) ([V32] Eq. 3). Sparse stage: all parameters, KL only over the selected set S_t (Eq. 4); indexer input
  detached, indexer trained only by L_I, main model only by the LM loss ([V32] §2.1.1).
- Short-sequence prefill: a masked MHA mode simulates DSA for efficiency ([V32] §2.3).
- Prefill path: per-query [ks, ke) ranges over packed KV, non-paged logits kernel; decode path: paged KV via block table,
  paged logits kernel with schedule metadata ([VLLM]; [DG]).
- GLM-5 RL: storing top-k indices for replay is impractical with k = 2048; non-deterministic top-k (CUDA / TileLang)
  degraded RL within a few steps with an entropy drop; GLM-5 uses deterministic `torch.topk` and freezes the indexer
  during RL ([GLM5] "DSA RL insights").

## Discrepancies (Notion row vs primary sources)

1. **"top-k (about 2048)"**: the source fixes k = 2,048 exactly ([V32] §2.1.1; [HF] index_topk). Slides say 2,048.
2. **"Large drop in long-context training … cost"**: DeepSeek quantifies only *inference* cost (Fig. 3); the training
   cost reduction is not measured in [V32]. GLM-5 asserts reduced training and inference cost qualitatively and gives
   "1.5–2×" for attention computation. Slides quantify serving cost only.
3. **"GLM-5 adopts it wholesale"**: GLM-5 changes the configuration (32 instead of 64 indexer heads, interleaved RoPE in
   the indexer per the transformers implementation, a 20B-token adaptation instead of 943.7B, deterministic top-k and a
   frozen indexer during RL). Slides say "adopts" and list the differences.
4. **Notion links NSA (arXiv 2502.11089)** as a paper for this row; NSA is a different, natively trained block-sparse
   method. It appears only as a failed-suitor mention ("native sparse pretraining"), not as part of DSA.
5. **Source-vs-source**: GLM-5's "roughly 1.5–2×" attention-compute reduction is much smaller than DeepSeek's measured
   3.6× (prefill) / 8.7× (decode) cost reduction at 128K; they measure different things (attention FLOPs vs end-to-end
   serving cost per token at a given position). Slides use DeepSeek's measured costs and attribute them.
