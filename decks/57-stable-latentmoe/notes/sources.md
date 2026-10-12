# Sources — deck 57: Stable LatentMoE

## Sources read

| Tag | Source | Where |
|---|---|---|
| [K3] | Kimi Team, "Kimi K3: Open Frontier Intelligence", arXiv 2607.24653 (cached text) | Abstract (p.1); §1 (p.2); Fig. 2 caption (p.3); §2.3 intro + Eq. 11 (p.6-7); §2.3.1 (p.7); §2.3.2 + Eq. 12 + Fig. 4 (p.7-8); §3.2 + Fig. 7 + Table 1 (p.10-11); §5.4.2 Stable LatentMoE kernels (p.24-25); App. B Eq. 18-19 (p.43) |
| [K3-cfg] | Kimi-K3 Hugging Face `config.json`, `modeling_kimi_linear.py` (https://huggingface.co/moonshotai/Kimi-K3, downloaded 2026-10-13) | `hidden_size` 7168, `routed_expert_hidden_size` 3584, `moe_intermediate_size` 3072, `num_experts` 896, `num_experts_per_token` 16, `num_shared_experts` 2, `latent_moe_use_norm` true, `activation_situ_beta` 4.0, `activation_situ_linear_beta` 25.0; `KimiSparseMoeBlock` (router on the full-width x; `routed_expert_down_proj` 7168→3584 before dispatch; experts `KimiBlockSparseMLP(hidden 3584, inter 3072)`; weighted sum → `routed_expert_norm` (RMSNorm 3584) → `routed_expert_up_proj` 3584→7168; shared experts = one `KimiMLP` with intermediate 3072 × 2 = 6144 on the full x); `SituAndMul` |
| [LMoE] | V. Elango et al. (NVIDIA), "LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts", arXiv 2601.18089v1 | Fig. 1 caption (p.2); §2.1-2.3 Design Principles I-V (p.3-6); §3 Eq. 1-2 (p.6-7); Table 1-2 (p.8); §4.1 + Fig. 3-4 (p.8-10); Table 3 (p.11); Table 4 + Table 5 + §4.3 (p.12); §4.3.1 (p.12-14) |
| [K3-blog] | Kimi K3 tech blog, https://www.kimi.com/en/blog/kimi-k3 | "16 out of 896 experts"; "SiTU ... improve activation control" |
| [SA] | SemiAnalysis, "Kimi K3 ..." newsletter (downloaded 2026-10-13) | LatentMoE section: same derivation of K2→K3 equal communication volume; comm/compute ratio (P·F)/(6·m·B)·(1−1/E) independent of d (used only in speaker notes, as commentary) |
| [DSV2] | DeepSeek-AI, "DeepSeek-V2", arXiv 2405.04434 | MLA compresses KV into a latent vector (used only for the analogy in notes) |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 2.8T total (2.78T), 104B active (104.2B) | K3 size | [K3] abstract p.1; Table 1 p.11 |
| 1.04T / 32.6B | Kimi K2 total / active | [K3] Table 1 p.11 |
| 896 routed / 16 active / 2 shared; sparsity 56 | K3 MoE | [K3] §2.3 p.6, Eq. 11 p.7, Table 1 |
| 384 routed / 8 active / 1 shared | K2 MoE | [K3] Table 1 |
| d = 7,168 (K2 and K3) | hidden size | [K3] Table 1; [K3-cfg] |
| ℓ = 3,584 (0.5×) | latent MoE dimension | [K3] Table 1; [K3-cfg] `routed_expert_hidden_size` |
| m = 3,072 (K3), 2,048 (K2) | MoE hidden dimension per expert | [K3] Table 1; [K3-cfg] |
| 93 layers, 1 dense | K3 depth | [K3] Table 1 |
| β1 = 4 (gate), β2 = 25 (up), bound β1β2 = 100 | SiTU-GLU caps | [K3] §2.3.2 p.8, App. B Eq. 19 p.43; [K3-cfg] |
| d/ℓ ≤ 4 preserves quality | LatentMoE compression sweep, 16BT-2BA model | [LMoE] §4.1 + Fig. 3 p.8-9 |
| Hybrid-73BT-8BA, 1T tokens: MMLU-Pro 48.30 → 52.87; MMLU 70.10 → 72.11; Code 51.95 → 55.14; Math 78.32 → 80.19; Commonsense 81.73 → 82.10 (baseline → ℓ-MoE_acc); active 8.09B vs 8.02B; total 72.6B vs 72.8B | accuracy at equal parameters | [LMoE] Table 4 p.12 |
| 95BT-8BA, 300B tokens: MMLU-Pro 29.26 → 34.91 | same, Transformer | [LMoE] Table 3 p.11 |
| Tokens/s/GPU (LatentMoE vs standard) at concurrency 1/4/16/64/128: 181.6/206.6, 528.5/509.8, 1130.8/1204.6, 1569.6/1549.3, 1625.8/1725.9 | H100 ×2, vLLM, FP8 per-tensor, Hybrid-73BT-8BA | [LMoE] Table 5 + §4.3 p.12 |
| "drops by only up to 6%" at higher concurrencies | paper's own reading of Table 5 | [LMoE] §4.3 p.12 |
| K2-1T-LatentMoE: projection overhead within ~9%; iso-accuracy standard MoE (K2-1.35T) 1.24×–3.46× slower; EPM λ ≈ 1.35 | simulator projections | [LMoE] §4.3.1 p.13-14 |
| comm/compute ratio ≈ 9 (Qwen3-235B-A22B on GB200 NVL72, throughput regime) | LatentMoE's analytic model, cited in speaker notes only | [LMoE] §2.2 p.4-5 |
| ≈ 2.5× scaling efficiency over K2 | joint: KDA, AttnRes, Stable LatentMoE, data, recipes | [K3] abstract, §1, §3.2, Fig. 7 |

### Derived numbers (written out)

- **2× all-to-all** (slide 3): per-token dispatch elements ∝ K·d ([LMoE] Design Principle II, §2.2 p.5).
  K2: 8 × 7,168 = 57,344; K3 if full width: 16 × 7,168 = 114,688 = 2×; K3 latent: 16 × 3,584 = 57,344 = K2.
  (Same derivation in [SA].) Analytic: ignores dtype and the shared-expert path, which does not cross EP.
- **Expert size** (slide 7 notes): gate + up + down = 3·(input × m) parameters. K2 expert 3 × 7,168 × 2,048 =
  44.0M; K3 latent expert 3 × 3,584 × 3,072 = 33.0M; a full-width expert with K3's m would be 3 × 7,168 × 3,072
  = 66.1M (2× the latent one).
- **Routed nonlinear width per token K·m** ([LMoE] Principle III): K2 8 × 2,048 = 16,384; K3 16 × 3,072 = 49,152 = 3×.
- **Sanity check**: 92 MoE layers × 896 × 33.0M = 2.72T routed parameters, consistent with 2.78T total.
- **W↓ + W↑ per layer**: 2 × 7,168 × 3,584 = 51.4M parameters (shared by all experts).
- **K3/K2 ratios** (chart `width.py`): from [K3] Table 1 ∆ column: routed experts ↑133% (2.33×), active experts
  ↑100% (2×), expert hidden ↑50% (1.5×), shared experts ↑100% (2×), hidden dim = (1×).
- **+4.6 MMLU-Pro** (slide 10): 52.87 − 48.30 = 4.57; notes: MMLU +2.0 (2.01), Code +3.2 (3.19), Math +1.9 (1.87);
  95B: 34.91 − 29.26 = 5.65 (≈ 5.7).
- **Throughput deltas** (slide 11): 181.6/206.6 − 1 = −12.1%; 528.5/509.8 − 1 = +3.7%; 1130.8/1204.6 − 1 =
  −6.1%; 1569.6/1549.3 − 1 = +1.3%; 1625.8/1725.9 − 1 = −5.8%.
- **SiTU chart** (`situ.py`): analytic plot of Eq. 12 with both branches given the same scalar x (as in [K3]
  Fig. 4): SiTU-GLU f(x) = 4·tanh(x/4)·σ(x) · 25·tanh(x/25); SwiGLU f(x) = x·σ(x)·x. No data.

## Mechanism facts used on slides

- Layer (Eq. 11): u = Σ_{i∈T_k(x)} p_i E_i^routed(W↓x), y = Σ_{j=1..Ns} E_j^shared(x) + W↑ RMSNorm(u);
  u ∈ R^ℓ; routed experts R^ℓ → R^ℓ; shared experts R^d → R^d; p_i from Quantile Balancing routing ([K3] p.7).
- Two failure modes at extreme sparsity: (1) W↓ → gated multi-branch expert FFN → W↑ = "nearly four consecutive
  matrix multiplications", ill-conditioned, with the 2.8T scale → exploding internal activations; (2) balancing
  ~10³ experts ([K3] p.6-7).
- RMSNorm between aggregation and W↑ reduces sensitivity to scale variation (scale of u varies with the
  selected experts and their routing weights) before combining with the shared branch; "consistently improves
  validation loss and downstream benchmarks" ([K3] §2.3.1 p.7; no numbers given).
- SiTU-GLU(x) = (β1 tanh(W_g x/β1) ⊙ σ(W_g x)) ⊙ (β2 tanh(W_u x/β2)); matches SwiGLU to first order near 0;
  recovers SwiGLU as β → ∞; smooth cap keeps nonzero gradients away from saturation, better than hard clamping
  ([K3] §2.3.2, App. B).
- Serving: K3 fuses the latent down-projection with the router into one GEMM, shards latent weights across
  ranks with the all-gather fused into the GEMM epilogue (multimem store), overlaps with shared-expert compute;
  routed-expert decode kernel is token-centric (WarpDecode) ([K3] §5.4.2 p.24-25).
- LatentMoE: W↓ ∈ R^{ℓ×d}, experts' FC1/gate ∈ R^{m×ℓ}, FC2 ∈ R^{ℓ×m}, W↑ ∈ R^{d×ℓ}; router and shared experts at
  full d; communication and per-expert weight traffic reduced by d/ℓ; ℓ-MoE_acc scales N and K by d/ℓ
  ([LMoE] §3).

## Discrepancies (Notion row vs primary source)

1. **"RMSNorm on routed experts"**: K3 applies one RMSNorm to the *aggregated* routed latent u (after the
   weighted sum over the selected experts, before W↑), not inside or per routed expert (Eq. 11; [K3-cfg]
   `routed_expert_norm` applied to the combined output). Slides follow K3.
2. **"an MLA analogue"**: K3 cites LatentMoE (NVIDIA, arXiv 2601.18089) as the origin of the latent routed path
   and does not describe it as an MLA analogue; the analogy (compress to a latent, expand later) is Notion's.
   The deck credits LatentMoE and mentions MLA only as an analogy in the notes.
3. **"Trains stably"**: K3 states the stabilizers suppress activation explosion but publishes no loss or
   activation curves isolating them; "trains stably" is an inference from the completed 2.8T run.
4. **Notion body** lists DeepSeek-V2/MLA as a paper but not LatentMoE, which is the actual base design.
