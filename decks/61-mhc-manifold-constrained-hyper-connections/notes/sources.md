# Sources — deck 61: mHC (manifold-constrained hyper-connections)

## Sources read

| Tag | Source | Where |
|---|---|---|
| [mHC] | Xie et al. (DeepSeek-AI), "mHC: Manifold-Constrained Hyper-Connections", arXiv 2512.24880v2 (5 Jan 2026). https://arxiv.org/abs/2512.24880 (cached `arxiv-2512.24880.txt`) | Abstract; §1 (p.3-4, eq. 1-4, "6.7%"); §3 (p.5-6, eq. 5, Table 1); §3.1 (p.6-7, Fig. 2-3); §3.2 + Table 2 (p.7-8); §4.1 (p.8-9, eq. 6, three properties); §4.2 (p.9, eq. 7-9, t_max = 20); §4.3.1-4.3.3 (p.9-11, eq. 10-20, Table 3, Fig. 4); §5.1-5.4 (p.12-14, Fig. 5-8, Table 4); App. A Table 5 (p.19) |
| [V4] | DeepSeek-AI, "DeepSeek-V4", arXiv 2606.19348 (cached) | §2 (p.6-7); §2.2 eq. 1-8 (p.7-8); §2.4 (p.14: AdamW for mHC static biases and gating factors); §3.3 (p.19: mHC matmul output dim 24, deterministic split-k); §3.4.2 (p.20: 6.7% of the overlapped 1F1B stage); §4.2.1 (p.24-25: n_hc = 4, t_max = 20, hidden 7168 / 4096) |
| [AttnRes] | Kimi Team, "Attention Residuals", arXiv 2603.15031 (cached) | p.8 Table 1 (mHC per-layer residual I/O 34d at m = 4, cross-check); §5.1 (p.9) |
| [lite] | Yang & Gao, "mHC-lite: You Don't Need 20 Sinkhorn-Knopp Iterations", arXiv 2601.05732 — abstract only, via the arXiv export API (`$SCRATCH/research/61/mhc-lite-abs.xml`) | abstract |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| n = 4 (n_hc), residual stream n × C | expansion rate in V4-Pro and V4-Flash, and in the mHC experiments | [V4] §4.2.1; [mHC] §5.1, Table 5 |
| t_max = 20 Sinkhorn-Knopp iterations | practical value | [mHC] §4.2 p.9; [V4] §2.2 p.8 and §4.2.1 |
| H_pre = σ(·), H_post = 2σ(·), H_res = Sinkhorn-Knopp(·) | constraints | [mHC] eq. 8; [V4] eq. 6-8 |
| α gates initialized to small values (0.01 in the experiments) | gating | [mHC] eq. 7, Table 5 |
| ‖H_res‖₂ ≤ 1; closed under multiplication; Birkhoff polytope = convex hull of permutation matrices | properties | [mHC] §4.1 p.8; [V4] §2.2 p.8 |
| H_res ablation: −0.022 of the −0.027 loss gap of full HC | why H_res matters | [mHC] Table 1 p.6 |
| ~3000 | peak Amax gain magnitude of HC's composite residual mapping, 27B model | [mHC] §3.1 p.6, Fig. 3(b) |
| loss surge around step 12k for HC (27B) | instability | [mHC] §3.1 p.6, Fig. 2 |
| ≈ 1.6 | maximum composite gain of mHC (27B); "three orders of magnitude" below HC | [mHC] §5.4 p.14, Fig. 7(b) |
| 27B MoE model: 30 layers (60 sublayers), 72 routed experts, 4.14B active; n = 4 | experiment model | [mHC] Table 5 p.19; Fig. 3 caption |
| HC residual I/O per token (forward): reads (5n+1)C + n² + 2n, writes (3n+1)C + n² + 2n; standard residual reads 2C, writes C | memory wall | [mHC] Table 2 p.8 |
| fused post/res kernel: reads (3n+1)C → (n+1)C, writes 3nC → nC | kernel fusion | [mHC] §4.3.1 p.10 |
| φ ∈ R^{nC × (n² + 2n)}; output dim 24 at n = 4 | one projection for all three mappings | [mHC] eq. 10; [V4] §3.3 p.19 ("output dimension of only 24") |
| recompute block L_r* ≈ √(nL/(n+2)); aligned with pipeline stages; stores x_l0 every L_r layers; transient (n+2)C·L_r | recomputation | [mHC] §4.3.2 eq. 20, Table 3 |
| n-fold more pipeline communication in HC | PP cost | [mHC] §3.2 p.7 |
| 6.7% | extra training time of mHC at n = 4 (in-house large-scale) | [mHC] §1 p.4, §4.3 p.9 |
| 6.7% of the overlapped 1F1B pipeline stage | V4 wall-time overhead | [V4] §3.4.2 p.20 |
| −0.021 | final training-loss reduction of mHC vs baseline, 27B | [mHC] §5.2 p.12 |
| 27B benchmarks Baseline / HC / mHC: BBH 43.8 / 48.9 / 51.0; DROP 47.0 / 51.6 / 53.9; GSM8K 46.7 / 53.2 / 53.8; HellaSwag 73.7 / 74.3 / 74.7; MATH 22.0 / 26.4 / 26.0; MMLU 59.0 / 63.0 / 63.4; PIQA 78.5 / 79.9 / 80.5; TriviaQA 54.3 / 56.3 / 57.6 | downstream results | [mHC] Table 4 p.13 |
| 34d | mHC/HC per-layer residual I/O at m = 4 (unfused), cross-check | [AttnRes] Table 1 p.8 |

## Derived numbers (labelled "derived", "analytic" or "toy" on slides)

| Derived | Formula | Value |
|---|---|---|
| HC residual I/O at n = 4 | (5n+1)C + (3n+1)C = 21C + 13C (+ 2·24 scalars) | **34C vs 3C** for a plain residual (matches [AttnRes] 34d) |
| post + res + merge elements, unfused vs fused, n = 4 | reads (3n+1)C = 13C → (n+1)C = 5C; writes 3nC = 12C → nC = 4C | **25C → 9C** (plain residual merge: 3C) — chart `io_merge.py` |
| V4-Pro mHC projection shape | nC = 4 × 7168 = 28,672 rows × 24 columns | **28,672 × 24** (V4-Flash: 16,384 × 24) |
| Sinkhorn toy example (slide figure) | made-up 4×4 input H̃; M⁽⁰⁾ = exp(H̃); M⁽ᵗ⁾ = T_r(T_c(M⁽ᵗ⁻¹⁾)) computed in Python (`figures/gen_sinkhorn.py`) | column sums 5.74 / 4.50 / 5.72 / 4.31 → 1.029 / 0.974 / 1.010 / 0.987 after 1 iteration → 1.000 after 20 |
| closure toy example (insight figure) | A = 0.7 I + 0.1 (J − I), B = ½ (I + cyclic shift); A·B computed | A·B rows and columns all sum to 1 |

## Discrepancies (primary source wins on the slides)

1. **"Compound gain at depth 64 goes from 1e16 to about 1"** (Notion). [mHC] reports a peak composite gain of about
   **3000** for HC and a maximum of about **1.6** for mHC in the 27B model (30 layers = 60 sublayers); no 1e16 value or
   depth-64 measurement appears in the paper. Slides use ~3000 → ≤ 1.6.
2. **"A single Sinkhorn iteration suffices"** (Notion). Both [mHC] and [V4] use **t_max = 20**; [mHC] §5.4 notes that
   the finite iteration count already makes the backward gain deviate slightly from 1. The follow-up mHC-lite [lite]
   argues for *replacing* Sinkhorn with explicit convex combinations of permutation matrices, not for one iteration.
   Slides use 20; the toy figure shows column sums after 1 iteration only as an illustration of convergence.
3. **"Also adopted by GLM-5.3"** (Notion). The cached GLM-5 report (arXiv 2602.15763) has no mention of mHC or
   hyper-connections; no GLM-5.3 source was available. The slides do not claim GLM adoption.
4. **"Eliminates HC's gradient explosion at 27B"** — consistent with [mHC] Fig. 2/5 (HC loss surge near step 12k,
   gradient-norm spikes; mHC stable).
