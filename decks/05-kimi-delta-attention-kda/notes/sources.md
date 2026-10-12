# Sources: deck 05, Kimi Delta Attention (KDA)

## Sources read

| Key | Source | Where | What was used |
|---|---|---|---|
| K3 | Kimi Team, "Kimi K3: Open Frontier Intelligence", arXiv 2607.24653 (https://arxiv.org/abs/2607.24653); cached text `$CACHE/arxiv-2607.24653.txt` | Abstract; §1; §2 + Fig. 2; §2.1, §2.1.1 Eq. 1-6 + Fig. 3; §2.1.2 Eq. 7; §3.2 + Fig. 7; Table 1; §3.4; §5.1 (5.1.1, 5.1.2); §5.4.1; §5.4.2 | Model scale, layer pattern, KDA equations, lower-bounded decay, full-rank gate, NoPE, scaling efficiency, context curriculum, system costs |
| KL | Kimi Team, "Kimi Linear: An Expressive, Efficient Attention Architecture", arXiv 2510.26692 (https://arxiv.org/abs/2510.26692); PDF in `$SCRATCH/research/05/` | Abstract; Fig. 1 (a,b); §2.2; §3.1 Eq. 1-9; §3.2 + Fig. 2; §4; §5.1 + Fig. 4; §5.2 Table 1; §5.3 Fig. 5; §5.4; Tables 3, 5; §5.6 Fig. 7; §6.1; §6.2; §6.3 Eq. 13-14 | Fair-comparison quality and speed numbers, head size, chunk size, hybrid-ratio ablation, delta rule as online gradient descent, KDA as positional encoding |
| GDN | Yang, Kautz, Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule", ICLR 2025, arXiv 2412.06464 (https://arxiv.org/abs/2412.06464) | Abstract; §1 (memory collisions); §2-3 gated delta rule | Capacity bound (orthogonal key-value pairs bounded by dimension), scalar gate of GDN/Mamba2 |
| DN | Yang et al., "Parallelizing Linear Transformers with the Delta Rule over Sequence Length", arXiv 2406.06484 | skimmed for chunkwise WY/UT background | background only (no numbers on slides) |
| KLrepo | https://github.com/MoonshotAI/Kimi-Linear README (via `gh api`, saved in `$SCRATCH/research/05/`) | Overview | Confirms 75% KV-cache reduction and 6x / 6.3x decoding claims; KDA kernels in flash-linear-attention |

`$CACHE` = scratchpad `research/cache`, `$SCRATCH` = session scratchpad.

## Numbers on slides (slide number = logical slide, title slide = 1)

| Slide | Number | Source location | Note |
|---|---|---|---|
| 2 | 69 KDA + 24 Gated MLA layers, 93 layers | K3 Table 1 ("#Layers 93", "Attention-Layer Composition 69 KDA + 24 MLA") | figure pattern 23 x [3 KDA + 1 MLA] + 1 final MLA is derived from §2.1 ("3 KDA followed by 1 Gated MLA ... An additional Gated MLA layer is placed at the end"): 23 x 4 + 1 = 93, 23 x 3 = 69 |
| 2 | 2.8T parameters, 104B active, 1M context | K3 abstract; Table 1 (2.78T, 104.2B) | |
| 3 | 11.48 ms per output token (MLA at 1M) | KL Fig. 1 caption ("6.3x faster TPOT (1.84ms vs. 11.48ms) than MLA at 1M tokens") | MLA baseline in the Kimi Linear 48B-A3B setting (§5.6: "all models are based on the Kimi Linear 48B setting, with the same number of layers and attention heads") |
| 3 (notes) | attention FLOPs 2 T^2 d_h per head; KDA 6 T d_h^2 + 3 T C d_h + T C^2 | KL §6.3 Eq. 13-14 | |
| 3 (notes) | K2: 61 MLA layers, 128K training context; K3: 1M (8x) | K3 Table 1 | |
| 4 | at most 128 orthogonal keys fit in 128-dim keys | GDN §1 ("the number of orthogonal key-value pairs they can store is bounded by the model's dimensionality ... memory collisions"); KL §4 (d_k = d_v = 128) | 128 is the Kimi Linear head size; K3 does not state its head dimension |
| 5, 7 | S_t = (I - beta k k^T) Diag(alpha) S_{t-1} + beta k v^T; o = S^T q | K3 Eq. 1; KL Eq. 1 | rearranged form S' = Diag(alpha) S; S <- S' + beta k (v - S'^T k)^T is algebra on Eq. 1 |
| 5 | delta rule = gradient step on 1/2 ||S^T k - v||^2 | KL §2.2 | |
| 6 | q,k ShortConv+Swish+L2Norm; v ShortConv+Swish; beta Sigmoid; alpha low-rank; head-wise RMSNorm; sigmoid output gate, full rank in K3 | K3 Eq. 2, 5, 6; Fig. 2 | |
| 8 (per-channel) | 256-2,048 tokens (synthetic tasks) | KL §5.1 + Fig. 4 ("KDA consistently achieves the highest accuracy as the sequence length increases from 256 to 2,048 tokens"; Mamba2 "fails on all tasks") | |
| 8 (chart) | retention curves alpha^distance | analytic, illustrative constant decays g = -0.5, -0.01, -0.0002 (alpha = e^g); not measured | caption says "analytic" |
| 9 | C = 64 tokens per chunk | KL §6.3 ("fixed chunk size C = 64 in the gated delta rule") | K3 does not state its chunk size |
| 10 | g_min = -5; alpha > e^-5 ~ 6.7e-3; 16-token tiles; cumulative log-decay in (-80, 0); 1/Gamma < e^80 within BF16 range | K3 §2.1.1 "Lower-bounded decay", Eq. 5, Fig. 3 | |
| 11 | validation PPL 0:1 = 5.77, 1:1 = 5.66, 3:1 = 5.65, 7:1 = 5.70, 15:1 = 5.82 | KL Table 1 (validation PPL column) | training PPL: 9.45 / 9.29 / 9.23 / 9.23 / 9.34 (not on slide) |
| 12 | 1.4T tokens, 48B total / 3B active | KL §5.4, §5.4.1 | |
| 12 | MMLU-Pro 47.2 (MLA) / 47.9 (GDN-H) / 51.0 (Kimi Linear) | KL Table 3 (base models) | |
| 12 | RULER@128K 81.3 / 80.5 / 84.3; long-context average 52.2 / 51.2 / 54.5 | KL Table 5 | |
| 12 | ~1.16x compute efficiency (scaling-law fit) | KL §5.3 + Fig. 5 | |
| 13 | 4.8x @256K, 5.7x @512K, 6.3x @1M TPOT speed-up | KL Fig. 1b annotations (read from the rendered figure) | larger batches enabled by the smaller cache (Fig. 1 caption) |
| 13 | 1.84 ms vs 11.48 ms | KL Fig. 1 caption | |
| 13 | up to 75% less KV cache | KL abstract; §1 | |
| 13 | batch size 1: 2.2x decode, 2.9x prefill at 1M (2.3x prefill at 512K, 1.8x decode at 512K) | KL Fig. 7a/7b annotations; Fig. 7 caption ("batch size = 1") | see Discrepancies (text says 2.3x for decode) |
| 14 | ~2.5x overall scaling efficiency over K2 | K3 abstract; §1; §3.2 + Fig. 7 | joint effect of KDA, AttnRes, Stable LatentMoE, refined data and training recipes |
| 14 | 1M context without RoPE rescaling or interpolation | K3 §3.4 "Positional encoding" | |
| 14 (notes) | context 8K -> 64K (pre-training) -> 256K -> 1M (cooldown), four stages | K3 §3.4 | |
| 15 | prefill comparable to MLA at 4K-16K | KL §5.6 | |
| 15 | 24 MLA layers keep a KV cache | K3 Table 1; §5.4.1 | |
| 15 (notes) | prefix reuse at any 512-token boundary via KDA state checkpoints | K3 §5.4.1 | |
| 15 (notes) | MTP speculative decoding: cache projected inputs and replay | K3 §5.4.2 | |
| 16 | 11.48 ms, 6.3x, 69 of 93 | as above | |

## Discrepancies

1. **Notion "Impact" credits 2.5x to KDA "with AttnRes and Stable LatentMoE".** K3 (abstract, §2, §3.2) credits the ~2.5x
   scaling-efficiency gain to KDA, AttnRes, Stable LatentMoE **and refined data and training recipes** together; there is
   no KDA-only ablation at K3 scale. The slide states the joint attribution.
2. **Notion "three KDA layers alternate with one gated-MLA layer".** K3 also adds one extra Gated MLA layer at the end of
   the backbone (§2.1), so the count is 69 KDA + 24 MLA (Table 1), not an exact 3:1. Slides show both.
3. **Notion "State does not grow with length, so 1M context needs no RoPE interpolation".** K3 §3.4 attributes the
   no-interpolation property to NoPE: position comes from KDA's recurrent gating/decay, so there is no RoPE to rescale.
   The fixed state size is a memory property, not the reason. Slides use K3's reasoning.
4. **Internal inconsistency in Kimi Linear (batch-1 decode at 1M):** Fig. 7b annotates 2.2x, while §6.3 text says
   "Kimi Linear achieves a 2.3x speedup at a 1M token context". Slides use the figure value (2.2x) and the notes mention
   the text value. Kimi Linear's abstract rounds the Fig. 1b 6.3x to "up to 6x".
5. **Scope of measured numbers.** All speed and fair-comparison quality numbers come from the Kimi Linear 48B-A3B model
   (KDA before the K3 changes: unbounded decay, low-rank output gate). K3 publishes no KDA-specific speed numbers.
   The deck says so on the costs slide.
