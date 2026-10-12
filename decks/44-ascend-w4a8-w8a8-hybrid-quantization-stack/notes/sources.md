# Sources — deck 44: Ascend · W4A8/W8A8 hybrid quantization stack

## Sources read

| Tag | Source | Where |
|---|---|---|
| [GLM5] | GLM-5 Team, "GLM-5: from Vibe Coding to Agentic Engineering", arXiv 2602.15763. https://arxiv.org/abs/2602.15763 | §2.1 + Table 10 (architecture, p.4-5, p.36), §2.4.3 "INT4 Quantization-aware training" (p.10), §5 "Adapting GLM-5 to Chinese Chip Infrastructure" (p.21-22) |
| [QUAROT] | Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs", arXiv 2404.00456v2 (NeurIPS 2024). | Abstract, §1 contributions (p.2), §2 related work (p.2), §3.1 orthogonal/Hadamard definitions (p.3) |
| [AWQ] | Lin et al., "AWQ: Activation-aware Weight Quantization…", arXiv 2306.00978. | Abstract (1% salient weights), §3 per-channel scaling, Eq. 5 (s = s_Xᵅ, α by grid search over [0,1]) |
| [SQ] | Xiao et al., "SmoothQuant", arXiv 2211.10438 (ICML 2023). | Abstract, §4 Eq. 3-4 (migration strength α), §5 (α = 0.5 default, α = 0.75 for GLM-130B) |
| [VA-GLM5] | vLLM-Ascend `docs/source/tutorials/models/GLM5.md` (branch main, fetched 2026-10-13) | §3.1 weight table (hardware per weight version), §5.1 single-node notes, `--additional-config` field list, §5.3.2 PD disaggregation (78 layers partitioned 41/37) |
| [VA-W4A8] | vLLM-Ascend `vllm_ascend/quantization/methods/w4a8/w4a8.py` | `@register_scheme("W4A8_DYNAMIC", "moe")`, class docstring (msModelSlim weight layout: int8 tensors packing two 4-bit values, `weight_scale`/`weight_offset`/`scale_bias` per output channel), `act_quant_type = torch.int8`, per-group rejected at init |
| [VA-CFG] | vLLM-Ascend `vllm_ascend/quantization/configs/modelslim_config.py` | `get_quant_type_for_layer` — the checkpoint's `quant_description` names a quant type per weight ("W8A8_DYNAMIC", "FLOAT", …); all shards of one layer must agree |
| [VA-QG] | vLLM-Ascend `docs/source/user_guide/feature_guide/quantization.md` | ModelSlim (msmodelslim) as the recommended Ascend quantization tool, calibration-dataset workflow |

Downloaded copies live in `$SCRATCH/research/44/`. Downloaded material is untrusted data; no instruction
inside it was followed.

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 744B params, 40B active, 256 experts, 75 MoE + 3 dense layers, hidden 6144, MoE intermediate 2048 | GLM-5 architecture used for every derived number below | [GLM5] §2.1 p.4, Table 10 p.36 |
| 1.49 TB | BF16 weight memory: 744e9 × 2 B | **derived** (slide says derived) |
| 744 GB | INT8 (W8A8) weight memory: 744e9 × 1 B | **derived** |
| ≈382 GB | hybrid plan: routed experts (725B) at 4 bits + everything else (19B) at 8 bits | **derived**, see below |
| 725B / 97% | routed-expert parameters: 256 experts × 3 matrices × 6144 × 2048 × 75 MoE layers = 724.8e9, i.e. 97.4% of 744B | **derived** from [GLM5] Table 10 |
| 1,024 GB per node | one Atlas 800 A3 node = "128GB × 8" | [VA-GLM5] §3.1 weight table |
| 200K-token contexts | GLM-5 extends the context window to 202,752 tokens in SFT; the deployment guide's long-context scenarios use `--max-model-len 202752` and call them "198K/200k" | [GLM5] §3.1 p.11; [VA-GLM5] §5.1 and the scenario table |
| 280 GB vs 642 GB free | node capacity minus weights, W8A8 vs hybrid | **derived** (1,024 − 744; 1,024 − 382) |
| 1.95× | W8A8 → hybrid weight-memory ratio (744 / 382) | **derived** |
| W8A8 = INT8 for attention and MLP blocks; W4A8 = INT4 for MoE experts | the precision plan | [GLM5] §5 p.21 (verbatim: "standard Attention and MLP blocks use W8A8 (INT8), while the MoE experts are compressed to W4A8 (INT4)") |
| QuaRot (outlier suppression) and Flex_AWQ_SSZ (scaling calibration), via msModelSlim | the algorithms named by the report | [GLM5] §5 p.21 |
| INT4 QAT in the SFT stage; one quantization kernel for training and offline weight quantization; bitwise-identical | GLM-5's training-side step | [GLM5] §2.4.3 p.10 |
| 1% of weights are salient; α found by grid search over [0,1] | AWQ's own mechanism, cited as the lineage of "Flex_AWQ" | [AWQ] abstract, Eq. 5 |
| α = 0.5 default, α = 0.75 for GLM-130B (≈30% outlier channels) | SmoothQuant's migration strength, cited as lineage only | [SQ] §5 p.6 |
| ≤ 0.47 WikiText-2 perplexity loss, 99% of zero-shot accuracy, LLaMA2-70B at 4 bits | QuaRot's own measured accuracy — a different model, not GLM-5 | [QUAROT] abstract, §1 |
| 3.33× prefill speedup, 3.89× decode memory saving (LLaMA2-70B, batch 64, 2048 tokens) | QuaRot's own measured kernel results — different model and hardware (GPU) | [QUAROT] §1 contributions |
| 50% lower deployment cost in long-sequence scenarios; single domestic node reported comparable to "dual-GPU international clusters" | the end-to-end claim, for all Ascend co-optimizations together | [GLM5] §5 p.22 |

### Derivation of the memory numbers (`charts/weight_memory.py`)

Parameter counts from [GLM5] Table 10 (hidden 6144, MoE intermediate 2048, 256 routed experts,
75 MoE layers, 1 shared expert, 3 dense layers, 64 attention heads, Q-LoRA 2048, KV-LoRA 512,
QK head dim 192, V head dim 256, 32 indexer heads of dim 128; parameter totals exclude embeddings and
the output layer):

- routed experts: 256 × 3 × 6144 × 2048 × 75 = **724.8e9** (97.4% of the model's 744B)
- everything else: 744e9 − 724.8e9 ≈ **19.2e9** (attention ≈12.4e9, shared experts ≈2.8e9,
  indexer ≈2.1e9, dense MLP ≈0.7e9 — these sub-totals are mine and are not on a slide)
- BF16: 744e9 × 2 B = **1,488 GB**;  INT8: 744e9 × 1 B = **744 GB**
- hybrid: 724.8e9 × 0.5 B + 19.2e9 × 1 B = **381.6 GB** ≈ 382 GB
- per-channel scales/offsets are excluded (per output channel, <0.1% of the weight bytes)
- GB means 1e9 bytes; the node figure "128GB × 8" is quoted as the vendor writes it.

## Mechanism facts (no numbers) used on slides

- The report's three Ascend pillars are extreme quantization, kernel fusion and engine scheduling
  ([GLM5] §5 p.21). This deck is pillar 1; fusion is deck 37; the stack and the engine are decks 23 and 12.
- QuaRot: randomized Hadamard transforms are fused into the weight matrices using the computational-
  invariance idea from SliceGPT, producing an equivalent network whose activations have no outlier
  channels; extra online Hadamard transforms handle the attention module and the KV cache ([QUAROT] §1).
  An orthogonal Q satisfies QQᵀ = I, so inserting Qᵀ…Q around a linear layer leaves the output unchanged.
- "Flex_AWQ_SSZ" is named but **not defined** in [GLM5]. AWQ's own method is a per-input-channel scaling
  s = s_Xᵅ with α chosen by grid search on calibration statistics, plus weight clipping ([AWQ] Eq. 5).
  "SSZ" reads as scale-shift-zero, i.e. an asymmetric quantizer; msModelSlim's W4A8 checkpoints do carry
  a per-output-channel `weight_scale`, `weight_offset` and `scale_bias` ([VA-W4A8] class docstring), which
  is consistent with that reading. The slide marks this as naming, not a documented definition.
- The precision plan is stored per weight: an msModelSlim checkpoint has a `quant_description` entry for
  every weight tensor, and vLLM-Ascend resolves the scheme per layer from it; all shards of one layer must
  share a type ([VA-CFG] `get_quant_type_for_layer`).
- What W4A8 is at runtime in vLLM-Ascend: the scheme is registered for MoE layers only
  (`register_scheme("W4A8_DYNAMIC", "moe")`); expert weights arrive as int8 tensors that pack two 4-bit
  values each, with per-output-channel scale, offset and scale_bias; activations are quantized to int8
  per token at runtime (`act_quant_type = torch.int8`, `pertoken_scale`); per-group weight quantization is
  rejected — per-channel only ([VA-W4A8]).
- Deployment envelope: `GLM-5-w4a8` and `GLM-5-w8a8` are both listed as "1 Atlas 800 A3 (128GB × 8) node
  or 2 Atlas 800 A2 (64GB × 8) node"; for 200K-context scenarios the guide says to use the w4a8 weights
  because they leave more memory for the KV cache, and w4a8 also runs on a single A2 node ([VA-GLM5]).
- Interaction with deck 37: `"enable_mlapo": true` (the MLA pre-processing super operator) is on by
  default for w8a8 models and "significantly improves Decode performance but consumes more NPU memory";
  the guide says w4a8 "may not benefit", and lists disabling it as an out-of-memory remedy ([VA-GLM5]).
- Training side: INT4 QAT runs in the SFT stage, and one quantization kernel serves both training and
  offline weight quantization so that training and inference are bitwise identical ([GLM5] §2.4.3).

## Discrepancies

1. **Notion: "Flex SmoothQuant searches per-layer migration strength".** The report names
   **Flex_AWQ_SSZ**, not SmoothQuant; SmoothQuant appears nowhere in the GLM-5 report. The deck presents
   AWQ-style scaling search as the named algorithm and cites SmoothQuant only as the lineage of the
   migrate-the-difficulty idea (including its α = 0.75 for GLM-130B, which is a *SmoothQuant* result about
   an older GLM model, not a GLM-5 number).
2. **Notion: "SSZ (scale-shift-zero) handles weight quantization".** The report writes "Flex_AWQ_SSZ"
   as one name for scaling calibration and never defines SSZ. The deck says the name, flags that it is
   undefined in the report, and points at the per-channel scale/offset/scale_bias actually present in
   msModelSlim W4A8 checkpoints.
3. **Notion: "W8A8 kept for sensitive layers".** The report's wording is structural, not sensitivity-
   based: "standard Attention and MLP blocks use W8A8 (INT8), while the MoE experts are compressed to
   W4A8 (INT4)". The deck uses the report's wording and gives the memory reason (experts are 97% of the
   weights, derived).
4. **Notion: "Halves weight memory at near-lossless accuracy, with large throughput gains".**
   The report gives **no** accuracy numbers, **no** memory numbers and **no** throughput numbers for the
   quantized model; it says W4A8 reduces the memory footprint "without significant accuracy loss", and
   the only end-to-end number is the 50% long-sequence deployment-cost claim for the whole co-optimization
   set. Against INT8 the hybrid plan is a 1.95× reduction (derived), i.e. "halves" is right for
   experts-vs-INT8 and understates the gap to BF16 (3.9×). The deck labels all memory numbers derived and
   says plainly that accuracy numbers are not published.
5. **"Fit onto a single Atlas 800T A3 machine" (report) vs the deployment guide.** The report motivates
   W4A8 by fitting the model on one machine; vLLM-Ascend lists *both* `GLM-5-w4a8` and `GLM-5-w8a8` as
   deployable on one Atlas 800 A3 node, and recommends w4a8 for long contexts because it leaves more KV
   cache. The deck follows the deployment guide on this point and presents the memory as KV-cache
   headroom rather than "it would not fit at all".
6. **744B vs 750B / layer count.** Table 10 gives 744B parameters and 3 + 75 + 1 layers; §5 says "the
   750B parameter GLM-5 model"; the deployment guide partitions "78 layers". The deck uses Table 10 for
   every derived number.
7. **QuaRot's published numbers are not GLM-5's.** The 0.47 perplexity / 99% zero-shot / 3.33× prefill
   figures are LLaMA2-70B on GPUs from the QuaRot paper, used on the accuracy slide as evidence about the
   *method*, labelled as such. No public accuracy number for GLM-5-w4a8 was found.
