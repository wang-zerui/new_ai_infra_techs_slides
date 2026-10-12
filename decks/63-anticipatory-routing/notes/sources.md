# Sources — deck 63: anticipatory routing

## Sources read

| Tag | Source | Where |
|---|---|---|
| [V4] | DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence", arXiv 2606.19348v1. https://arxiv.org/abs/2606.19348 (cached text) | Abstract p.1; §2.1 p.7 (DeepSeekMoE routing in V4); §2.2 p.7-8 (mHC stability); §4.2.1-4.2.2 p.24-26 (configs, training setup); §4.2.3 p.26 (Anticipatory Routing, SwiGLU Clamping); §6 p.44 (limitations) |
| [V3] | DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv 2412.19437 (cached) | §2.1.2 p.8-9, eqs. 12-16 (top-K routing, bias term used only for routing, gating values from the original affinity) |
| [DG-MOE] | DeepGEMM @7f2a703 `tests/test_mega_moe.py` and `sm100_fp8_fp4_mega_moe.cuh` (read for deck 14): SwiGLU with `activation_clamp` in the fused MoE kernel | speaker notes only |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 1.6T / 49B | DeepSeek-V4-Pro total / activated parameters | [V4] abstract; §4.2.1 |
| 384 routed experts, top-6, 1 shared expert, 61 layers | V4-Pro MoE configuration | [V4] §4.2.1 p.25 |
| first 3 MoE layers use hash routing | no learned router there | [V4] §4.2.1, §2.1 |
| 33T tokens; batch up to 94.4M tokens | V4-Pro pre-training | [V4] §4.2.2 p.25 |
| 32T tokens; batch 75.5M tokens | V4-Flash pre-training | [V4] §4.2.2 p.25 (speaker notes) |
| bias update speed 0.001; balance-loss weight 0.0001 | auxiliary-loss-free balancing settings | [V4] §4.2.2 p.25-26 (speaker notes) |
| ≈ 20% | additional wall-clock time of anticipatory routing (while it is active), after orchestrating the pipeline and overlapping with EP communication | [V4] §4.2.3 p.26 |
| "negligible" overall overhead | with spike-triggered activation | [V4] §4.2.3 p.26 (qualitative) |
| [−10, 10]; gate ≤ 10 | SwiGLU clamping: linear component clamped to [−10, 10], gate component capped at 10, throughout V4-Flash and V4-Pro training | [V4] §4.2.3 p.26 |
| overhead = 20% × f | **analytic**: if anticipatory mode is active for a fraction f of steps and costs ≈ 20% extra wall-clock while active, the run-level overhead is ≈ 0.2·f. Plotted in `charts/overhead.py`; f values are illustrative, not reported | derived from [V4] §4.2.3 |

## Mechanism facts (no numbers) used on slides

- Routing in V4 (inherited from V3 with changes): affinity s_{i,t} = Sqrt(Softplus(u_tᵀ e_i)) (V4 replaces V3's
  Sigmoid); top-K selection uses s_{i,t} + b_i with a per-expert bias b_i that is only used for routing; the gating
  value multiplied with the expert output comes from the original affinity, normalized over the selected experts
  ([V3] eqs. 12-16; [V4] §2.1).
- Anticipatory Routing ([V4] §4.2.3): decoupling the synchronous updates of the backbone and the routing network
  improves stability; at step t the current parameters θ_t are used for feature computation, but routing indices
  are computed and applied using historical parameters θ_{t−Δt}; to avoid loading parameters twice, the data of step
  t is fetched in advance at step t−Δt and the routing indices are computed and cached then ("anticipatorily");
  pre-computing indices needs a single forward pass over the data; pipeline execution and EP-communication overlap
  bound the extra wall-clock time to about 20%; an automatic detection mechanism triggers a short rollback and turns
  Anticipatory Routing on only when a loss spike occurs, and after a period the system returns to standard training;
  result: loss spikes averted with negligible overall overhead and without compromising model performance.
- Motivation ([V4] §4.2.3): rollbacks restored the training state only temporarily and did not prevent recurrence;
  spikes were consistently tied to outliers in the MoE layers, and the routing mechanism appeared to exacerbate the
  outliers; two directions: break the vicious cycle induced by routing, and suppress anomalous values directly.
- SwiGLU Clamping is the second technique; both lack a comprehensive theoretical understanding ([V4] §4.2.3, §6).
- The paper does not state Δt, the detection criterion, the rollback length, or how long the mode stays on.
- The paper says only that routing *indices* come from θ_{t−Δt}; it does not say whether the gating values (affinity
  weights that multiply expert outputs) are recomputed with θ_t. Slides say "which experts" for the θ_{t−Δt} part
  and flag gate values as unspecified.
- Other V4 stability components named elsewhere in the paper: mHC (doubly stochastic residual mapping, spectral norm
  ≤ 1, §2.2) and Muon ("faster convergence and greater training stability", abstract). They are not described as part
  of the spike mitigation in §4.2.3.

## Discrepancies (Notion row vs primary sources)

1. **"Asynchronous router parameters" / "routing decisions use cached older ones".** [V4] computes routing indices
   with the historical *network* parameters θ_{t−Δt}, i.e. the backbone layers that produce the router input as
   well as the router itself, in a forward pass over step t's data run at step t−Δt; what is cached is the indices,
   not router parameters. Slides follow the paper.
2. **"Breaking the anomaly → routing drift → larger anomaly feedback loop."** [V4] says spikes are tied to MoE
   outliers, routing "appears to exacerbate" them, and the technique breaks "the vicious cycle induced by routing";
   it does not describe the loop's mechanism and states the principles are insufficiently understood. Slides draw
   the loop as a hypothesis.
3. **"With mHC and Muon, suppresses loss spikes and divergence."** In [V4] §4.2.3 the companion technique is SwiGLU
   clamping; mHC and Muon are stability-related elsewhere in the paper but not tied to spike mitigation. No
   divergence results are reported. Slides pair it with SwiGLU clamping and mention mHC/Muon as separate decks.
4. Notion omits the key operating detail: the mode is spike-triggered (with a short rollback) and costs ≈ 20%
   wall-clock while active; always-on use is not what DeepSeek does.
