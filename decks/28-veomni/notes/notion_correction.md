# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Container entry: ByteDance Seed's PyTorch-native omni-modal training framework; its core claim is decoupling parallelism/communication logic from model definitions, providing a composable distributed recipe zoo (FSDP1/2, Ulysses SP, EP) plus dynamic batching, DCP checkpointing and multiple hardware backends.
**After:** Container entry: ByteDance Seed's PyTorch-native omni-modal training framework; its core claim is decoupling parallelism/communication logic from model definitions, providing a composable distributed recipe zoo (FSDP2/HSDP, Ulysses SP, EP; the paper also used FSDP1, since removed) plus dynamic batching, checkpointing (ByteCheckpoint in the paper, PyTorch DCP in the repo) and multiple hardware backends.
**Why:**
- The paper integrates FSDP1 and FSDP2 ([VO] §3.2.1), but the repo says FSDP1 "has since been deprecated upstream and removed from VeOmni" ([DOC-EP]).
- The paper uses ByteCheckpoint ([VO] §3.2.5); the open-source repo uses PyTorch DCP ([README], [DOC-CKPT]).

## Impact / Measured Results
**Before:** More than 2800 tokens/s/GPU for a 30B omni-modal MoE on 128 GPUs with 160K context; adding a new modality takes very little code.
**After:** Over 2,800 tokens/s/GPU for a 30B omni-modal MoE on 128 GPUs at short context; the same model scales to 160K context with larger SP/EP at much lower throughput; adding a modality needs minimal code change (repo guide: five files for an omni-modal MoE).
**Why:**
- The paper states throughput and 160K context as two separate facts; Fig. 6(a) has the highest throughput at 16K (SP1/EP1) and reaches 160K only with larger SP/EP at much lower throughput ([VO] Abstract, §4.2, Fig. 6; same pattern in Table 4: 2,853 at 16K vs 1,075 tokens/s/GPU at 128K for text-only Qwen3-30B-A3B).
- "Very little code" is the paper's unquantified "minimal code change"; the repo guide lists `__init__.py` plus 4 more files for an omni-modal MoE ([DOC-NEW]).

## Other properties
none
