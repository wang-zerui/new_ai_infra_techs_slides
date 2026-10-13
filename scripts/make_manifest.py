"""Build data/manifest.json: one entry per Notion row (71 techniques), numbered by category then title."""
import json, re, pathlib

ROWS = """\
Attention & Long Context|CSA compressed sparse attention|e36ba1f7295647b897468dc2b4d30590|https://arxiv.org/abs/2606.19348|DeepSeek V4
Attention & Long Context|DeepSeek Sparse Attention (DSA) + Lightning Indexer|cf0641f4eac74fd9bd6c903a15b59ca3|https://arxiv.org/abs/2512.02556|DeepSeek V3.2, GLM-5 series
Attention & Long Context|HCA heavily-compressed attention + hybrid attention stacking|ceae2901ccd44561a14abacc7f6574bf|https://arxiv.org/abs/2606.19348|DeepSeek V4
Attention & Long Context|IndexShare (cross-layer shared indexer)|e18d1e0afe6f4d3d9c37bfdf815b0738|https://arxiv.org/abs/2603.12201|GLM-5 series
Attention & Long Context|Kimi Delta Attention (KDA)|0a25e512d50542b08af4e090523e14b2|https://arxiv.org/abs/2607.24653|Kimi K3
Attention & Long Context|Staged context extension + annealing / cooldown|ff95a3d4ac0745728425571a7344ee70|https://arxiv.org/abs/2607.24653|Kimi K2, Kimi K3, GLM-4.5
Attention Kernels|Dynamic Decode Attention (dynamic task scheduling)|395fb76ae58944d0ae141c2a047478b6|https://github.com/Tencent/hpc-ops|HPC-Ops
Attention Kernels|FlashKDA (chunkwise linear-attention kernel)|addf5d5fcfad445081c78ebac0e50d60|https://arxiv.org/abs/2607.24653|Kimi K3
Attention Kernels|FP8 block-sparse prefill attention|6ee53460854a4c988e083e2ba08d1f88|https://github.com/Tencent/hpc-ops|HPC-Ops
Attention Kernels|FP8 MQA logits / sparse indexer scoring kernel|ac84a3f18a034bac9f7c33b63a8eb6e3|https://github.com/deepseek-ai/DeepGEMM|DeepGEMM, DeepSeek V3.2
Comm/Compute Overlap|1F1B / batch-level EP all-to-all overlap|0e6015aa2a8449ff917574f6139df4c2|https://arxiv.org/abs/2603.07685|Megatron-LM / MCore
Comm/Compute Overlap|Ascend · comm/compute fusion and PD-disaggregated scheduling|b1ba69567b7740fdbeead2013a32881c|https://arxiv.org/abs/2602.15763|GLM-5 series
Comm/Compute Overlap|Fused AllReduce + residual + RMSNorm|2a55d26c7c7a4f858f5816d7ca24b473|https://github.com/Tencent/hpc-ops|HPC-Ops
Comm/Compute Overlap|Wave-based fine-grained communication/compute overlap|0a5b3fd612d24e81a538b34ef561d06c|https://arxiv.org/abs/2606.19348|DeepSeek V4
Expert-Parallel Communication|0-SM PP / CP / Engram communication primitives|c3f064d3adbd46a983472f8d7992f155|https://github.com/deepseek-ai/DeepEP|DeepEP
Expert-Parallel Communication|Elastic GPU/CPU hybrid memory buffer|f8ca122ace3b41dba7d867f9c2731da3|https://github.com/deepseek-ai/DeepEP|DeepEP
Expert-Parallel Communication|ElasticBuffer unified API + analytic SM/QP budgeting|288a723bba1048b2964f7c67a24d4a3a|https://github.com/deepseek-ai/DeepEP|DeepEP
Expert-Parallel Communication|FP8 dispatch + BF16 combine low-precision communication|8f8e53cef45c40b58e4079637b461706|https://github.com/deepseek-ai/DeepEP|DeepEP, DeepSeek V4
Expert-Parallel Communication|HybridEP + FlexDispatcher|3e7e3ebd211b43769e3a8acb1883c7f6|https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md|Megatron-LM / MCore, DeepEP
Expert-Parallel Communication|MoonEP · dynamic redundant experts + online GPU planning + zero-copy static shapes|dfc27fd2e9d443c9a0a0efd215bdc3da|https://github.com/MoonshotAI/MoonEP|MoonEP
Expert-Parallel Communication|NCCL Gin backend (GPU-initiated RDMA)|50c0563d46924e00830d20a62dd6cbde|https://github.com/deepseek-ai/DeepEP|DeepEP
Expert-Parallel Communication|Network-side optimization: VL traffic isolation / adaptive routing / SM-free RDMA|c0f6478d3ece4c3c8cef0965065c4c0a|https://github.com/deepseek-ai/DeepEP|DeepEP
Frameworks & Platform Stacks|Ascend adaptation stack (GLM-5)|3eda454335e04ef09f6a0b6991ee5608|https://arxiv.org/abs/2602.15763|GLM-5 series, VeOmni
Frameworks & Platform Stacks|DeepEP v2 (EP communication library)|d471baa2bcdf452592855568ee99f726|https://github.com/deepseek-ai/DeepEP|DeepEP
Frameworks & Platform Stacks|DeepGEMM (unified tensor-core kernel library)|c0133413563f4ee390f9e4fb55027780|https://github.com/deepseek-ai/DeepGEMM|DeepGEMM
Frameworks & Platform Stacks|HPC-Ops (Tencent Hunyuan inference operator library)|85f1abe2180f4b86bf1e9ea6d23705ce|https://github.com/Tencent/hpc-ops|HPC-Ops
Frameworks & Platform Stacks|MoonEP (perfectly balanced EP communication library)|65b197b767bb4d5c9dc17bbce5c0ca13|https://github.com/MoonshotAI/MoonEP|MoonEP, Kimi K3
Frameworks & Platform Stacks|VeOmni (model-centric omni-modal training framework)|9b9145984b6a433abf69f32d2c06db67|https://arxiv.org/abs/2508.02317|VeOmni
GEMM & MoE Kernels|FP8 grouped GEMM (per-tensor / block-wise scaling)|4b75d7ab1bfc40239d6b9566c5d6fc14|https://github.com/Tencent/hpc-ops|HPC-Ops
GEMM & MoE Kernels|FP8/FP4/BF16 dense GEMM + UE8M0 scaling layout|eedb72bc30e14aa8ba49b28b4344d0b9|https://github.com/deepseek-ai/DeepGEMM|DeepGEMM
GEMM & MoE Kernels|Fused FP8 MoE low-latency pipeline|4d0436aff16240398bbe1c41ab56d5ad|https://github.com/Tencent/hpc-ops|HPC-Ops
GEMM & MoE Kernels|K-axis grouped GEMM (MoE weight backward)|56c25e6b3cbe4425b251e2b59530b802|https://github.com/deepseek-ai/DeepGEMM|DeepGEMM
GEMM & MoE Kernels|M-axis grouped GEMM (contiguous layout)|3e1e63852a934fad8b5d53773913bce5|https://github.com/deepseek-ai/DeepGEMM|DeepGEMM
GEMM & MoE Kernels|Masked grouped GEMM (CUDA Graph friendly)|d8af0edc72e64994bbb502dc9a3bfcea|https://github.com/deepseek-ai/DeepGEMM|DeepGEMM, DeepEP
GEMM & MoE Kernels|Mega MoE fused MegaKernel|77562a16a085453aa50e5744e3382f7e|https://github.com/deepseek-ai/DeepGEMM|DeepGEMM
GEMM & MoE Kernels|Route GEMM (BF16 x FP32 via dual-BF16 split)|1ca14bb9a0ce4d439cc9a0686b60a02d|https://github.com/Tencent/hpc-ops|HPC-Ops
Kernel Fusion & Runtime|Ascend · super-operator fusion (MLAPO / Sparse FA / MoE Mega-Fusion)|5926e0ea48474805988cb4d1419bac18|https://arxiv.org/abs/2602.15763|GLM-5 series
Kernel Fusion & Runtime|CUDA Graphs (full-iteration / per-layer / TE graphed callables)|7d97c06866b54afdb590597a7dc39756|https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/cuda_graph.html|Megatron-LM / MCore, TransformerEngine
Kernel Fusion & Runtime|DeepJIT runtime compilation + PDL + determinism switches|1ebac796b72b428f9c32afc18e7b5fd4|https://github.com/deepseek-ai/DeepGEMM|DeepGEMM
Kernel Fusion & Runtime|Fused Sampler (decode post-processing fusion)|3070daaed63740e3be901cba2641e152|https://github.com/Tencent/hpc-ops|HPC-Ops
Kernel Fusion & Runtime|Mega Gate / Mega mHC fused kernels|0bb7a1acbf924329aab698ccc8c5ffee|https://github.com/deepseek-ai/DeepGEMM|DeepGEMM, DeepSeek V4
Kernel Fusion & Runtime|Router / permutation / MLA-RoPE fused kernels|9d82ddbf31244093bc930a84ed679c2d|https://arxiv.org/abs/2603.07685|Megatron-LM / MCore, TransformerEngine
Kernel Fusion & Runtime|TileLang DSL + batch-invariant deterministic kernels|22c7e2d5f253481981afb6c6ee06e3f5|https://arxiv.org/abs/2606.19348|DeepSeek V4
Low-Precision & Quantization|Ascend · W4A8/W8A8 hybrid quantization stack|38d6d033f1b943b3b79615c26f4f0a2e|https://arxiv.org/abs/2602.15763|GLM-5 series
Low-Precision & Quantization|FP4 QAT (MXFP4 expert weights + indexer QK path)|ad31c31051744ea2b53ea97c1417ac71|https://arxiv.org/abs/2606.19348|DeepSeek V4
Low-Precision & Quantization|FP8 training (per-tensor / blockwise) and FP8 activation compression|8406549e9fd645e49d2f2e90a838f1bd|https://github.com/NVIDIA/TransformerEngine|TransformerEngine, Megatron-LM / MCore, Kimi K2
Low-Precision & Quantization|MXFP8 micro-scaling training|02c9479032474c7e8e387fb2e6b7a964|https://arxiv.org/abs/2506.08027|TransformerEngine
Low-Precision & Quantization|NVFP4 pretraining recipe|eae9504538664595866784566f5d14cd|https://arxiv.org/abs/2509.25149|TransformerEngine, Megatron-LM / MCore
Low-Precision & Quantization|Precision-aware optimizer (BF16/FP8 optimizer states)|893ccbb87b354e7e8fedbc0a18ca7117|https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md|Megatron-LM / MCore, TransformerEngine
Memory & Offload Scheduling|Distributed checkpointing and elastic-resource training|a065908961604600811834b0c1800744|https://github.com/ByteDance-Seed/VeOmni|VeOmni, Kimi K2
Memory & Offload Scheduling|Dynamic batching / sequence packing|5cfd3adb77f74e0c8d09435a3af5d9b3|https://arxiv.org/abs/2508.02317|VeOmni, Megatron-LM / MCore, GLM-5 series
Memory & Offload Scheduling|Fine-grained selective recomputation|2e31e16f18ce45edba1e3d9058756c08|https://arxiv.org/abs/2603.07685|Megatron-LM / MCore
Memory & Offload Scheduling|Memory-efficient token permutation|3a91e54ff5c7400e9b2e65e0ba6479f5|https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md|Megatron-LM / MCore
Memory & Offload Scheduling|Pipeline-aware activation / optimizer offload|431ab65669934bd2af65da396bb96f7f|https://arxiv.org/abs/2603.07685|Megatron-LM / MCore, Kimi K2, Kimi K3
MoE & Routing|Loss-free balance routing + sigmoid gating|efa722c88bed4a4cb1baa6d8d0a0d6fa|https://arxiv.org/abs/2508.06471|GLM-4.5, DeepSeek V3.2, Megatron-LM / MCore
MoE & Routing|Quantile Balancing|b59b1fb31a654c95a2957772ad9e4d92|https://arxiv.org/abs/2607.24653|Kimi K3
MoE & Routing|Stable LatentMoE|fafccf6521d44c569d9d5cfad419049f|https://arxiv.org/abs/2607.24653|Kimi K3
MoE & Routing|Ultra-sparse MoE (fine-grained routed experts + shared expert)|b2283eba9cf14a8fb2fa3f7b2ba4127f|https://arxiv.org/abs/2606.19348|DeepSeek V4, Kimi K2, GLM-5 series
Model Shape & Depth Scaling|Attention Residuals (AttnRes / Block AttnRes)|8b86f64fd2d44d4bac732d2321f7722a|https://arxiv.org/abs/2603.15031|Kimi K3
Model Shape & Depth Scaling|Deep-and-narrow + more attention heads + partial RoPE|ba32c8b828d748ea9f32a65e15598576|https://arxiv.org/abs/2508.06471|GLM-4.5
Model Shape & Depth Scaling|mHC: manifold-constrained hyper-connections|99ccc6cd324244c0bf114f19b5440674|https://arxiv.org/abs/2512.24880|DeepSeek V4, GLM-5 series
Model Shape & Depth Scaling|MTP (multi-token prediction) layer|354b71e013554721a039936a8fff8b2a|https://arxiv.org/abs/2508.06471|DeepSeek V4, GLM-4.5, GLM-5 series, Megatron-LM / MCore
Optimizer & Stability|Anticipatory routing (asynchronous router parameters)|ecc924eb12db4b0fa39609cf42a53369|https://arxiv.org/abs/2606.19348|DeepSeek V4
Optimizer & Stability|Distributed engineering of Muon at scale|9069ea0863a741658d052a19a186169b|https://arxiv.org/abs/2606.19348|DeepSeek V4, GLM-5 series
Optimizer & Stability|MuonClip (Muon + QK-Clip)|63cf5bee70464919acbac2e6ed917cdd|https://arxiv.org/abs/2507.20534|Kimi K2
Optimizer & Stability|QK-Norm|319387d19fc64c6d8079c56a5c3b7a69|https://arxiv.org/abs/2508.06471|GLM-4.5
Parallelism Strategy|Asymmetric VPP / custom pipeline layout|0b667c3fda7240be99b87c3b35fdcfcd|https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md|Megatron-LM / MCore
Parallelism Strategy|Context Parallelism / Dynamic CP|82d104d9bcc74fab931db687fed76e81|https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/context_parallel.md|Megatron-LM / MCore, TransformerEngine
Parallelism Strategy|KDA Context Parallelism (KCP)|a5e78a14357748deb3032985bef70062|https://arxiv.org/abs/2607.24653|Kimi K3
Parallelism Strategy|Megatron-FSDP / HSDP (with EP support)|bbdbb22301864a05bcbc8e4097a6b284|https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md|Megatron-LM / MCore
Parallelism Strategy|MoE Parallel Folding|ef4d75b8eee841a79ada63b79bff78d8|https://arxiv.org/abs/2603.07685|Megatron-LM / MCore
Frameworks & Platform Stacks|MagiAttention (distributed attention for ultra-long, heterogeneous-mask training)|3f895647fb2a81aa8f07fc1ab10670bb|https://github.com/SandAI-org/MagiAttention|MagiAttention, MAGI-1
"""

def slugify(s, n=48):
    s = s.lower().replace("·", " ").replace("+", " ").replace("/", " ")
    s = re.sub(r"\([^)]*\)", " ", s) if len(s) > n else s
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    if len(s) > n:
        s = s[:n + 1].rsplit("-", 1)[0]  # cut at a word boundary
    return s

root = pathlib.Path(__file__).resolve().parents[1]
mpath = root / "data" / "manifest.json"
# Slugs are folder names: once assigned, keep them even if a row's title is later corrected.
old_slugs = {e["notion_page_id"]: e["slug"] for e in json.loads(mpath.read_text())} if mpath.exists() else {}
out = []
for i, line in enumerate(l for l in ROWS.splitlines() if l.strip()):
    cat, tech, pid, ref, src = line.split("|")
    nn = f"{i+1:02d}"
    out.append({"id": nn, "slug": old_slugs.get(pid, f"{nn}-{slugify(tech)}"), "technique": tech, "category": cat,
                "notion_page_id": pid, "notion_url": f"https://app.notion.com/p/{pid}",
                "reference": ref, "source": [x.strip() for x in src.split(",")]})
mpath.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print(len(out))
for e in out: print(e["slug"])
