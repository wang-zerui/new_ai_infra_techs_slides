"""Router GEMM latency vs M on H20 (K=4096, N=192). Source: LMSYS blog 'HPC-Ops x SGLang' Table 2."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

M = [1, 16, 48, 96, 208, 512, 1024, 2048, 4096]
hpc = [11.200, 11.744, 12.144, 13.904, 17.088, 26.992, 50.640, 76.688, 141.120]
fp32 = [14.576, 23.808, 31.008, 31.760, 39.280, 86.976, 110.480, 198.576, 403.728]
tf32 = [14.048, 18.752, 20.064, 24.720, 28.928, 44.736, 68.544, 100.800, 205.760]

sc.setup()
fig, ax = sc.figure("right")
ax.plot(M, fp32, color=sc.GRAYS[3], ls="--", marker="s", label="FP32 cuBLAS")
ax.plot(M, tf32, color=sc.GRAYS[1], ls=":", marker="^", label="TF32 cuBLAS")
ax.plot(M, hpc, color=sc.ACCENT, marker="o", label="Route GEMM")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks([1, 16, 256, 4096]); ax.set_xticklabels(["1", "16", "256", "4096"])
ax.set_yticks([10, 30, 100, 300]); ax.set_yticklabels(["10", "30", "100", "300"])
ax.set_xlabel("M (tokens)")
ax.set_ylabel("Latency (µs, log)")
sc.top_legend(ax, ncol=3, fontsize=16)
ax.text(580, 20, "3.22× at M = 512", color=sc.ACCENT, fontsize=20, fontweight="bold", ha="left", va="center")
ax.annotate("", xy=(512, hpc[5] * 1.08), xytext=(512, fp32[5] / 1.08),
            arrowprops=dict(arrowstyle="<->", color=sc.ACCENT, lw=2))
sc.better(ax, "Lower is better", where="upper left")
sc.save(fig, __file__)
