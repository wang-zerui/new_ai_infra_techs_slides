import numpy as np
def rne_bits(a, keep):  # round fp32 array to `keep` explicit mantissa bits (RNE), return fp32
    a = np.asarray(a, dtype=np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    drop = 23 - keep
    lsb = (u >> drop) & 1
    bias = (1 << (drop - 1)) - 1 + lsb
    u = ((u + bias) >> drop) << drop
    return u.astype(np.uint32).view(np.float32)
bf16 = lambda a: rne_bits(a, 7)
tf32 = lambda a: rne_bits(a, 10)
def trunc_bf16(a):
    u = np.asarray(a, dtype=np.float32).view(np.uint32)
    return (u & np.uint32(0xFFFF0000)).view(np.float32)

rng = np.random.default_rng(0)
w = rng.standard_normal(2_000_000).astype(np.float32)
hi = bf16(w); r = (w.astype(np.float64) - hi.astype(np.float64)).astype(np.float32)
lo = bf16(r * np.float32(256.0))
rec = hi.astype(np.float64) + lo.astype(np.float64) / 256.0
rel = lambda approx: np.max(np.abs(approx - w.astype(np.float64)) / np.abs(w.astype(np.float64)))
print("max rel err bf16 ", rel(hi.astype(np.float64)), "bound 2^-8 =", 2**-8)
print("max rel err tf32 ", rel(tf32(w).astype(np.float64)), "bound 2^-11 =", 2**-11)
print("max rel err split", rel(rec), "bound 2^-16 =", 2**-16)
# truncation variant (blog wording)
hit = trunc_bf16(w); rt = (w.astype(np.float64) - hit).astype(np.float32); lot = bf16(rt*np.float32(256))
print("max rel err split(trunc hi)", rel(hit.astype(np.float64) + lot.astype(np.float64)/256))

# running example
for val in [0.1234567, -0.0371234, 0.0123456789]:
    v = np.float32(val); h = bf16(np.array([v]))[0]; rr = np.float32(np.float64(v) - np.float64(h)); l = bf16(np.array([rr*np.float32(256)]))[0]
    print(f"w={v:.9g} hi={h:.9g} r={rr:.6e} r*256={rr*256:.9g} lo={l:.9g} hi+lo/256={np.float64(h)+np.float64(l)/256:.12g} err={abs(np.float64(h)+np.float64(l)/256-np.float64(v)):.3e} relerr={abs(np.float64(h)+np.float64(l)/256-np.float64(v))/abs(v):.3e}  tf32={tf32(np.array([v]))[0]:.9g} tf32err={abs(np.float64(tf32(np.array([v]))[0])-np.float64(v)):.3e}")

# GEMM error, K=4096 N=192 M=512 (same setup as benchmark: x~N(0,1) bf16, w~N(0,1) fp32)
M,N,K=512,192,4096
x = bf16(rng.standard_normal((M,K)).astype(np.float32)).astype(np.float64)
W = rng.standard_normal((N,K)).astype(np.float32)
ref = x @ W.astype(np.float64).T
Wh = bf16(W); Wl = bf16(((W.astype(np.float64)-Wh).astype(np.float32))*np.float32(256))
ysplit = x @ Wh.astype(np.float64).T + (x @ Wl.astype(np.float64).T)/256
ytf = x @ tf32(W).astype(np.float64).T
ybf = x @ Wh.astype(np.float64).T
for n,y in [("split",ysplit),("tf32",ytf),("bf16",ybf)]:
    d=np.abs(y-ref); print(n, "max abs", d.max(), "mean abs", d.mean())
print("ref std", ref.std())
