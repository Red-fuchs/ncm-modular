"""Compare MLX codec vs torch dump."""
import sys, os, numpy as np, mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK
import codec_mlx
mx.set_default_device(mx.cpu)
task = sys.argv[1]
R = np.load(f"{WORK}/ref_{task}.npz")
for dt, nm in [(mx.float32, "fp32"), (mx.float64, "fp64")]:
    c = codec_mlx.load(f"{WORK}/mlx/{task}", dt)
    x = mx.array(R["ids"])
    z = c.embed(x); zq = z
    codes = c.encode(x); zh = c.cumulative(codes)
    mx.eval(codes, zh)
    # pre-quant latent
    print(f"--- {task} {nm}: pre-quant z max|diff| {np.abs(np.array(z) - R['z']).max():.2e}")
    off = 0
    for k, cd in enumerate(codes):
        n = c.levels[k]
        ref = R["idx"][:, off:off + n]; off += n
        eq = (np.array(cd) == ref)
        print(f"  level {k:2d} (len {n:2d}) codes equal {eq.mean()*100:6.2f}%  zhat max|diff| {np.abs(np.array(zh[k]) - R['zh'][k]).max():.2e}")
    dec = np.stack([np.array(mx.argmax(c.decode_logits(h), -1)) for h in zh])
    print(f"  decoded ids per level equal: " + " ".join(f"{(dec[k]==R['dec'][k]).mean()*100:.1f}" for k in range(len(zh))))
    print(f"  final decoded ids equal: {(dec[-1]==R['dec_full']).mean()*100:.2f}%")
    # decode-only parity: feed torch codes into MLX decode
    cs, off = [], 0
    for n in c.levels: cs.append(mx.array(R["idx"][:, off:off + n])); off += n
    dd = np.array(c.decode_codes(cs))
    zz = c.cumulative(cs)[-1]
    print(f"  decode-from-torch-codes: ids equal {(dd==R['dec_full']).mean()*100:.2f}%  zhat max|diff| {np.abs(np.array(zz)-R['zh'][-1]).max():.2e}")
