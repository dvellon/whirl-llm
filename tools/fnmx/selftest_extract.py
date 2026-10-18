"""FNMX self-test input: real BF16 rows + imatrix weights -> raw float32 files (nrows, ncols, w[ncols], x[nrows*ncols])."""
import sys, struct
import numpy as np
sys.path.insert(0, r"C:\ClaudeWorkspace\tools\llamacpp_fn\gguf-py")
import gguf
D = r"C:\ClaudeWorkspace\models\fnmx"
m = gguf.GGUFReader(D + r"\gguf\swift-fn-bf16.gguf")
im = gguf.GGUFReader(D + r"\imatrix\imatrix-swiftfn-v1mix.gguf")
mt = {t.name: t for t in m.tensors}
it = {t.name: t for t in im.tensors}
def bf16(t):
    a = np.asarray(t.data).view(np.uint16).astype(np.uint32) << 16
    return a.view(np.float32)
cases = [("blk.10.ffn_up_exps.weight", 7), ("blk.30.ffn_down_exps.weight", 300), ("blk.10.attn_qkv.weight", None), ("blk.46.ssm_out.weight", None)]
for name, ex in cases:
    t = mt[name]
    x = bf16(t)
    ncols = int(t.shape[0])
    s2 = np.asarray(it[name + ".in_sum2"].data, dtype=np.float32)
    cnt = np.asarray(it[name + ".counts"].data, dtype=np.float32)
    if ex is not None:
        x = x.reshape(-1, int(t.shape[1]), ncols)[ex]
        s2 = s2.reshape(-1, ncols)[ex]; c = cnt.reshape(-1)[ex]
    else:
        x = x.reshape(-1, ncols); c = cnt.reshape(-1)[0]
    w = s2 / max(float(c), 1.0)
    x = x[:256]
    fn = D + rf"\selftest\{name}{'' if ex is None else '.e%d' % ex}.bin"
    with open(fn, "wb") as f:
        f.write(struct.pack("<ii", x.shape[0], ncols)); f.write(w.astype(np.float32).tobytes()); f.write(np.ascontiguousarray(x, dtype=np.float32).tobytes())
    print(fn, x.shape, "w>0", int((w > 0).sum()))
