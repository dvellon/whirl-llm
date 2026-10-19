"""FNMX: summarize tensor types of a GGUF (or first shard of a split) by group. Usage: python dump_types.py FILE.gguf [--all]"""
import sys, re, collections
sys.path.insert(0, r"C:\ClaudeWorkspace\tools\llamacpp_fn\gguf-py")
import gguf

def group(n):
    n2 = re.sub(r"blk\.\d+\.", "blk.N.", n)
    return n2

r = gguf.GGUFReader(sys.argv[1])
arch = r.fields["general.architecture"]
print("arch", bytes(arch.parts[arch.data[0]]).decode())
for k in r.fields:
    if any(s in k for s in ("nextn", "block_count", "ple", "file_type", "split")):
        f = r.fields[k]
        try:
            v = f.contents()
        except Exception:
            v = "?"
        if isinstance(v, list) and len(v) > 8:
            v = f"[{len(v)} items] {v[:4]}..."
        print(" ", k, "=", v)
g = collections.OrderedDict()
tot = collections.Counter()
for t in r.tensors:
    key = (group(t.name), t.tensor_type.name)
    c = g.setdefault(key, [0, 0, []])
    c[0] += 1; c[1] += int(t.n_bytes)
    m = re.match(r"blk\.(\d+)\.", t.name)
    if m: c[2].append(int(m.group(1)))
    tot[t.tensor_type.name] += int(t.n_bytes)
print("tensors", len(r.tensors))
for (name, ty), (n, b, layers) in g.items():
    lr = f" layers {min(layers)}-{max(layers)}" if layers else ""
    print(f"{name:55s} {ty:8s} x{n:<3d} {b/1e9:9.3f} GB{lr}")
print("bytes by type:", {k: round(v/1e9, 3) for k, v in tot.items()}, "total GB", round(sum(tot.values())/1e9, 3))
