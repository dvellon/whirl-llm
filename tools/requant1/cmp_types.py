"""REQUANT-1: compare tensor names / types / shapes of two GGUFs (old published vs new build). Usage: python cmp_types.py OLD.gguf NEW.gguf"""
import sys, collections
sys.path.insert(0, r"C:\ClaudeWorkspace\tools\llamacpp_fn\gguf-py")
import gguf

def load(p):
    r = gguf.GGUFReader(p)
    return {t.name: (t.tensor_type.name, tuple(int(x) for x in t.shape), int(t.n_bytes)) for t in r.tensors}

a, b = load(sys.argv[1]), load(sys.argv[2])
diff = 0
for n in sorted(set(a) | set(b)):
    if a.get(n) != b.get(n):
        diff += 1
        if diff <= 30: print("DIFF", n, a.get(n), b.get(n))
c = collections.Counter(v[0] for v in b.values())
print(f"tensors old {len(a)} new {len(b)}; differing {diff}; new types {dict(c)}")
print("RESULT", "SAME_LAYOUT" if diff == 0 else "LAYOUT_DIFFERS")
