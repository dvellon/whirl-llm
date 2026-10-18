#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# PROF-8S: add a model's tensor shapes / types / bytes and the metadata sol.py needs to sol_shapes.json
#   python dump_shapes.py NAME GGUF
import json, os, sys
sys.path.insert(0, r"C:\ClaudeWorkspace\refsrc\llama.cpp\gguf-py")
from gguf import GGUFReader  # noqa: E402
name, path = sys.argv[1], sys.argv[2]
r = GGUFReader(path)
arch = None
meta = {}
for k, f in r.fields.items():
    if k == "general.architecture":
        arch = bytes(f.parts[f.data[0]]).decode()
for k, f in r.fields.items():
    if arch and k.startswith(arch + "."):
        v = [f.parts[i].tolist() for i in f.data]
        v = [x[0] if isinstance(x, list) and len(x) == 1 else x for x in v]
        meta[k[len(arch) + 1:]] = v[0] if len(v) == 1 else v
ts = [[t.name, t.tensor_type.name, [int(x) for x in t.shape], int(t.n_bytes)] for t in r.tensors]
p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sol_shapes.json")
d = json.load(open(p, encoding="utf-8"))
d[name] = {"path": path, "arch": arch, "meta": meta, "tensors": ts}
json.dump(d, open(p, "w", encoding="utf-8"), indent=0)
print(name, arch, len(ts), {k: meta[k] for k in list(meta)[:30]})
