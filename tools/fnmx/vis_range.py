"""FNMX: max |w| of the BF16 vision tensors, to check F16 (max 65504) is safe for the mmproj."""
import json, os
from safetensors import safe_open
d = r"C:\ClaudeWorkspace\models\fnmx\bf16"
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
vis = [k for k in idx if "visual" in k]
print(len(vis), "visual tensors")
mx, files = 0.0, {}
for k in vis:
    f = idx[k]
    if f not in files:
        files[f] = safe_open(os.path.join(d, f), "pt")
    m = files[f].get_tensor(k).float().abs().max().item()
    mx = max(mx, m)
print("max abs visual weight", mx)
