#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# PROF-8S: per-kernel times of a prefill capture, restricted to the prefill request (dispatches before the first idle
# gap > 50 ms, host-side copyBuffer waits listed separately), per layer and side by side.
#   python pf.py LAYERKERNEL A_dispatches.csv [B_dispatches.csv ...]
import csv, sys
from collections import defaultdict

def load(p, layer_k):
    rows = [r for r in csv.DictReader(open(p, newline="")) if r.get("is_artifact", "").lower() not in ("1", "true")]
    rows.sort(key=lambda r: float(r["ts_us"]))
    out = []
    for i, r in enumerate(rows):
        if i and float(r.get("gap_before_us") or 0) > 50000:
            break
        out.append(r)
    agg = defaultdict(float); cnt = defaultdict(int)
    for r in out:
        d = float(r["dur_us"] or 0)
        if r["kernel"].startswith("__amd_rocclr_copy") and d > 2000:
            agg["(long copyBuffer, host wait?)"] += d; continue
        agg[r["kernel"]] += d; cnt[r["kernel"]] += 1
    L = max(1, cnt.get(layer_k, 0))
    span = float(out[-1]["ts_us"]) + float(out[-1]["dur_us"]) - float(out[0]["ts_us"])
    return L, agg, cnt, span, len(out)

lk = sys.argv[1]
caps = [load(p, lk) for p in sys.argv[2:]]
print("layers (count of", lk, "):", [c[0] for c in caps], " dispatches:", [c[4] for c in caps], " span ms:", [round(c[3] / 1e3, 1) for c in caps])
tot = [sum(v for k, v in c[1].items() if not k.startswith("(")) for c in caps]
print("busy ms:", [round(t / 1e3, 1) for t in tot], " busy/layer ms:", [round(t / c[0] / 1e3, 3) for t, c in zip(tot, caps)])
keys = sorted(set().union(*[c[1] for c in caps]), key=lambda k: -max(c[1].get(k, 0) / c[0] for c in caps))
for k in keys:
    v = [c[1].get(k, 0) / c[0] for c in caps]
    sh = [100 * c[1].get(k, 0) / t for c, t in zip(caps, tot)]
    if max(v) < 50: continue
    print(f"{k[:40]:40s} " + " ".join(f"{x:9.1f}us/L {s:5.1f}%" for x, s in zip(v, sh)) + f"  n={[c[2].get(k,0) for c in caps]}")
