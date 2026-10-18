#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# PROF-8S: per-verify-cycle kernel times of an MTP decode capture (window = first..last LM-head launch, the
# widest gemv launch), side by side for several captures.   python cyc.py A_dispatches.csv B_dispatches.csv ...
import csv, re, sys
from collections import defaultdict

def load(p):
    rows = [r for r in csv.DictReader(open(p, newline="")) if r.get("is_artifact", "").lower() not in ("1", "true")]
    rows.sort(key=lambda r: float(r["ts_us"]))
    gem = [r for r in rows if re.match(r"g?gemv", r["kernel"]) and not r["kernel"].startswith("gemv_d2")]
    hmax = max(int(r["threads"] or 0) for r in gem)
    heads = [i for i, r in enumerate(rows) if r in gem and int(r["threads"] or 0) == hmax]
    lo, hi = heads[0], heads[-1]
    win = rows[lo + 1:hi + 1]
    cyc = len(heads) - 1
    agg = defaultdict(float)
    for r in win:
        agg[r["kernel"]] += float(r["dur_us"] or 0)
    span = float(win[-1]["ts_us"]) + float(win[-1]["dur_us"]) - float(rows[lo]["ts_us"]) - float(rows[lo]["dur_us"])
    busy = sum(agg.values())
    return cyc, {k: v / cyc for k, v in agg.items()}, span / cyc, busy / cyc, len(win) / cyc

caps = [load(p) for p in sys.argv[1:]]
print("cycles:", [c[0] for c in caps])
print("span/cycle us:", [round(c[2], 1) for c in caps], " busy/cycle:", [round(c[3], 1) for c in caps], " gap/cycle:", [round(c[2] - c[3], 1) for c in caps], " dispatches/cycle:", [round(c[4], 1) for c in caps])
keys = sorted(set().union(*[c[1] for c in caps]), key=lambda k: -max(c[1].get(k, 0) for c in caps))
for k in keys:
    v = [c[1].get(k, 0) for c in caps]
    if max(v) < 20: continue
    print(f"{k[:44]:44s} " + " ".join(f"{x:9.1f}" for x in v) + (f"   d={v[1]-v[0]:+8.1f}" if len(v) > 1 else ""))
