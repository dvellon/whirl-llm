#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# FAST-8S-1: summarize ab.py TSV: per engine x test/length mean decode tok/s over rounds, mean acceptance,
# and the acceptance-independent cycle time (gen_tokens / tps / drafts; one MTP draft per cycle on Ornith).
import re, sys, collections, statistics as st
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1], encoding="utf-8")]
H = rows[0]; idx = {c: i for i, c in enumerate(H)}
d = collections.defaultdict(list)
for r in rows[1:]:
    if len(r) < len(H) or not r[idx["decode_tps"]]:
        continue
    m = re.search(r"drafts (\d+)/(\d+)", r[idx["note"]])
    acc, dr = (int(m[1]), int(m[2])) if m else (0, 0)
    tps = float(r[idx["decode_tps"]]); gen = int(r[idx["gen_tokens"]] or 0)
    pp = float(r[idx["prefill_tps"]]) if r[idx["prefill_tps"]] else None
    key = (r[idx["engine"]], r[idx["test"]] + (r[idx["target"]] if r[idx["test"]] == "qa" else ""))
    d[key].append((tps, acc, dr, gen, pp, r[idx["note"]].split()[0]))
base = sys.argv[2] if len(sys.argv) > 2 else None
tests = sorted({k[1] for k in d}, key=lambda t: (t == "zh_chat", int(t[2:]) if t.startswith("qa") else 0))
engines = list(dict.fromkeys(k[0] for k in d))
print("engine\ttest\tn\ttg_mean\ttg_rounds\tvs_base\tacc_mean\tms_per_cycle\tcyc_vs_base\tpp_mean")
for t in tests:
    bm = None; bc = None
    for e in engines:
        v = d.get((e, t))
        if not v:
            continue
        tg = st.mean(x[0] for x in v)
        cyc = st.mean(1000 * x[3] / x[0] / x[2] for x in v if x[2])
        acc = st.mean(x[1] / x[2] for x in v if x[2])
        pps = [x[4] for x in v if x[4]]
        if e == base:
            bm, bc = tg, cyc
        vs = f"{100 * (tg / bm - 1):+.1f}%" if bm else ""
        cv = f"{100 * (cyc / bc - 1):+.1f}%" if bc else ""
        print(f"{e}\t{t}\t{len(v)}\t{tg:.1f}\t{'/'.join(f'{x[0]:.1f}' for x in v)}\t{vs}\t{acc:.3f}\t{cyc:.2f}\t{cv}\t{st.mean(pps):.0f}" if pps else
              f"{e}\t{t}\t{len(v)}\t{tg:.1f}\t{'/'.join(f'{x[0]:.1f}' for x in v)}\t{vs}\t{acc:.3f}\t{cyc:.2f}\t{cv}\t")
