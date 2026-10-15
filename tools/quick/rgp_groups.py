#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# FASTLANE: group a rgp_capture <name>_kernels.csv into op classes (GEMM/GEMV, dequant, attention, DeltaNet,
# norm/elementwise, copies, other) with time share.   python rgp_groups.py X_kernels.csv [...]
import csv
import sys
from collections import defaultdict

RULES = [("attention", ("attn", "kv_dq", "kv_store", "fa_")), ("deltanet", ("gdn",)),
         ("dequant", ("dequant",)), ("gemm/gemv", ("gemm", "gemv", "ggemv", "gemms")),
         ("copy", ("__amd_rocclr",)), ("lm-head/sample", ("argmax", "topk", "sample")),
         ("norm/elementwise", ("rmsnorm", "silu", "gated_norm", "quantize", "f32_to", "get_rows", "set_tokens", "add", "mul"))]


def cls(k):
    for name, pre in RULES:
        if any(k.startswith(p) for p in pre):
            return name
    return "other"


def groups(path):
    g = defaultdict(lambda: [0.0, 0])
    tot = 0.0
    for r in csv.DictReader(open(path, newline="")):
        c = cls(r["kernel"])
        g[c][0] += float(r["total_us"])
        g[c][1] += int(r["count"])
        tot += float(r["total_us"])
    return tot, sorted(g.items(), key=lambda x: -x[1][0])


if __name__ == "__main__":
    for p in sys.argv[1:]:
        tot, gs = groups(p)
        print(f"{p}: busy {tot / 1000:.1f} ms")
        for c, (us, n) in gs:
            print(f"   {c:18s} {100 * us / tot:6.2f}%  {us / 1000:9.2f} ms  x{n}")
