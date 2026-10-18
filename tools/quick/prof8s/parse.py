#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# PROF-8S: parse a capture left by prof8s.py --no-parse (CPU only; runs beside the next GPU capture).
#   python parse.py CAP.rgp [prefill|decode]
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import rgp_capture as rc  # noqa: E402
f = sys.argv[1]; base = f[:-4]
p = subprocess.run([rc.PARSER_PY, os.path.join(rc.PARSER, "main.py"), f, base], cwd=rc.PARSER, capture_output=True, text=True)
if p.returncode:
    print("parse failed", (p.stderr or p.stdout)[-800:]); sys.exit(1)
sm = rc.summarize(base + "_dispatches.csv", base + "_kernels.csv", 300000 if "prefill" in f else 0)
print(os.path.basename(base), sm["dispatches"], "disp, span %.1f ms busy %.1f gaps %.1f" % (sm["span_us"] / 1e3, sm["busy_us"] / 1e3, sm["gap_us"] / 1e3))
for k, us, sh, n in sm["top"][:12]:
    print(f"  {sh:6.2f}% {us / 1e3:8.3f} ms x{n:<5d} {k[:90]}")
if os.path.getsize(f) > 200e6:
    os.remove(f)
