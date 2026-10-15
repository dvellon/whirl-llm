#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# FASTLANE FL-1a: KL quick check (tier 0): tools/klgate/klgate.py on the dense model at 2k + 8k, ONE 512-row
# window per length (--positions 1: rows [L-512, L)), code + zh prompts, mode vs precise, the gate's dense limits
# (balance KL <= 0.005 / top-1 >= 98%, fast KL <= 0.03 / top-1 >= 93%).
# Screening only: a single window jitters by +-20-35% KL / +-1.6 points top-1 with the forward split before it
# (KL-ROBUST-2); the full 3-position x 4-length gate stays the rc / release check.
#
#   python kl_quick.py --device r97|8s [--modes balance] [--exe whirl.exe] [--prompts code,zh] [--lengths 2048,8192]
#                      [--env K=V;K2=V2] [--as MODE=--flag] [--tag x]
# --env sets WHIRL_* variables for the MODE runs only (not the precise reference): e.g. WHIRL_KV=q8h with --modes fast.
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KLGATE = os.path.join(HERE, "..", "klgate", "klgate.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["r97", "8s"], required=True)
    ap.add_argument("--modes", default="balance")
    ap.add_argument("--exe", default=qlib.DEF_EXE)
    ap.add_argument("--prompts", default="code,zh")
    ap.add_argument("--lengths", default="2048,8192")
    ap.add_argument("--env", default="")
    ap.add_argument("--as", dest="as_", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tw0 = time.time()
    name = f"{time.strftime('%Y%m%d-%H%M%S')}_kl_{a.device}_{a.modes.replace(',', '+')}{'_' + a.tag if a.tag else ''}"
    out = os.path.join(qlib.WORK, "fl1a", "kl", name)
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, KLGATE, "--exe", a.exe, "--corpus", qlib.CORPUS, "--model", "dense=" + qlib.MODEL,
           "--lengths", a.lengths, "--modes", a.modes, "--positions", "1", "--truncate", "--report", os.path.join(out, "report.json"),
           "--work", out, "--device", qlib.DEVICES[a.device],
           "--only", ",".join("dense:" + p for p in a.prompts.split(","))]
    if a.as_:
        cmd += ["--as", a.as_]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if a.env:
        # klgate strips WHIRL_* from the environment; pass them through its --wrap prefix for the mode runs only
        # is not possible, so use a tiny env-setting wrapper that applies them unless the run is precise
        env["FL_KL_ENV"] = a.env
        cmd += ["--wrap", f"{sys.executable} {os.path.join(HERE, 'envwrap.py')} --"]
    with qlib.GpuLock():
        with open(os.path.join(out, "klgate.log"), "w", encoding="utf-8") as fo:
            rc = subprocess.run(cmd, env=env, stdout=fo, stderr=subprocess.STDOUT).returncode
    for f in os.listdir(out):
        if f.endswith(".f16"):
            os.remove(os.path.join(out, f))
    wall = time.time() - tw0
    try:
        rep = json.load(open(os.path.join(out, "report.json"), encoding="utf-8"))
    except Exception:
        print(open(os.path.join(out, "klgate.log"), encoding="utf-8").read()[-2000:])
        print(f"kl_quick: no report (klgate exit {rc}), wall {wall:.0f} s")
        return 2
    print(f"== kl_quick {a.device} {a.modes} {a.env} exe sha256 {rep['exe_sha256'][:16]}  ({name})")
    for c in rep["cases"]:
        p = c.get("prompt", c.get("case", ""))
        print(f"  {'PASS' if c['pass'] else 'FAIL'} {c.get('model', '')}:{p:5s} {c.get('mode', ''):8s} {c['length']:6d}  KL {c['kl_mean']:.5f} "
              f"(<= {c['kl_limit']:.3f})  top1 {c['top1'] * 100:6.2f}% (>= {c['top1_limit'] * 100:.0f}%)  p99 {c['kl_p99']:.4f}")
    for e in rep.get("errors", []):
        print("  ERROR", e)
    print(f"klgate exit {rc}; wall {wall:.0f} s; report {os.path.join(out, 'report.json')}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
