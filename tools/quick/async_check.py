#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# FASTLANE FL-1a / REF-MINE T0-A: async regression test. Runs the golden suites (tools/golden/golden.ps1 check:
# plain prefill short + 4000 tokens, decode steps, MTP decode, forced n-gram drafts, two-sequence server paths =
# segmented prefill / batched rows / batched verify / one batched MTP cycle) twice on one GPU: once as usual and once
# with HIP_LAUNCH_BLOCKING=1 (every kernel launch synchronous). Every output file ('G' lines: top-5 ids + FNV-1a
# hash of the logits per step, token streams) must be byte-identical between the two runs. A difference means a
# missing dependency between streams / host copies (a race the synchronous run hides); it is a correctness bug.
# Both runs are also checked against the recorded references by golden.ps1 itself.
#
#   python async_check.py --device r97|8s [--models dense,moe] [--modes precise,balance,fast] [--suites single,pair,ngram]
#                         [--exe whirl.exe]
# Exit 0 = identical (and both runs matched the references), 1 = a difference, 2 = a run error.
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.abspath(os.path.join(HERE, "..", "golden", "golden.ps1"))
ARCH = {"r97": "gfx1201", "8s": "gfx1151"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["r97", "8s"], required=True)
    ap.add_argument("--models", default="dense,moe")
    ap.add_argument("--modes", default="precise,balance,fast")
    ap.add_argument("--suites", default="single,pair,ngram")
    ap.add_argument("--exe", default=qlib.DEF_EXE)
    a = ap.parse_args()
    tw0 = time.time()
    name = f"{time.strftime('%Y%m%d-%H%M%S')}_async_{a.device}"
    base = os.path.join(qlib.WORK, "fl1a", "async", name)
    runs = {"default": {}, "blocking": {"HIP_LAUNCH_BLOCKING": "1"}}
    rcs, walls = {}, {}
    with qlib.GpuLock():
        for tag, extra in runs.items():
            out = os.path.join(base, tag)
            os.makedirs(out, exist_ok=True)
            env = dict(os.environ)
            env.pop("HIP_LAUNCH_BLOCKING", None)
            env.update(extra)
            t = time.time()
            with open(os.path.join(base, tag + ".log"), "w", encoding="utf-8") as fo:
                rcs[tag] = subprocess.run(["pwsh", "-NoProfile", "-File", GOLDEN, "check", "-Arch", ARCH[a.device], "-Models", a.models,
                                           "-Modes", a.modes, "-Suites", a.suites, "-Exe", a.exe, "-Out", out], env=env, stdout=fo,
                                          stderr=subprocess.STDOUT).returncode
            walls[tag] = time.time() - t
            qlib.log(f"{tag}: golden.ps1 exit {rcs[tag]}, {walls[tag]:.0f} s")
    da, db = os.path.join(base, "default"), os.path.join(base, "blocking")
    fa, fb = sorted(os.listdir(da)), sorted(os.listdir(db))
    diffs = []
    if fa != fb:
        diffs.append(f"file sets differ: {sorted(set(fa) ^ set(fb))}")
    for f in sorted(set(fa) & set(fb)):
        la = open(os.path.join(da, f), encoding="utf-8").read().splitlines()
        lb = open(os.path.join(db, f), encoding="utf-8").read().splitlines()
        if la != lb:
            i = next((k for k in range(min(len(la), len(lb))) if la[k] != lb[k]), min(len(la), len(lb)))
            diffs.append(f"{f}: first difference at line {i + 1}:\n    default : {la[i] if i < len(la) else '<end>'}\n"
                         f"    blocking: {lb[i] if i < len(lb) else '<end>'}")
    nlines = sum(len(open(os.path.join(da, f), encoding="utf-8").read().splitlines()) for f in fa)
    print(f"== async_check {a.device} ({ARCH[a.device]}) models {a.models} modes {a.modes} suites {a.suites}")
    print(f"   files {len(fa)} / {len(fb)}, {nlines} G lines; golden.ps1 exit default {rcs['default']} blocking {rcs['blocking']}; "
          f"time default {walls['default']:.0f} s, HIP_LAUNCH_BLOCKING=1 {walls['blocking']:.0f} s")
    for d in diffs:
        print("   DIFF " + d)
    ok = not diffs and len(fa) > 0
    print(f"   {'IDENTICAL' if ok else 'DIFFERENT'}; wall {time.time() - tw0:.0f} s; outputs {base}")
    if not ok:
        return 1
    return 0 if rcs["default"] == 0 and rcs["blocking"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
