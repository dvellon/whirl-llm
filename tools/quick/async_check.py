#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# FASTLANE FL-1a / REF-MINE T0-A: async regression test. The golden suites (plain prefill short + 4000 tokens,
# decode steps, MTP decode, forced n-gram drafts, two-sequence server paths = segmented prefill / batched rows /
# batched verify / one batched MTP cycle) must give byte-identical 'G' lines (top-5 ids + FNV-1a hash of the logits
# per step, token streams) when every kernel launch is synchronous. A difference means a missing dependency between
# streams / host copies that the synchronous run hides: a correctness bug.
#   run "default"  : tools/golden/golden.ps1 check (also compared with the recorded references)
#   run "blocking" : whirl golden directly with HIP_LAUNCH_BLOCKING=1 + WHIRL_NO_GRAPH=1
#   run "nograph"  : whirl golden directly with WHIRL_NO_GRAPH=1 (suite single only: separates the graph switch)
# WHIRL_NO_GRAPH=1 is needed with HIP_LAUNCH_BLOCKING=1: the HIP runtime cannot replay the decode-step HIP graph in
# launch-blocking mode ('hip_graph_internal.cpp UpdateStreams failed', whirl exits 0xC0000374 at the first decode
# step; FL-1a 2026-10-15), so the plain decode step runs as individual launches there.
#
#   python async_check.py --device r97|8s [--models dense,moe] [--modes precise,balance,fast] [--suites single,pair,ngram]
#                         [--exe whirl.exe]
# Exit 0 = identical (and the default run matched the references), 1 = a difference, 2 = a run error.
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
MODELS = {"dense": qlib.MODEL,
          "moe": r"C:\Users\tsaip\.lmstudio\models\ornith-ai\Ornith-1.5-35B-A3B-MXFP4\Ornith-1.5-35B-A3B-MXFP4.gguf"}


def direct(a, out, extra, suites, logf):
    """whirl golden for every model x mode x suite with extra env; writes <model>_<mode>_<suite>.txt (G lines)."""
    os.makedirs(out, exist_ok=True)
    env = qlib.clean_env(**extra)
    env.pop("HIP_LAUNCH_BLOCKING", None)
    env.update(extra)
    bad = 0
    with open(logf, "w", encoding="utf-8") as lf:
        for m in a.models.split(","):
            for md in a.modes.split(","):
                for s in suites:
                    t = time.time()
                    r = subprocess.run([a.exe, "golden", MODELS[m], "--suite", s, "--" + md, "--device", ARCH[a.device]], env=env,
                                       capture_output=True, text=True, encoding="utf-8", errors="replace")
                    g = [l for l in r.stdout.splitlines() + r.stderr.splitlines() if l.startswith("G ")]
                    open(os.path.join(out, f"{m}_{md}_{s}.txt"), "w", encoding="utf-8").write("\n".join(g) + "\n")
                    lf.write(f"{m}/{md}_{s}: exit {r.returncode}, {len(g)} G lines, {time.time() - t:.0f} s\n")
                    if r.returncode != 0:
                        bad += 1
                        lf.write(r.stderr[-2000:] + "\n")
    return bad


def compare(ref_dir, test_dir, skip_ref_model_line=True):
    diffs, n = [], 0
    for f in sorted(os.listdir(test_dir)):
        rp = os.path.join(ref_dir, f)
        lb = open(os.path.join(test_dir, f), encoding="utf-8").read().splitlines()
        if not os.path.isfile(rp):
            diffs.append(f"{f}: no default-run file")
            continue
        la = [l for l in open(rp, encoding="utf-8").read().splitlines() if l.startswith("G ") and not l.startswith("G model ")]
        n += 1
        if la != lb:
            i = next((k for k in range(min(len(la), len(lb))) if la[k] != lb[k]), min(len(la), len(lb)))
            diffs.append(f"{f}: first difference at G line {i + 1} ({len(la)} vs {len(lb)} lines):\n      default: "
                         f"{la[i] if i < len(la) else '<end>'}\n      test   : {lb[i] if i < len(lb) else '<end>'}")
    return n, diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["r97", "8s"], required=True)
    ap.add_argument("--models", default="dense,moe")
    ap.add_argument("--modes", default="precise,balance,fast")
    ap.add_argument("--suites", default="single,pair,ngram")
    ap.add_argument("--exe", default=qlib.DEF_EXE)
    a = ap.parse_args()
    tw0 = time.time()
    base = os.path.join(qlib.WORK, "fl1a", "async", f"{time.strftime('%Y%m%d-%H%M%S')}_async_{a.device}")
    os.makedirs(base, exist_ok=True)
    suites = a.suites.split(",")
    walls = {}
    with qlib.GpuLock():
        t = time.time()
        env = dict(os.environ)
        env.pop("HIP_LAUNCH_BLOCKING", None)
        with open(os.path.join(base, "default.log"), "w", encoding="utf-8") as fo:
            rc_def = subprocess.run(["pwsh", "-NoProfile", "-File", GOLDEN, "check", "-Arch", ARCH[a.device], "-Models", a.models,
                                     "-Modes", a.modes, "-Suites", a.suites, "-Exe", a.exe, "-Out", os.path.join(base, "default")],
                                    env=env, stdout=fo, stderr=subprocess.STDOUT).returncode
        walls["default"] = time.time() - t
        qlib.log(f"default: golden.ps1 exit {rc_def}, {walls['default']:.0f} s")
        t = time.time()
        bad_b = direct(a, os.path.join(base, "blocking"), {"HIP_LAUNCH_BLOCKING": "1", "WHIRL_NO_GRAPH": "1"}, suites,
                       os.path.join(base, "blocking.log"))
        walls["blocking"] = time.time() - t
        qlib.log(f"blocking (+NO_GRAPH): {bad_b} failed runs, {walls['blocking']:.0f} s")
        bad_n = 0
        if "single" in suites:
            t = time.time()
            bad_n = direct(a, os.path.join(base, "nograph"), {"WHIRL_NO_GRAPH": "1"}, ["single"], os.path.join(base, "nograph.log"))
            walls["nograph"] = time.time() - t
            qlib.log(f"nograph: {bad_n} failed runs, {walls['nograph']:.0f} s")
    nb, db = compare(os.path.join(base, "default"), os.path.join(base, "blocking"))
    nn, dn = compare(os.path.join(base, "default"), os.path.join(base, "nograph")) if "single" in suites else (0, [])
    print(f"== async_check {a.device} ({ARCH[a.device]}) models {a.models} modes {a.modes} suites {a.suites}")
    print(f"   default (golden.ps1 vs references): exit {rc_def}, {walls['default']:.0f} s")
    print(f"   HIP_LAUNCH_BLOCKING=1 + WHIRL_NO_GRAPH=1: {nb} files compared, {bad_b} failed runs, {walls['blocking']:.0f} s")
    if "nograph" in walls:
        print(f"   WHIRL_NO_GRAPH=1 (single): {nn} files compared, {bad_n} failed runs, {walls['nograph']:.0f} s")
    for d in db:
        print("   DIFF blocking " + d)
    for d in dn:
        print("   DIFF nograph " + d)
    ok = not db and not dn and nb > 0
    print(f"   {'IDENTICAL' if ok else 'DIFFERENT'}; wall {time.time() - tw0:.0f} s; outputs {base}")
    if not ok:
        return 1
    return 0 if rc_def == 0 and bad_b == 0 and bad_n == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
