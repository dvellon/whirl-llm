#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# FASTLANE FL-1a: 5-minute tok/s quick test (Qwen3.8-27B Q4_K_M): one whirl-server start, then the REL3 sweep
# requests (rel3rc5\rc5.py: unrelated warm-up, then per length a chat request of the REL3 qa prompt with a unique
# first line, 256 tokens, thinking off, default MTP + n-gram) -> decode tok/s at each length and prefill tok/s.
# The numbers are directly comparable with rel3rc5\results_*.tsv (same prompts, same client).
#
#   python toks.py --device r97|8s [--mode balance] [--exe whirl.exe] [--lengths 2048,8192,32768] [--tag x]
#                  [--env K=V;K2=V2] [--out results.tsv]
# Default lengths: 2048 + 8192 + 32768 (decode at 2k / 32k, prefill at 8k); every length reports both.
import argparse
import os
import sys
import time
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlib  # noqa: E402

sys.path.insert(0, os.path.join(qlib.WORK, "rel3rc5"))
import rc5  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["r97", "8s"], required=True)
    ap.add_argument("--mode", default="balance")
    ap.add_argument("--exe", default=qlib.DEF_EXE)
    ap.add_argument("--lengths", default="2048,8192,32768")
    ap.add_argument("--tag", default="")
    ap.add_argument("--env", default="", help="extra env for the server, K=V;K2=V2 (e.g. WHIRL_KV=q8h)")
    ap.add_argument("--out", default=os.path.join(qlib.WORK, "fl1a", "toks.tsv"))
    ap.add_argument("--fixed-nonce", action="store_true",
                    help="same first prompt line every run (FL-1b gate): a bitwise-equal candidate generates the same tokens, "
                         "so MTP acceptance (the main decode jitter) and the output hash match the reference run")
    a = ap.parse_args()
    if a.fixed_nonce:
        orig = rc5.doc_prompt
        rc5.doc_prompt = lambda n, kind, nonce: orig(n, kind, f"fl-fixed-{n}")
    env = dict(kv.split("=", 1) for kv in a.env.split(";") if kv)
    lens = [int(x) for x in a.lengths.split(",")]
    tw0 = time.time()
    ctx = max(lens) + 2048
    eng = "whirl" + (("_" + a.tag) if a.tag else "")
    slog = os.path.join(qlib.WORK, "fl1a", "logs", f"{time.strftime('%Y%m%d-%H%M%S')}_toks_{a.device}_{eng}_{a.mode}.server.log")
    os.makedirs(os.path.dirname(slog), exist_ok=True)
    rc5.OUT = Path(a.out)
    n0 = sum(1 for _ in open(a.out, encoding="utf-8")) if os.path.isfile(a.out) else 0
    with qlib.GpuLock():
        tl = time.time()
        with qlib.Server(a.exe, a.device, a.mode, ctx, slog, env=env) as s:
            qlib.log(f"server up in {s.t_load:.0f} s ({a.device} {a.mode} {a.env or 'default env'})")
            ns = SimpleNamespace(url=s.url, engine=eng, model="qwen", mode=a.mode + (f"[{a.env}]" if a.env else ""),
                                 lengths=a.lengths, max_tokens=256, fresh="")
            rc5.cmd_sweep(ns)
        tg = time.time() - tl
    rows = [l.rstrip("\n").split("\t") for l in open(a.out, encoding="utf-8")][max(n0, 1):]
    print(f"\n== toks {a.device} {a.mode} {a.env} exe {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(a.exe))))}")
    print(f"{'length':>7} {'prompt':>7} {'prefill t/s':>11} {'ttft s':>7} {'decode t/s':>10}  note")
    for r in rows:
        if len(r) >= 11:
            print(f"{r[5]:>7} {r[6]:>7} {r[7]:>11} {r[8]:>7} {r[9]:>10}  {r[11][:60] if len(r) > 11 else ''}")
    print(f"wall {time.time() - tw0:.0f} s (GPU part {tg:.0f} s, server load {s.t_load:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
