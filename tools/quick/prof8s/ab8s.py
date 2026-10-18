#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# PROF-8S: tok/s A/B on the 8060S without the profiler: one p8sprof-server start per variant, the same b8s.py sweep
# (thermal warm-up + qa doc prompt, 256 tokens) the published 0.2.0 numbers used. Variants run interleaved per round.
#   python ab8s.py --model ornith --variants "balance;fast;balance:WHIRL_KV=q4" --lengths 2048,8192 --rounds 2 --out F
# A variant = MODE[:K=V,K=V...].
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import qlib  # noqa: E402
import prof8s  # noqa: E402

PY = r"C:\ClaudeWorkspace\tools\pyref\Scripts\python.exe"
B8S = r"C:\ClaudeWorkspace\whirl-work\base8s\b8s.py"


def run(model, variant, lengths, rnd, out, ctxslot, logdir):
    mode, _, kv = variant.partition(":")
    env = qlib.clean_env(HIP_VISIBLE_DEVICES="0")
    env.update(dict(x.split("=", 1) for x in kv.split(",") if x))
    label = "p8s-" + variant.replace(":", "+").replace(",", "+")
    slog = os.path.join(logdir, f"{time.strftime('%Y%m%d-%H%M%S')}_{model}_{label}_r{rnd}.server.log")
    args = [prof8s.SERVER, prof8s.MODELS[model], "--port", str(prof8s.PORT), "--log-file", slog, "--kv-ssd-mode", "off", "-np", "1",
            "--ctx-per-slot", str(ctxslot), "--device", "8060s", "--" + mode]
    p = subprocess.Popen(args, env=env, stdout=open(slog + ".stderr", "w"), stderr=subprocess.STDOUT,
                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    qlib.add_pid("prof8s", p.pid, "p8sprof-server ab " + label)
    url = f"http://127.0.0.1:{prof8s.PORT}"
    try:
        while True:
            if p.poll() is not None:
                raise RuntimeError(f"server exited ({p.returncode}) {slog}")
            try:
                qlib.http(url + "/health", timeout=5)
                break
            except Exception:
                time.sleep(1)
        subprocess.run([PY, B8S, "sweep", "--url", url, "--engine", label, "--model", model, "--mode", mode, "--out", out,
                        "--round", str(rnd), "--lengths", lengths, "--max-tokens", "256"], check=False)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait(60)
        qlib.drop_pid("prof8s", p.pid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--lengths", default="2048")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--ctx-per-slot", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(qlib.WORK, "prof8s", "ab.tsv"))
    a = ap.parse_args()
    logdir = os.path.join(qlib.WORK, "prof8s", "ablogs")
    os.makedirs(logdir, exist_ok=True)
    ctxslot = a.ctx_per_slot or (max(int(x) for x in a.lengths.split(",")) + 4096)
    vs = a.variants.split(";")
    for r in range(1, a.rounds + 1):
        for v in (vs if r % 2 else vs[::-1]):
            run(a.model, v, a.lengths, r, a.out, ctxslot, logdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
