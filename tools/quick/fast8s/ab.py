#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# FAST-8S-1: tok/s A/B on the 8060S (ab8s.py of PROF-8S with a chosen server exe and a chat run).
# One server start per variant per round, b8s.py sweep (25 s thermal warm-up, qa doc prompt + nonce,
# 256 tokens) then b8s.py chat (zh coding prompt, 512 tokens). Variants interleaved, order reversed on
# even rounds. A variant = [EXETAG@]MODE[:K=V,K=V...]; EXETAG picks --exe NAME=PATH.
#   python ab.py --model ornith --exe new=...\whirl-server.exe --variants "new@balance;new@fast:WHIRL_DQ4_DYN=0" \
#       --lengths 2048,8192 --rounds 1 --round0 1 --out F
import argparse, os, subprocess, sys, time

sys.path.insert(0, r"C:\ClaudeWorkspace\whirl-work\wt_prof8s\tools\quick")
import qlib  # noqa: E402

PY = r"C:\ClaudeWorkspace\tools\pyref\Scripts\python.exe"
B8S = r"C:\ClaudeWorkspace\whirl-work\base8s\b8s.py"
MODELS = {
    "swift": r"C:\Users\tsaip\.lmstudio\models\ukisai\Swift-1.5-Qwen3.8-27b-MXFP4\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf",
    "ornith": r"C:\Users\tsaip\.lmstudio\models\ornith-ai\Ornith-1.5-35B-A3B-MXFP4\Ornith-1.5-35B-A3B-MXFP4.gguf",
}
PORT = 8198
TASK = "fast8s"


def run(a, exes, variant, rnd, ctxslot, logdir):
    tag, _, rest = variant.rpartition("@")
    exe = exes[tag or "new"]
    mode, _, kv = rest.partition(":")
    env = qlib.clean_env(HIP_VISIBLE_DEVICES="0")
    env.update(dict(x.split("=", 1) for x in kv.split(",") if x))
    label = "f8s-" + variant.replace("@", "-").replace(":", "+").replace(",", "+")
    slog = os.path.join(logdir, f"{time.strftime('%Y%m%d-%H%M%S')}_{a.model}_{label}_r{rnd}.server.log")
    args = [exe, MODELS[a.model], "--port", str(PORT), "--log-file", slog, "--kv-ssd-mode", "off", "-np", "1",
            "--ctx-per-slot", str(ctxslot), "--device", "8060s", "--" + mode]
    p = subprocess.Popen(args, env=env, stdout=open(slog + ".stderr", "w"), stderr=subprocess.STDOUT,
                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    qlib.add_pid(TASK, p.pid, "whirl-server ab " + label)
    url = f"http://127.0.0.1:{PORT}"
    try:
        while True:
            if p.poll() is not None:
                raise RuntimeError(f"server exited ({p.returncode}) {slog}")
            try:
                qlib.http(url + "/health", timeout=5)
                break
            except Exception:
                time.sleep(1)
        common = ["--url", url, "--engine", label, "--model", a.model, "--mode", mode, "--out", a.out, "--round", str(rnd),
                  "--expect-device", "gfx1151", "--server-log", slog]
        if a.lengths:
            subprocess.run([PY, B8S, "sweep", *common, "--lengths", a.lengths, "--max-tokens", "256"], check=False)
        if a.chat:
            subprocess.run([PY, B8S, "chat", *common, "--runs", "1"], check=False)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait(60)
        qlib.drop_pid(TASK, p.pid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exe", action="append", default=[])
    ap.add_argument("--variants", required=True)
    ap.add_argument("--lengths", default="2048")
    ap.add_argument("--chat", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--round0", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    exes = dict(x.split("=", 1) for x in a.exe)
    logdir = os.path.join(qlib.WORK, "fast8s", "logs")
    os.makedirs(logdir, exist_ok=True)
    ctxslot = max([int(x) for x in a.lengths.split(",") if x] + [8192]) + 4096
    vs = a.variants.split(";")
    for r in range(a.round0, a.round0 + a.rounds):
        for v in (vs if r % 2 else vs[::-1]):
            run(a, exes, v, r, ctxslot, logdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
