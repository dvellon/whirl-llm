#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# PROF-8S: RGP capture of one phase (prefill chunk or MTP decode) of a WHIRL server on the Radeon 8060S only,
# for the MXFP4 product models (Swift dense / Ornith MoE), any numerics mode. Derived from tools/quick/rgp_capture.py
# (FL-1a) with these differences:
#   - model selectable (--model swift|ornith|PATH); prompt token counts come from the shared fl1a prompt files
#     (Qwen tokenizer, same family);
#   - the server binary is a renamed copy (p8sprof-server.exe) and the CLI filter is -p p8sprof, so no other whirl
#     process on the machine (e.g. another session's R9700 server) can be hooked by the profiler;
#   - no machine-wide GPU lock (owner-approved 8060S use alongside an R9700 job); HIP_VISIBLE_DEVICES=0 = 8060S;
#   - decode is captured at depth ctx (request A only, C = A again), prefill = request B (A + 2048 rows) at depth
#     ctx; ctx 0 = a 2048-token prompt from an empty cache.
#   python prof8s.py --model ornith --mode fast --phase decode --ctx 2048 [--delay S] [--env K=V;...]
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import qlib  # noqa: E402
import rgp_capture as rc  # noqa: E402

WORK = qlib.WORK
BIN = os.path.join(WORK, "prof8s", "bin")
SERVER = os.path.join(BIN, "p8sprof-server.exe")
MODELS = {
    "swift": r"C:\Users\tsaip\.lmstudio\models\ukisai\Swift-1.5-Qwen3.8-27b-MXFP4\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf",
    "ornith": r"C:\Users\tsaip\.lmstudio\models\ornith-ai\Ornith-1.5-35B-A3B-MXFP4\Ornith-1.5-35B-A3B-MXFP4.gguf",
}
PORT = 8197


def log(m, f=None):
    s = time.strftime("%H:%M:%S ") + m
    print(s, flush=True)
    if f:
        f.write(s + "\n")
        f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="balance")
    ap.add_argument("--phase", choices=["decode", "prefill"], required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--ops", type=int, default=3000)
    ap.add_argument("--skip", type=int, default=2000, help="decode: dispatches skipped after arming")
    ap.add_argument("--buffer", default="high")
    ap.add_argument("--delay", type=float, required=True, help="seconds from HIP connect to arming (server idle then)")
    ap.add_argument("--env", default="")
    ap.add_argument("--reserve-mb", type=int, default=6144)
    ap.add_argument("--dec-tokens", type=int, default=256)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-parse", action="store_true", help="leave the .rgp for parse.py (pipelined with the next capture)")
    ap.add_argument("--out", default=os.path.join(WORK, "prof8s", "cap"))
    a = ap.parse_args()
    model = MODELS.get(a.model, a.model)
    os.makedirs(a.out, exist_ok=True)
    tag = f"{time.strftime('%Y%m%d-%H%M%S')}_{a.model}_{a.mode}_{a.phase}_ctx{a.ctx}{('_' + a.tag) if a.tag else ''}"
    lf = open(os.path.join(a.out, tag + ".log"), "w", encoding="utf-8")
    exe = os.path.join(BIN, "whirl.exe")
    ta = open(qlib.prompt_file(exe, a.ctx), encoding="utf-8").read() if a.ctx else ""
    tb = open(qlib.prompt_file(exe, a.ctx + 2048), encoding="utf-8").read()
    assert tb.startswith(ta)
    res = {"tag": tag, "model": model, "mode": a.mode, "phase": a.phase, "ctx": a.ctx, "env": a.env}
    rgp_base = os.path.join(a.out, tag + ".rgp")
    start = 1 if a.phase == "prefill" else a.skip
    cli_args = [os.path.join(rc.RDTS, "RadeonDeveloperPanelCLI.exe"), "-m", "profiling", "-p", "p8sprof",
                "-o", rgp_base, "--rgp-capture-mode", "dispatch", "--rgp-render-op-count", str(a.ops),
                f"--rgp-auto-capture=dispatch:{start}:{a.ops}", "--rgp-auto-capture-delay", str(int(a.delay * 1000)),
                "--rgp-sqtt-buffer-size", a.buffer, "--verbose"]
    cli = subprocess.Popen(cli_args, cwd=rc.RDTS, stdin=subprocess.PIPE, stdout=open(os.path.join(a.out, tag + ".cli.log"), "w"),
                           stderr=subprocess.STDOUT, text=True)
    qlib.add_pid("prof8s", cli.pid, "RadeonDeveloperPanelCLI " + tag)
    srv_p = None
    try:
        time.sleep(5)
        env = qlib.clean_env(HIP_VISIBLE_DEVICES="0", WHIRL_POOL_RESERVE_MB=str(a.reserve_mb), AMD_LOG_LEVEL="1")
        env.update(dict(kv.split("=", 1) for kv in a.env.split(";") if kv))
        slog = os.path.join(a.out, tag + ".server.log")
        args = [SERVER, model, "--port", str(PORT), "--log-file", slog, "--kv-ssd-mode", "off", "-np", "1",
                "--ctx-per-slot", str(a.ctx + 4096 + 512), "--device", "8060s", "--" + a.mode]
        err = open(slog + ".stderr", "w")
        t_srv = time.time()
        srv_p = subprocess.Popen(args, env=env, stdout=err, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        qlib.add_pid("prof8s", srv_p.pid, "p8sprof-server " + tag)
        url = f"http://127.0.0.1:{PORT}"
        while True:
            if srv_p.poll() is not None:
                raise RuntimeError(f"server exited during startup ({srv_p.returncode})")
            try:
                qlib.http(url + "/health", timeout=5)
                break
            except Exception:
                time.sleep(1)
        res["load_s"] = round(time.time() - t_srv, 1)
        log(f"server up in {res['load_s']} s", lf)
        comp = lambda p, n: qlib.http(url + "/v1/completions", {"prompt": p, "max_tokens": n, "temperature": 0, "stream": False})
        if ta:
            t = time.time()
            r = comp(ta, 1)
            res["A_s"] = round(time.time() - t, 1)
            log(f"A ({a.ctx}) done {res['A_s']} s timings {r.get('timings')}", lf)
        arm = t_srv + a.delay + 4
        if time.time() > arm:
            res["warn"] = f"armed while busy ({time.time() - arm:.0f} s late)"
            log("WARNING " + res["warn"], lf)
        while time.time() < arm:
            time.sleep(0.5)
        before = set(glob.glob(os.path.join(a.out, tag + "*.rgp")))
        pd = ta if (a.phase == "decode" and ta) else tb
        if a.phase == "prefill":
            t = time.time()
            r = comp(tb, 1)
            res["B"] = {"wall_s": round(time.time() - t, 2), "timings": r.get("timings")}
            log(f"B (prefill) {res['B']}", lf)
            time.sleep(2)
        f = None
        for k in range(8):
            t = time.time()
            r = comp(pd, a.dec_tokens if a.phase == "decode" else 64)
            tm = r.get("timings") or {}
            res.setdefault("C", []).append(tm)
            log(f"C{k} {time.time() - t:.1f} s {tm}", lf)
            f = rc.wait_file(os.path.join(a.out, tag + "*.rgp"), before, 25)
            if f:
                break
        if not f:
            f = rc.wait_file(os.path.join(a.out, tag + "*.rgp"), before, 120)
        res["rgp"] = f
        log(f"rgp {f}", lf)
    finally:
        try:
            cli.stdin.write("q\n")
            cli.stdin.flush()
            cli.wait(20)
        except Exception:
            pass
        if cli.poll() is None:
            cli.kill()
        qlib.drop_pid("prof8s", cli.pid)
        if srv_p is not None:
            if srv_p.poll() is None:
                srv_p.kill()
                srv_p.wait(60)
            qlib.drop_pid("prof8s", srv_p.pid)
    if res.get("rgp") and not a.no_parse:
        base = os.path.join(a.out, tag)
        p = subprocess.run([rc.PARSER_PY, os.path.join(rc.PARSER, "main.py"), res["rgp"], base], cwd=rc.PARSER, capture_output=True, text=True)
        res["parse_rc"] = p.returncode
        if p.returncode == 0:
            sm = rc.summarize(base + "_dispatches.csv", base + "_kernels.csv", 300000 if a.phase == "prefill" else 0)
            res["summary"] = sm
            log(f"{sm['dispatches']} dispatches span {sm['span_us'] / 1e3:.2f} ms busy {sm['busy_us'] / 1e3:.2f} ms gaps {sm['gap_us'] / 1e3:.2f} ms", lf)
            for k, us, sh, n in sm["top"][:12]:
                log(f"  {sh:6.2f}% {us / 1e3:8.3f} ms x{n:<5d} {k[:100]}", lf)
            if os.path.getsize(res["rgp"]) > 200e6:
                os.remove(res["rgp"])
                res["rgp_deleted"] = True
        else:
            res["parse_err"] = (p.stderr or p.stdout)[-1500:]
            log("parse failed " + res["parse_err"][-400:], lf)
    json.dump(res, open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
    log("done " + tag, lf)
    return 0 if res.get("summary") else 1


if __name__ == "__main__":
    sys.exit(main())
