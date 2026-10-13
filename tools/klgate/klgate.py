#!/usr/bin/env python3
# KL gate (FIX-TEST-B, MOE-BAL): balance and fast against precise, on a fixed corpus, at fixed context lengths.
# SPDX-License-Identifier: Apache-2.0
#
# For every model x prompt x mode, one `whirl chat` run prefills the whole prompt (raw, no MTP) and
# dumps the teacher-forced next-token logits of the --rows (default 512) positions before each length
# in --lengths (WHIRL_DUMP_LOGITS + WHIRL_DUMP_AT; 16-row forwards, every row computed: d61f6df). Per
# window it computes KL(precise || mode) (mean over the rows) and the top-1 agreement with precise.
# Limits (MOE-BAL, owner decision 2026-10-13):
#   dense models (absolute, per window):
#     balance: KL <= 0.005 and top-1 >= 98%      fast: KL <= 0.03 and top-1 >= 93%
#     (512 rows per window since MOE-BAL: with 128 rows one row is 0.78 points of top-1 and dense
#     fast zh 8k missed 93% by one row; a 256-row run still failed it, and the window start also moves
#     the prefill split, so a window's numbers carry chunk-schedule jitter: audit/MOE-BAL.md)
#   MoE models (arch *moe): relative to the noise floor N measured in the same gate run = precise vs
#     precise with the sequential DeltaNet prefill (WHIRL_GDN_SEQ=1, an equally exact f32 path), same
#     windows; N = the mean of the noise run's window KLs, N_top1(w) = its top-1 in window w:
#     balance: mean KL over the windows <= 2 N, each window KL <= 3 N, top-1(w) >= N_top1(w) - 2.4 points
#     fast:    mean KL <= 8 N, each window KL <= 12 N, top-1(w) >= N_top1(w) - 8 points
#     (two equally exact paths of a 256-expert top-8 router already differ by KL 0.002-0.006 on
#     these prompts, above the dense balance limit; see audit/MOE-KL.md)
# Exit 0 only when every case ran and passed; 1 = a limit failed; 2 = a run / input error or 0 cases.
# The JSON report (--report) holds the exe's SHA-256: tools/golden/golden.ps1 record requires a
# passing report for the same exe before it writes fast / balance references.
#
# Corpus: --corpus DIR with the files listed in corpus.sha256 next to this script (fixed texts; a
# changed file is an error, so numbers stay comparable across releases). Defaults: code = C++/HIP
# source with Chinese design notes, zh = Chinese prose docs with some code, each >= 129k tokens.
#
# usage: python klgate.py --exe build\Release\whirl.exe --model dense=Q.gguf --model moe=M.gguf
#            --corpus DIR [--device r9700|8060s] [--lengths 8192,32768,98304,131072] [--modes balance,fast]
#            [--report out.json] [--work DIR] [--wrap "python gpu_lock.py --"] [--only dense:code] [--rows 512]

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time

import numpy as np

LIMITS = {"balance": (0.005, 0.98), "fast": (0.03, 0.93)}  # dense: absolute (KL max, top-1 min) per window
# MoE: (mean KL <= k_mean * N, window KL <= k_win * N, top-1(w) >= N_top1(w) - drop)
MOE_LIMITS = {"balance": (2.0, 3.0, 0.024), "fast": (8.0, 12.0, 0.08)}
ROWS = 512
HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def load_corpus(d):
    want = {}
    with open(os.path.join(HERE, "corpus.sha256"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                h, name = line.split()
                want[name] = h
    out = {}
    for name, h in want.items():
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            raise SystemExit(f"klgate: corpus file missing: {p}")
        got = sha256(p)
        if got != h:
            raise SystemExit(f"klgate: corpus file changed: {p} sha256 {got}, want {h}")
        out[os.path.splitext(name)[0]] = p
    return out


def clean_env():
    # a stray WHIRL_* in the shell (WHIRL_KV, WHIRL_MODE, A/B switches) would test another config
    return {k: v for k, v in os.environ.items() if not k.upper().startswith("WHIRL_")}


def ntokens(exe, model, prompt):
    env = clean_env()
    env["WHIRL_TOKENIZE_ONLY"] = "1"
    args = [exe, "chat", model, "@" + prompt, "--raw", "--max-tokens", "1"]  # tokenizer only: no GPU, no lock
    r = subprocess.run(args, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    ids = [l for l in r.stdout.splitlines() if l and l[0].isdigit()]
    if r.returncode != 0 or not ids:
        raise RuntimeError(f"tokenize failed (exit {r.returncode}): {r.stderr[-400:]}")
    return ids[-1].count(",") + 1


def run_dump(exe, model, prompt, mode, ends, n, dump, logf, wrap, device, extra=None):
    env = clean_env()
    env.update(extra or {})
    env.update({"WHIRL_MTP": "0", "WHIRL_DUMP_LOGITS": dump, "WHIRL_DUMP_AT": ",".join(map(str, ends)), "WHIRL_DUMP_ROWS": str(ROWS),
                "AMD_LOG_LEVEL": "1"})
    ctx = (n + 1 + 255) // 256 * 256 + 256
    args = wrap + [exe, "chat", model, "@" + prompt, "--raw", "--max-tokens", "1", "--ctx", str(ctx), "--" + mode]
    if device:
        args += ["--device", device]
    t0 = time.time()
    with open(logf, "w", encoding="utf-8") as fo:
        fo.write(" ".join(args) + (" [" + " ".join(f"{k}={v}" for k, v in extra.items()) + "]" if extra else "") + "\n")
        fo.flush()
        r = subprocess.run(args, env=env, stdout=fo, stderr=subprocess.STDOUT)
    return r.returncode, time.time() - t0


def load_dump(p, nwin):
    # kept in f16 (512 rows x 4 windows x a 248k vocab = 1 GB); compare() widens one window at a time
    a = np.fromfile(p, dtype=np.float16)
    rows = nwin * ROWS
    if a.size == 0 or a.size % rows:
        raise RuntimeError(f"dump {p}: {a.size} halfs, not a multiple of {rows} rows")
    return a.reshape(rows, a.size // rows)


def log_info(logf):
    # the model arch ("  arch qwen35moe  layers ...") and the numerics line of a run's log
    arch, num = "", ""
    try:
        with open(logf, encoding="utf-8", errors="replace") as f:
            for line in f:
                if not arch:
                    m = re.match(r"\s*arch (\S+)", line)
                    if m:
                        arch = m.group(1)
                if not num and line.strip().startswith("numerics:"):
                    num = line.strip()
    except OSError:
        pass
    return arch, num


def logsm(x):
    z = x - x.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def compare(ref, test):
    ref = ref.astype(np.float32)
    test = test.astype(np.float32)
    lr = logsm(ref)
    kl = (np.exp(lr) * (lr - logsm(test))).sum(1)
    top1 = (ref.argmax(1) == test.argmax(1)).mean()
    return float(kl.mean()), float(np.quantile(kl, 0.99)), float(kl.max()), float(top1)


def judge_moe(res, noise, md):
    """MoE limits relative to the noise floor. res = [(length, zero_rows, kl, p99, max, top1)] per window,
    noise = {"n_mean": N, "windows": [{"top1": ...}, ...]}. Returns (window cases, mean case) as dicts."""
    k_mean, k_win, drop = MOE_LIMITS[md]
    nm = noise["n_mean"]
    wins = []
    for w, (e, zt, kl, p99, mx, top1) in enumerate(res):
        nt1 = noise["windows"][w]["top1"]
        ok = zt == 0 and kl <= k_win * nm and top1 >= nt1 - drop - 1e-9
        wins.append({"length": e, "kl_mean": kl, "kl_p99": p99, "kl_max": mx, "top1": top1, "zero_rows": zt, "rule": "moe-window",
                     "kl_limit": k_win * nm, "top1_limit": nt1 - drop, "x_noise": kl / nm, "pass": ok})
    mean = float(np.mean([x[2] for x in res]))
    mcase = {"length": 0, "kl_mean": mean, "rule": "moe-mean", "kl_limit": k_mean * nm, "x_noise": mean / nm, "pass": mean <= k_mean * nm}
    return wins, mcase


def main():
    global ROWS
    ap = argparse.ArgumentParser(description="KL gate: balance / fast vs precise")
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", action="append", required=True, help="NAME=PATH (repeat)")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--device", default="")
    ap.add_argument("--lengths", default="8192,32768,98304,131072")
    ap.add_argument("--modes", default="balance,fast")
    ap.add_argument("--report", default="")
    ap.add_argument("--work", default="")
    ap.add_argument("--wrap", default="", help="command prefix for every run (a GPU lock)")
    ap.add_argument("--only", default="", help="MODEL:PROMPT[,...] subset")
    ap.add_argument("--keep", action="store_true", help="keep the logits dumps")
    ap.add_argument("--rows", type=int, default=ROWS, help=f"rows per window (default {ROWS})")
    a = ap.parse_args()
    ROWS = a.rows

    exe = os.path.abspath(a.exe)
    if not os.path.isfile(exe):
        print(f"klgate: no exe {exe}")
        return 2
    models = {}
    for m in a.model:
        name, _, path = m.partition("=")
        if not path or not os.path.isfile(path):
            print(f"klgate: bad --model {m}")
            return 2
        models[name] = path
    modes = [x for x in a.modes.split(",") if x]
    for md in modes:
        if md not in LIMITS:
            print(f"klgate: unknown mode {md}")
            return 2
    lengths = sorted(int(x) for x in a.lengths.split(","))
    corpus = load_corpus(a.corpus)
    only = set(x for x in a.only.split(",") if x)
    wrap = shlex.split(a.wrap, posix=False) if a.wrap else []
    work = os.path.abspath(a.work or os.path.join(os.path.dirname(exe), "klgate_work"))
    os.makedirs(work, exist_ok=True)
    rep = {"exe": exe, "exe_sha256": sha256(exe), "device": a.device or "default", "limits": LIMITS, "moe_limits": MOE_LIMITS,
           "rows": ROWS, "lengths": lengths, "modes": modes, "models": {k: os.path.basename(v) for k, v in models.items()},
           "corpus": {k: sha256(v) for k, v in corpus.items()}, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": [],
           "noise": [], "numerics": {}, "errors": []}
    log(f"klgate: exe {exe} sha256 {rep['exe_sha256'][:16]} device {rep['device']} rows {ROWS}")

    for mname, mpath in models.items():
        for pname, ppath in corpus.items():
            if only and f"{mname}:{pname}" not in only:
                continue
            try:
                n = ntokens(exe, mpath, ppath)
            except Exception as e:
                rep["errors"].append(f"{mname}:{pname}: {e}")
                log(f"ERROR {mname}:{pname}: {e}")
                continue
            # the windows end at each length (the longest: the prompt end if the prompt is shorter;
            # it must be within 4% of the length, so the corpus really reaches it)
            ends = []
            for L in lengths:
                e = min(L, n)
                if e < L * 0.96:
                    rep["errors"].append(f"{mname}:{pname}: prompt has {n} tokens, too short for {L}")
                    continue
                if not ends or e - ends[-1] >= ROWS:
                    ends.append(e)
            if not ends:
                continue
            log(f"{mname}:{pname}: {n} tokens, windows end at {ends}")
            ref = None
            res_by = {}  # mode -> per-window results (each dump is compared right after its run; only precise stays in memory)
            moe = False
            ok_runs = True
            runs = ["precise"]
            ri = 0

            def windows(t):
                out = []
                for w, e in enumerate(ends):
                    s = slice(w * ROWS, (w + 1) * ROWS)
                    zt = int((np.abs(t[s]).max(1) == 0).sum())
                    out.append((e, zt) + compare(ref[s], t[s]))
                return out

            while ri < len(runs):
                md = runs[ri]
                ri += 1
                tag = f"{mname}_{pname}_{md}"
                dump = os.path.join(work, tag + ".f16")
                logf = os.path.join(work, tag + ".log")
                # the MoE noise floor: precise with the sequential f32 DeltaNet prefill
                extra = {"WHIRL_GDN_SEQ": "1"} if md == "noise" else None
                rc, wall = run_dump(exe, mpath, ppath, "precise" if md == "noise" else md, ends, n, dump, logf, wrap, a.device, extra)
                log(f"  run {tag}: exit {rc}, {wall:.0f} s")
                if rc != 0 or not os.path.isfile(dump):
                    rep["errors"].append(f"{tag}: exit {rc} (log {logf})")
                    ok_runs = False
                    break
                arch, num = log_info(logf)
                rep["numerics"][tag] = num
                try:
                    t = load_dump(dump, len(ends))
                except Exception as e:
                    rep["errors"].append(f"{tag}: {e}")
                    ok_runs = False
                    break
                if not a.keep:
                    os.remove(dump)
                if md == "precise":
                    ref = t
                    zero = int((np.abs(ref).max(1) == 0).sum())
                    if zero:
                        rep["errors"].append(f"{mname}:{pname}: {zero} all-zero rows in the precise dump")
                        ok_runs = False
                        break
                    moe = arch.endswith("moe")
                    runs += (["noise"] if moe else []) + modes
                    log(f"  {mname}: arch {arch or '?'} -> {'MoE: limits relative to the noise floor' if moe else 'dense: absolute limits'}")
                    continue
                if t.shape != ref.shape:
                    rep["errors"].append(f"{mname}:{pname}:{md}: dump shape {t.shape} != precise {ref.shape}")
                    continue
                res_by[md] = windows(t)
                del t
            ref = None
            if not ok_runs:
                continue

            noise = None
            if moe:
                if "noise" not in res_by:
                    rep["errors"].append(f"{mname}:{pname}: MoE noise floor run missing")
                    continue
                nw = res_by["noise"]
                n_mean = float(np.mean([x[2] for x in nw]))
                noise = {"model": mname, "prompt": pname, "n_mean": n_mean,
                         "windows": [{"length": x[0], "kl_mean": x[2], "kl_p99": x[3], "kl_max": x[4], "top1": x[5]} for x in nw]}
                rep["noise"].append(noise)
                log(f"  noise  {mname:6s} {pname:5s}: N = {n_mean:.5f}  (" +
                    ", ".join(f"{x[0]}: KL {x[2]:.5f} top1 {x[5] * 100:.2f}%" for x in nw) + ")")
                if n_mean <= 0:
                    rep["errors"].append(f"{mname}:{pname}: noise floor N = 0 (WHIRL_GDN_SEQ=1 did not change the bits)")
                    continue
            for md in modes:
                if md not in res_by:
                    continue
                res = res_by[md]
                base = {"model": mname, "prompt": pname, "mode": md}
                if moe:
                    wins, mcase = judge_moe(res, noise, md)
                    k_mean, k_win, _ = MOE_LIMITS[md]
                    for c in wins:
                        rep["cases"].append({**base, **c})
                        log(f"  {'PASS' if c['pass'] else 'FAIL'} {mname:6s} {pname:5s} {md:8s} {c['length']:7d}: KL mean {c['kl_mean']:.5f} "
                            f"= {c['x_noise']:.2f} N (limit {k_win:g} N) p99 {c['kl_p99']:.4f} max {c['kl_max']:.4f}  top1 {c['top1'] * 100:.2f}% "
                            f"(limit {c['top1_limit'] * 100:.2f}%)" + (f"  zero rows {c['zero_rows']}" if c["zero_rows"] else ""))
                    rep["cases"].append({**base, **mcase})
                    log(f"  {'PASS' if mcase['pass'] else 'FAIL'} {mname:6s} {pname:5s} {md:8s}    mean: KL {mcase['kl_mean']:.5f} = "
                        f"{mcase['x_noise']:.2f} N (limit {k_mean:g} N = {mcase['kl_limit']:.5f})")
                else:
                    kl_max, top1_min = LIMITS[md]
                    for e, zt, kl, p99, mx, top1 in res:
                        ok = zt == 0 and kl <= kl_max and top1 >= top1_min
                        rep["cases"].append({**base, "length": e, "kl_mean": kl, "kl_p99": p99, "kl_max": mx, "top1": top1, "zero_rows": zt,
                                             "rule": "dense-abs", "kl_limit": kl_max, "top1_limit": top1_min, "pass": ok})
                        log(f"  {'PASS' if ok else 'FAIL'} {mname:6s} {pname:5s} {md:8s} {e:7d}: KL mean {kl:.5f} p99 {p99:.4f} max {mx:.4f}  top1 {top1 * 100:.2f}%"
                            + (f"  zero rows {zt}" if zt else ""))

    npass = sum(1 for c in rep["cases"] if c["pass"])
    nfail = len(rep["cases"]) - npass
    rc = 2 if rep["errors"] or not rep["cases"] else (1 if nfail else 0)
    rep["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    rep["pass"] = rc == 0
    rep["summary"] = f"{npass} passed, {nfail} failed, {len(rep['errors'])} errors"
    for e in rep["errors"]:
        log("ERROR " + e)
    log(f"klgate: {rep['summary']} -> {'PASS' if rc == 0 else 'FAIL'} (exit {rc})")
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
