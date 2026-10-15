#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# FASTLANE FL-1a: one-command RGP capture + parse of a WHIRL server prefill chunk and / or decode steps at a
# given context depth (Qwen3.8-27B Q4_K_M), on the R9700 or the Radeon 8060S.
#
#   python rgp_capture.py --device r97|8s --phase decode|prefill|both [--ctx 2048] [--mode balance] [--exe whirl.exe]
#                         [--ops 3000] [--buffer minimum] [--out DIR] [--no-mtp]
#
# How (audit/rgp_setup.md + ROCm/hip-ep tools/rgp_parser README):
#   1. RadeonDeveloperPanelCLI starts first in interactive profiling mode (-p whirl, dispatch capture mode,
#      --rgp-render-op-count OPS, SQTT buffer BUFFER), so the driver hooks whirl-server when its HIP context opens.
#   2. whirl-server loads; request A = the corpus text of CTX tokens (raw /v1/completions, max_tokens 1) fills the
#      KV to depth CTX and leaves a prompt-end checkpoint there.
#   3. prefill: 'c' is sent to the CLI while the server is idle, so the capture starts at the first dispatch of
#      request B = A's text + about 2048 more tokens: B resumes from A's checkpoint and prefills ~2k new rows at
#      depth CTX (one server prefill chunk, WHIRL_PREFILL_CHUNK default 2048).
#      decode: request C (= B again, streamed, 200 tokens) decodes at depth CTX + 2k; 'c' is sent after 24 tokens.
#   4. The .rgp is decoded by ROCm/hip-ep tools/rgp_parser (MIT, C:\ClaudeWorkspace\tools\rgp_parser) into
#      <name>_dispatches.csv etc., then summarised here: per kernel total busy time, share, count
#      (<name>_kernels.csv) + the captured span and the summed idle gaps.
# The driver runs the GPU at peak clock while capturing (RGP behaviour): use shares and gaps, not absolute tok/s.
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlib  # noqa: E402

RDTS_OLD = r"C:\ClaudeWorkspace\tools\rdts\RadeonDeveloperToolSuite-2026-05-28-1806"  # RDP 3.5.0 / RGP 2.7.0
RDTS_NEW = r"C:\ClaudeWorkspace\tools\rdts\RadeonDeveloperToolSuite-2026-10-02-1872"  # RDP 3.5.1 / RGP 2.7.1
RDTS = RDTS_NEW if os.path.isdir(RDTS_NEW) else RDTS_OLD
# R9700 (gfx1201): the HIP process dies with 0xC0000005 inside amdhip64_7.dll the moment the SQTT trace starts (null
# pointer read at +0x80 in the runtime's RGP/PAL trace path). Same with a 20-line HIP loop, both RDTS versions, the
# 26.10 driver runtime and the ROCm 7.2 / 7.1 SDK runtimes; the 8060S works with the same DLL. AMD bug, see
# audit/RGP-R97.md: --device r97 stops unless --force-r97 (retest after a driver / runtime update).
PARSER = r"C:\ClaudeWorkspace\tools\rgp_parser"
HIP_INDEX = {"8s": "0", "r97": "1"}  # `whirl devices` order on this machine (device 0 8060S, device 1 R9700)
PARSER_PY = os.path.join(PARSER, r".venv\Scripts\python.exe")


def wait_file(pattern, before, timeout):
    """The newest file matching pattern that was not in before, once its size is stable (or None)."""
    t0 = time.time()
    last = (None, -1)
    while time.time() - t0 < timeout:
        new = [f for f in glob.glob(pattern) if f not in before]
        if new:
            f = max(new, key=os.path.getmtime)
            sz = os.path.getsize(f)
            if sz > 0 and last == (f, sz):
                return f
            last = (f, sz)
        time.sleep(2)
    return None


def summarize(csv_path, out_csv, cut_gap_us=0):
    """Per-kernel totals over the captured dispatches; cut_gap_us > 0: only the dispatches before the first idle gap
    longer than that (prefill: request B alone, not the decode request that filled the capture window)."""
    rows = list(csv.DictReader(open(csv_path, newline="")))
    rows = [r for r in rows if r.get("is_artifact", "").lower() not in ("1", "true")]
    rows.sort(key=lambda r: float(r["ts_us"]))
    n_all = len(rows)
    if cut_gap_us:
        for i in range(1, len(rows)):
            if float(rows[i].get("gap_before_us") or 0) > cut_gap_us:
                rows = rows[:i]
                break
    if not rows:
        return None
    agg = defaultdict(lambda: [0.0, 0, 0.0])
    gaps = 0.0
    for r in rows:
        k = r["kernel"] or r["label"] or r["pc"]
        a = agg[k]
        d = float(r["dur_us"] or 0)
        a[0] += d
        a[1] += 1
        a[2] = max(a[2], d)
        if r is not rows[0]:
            gaps += float(r.get("gap_before_us") or 0)
    t0 = min(float(r["ts_us"]) for r in rows)
    t1 = max(float(r["ts_us"]) + float(r["dur_us"] or 0) for r in rows)
    busy = sum(a[0] for a in agg.values())
    out = sorted(agg.items(), key=lambda x: -x[1][0])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "total_us", "share_pct", "count", "mean_us", "max_us"])
        for k, (tot, n, mx) in out:
            w.writerow([k, f"{tot:.1f}", f"{100 * tot / busy:.2f}", n, f"{tot / n:.2f}", f"{mx:.1f}"])
    cnt = lambda pre: sum(1 for r in rows if (r["kernel"] or "").startswith(pre))
    return {"gdn_layers": cnt("gdn_chunk_scan") or cnt("gdn_step"), "attn_dispatches": cnt("attn"),
            "dispatches": len(rows), "dispatches_captured": n_all, "span_us": t1 - t0, "busy_us": busy, "gap_us": gaps, "kernels": len(agg),
            "top": [(k, round(v[0], 1), round(100 * v[0] / busy, 2), v[1]) for k, v in out[:15]]}


def capture(a, ph, tag, ta, tb):
    """One CLI session + one server start for one phase. The CLI's interactive trigger fails for HIP clients
    (begin_trace -> 'Failed to update config', result -2, 2026-10-15), so the capture is armed with dispatch
    auto-capture: --rgp-auto-capture=dispatch:START:OPS --rgp-auto-capture-delay DELAY. The server is idle when the
    delay ends (request A finished before it), so the counted dispatches are those of the phase's request:
    prefill START=1 (B's first dispatch), decode START=--skip (past the restore / first cycles of C)."""
    tw = time.time()
    out = {}
    with qlib.GpuLock():
        qlib.log(f"[{ph}] lock taken")
        cli_log = os.path.join(a.out, f"{tag}_{ph}.cli.log")
        rgp_base = os.path.join(a.out, f"{tag}_{ph}.rgp")
        start = 1 if ph == "prefill" else a.skip
        delay_s = a.delay if a.delay else (40 + a.ctx / (900 if a.device == "r97" else 150))
        cli_args = [os.path.join(RDTS, "RadeonDeveloperPanelCLI.exe"), "-m", "profiling", "-p", "whirl-server",
                    "-o", rgp_base, "--rgp-capture-mode", "dispatch", "--rgp-render-op-count", str(a.ops),
                    f"--rgp-auto-capture=dispatch:{start}:{a.ops}", "--rgp-auto-capture-delay", str(int(delay_s * 1000)),
                    "--rgp-sqtt-buffer-size", a.buffer, "--verbose"]
        cli = subprocess.Popen(cli_args, cwd=RDTS, stdin=subprocess.PIPE, stdout=open(cli_log, "w"), stderr=subprocess.STDOUT, text=True)
        qlib.add_pid("fl1a", cli.pid, "RadeonDeveloperPanelCLI " + tag + " " + ph)
        try:
            time.sleep(5)
            env = {"WHIRL_MTP": "0"} if a.no_mtp else {}
            # RGP traces one GPU per process: with both AMD GPUs visible the trace start fails ('Failed to update
            # config', result -2, manual and auto trigger alike; 2026-10-15). Show the process only its GPU.
            env["HIP_VISIBLE_DEVICES"] = HIP_INDEX[a.device]
            # the SQTT buffer is allocated in VRAM when the trace starts; the KV pool normally fills the R9700 to
            # ~1 GiB free and the server then dies at trace start (2026-10-15): leave room for the trace buffer
            env["WHIRL_POOL_RESERVE_MB"] = str(a.reserve_mb)
            env["AMD_LOG_LEVEL"] = "1"
            env.update(dict(kv.split("=", 1) for kv in a.env.split(";") if kv))
            slog = os.path.join(a.out, f"{tag}_{ph}.server.log")
            t_srv = time.time()
            srv = qlib.Server(a.exe, a.device, a.mode, a.ctx + 4096 + 512, slog, env=env)
            out["server"] = srv
            with srv as s:
                qlib.log(f"[{ph}] server up in {s.t_load:.0f} s; request A ({a.ctx} tokens); capture arms ~{delay_s:.0f} s after connect")
                t = time.time()
                r = s.complete(ta, 1)
                qlib.log(f"[{ph}] A done {time.time() - t:.1f} s cached {r.get('usage', {}).get('prompt_tokens_details')}")
                if ph == "decode":  # B: extend to ctx + 2k (not captured: the capture is not armed yet)
                    r = s.complete(tb, 1)
                arm = t_srv + delay_s + 4
                if time.time() > arm:
                    out["warn"] = f"delay {delay_s:.0f} s too short: the server was busy when the capture armed"
                    qlib.log("[%s] WARNING %s" % (ph, out["warn"]))
                while time.time() < arm:
                    time.sleep(0.5)
                before = set(glob.glob(os.path.join(a.out, f"{tag}_{ph}*.rgp")))
                if ph == "prefill":
                    t = time.time()
                    r = s.complete(tb, 1)
                    out["request"] = {"wall_s": round(time.time() - t, 2), "usage": r.get("usage"), "timings": r.get("timings")}
                    qlib.log(f"[{ph}] B (prefill) done {time.time() - t:.2f} s timings {r.get('timings')}")
                    time.sleep(2)  # an idle gap marks the end of B in the dispatch list
                f = None
                for k in range(6):  # feed more dispatches until the capture window is full
                    t = time.time()
                    r = s.complete(tb, 256 if ph == "decode" else 64)
                    tm = r.get("timings") or {}
                    qlib.log(f"[{ph}] C{k} (decode) {time.time() - t:.1f} s, {tm.get('predicted_n')} tokens, "
                             f"{tm.get('predicted_per_second', 0):.1f} tok/s, drafts {tm.get('draft_n_accepted')}/{tm.get('draft_n')}")
                    if k == 0 and ph == "decode":
                        out["request"] = {"timings": tm}
                    f = wait_file(os.path.join(a.out, f"{tag}_{ph}*.rgp"), before, 25)
                    if f:
                        break
                if not f:
                    f = wait_file(os.path.join(a.out, f"{tag}_{ph}*.rgp"), before, 120)
                qlib.log(f"[{ph}] rgp {f} ({os.path.getsize(f) / 1e6:.1f} MB)" if f else f"[{ph}] NO rgp file")
                out["rgp"] = f
        except OSError as e:  # the server vanished mid-request (R9700: AMD runtime crash at trace start)
            out["error"] = repr(e)
            srv = out.get("server")
            try:
                rc = srv.p.wait(15) if srv is not None and srv.p is not None else None
            except Exception:
                rc = None
            if rc is not None:
                out["server_exit"] = f"0x{rc & 0xffffffff:08X}"  # 0xC0000005 = the R9700 trace-start crash
            qlib.log(f"[{ph}] server died: {out['error']}, exit code {out.get('server_exit')}")
        finally:
            try:
                cli.stdin.write("q\n")
                cli.stdin.flush()
                cli.wait(20)
            except Exception:
                pass
            if cli.poll() is None:
                cli.kill()  # our own PID only
            qlib.drop_pid("fl1a", cli.pid)
    out.pop("server", None)
    out["capture_wall_s"] = round(time.time() - tw, 1)
    return out


def main():
    global RDTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["r97", "8s"], required=True)
    ap.add_argument("--phase", choices=["decode", "prefill", "both"], default="both")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--mode", default="balance")
    ap.add_argument("--exe", default=qlib.DEF_EXE)
    ap.add_argument("--ops", type=int, default=3000, help="dispatches per capture (--rgp-render-op-count)")
    ap.add_argument("--buffer", default="high", help="SQTT buffer size: minimum|low|default|high")
    ap.add_argument("--out", default=os.path.join(qlib.WORK, "fl1a", "rgp"))
    ap.add_argument("--no-mtp", action="store_true", help="plain decode (WHIRL_MTP=0) instead of the default MTP")
    ap.add_argument("--skip", type=int, default=3000, help="decode: dispatches skipped after the capture arms")
    ap.add_argument("--delay", type=float, default=0, help="seconds from connect to arming (default: estimated load + A)")
    ap.add_argument("--env", default="", help="extra server env K=V;K2=V2 (e.g. HIP_VISIBLE_DEVICES=0)")
    ap.add_argument("--reserve-mb", type=int, default=6144, help="WHIRL_POOL_RESERVE_MB for the server (room for SQTT)")
    ap.add_argument("--keep-big", action="store_true", help="keep .rgp files over 200 MB")
    ap.add_argument("--force-r97", action="store_true", help="try the R9700 anyway (known AMD crash, audit/RGP-R97.md)")
    ap.add_argument("--rdts", default=RDTS, help="Radeon Developer Tool Suite folder (default: newest present)")
    a = ap.parse_args()
    if a.device == "r97" and not a.force_r97:
        print("rgp_capture: R9700 RGP capture crashes inside AMD's HIP runtime at trace start (audit/RGP-R97.md); "
              "use --force-r97 to retest after a driver/runtime update", file=sys.stderr)
        return 2
    RDTS = a.rdts
    os.makedirs(a.out, exist_ok=True)
    tw0 = time.time()
    tag = f"{time.strftime('%Y%m%d-%H%M%S')}_{a.device}_{a.mode}_ctx{a.ctx}"
    phases = ["prefill", "decode"] if a.phase == "both" else [a.phase]
    pa = qlib.prompt_file(a.exe, a.ctx)
    pb = qlib.prompt_file(a.exe, a.ctx + 2048)
    ta, tb = open(pa, encoding="utf-8").read(), open(pb, encoding="utf-8").read()
    assert tb.startswith(ta)
    results = {"tag": tag, "device": a.device, "mode": a.mode, "ctx": a.ctx, "ops": a.ops, "buffer": a.buffer, "phases": {}}
    for ph in phases:
        results["phases"][ph] = capture(a, ph, tag, ta, tb)
    # parse outside the lock (CPU only)
    for ph, d in results["phases"].items():
        f = d.get("rgp")
        if not f:
            continue
        base = os.path.join(a.out, f"{tag}_{ph}")
        t = time.time()
        p = subprocess.run([PARSER_PY, os.path.join(PARSER, "main.py"), f, base], cwd=PARSER, capture_output=True, text=True)
        d["parse_s"] = round(time.time() - t, 1)
        d["parse_rc"] = p.returncode
        if p.returncode != 0:
            d["parse_err"] = (p.stderr or p.stdout)[-1500:]
            qlib.log(f"{ph}: parser failed rc {p.returncode}: {d['parse_err'][-500:]}")
            continue
        sm = summarize(base + "_dispatches.csv", base + "_kernels.csv", 300000 if ph == "prefill" else 0)
        d["summary"] = sm
        d["rgp_mb"] = round(os.path.getsize(f) / 1e6, 1)
        if sm:
            qlib.log(f"{ph}: {sm['dispatches']} dispatches, span {sm['span_us'] / 1000:.2f} ms, busy {sm['busy_us'] / 1000:.2f} ms, "
                     f"gaps {sm['gap_us'] / 1000:.2f} ms")
            for k, us, sh, n in sm["top"][:10]:
                qlib.log(f"   {sh:6.2f}%  {us / 1000:8.3f} ms  x{n:<5d} {k[:90]}")
        if d["rgp_mb"] > 200 and not a.keep_big:
            os.remove(f)
            d["rgp_deleted"] = True
    results["wall_s"] = round(time.time() - tw0, 1)
    with open(os.path.join(a.out, tag + ".json"), "w") as fo:
        json.dump(results, fo, indent=1)
    qlib.log(f"done in {results['wall_s']} s -> {os.path.join(a.out, tag + '.json')}")
    return 0 if all(d.get("summary") for d in results["phases"].values()) and results["phases"] else 1


if __name__ == "__main__":
    sys.exit(main())
