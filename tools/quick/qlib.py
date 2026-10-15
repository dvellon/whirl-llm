#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# FASTLANE (FL-1a) shared helpers for the tools/quick scripts: the machine-wide GPU lock, prompt files of an
# exact token count, a whirl-server life cycle and the progress / PID bookkeeping.
# Developer tooling for this machine (paths below are the WHIRL workstation's); not shipped.
import ctypes
import json
import os
import subprocess
import sys
import time
import urllib.request
from ctypes import wintypes

WORK = r"C:\ClaudeWorkspace\whirl-work"
LOCK = os.path.join(WORK, "r9700.lock")
DEF_EXE = os.path.join(WORK, r"wt_moebal3\build\Release\whirl.exe")
MODEL = r"C:\Users\tsaip\.lmstudio\models\unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-Q4_K_M.gguf"
CORPUS = os.path.join(WORK, r"klgate\corpus")
PROMPTS = os.path.join(WORK, r"fl1a\prompts")
DEVICES = {"r97": "r9700", "8s": "8060s"}
PORTS = {"r97": 8195, "8s": 8196}
BUSY = ("whirl.exe", "zynfer.exe", "llama-server.exe", "llama-bench.exe", "llama-cli.exe")


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("WHIRL_") or k == "WHIRL_R9700_LOCK_HELD"}
    env.update({k: str(v) for k, v in extra.items()})
    return env


class GpuLock:
    """The one machine-wide GPU lock (same file and share mode as r9700_lock.ps1 / item8S\\gpu_lock.py), taken for
    both GPUs: one GPU job at a time on this machine. Skipped when a parent already holds it
    (WHIRL_R9700_LOCK_HELD, set by r9700_lock.ps1 / gpu_lock.py)."""

    def __init__(self):
        self.h = None

    def __enter__(self):
        if os.environ.get("WHIRL_R9700_LOCK_HELD"):
            return self
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateFileW.restype = wintypes.HANDLE
        k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                    wintypes.DWORD, wintypes.HANDLE]
        bad = wintypes.HANDLE(-1).value
        waited = 0
        while True:
            h = k32.CreateFileW(LOCK, 0xC0000000, 0, None, 4, 0x80, None)  # GENERIC_RW, share 0, OPEN_ALWAYS
            if h != bad:
                break
            if waited % 60 == 0:
                log(f"lock: GPU lock held by another job; waiting ({waited} s)")
            time.sleep(5)
            waited += 5
        w2 = 0
        while True:
            out = subprocess.run(["tasklist", "/NH"], capture_output=True, text=True, errors="replace").stdout.lower()
            busy = [p for p in BUSY if p in out]
            if not busy:
                break
            if w2 >= 600:
                k32.CloseHandle(h)
                raise SystemExit(f"lock: {busy} still alive after 10 min")
            if w2 % 60 == 0:
                log(f"lock: waiting for {busy} to exit")
            time.sleep(5)
            w2 += 5
        self.h, self.k32 = h, k32
        os.environ["WHIRL_R9700_LOCK_HELD"] = str(os.getpid())
        msg = f"pid {os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(sys.argv)[:200]}\n".encode()
        n = wintypes.DWORD()
        k32.SetFilePointer(h, 0, None, 0)
        k32.WriteFile(h, msg, len(msg), ctypes.byref(n), None)
        k32.SetEndOfFile(h)
        return self

    def __exit__(self, *a):
        if self.h:
            self.k32.CloseHandle(self.h)
            os.environ.pop("WHIRL_R9700_LOCK_HELD", None)
            self.h = None


def ntokens(exe, path):
    env = clean_env(WHIRL_TOKENIZE_ONLY=1)
    r = subprocess.run([exe, "chat", MODEL, "@" + path, "--raw", "--max-tokens", "1"], env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    ids = [l for l in r.stdout.splitlines() if l and l[0].isdigit()]
    if r.returncode != 0 or not ids:
        raise RuntimeError(f"tokenize failed (exit {r.returncode}): {r.stderr[-300:]}")
    return ids[-1].count(",") + 1


def prompt_file(exe, n, src="code", skip_tokens=0):
    """A prompt file of about n tokens (within 0.5%, cut at a line end) from the fixed KL corpus text, starting
    after about skip_tokens tokens of it. Cached under fl1a\\prompts."""
    os.makedirs(PROMPTS, exist_ok=True)
    out = os.path.join(PROMPTS, f"{src}_{skip_tokens}_{n}.txt")
    if os.path.isfile(out):
        return out
    text = open(os.path.join(CORPUS, src + ".txt"), encoding="utf-8").read()
    cpt = 3.0  # chars per token, refined below
    start = int(skip_tokens * cpt)
    if start:
        start = text.find("\n", start) + 1
    nch = int(n * cpt)
    for _ in range(6):
        end = text.find("\n", start + nch)
        end = len(text) if end < 0 else end + 1
        with open(out + ".tmp", "w", encoding="utf-8", newline="") as f:
            f.write(text[start:end])
        got = ntokens(exe, out + ".tmp")
        if abs(got - n) <= max(8, n * 0.005) or end >= len(text):
            break
        nch = int(nch * n / got)
    os.replace(out + ".tmp", out)
    log(f"prompt {out}: {got} tokens (target {n})")
    return out


def add_pid(task, pid, what):
    with open(os.path.join(WORK, "progress", task + ".pids"), "a", encoding="utf-8") as f:
        f.write(f"{pid} {time.strftime('%Y-%m-%dT%H:%M:%S')} {what}\n")


def drop_pid(task, pid):
    p = os.path.join(WORK, "progress", task + ".pids")
    try:
        lines = [l for l in open(p, encoding="utf-8") if not l.startswith(f"{pid} ")]
        open(p, "w", encoding="utf-8").writelines(lines)
    except OSError:
        pass


def http(url, body=None, timeout=3600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


class Server:
    """whirl-server for one device / mode (--kv-ssd-mode off, one slot); kill-by-PID on exit (only our PID)."""

    def __init__(self, exe, dev, mode, ctx, log_path, task="fl1a", env=None, extra=()):
        self.exe = os.path.join(os.path.dirname(exe), "whirl-server.exe")
        self.port = PORTS[dev]
        self.url = f"http://127.0.0.1:{self.port}"
        self.args = [self.exe, MODEL, "--port", str(self.port), "--log-file", log_path, "--kv-ssd-mode", "off", "-np", "1",
                     "--ctx-per-slot", str(ctx), "--device", DEVICES[dev], "--" + mode] + list(extra)
        self.env = clean_env(**(env or {}))
        self.task, self.p = task, None

    def __enter__(self):
        self.err = open(self.args[self.args.index("--log-file") + 1] + ".stderr", "w")
        self.p = subprocess.Popen(self.args, env=self.env, stdout=self.err, stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        add_pid(self.task, self.p.pid, "whirl-server " + " ".join(self.args[1:]))
        t0 = time.time()
        while True:
            if self.p.poll() is not None:
                raise RuntimeError(f"server exited during startup (code {self.p.returncode})")
            try:
                urllib.request.urlopen(self.url + "/health", timeout=5)
                break
            except Exception:
                pass
            if time.time() - t0 > 900:
                raise RuntimeError("server not ready after 900 s")
            time.sleep(1)
        self.t_load = time.time() - t0
        return self

    def complete(self, prompt, max_tokens, stream=False):
        body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": stream}
        if not stream:
            return http(self.url + "/v1/completions", body)
        body["stream_options"] = {"include_usage": True}
        req = urllib.request.Request(self.url + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=3600)

    def __exit__(self, *a):
        if self.p and self.p.poll() is None:
            self.p.kill()  # our own PID only
            self.p.wait(60)
        if self.p:
            drop_pid(self.task, self.p.pid)
