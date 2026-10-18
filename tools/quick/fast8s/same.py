#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# FAST-8S-1 output-identity check: per variant one server start, fixed greedy prompts (qa doc prompts with a
# fixed first line at each length + the zh coding chat), full texts saved to OUT/<label>.json. Compare two
# labels with:  same.py cmp OUT A B
#   same.py run --model ornith --exe PATH --variant "fast:WHIRL_DQ4_DYN=0" --lengths 2048,32768 --out DIR
import argparse, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, r"C:\ClaudeWorkspace\whirl-work\wt_prof8s\tools\quick")
sys.path.insert(0, r"C:\ClaudeWorkspace\whirl-work\rel3rc5")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlib  # noqa: E402
import rc5  # noqa: E402
from ab import MODELS, PORT, TASK  # noqa: E402


def run(a):
    mode, _, kv = a.variant.partition(":")
    env = qlib.clean_env(HIP_VISIBLE_DEVICES="0")
    env.update(dict(x.split("=", 1) for x in kv.split(",") if x))
    label = a.label or (a.model + "-" + a.variant.replace(":", "+").replace(",", "+").replace("=", ""))
    os.makedirs(a.out, exist_ok=True)
    slog = os.path.join(a.out, label + ".server.log")
    ctx = max(int(x) for x in a.lengths.split(",")) + 4096
    args = [a.exe, MODELS[a.model], "--port", str(PORT), "--log-file", slog, "--kv-ssd-mode", "off", "-np", "1",
            "--ctx-per-slot", str(ctx), "--device", "8060s", "--" + mode]
    p = subprocess.Popen(args, env=env, stdout=open(slog + ".stderr", "w"), stderr=subprocess.STDOUT,
                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    qlib.add_pid(TASK, p.pid, "whirl-server same " + label)
    url = f"http://127.0.0.1:{PORT}"
    res = {}
    try:
        while True:
            if p.poll() is not None:
                raise RuntimeError(f"server exited ({p.returncode}) {slog}")
            try:
                qlib.http(url + "/health", timeout=5)
                break
            except Exception:
                time.sleep(1)
        ns = argparse.Namespace(url=url)
        for L in [int(x) for x in a.lengths.split(",")]:
            s = rc5.one(ns, rc5.doc_prompt(L, "qa", "fast8s-fixed"), a.max_tokens)
            res[f"qa{L}"] = dict(text=s.text, drafts=s.timings.get("draft_n"), acc=s.timings.get("draft_n_accepted"),
                                 tps=s.decodeTps(), err=str(s.error) if s.error else "")
            print(label, L, res[f"qa{L}"]["acc"], res[f"qa{L}"]["drafts"], rc5.R.textHash(s.text), flush=True)
        s = rc5.one(ns, Path(rc5.ZH).read_text(encoding="utf-8"), 512)
        res["chat"] = dict(text=s.text, drafts=s.timings.get("draft_n"), acc=s.timings.get("draft_n_accepted"),
                           tps=s.decodeTps(), err=str(s.error) if s.error else "")
        print(label, "chat", res["chat"]["acc"], res["chat"]["drafts"], rc5.R.textHash(s.text), flush=True)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait(60)
        qlib.drop_pid(TASK, p.pid)
    Path(a.out, label + ".json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def cmp(out, x, y):
    A = json.loads(Path(out, x + ".json").read_text(encoding="utf-8"))
    B = json.loads(Path(out, y + ".json").read_text(encoding="utf-8"))
    bad = 0
    for k in A:
        same = A[k]["text"] == B.get(k, {}).get("text")
        bad += not same
        print(f"{k}: {'IDENTICAL' if same else 'DIFFERENT'} ({len(A[k]['text'])} chars) acc {A[k]['acc']}/{A[k]['drafts']} vs "
              f"{B.get(k, {}).get('acc')}/{B.get(k, {}).get('drafts')}")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1] == "cmp":
        sys.exit(cmp(*sys.argv[2:5]))
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--model", required=True)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--lengths", default="2048,8192,32768,131072")
    ap.add_argument("--max-tokens", type=int, default=256, dest="max_tokens")
    ap.add_argument("--out", required=True)
    run(ap.parse_args())
