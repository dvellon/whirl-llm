# SPDX-License-Identifier: Apache-2.0
# RGP-R97 step 3: RGP capture of the minimal HIP loop (minloop.exe) on one device, under minidbg.
# Build: hipcc -O2 --offload-arch=gfx1201 --offload-arch=gfx1151 -o minloop.exe minloop.hip (R97_MINEXE overrides the path).
# R97_CLIARGS: extra RadeonDeveloperPanelCLI args. usage: python mincap.py RDTS_DIR HIP_INDEX TAG [secs] [delay_ms] [buffer] [bigMiB] [extra env K=V;...]
import os, subprocess, sys, time, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # tools/quick (qlib)
import qlib

rdts, idx, tag = sys.argv[1], sys.argv[2], sys.argv[3]
secs = sys.argv[4] if len(sys.argv) > 4 else "40"
delay = sys.argv[5] if len(sys.argv) > 5 else "10000"
buf = sys.argv[6] if len(sys.argv) > 6 else "minimum"
big = sys.argv[7] if len(sys.argv) > 7 else "0"
extra = sys.argv[8] if len(sys.argv) > 8 else ""
out = os.environ.get("R97_OUT", os.path.join(qlib.WORK, "rgp97", "cap"))
os.makedirs(out, exist_ok=True)
stamp = time.strftime("%H%M%S")
rgp = os.path.join(out, f"min_{tag}_{stamp}.rgp")
with qlib.GpuLock():
    cli_log = rgp + ".cli.log"
    cli = subprocess.Popen([os.path.join(rdts, "RadeonDeveloperPanelCLI.exe"), "-m", "profiling", "-p", "minloop", "-o", rgp,
                            "--rgp-capture-mode", "dispatch", "--rgp-render-op-count", "2000",
                            "--rgp-auto-capture=dispatch:1:2000", "--rgp-auto-capture-delay", delay,
                            "--rgp-sqtt-buffer-size", buf, "--verbose"] + [x for x in os.environ.get("R97_CLIARGS", "").split(" ") if x],
                           cwd=rdts, stdin=subprocess.PIPE, stdout=open(cli_log, "w"), stderr=subprocess.STDOUT, text=True)
    qlib.add_pid("rgp97", cli.pid, "RadeonDeveloperPanelCLI min " + tag)
    time.sleep(5)
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = idx
    env.update(dict(kv.split("=", 1) for kv in extra.split(";") if kv))
    dlog = os.path.join(out, f"dbg_min_{tag}_{stamp}.log")
    t = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, "minidbg.py"), dlog, "--", os.environ.get("R97_MINEXE", os.path.join(HERE, "minloop.exe")), secs, big],
                       env=env, capture_output=True, text=True, timeout=int(secs) + 120)
    print("minloop stdout tail:", p.stdout[-600:])
    time.sleep(5)
    f = [x for x in glob.glob(rgp.replace(".rgp", "*.rgp"))]
    try:
        cli.stdin.write("q\n"); cli.stdin.flush(); cli.wait(20)
    except Exception:
        pass
    if cli.poll() is None:
        cli.kill()
    qlib.drop_pid("rgp97", cli.pid)
print("rgp files:", [(x, os.path.getsize(x)) for x in f])
print(open(dlog).read()[-3000:])
print("---- cli log tail ----")
print(open(cli_log, errors="replace").read()[-1500:])
