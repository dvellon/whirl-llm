#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# kl_quick helper: run the command with the WHIRL_* settings of FL_KL_ENV (K=V;K2=V2) unless it is a --precise run
# (klgate.py strips WHIRL_* from its children, so the A/B setting of the tested mode comes in through here).
import os
import subprocess
import sys

args = sys.argv[1:]
if args[:1] == ["--"]:
    args = args[1:]
env = dict(os.environ)
if "--precise" not in args:
    for kv in os.environ.get("FL_KL_ENV", "").split(";"):
        if kv:
            k, v = kv.split("=", 1)
            env[k] = v
sys.exit(subprocess.run(args, env=env).returncode)
