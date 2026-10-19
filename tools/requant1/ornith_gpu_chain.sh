#!/usr/bin/env bash
# REQUANT-1 Ornith 8060S chain for the new file: selftest x3, seqtest x3, server chat+vision (MTP), plain text-only (MTP == plain), llama.cpp vision.
set -uo pipefail
R=/c/ClaudeWorkspace/models/requant1; T=/c/ClaudeWorkspace/whirl-work/wt_requant1/tools/requant1; L=$R/logs
N='C:\ClaudeWorkspace\models\requant1\staging\ornith\Ornith-1.5-35B-A3B-MXFP4.gguf'
MM='C:\Users\tsaip\.lmstudio\models\ornith-ai\Ornith-1.5-35B-A3B-MXFP4\mmproj-Ornith-1.5-35B-A3B-F16.gguf'
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $L/ornith_gpu_chain.log; }
bash $T/gpu_checks.sh "$N" ornith_new "${STM-precise balance fast}" "${SQM-precise balance fast}" > $L/gpuchk_ornith_new${SUF:-}.txt 2>&1; say "$(cat $L/gpuchk_ornith_new${SUF:-}.txt | tr '\n' ' ')"
pwsh -NoProfile -File $T/serve_test.ps1 -Model "$N" -Tag ornith_new_mtp -Mmproj "$MM" > $L/st_ornith_mtp.out 2>&1; say "serve new mtp rc=$?"
pwsh -NoProfile -File $T/serve_test.ps1 -Model "$N" -Tag ornith_new_plain -Plain -TextOnly > $L/st_ornith_plain.out 2>&1; say "serve new plain rc=$?"
say "CHAIN DONE"
