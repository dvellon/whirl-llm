#!/usr/bin/env bash
# REQUANT-1 Ornith 8060S: new file with the OLD file's prefill tune table vs old file, 3 interleaved plain rounds (2k/32k).
set -uo pipefail
R=/c/ClaudeWorkspace/models/requant1; T=/c/ClaudeWorkspace/whirl-work/wt_requant1/tools/requant1; L=$R/logs
N='C:\ClaudeWorkspace\models\requant1\staging\ornith\Ornith-1.5-35B-A3B-MXFP4.gguf'
O='C:\Users\tsaip\.lmstudio\models\ornith-ai\Ornith-1.5-35B-A3B-MXFP4\Ornith-1.5-35B-A3B-MXFP4.gguf'
TD="$LOCALAPPDATA/whirl"
cp "$TD/tune4-Ornith-1.5-35B-A3B-MXFP4.gguf-19819767136-gfx1151r2.txt" "$TD/tune4-Ornith-1.5-35B-A3B-MXFP4.gguf-19819767584-gfx1151r2.txt"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $L/ornith_speed_oldtable.log; }
for r in 1 2 3; do
  for side in old new; do m=$([[ $side == new ]] && echo "$N" || echo "$O"); lab=$([[ $side == new ]] && echo newOT || echo old)
    pwsh -NoProfile -File $T/bench8s.ps1 -Model "$m" -Label $lab -ModelName ornith -Round ot$r -Plain > $L/bench/orn_${lab}_ot$r.out 2>&1; say "plain $lab ot$r rc=$?"
  done
done
say "OLDTABLE DONE"
