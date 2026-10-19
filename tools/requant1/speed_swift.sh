#!/usr/bin/env bash
# REQUANT-1: 3 interleaved rounds old/new Swift-A, balance, 2k + 32k, on the 8060S (CPU otherwise idle).
T=/c/ClaudeWorkspace/whirl-work/wt_requant1/tools/requant1
NEW='C:\ClaudeWorkspace\models\requant1\staging\swift\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf'
OLD='C:\Users\tsaip\.lmstudio\models\ukisai\Swift-1.5-Qwen3.8-27b-MXFP4\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf'
for r in 1 2 3; do
  if (( r % 2 )); then order="new old"; else order="old new"; fi
  for s in $order; do
    m=$([[ $s == new ]] && echo "$NEW" || echo "$OLD")
    pwsh -NoProfile -File $T/bench8s.ps1 -Model "$m" -Label $s -ModelName swift -Round $r > /c/ClaudeWorkspace/models/requant1/logs/bench/swift_${s}_r$r.out 2>&1
    echo "[$(date +%H:%M:%S)] round $r $s rc=$?"
  done
done
echo SPEED DONE
