#!/usr/bin/env bash
# REQUANT-1: plain decode (no MTP / n-gram) old vs new Swift-A, 3 interleaved rounds, 2k + 32k, balance, 8060S.
T=/c/ClaudeWorkspace/whirl-work/wt_requant1/tools/requant1
NEW='C:\ClaudeWorkspace\models\requant1\staging\swift\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf'
OLD='C:\Users\tsaip\.lmstudio\models\ukisai\Swift-1.5-Qwen3.8-27b-MXFP4\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf'
for r in 1 2; do
  if (( r % 2 )); then order="old new"; else order="new old"; fi
  for s in $order; do
    m=$([[ $s == new ]] && echo "$NEW" || echo "$OLD")
    pwsh -NoProfile -File $T/bench8s.ps1 -Model "$m" -Label $s -ModelName swift -Round t$r -Plain > /c/ClaudeWorkspace/models/requant1/logs/bench/swift_${s}_plain3_r$r.out 2>&1
    echo "[$(date +%H:%M:%S)] plain round $r $s rc=$?"
  done
done
echo PLAIN3 DONE
