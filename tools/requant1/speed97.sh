#!/usr/bin/env bash
# REQUANT-1-R97: 3 interleaved rounds old/new Swift-A, balance, 2k + 32k, R9700 (b8s.py --expect-device gfx1201 --server-log).
#   bash speed97.sh TAG [plain]   (TAG goes into the round label, e.g. fresh / oldtab)
TAG=${1:-fresh}; PL=${2:-}   # 2nd arg "plain": no MTP / n-gram (bench97.ps1 -Plain)
T=/c/ClaudeWorkspace/whirl-work/wt_requant1/tools/requant1
NEW='C:\Users\tsaip\.lmstudio\models\ukisai\Swift-1.5-Qwen3.8-27b-MXFP4\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf'
OLD='C:\ClaudeWorkspace\models\archive\swift-1.5-27b-mxfp4-first-release-2026-10-01\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf'
L=/c/ClaudeWorkspace/models/requant1/logs/bench97; mkdir -p $L
for r in 1 2 3; do
  if (( r % 2 )); then order="new old"; else order="old new"; fi
  for s in $order; do
    m=$([[ $s == new ]] && echo "$NEW" || echo "$OLD")
    pwsh -NoProfile -File $T/bench97.ps1 -Model "$m" -Label $s -ModelName swift -Round ${TAG}$r ${PL:+-Plain} > $L/swift_${s}_${TAG}_r$r.out 2>&1
    echo "[$(date +%H:%M:%S)] $TAG round $r $s rc=$?"
  done
done
echo "SPEED $TAG DONE"
