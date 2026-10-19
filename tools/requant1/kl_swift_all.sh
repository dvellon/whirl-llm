#!/usr/bin/env bash
# REQUANT-1: KL/PPL of new and published Swift A/B/C vs the BF16 base (interleaved new/old).
set -uo pipefail
R=/c/ClaudeWorkspace/models/requant1; T=/c/ClaudeWorkspace/whirl-work/wt_requant1/tools/requant1; L=$R/logs
PUB=/c/Users/tsaip/.lmstudio/models/ukisai/Swift-1.5-Qwen3.8-27b-MXFP4
for v in A-outQ6_K B-outQ8_0 C-outQ4_K; do
  f=Swift-1.5-Qwen3.8-27B-MXFP4-$v.gguf
  for side in new old; do
    m=$([[ $side == new ]] && echo $R/staging/swift/$f || echo $PUB/$f)
    [[ -f $L/kl_swift_${side}_$v.log ]] && grep -q "Same top p" $L/kl_swift_${side}_$v.log && continue
    echo "[$(date +%H:%M:%S)] KL $side $v start" >> $L/kl_swift_all.log
    bash $T/kl.sh eval $m $R/kl/swift_base_bf16.bin $L/kl_swift_${side}_$v.log
    echo "[$(date +%H:%M:%S)] KL $side $v: $(grep -E '^Mean    KLD|^Same top p|^Mean PPL\(Q\)  ' $L/kl_swift_${side}_$v.log | tr -s ' ' | tr '\n' ';')" >> $L/kl_swift_all.log
  done
done
echo "[$(date +%H:%M:%S)] KL ALL DONE" >> $L/kl_swift_all.log
