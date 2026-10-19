#!/usr/bin/env bash
# REQUANT-1 Swift GPU chain on the 8060S (after gpu_checks A): chat/vision (MTP) + plain text, B/C selftest + chat, klgate.
set -uo pipefail
R=/c/ClaudeWorkspace/models/requant1; T=/c/ClaudeWorkspace/whirl-work/wt_requant1/tools/requant1; L=$R/logs
S='C:\ClaudeWorkspace\models\requant1\staging\swift'
MM='C:\Users\tsaip\.lmstudio\models\ukisai\Swift-1.5-Qwen3.8-27b-MXFP4\mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf'
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $L/swift_gpu_chain.log; }
pwsh -NoProfile -File $T/serve_test.ps1 -Model "$S\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf" -Tag swiftA_new_mtp -Mmproj "$MM" > $L/st_swiftA_mtp.out 2>&1; say "serve A mtp rc=$?"
pwsh -NoProfile -File $T/serve_test.ps1 -Model "$S\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf" -Tag swiftA_new_plain -Plain -TextOnly > $L/st_swiftA_plain.out 2>&1; say "serve A plain rc=$?"
for v in B-outQ8_0 C-outQ4_K; do
  bash $T/gpu_checks.sh "$S\Swift-1.5-Qwen3.8-27B-MXFP4-$v.gguf" swift${v:0:1} "balance" "" >> $L/gpuchk_swift${v:0:1}.txt 2>&1; say "$(cat $L/gpuchk_swift${v:0:1}.txt)"
  pwsh -NoProfile -File $T/serve_test.ps1 -Model "$S\Swift-1.5-Qwen3.8-27B-MXFP4-$v.gguf" -Tag swift${v:0:1}_new_mtp -TextOnly > $L/st_swift${v:0:1}_mtp.out 2>&1; say "serve $v mtp rc=$?"
done
say "klgate swift start"
mkdir -p $R/klgate
/c/ClaudeWorkspace/tools/pyref/Scripts/python.exe /c/ClaudeWorkspace/whirl-work/wt_requant1/tools/klgate/klgate.py --exe 'C:\ClaudeWorkspace\whirl-work\wt_rel020f\build\Release\whirl.exe' \
  --corpus 'C:\ClaudeWorkspace\whirl-work\klgate\corpus' --model "swift=$S\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf" --device 8060s \
  --report "$R/klgate/kl_swift_8060s.json" --work "$R/klgate/work_swift" > $L/klgate_swift.log 2>&1; say "klgate swift rc=$? $(tail -1 $L/klgate_swift.log)"
say "CHAIN DONE"
