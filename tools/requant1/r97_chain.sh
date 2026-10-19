#!/usr/bin/env bash
# REQUANT-1-R97: R9700 (gfx1201) chain for the rebuilt Swift-1.5 27B MXFP4-A.
#   1 golden check gfx1201 on the OLD (first-release) file (temp GoldenRoot with the old models.txt hash)
#   2 klgate balance/fast vs precise, code/zh, 8k-128k, --device r9700, NEW file
#   3 golden record gfx1201 swift (NEW, with the klgate report), then golden check
#   4 seqtest precise/balance/fast on the R9700
# Stops at the first failure (no record after a failing gate). Usage: bash r97_chain.sh [START_STEP]
set -uo pipefail
START=${1:-1}
R=/c/ClaudeWorkspace/models/requant1; L=$R/logs/r97; mkdir -p $L $R/klgate
WT='C:\ClaudeWorkspace\whirl-work\wt_requant1'
EXE='C:\ClaudeWorkspace\whirl-work\wt_rel020f\build\Release\whirl.exe'
NEW='C:\Users\tsaip\.lmstudio\models\ukisai\Swift-1.5-Qwen3.8-27b-MXFP4\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf'
OLD='C:\ClaudeWorkspace\models\archive\swift-1.5-27b-mxfp4-first-release-2026-10-01\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf'
OLDROOT='C:\Users\tsaip\AppData\Local\Temp\claude\C--ClaudeWorkspace-Qwen3-8-27b-Research\10821dde-6d6e-4a78-bcb1-55026a5afaf2\scratchpad\golden_old'
PY=/c/ClaudeWorkspace/tools/pyref/Scripts/python.exe
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $L/chain.log; }
devchk(){ local d; d=$(grep -o -m1 -E 'gfx1(201|151)' "$1"); echo "${d:-NONE}"; }

"$EXE" devices > $L/devices.log 2>&1
"$EXE" golden "$NEW" --suite ngram --balance --device gfx1201 > $L/devprobe.log 2>&1
say "0 device probe (golden --device gfx1201, full stderr) rc=$? dev=$(devchk $L/devprobe.log)"
[[ $(devchk $L/devprobe.log) == gfx1201 ]] || { say "STOP: device probe did not report gfx1201"; exit 1; }
if (( START <= 1 )); then
  say "1 golden check OLD gfx1201 start"
  WHIRL_GOLDEN_SWIFT="$OLD" pwsh -NoProfile -File "$WT\tools\golden\golden.ps1" check -Arch gfx1201 -Models swift -Exe "$EXE" \
    -GoldenRoot "$OLDROOT" -Out "$(cygpath -w $L/old_check)" > $L/golden_old_check.log 2>&1; rc=$?
  say "1 golden check OLD rc=$rc $(tail -1 $L/golden_old_check.log)"
  (( rc == 0 )) || { say "STOP: old-file golden check failed"; exit 1; }
fi
if (( START <= 2 )); then
  say "2 klgate swift r9700 start"
  $PY "/c/ClaudeWorkspace/whirl-work/wt_requant1/tools/klgate/klgate.py" --exe "$EXE" \
    --corpus 'C:\ClaudeWorkspace\whirl-work\klgate\corpus' --model "swift=$NEW" --device r9700 \
    --report "$R/klgate/kl_swift_r9700.json" --work "$R/klgate/work_swift_r97" > $L/klgate_swift_r97.log 2>&1; rc=$?
  say "2 klgate rc=$rc $(tail -1 $L/klgate_swift_r97.log)"
  (( rc == 0 )) || { say "STOP: klgate failed"; exit 1; }
fi
if (( START <= 3 )); then
  say "3 golden record NEW gfx1201 start"
  WHIRL_GOLDEN_SWIFT="$NEW" pwsh -NoProfile -File "$WT\tools\golden\golden.ps1" record -Arch gfx1201 -Models swift -Exe "$EXE" \
    -KlReport "$(cygpath -w $R/klgate/kl_swift_r9700.json)" -Out "$(cygpath -w $L/new_record)" > $L/golden_new_record.log 2>&1; rc=$?
  say "3 golden record rc=$rc $(tail -1 $L/golden_new_record.log)"
  (( rc == 0 )) || { say "STOP: golden record failed"; exit 1; }
  WHIRL_GOLDEN_SWIFT="$NEW" pwsh -NoProfile -File "$WT\tools\golden\golden.ps1" check -Arch gfx1201 -Models swift -Exe "$EXE" \
    -Out "$(cygpath -w $L/new_check)" > $L/golden_new_check.log 2>&1; rc=$?
  say "3 golden check NEW rc=$rc $(tail -1 $L/golden_new_check.log)"
  (( rc == 0 )) || { say "STOP: golden check after record failed"; exit 1; }
fi
if (( START <= 4 )); then
  for m in precise balance fast; do
    "$EXE" seqtest "$NEW" --decode 24 --$m --device r9700 > $L/seqtest_$m.log 2>&1; rc=$?
    say "4 seqtest $m rc=$rc $(grep -E '^seqtest:' $L/seqtest_$m.log | head -1) dev=$(devchk $L/seqtest_$m.log)"
  done
fi
say "CHAIN DONE"
