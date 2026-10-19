#!/usr/bin/env bash
# REQUANT-1 Ornith R9700 (gfx1201) chain for the rebuilt Ornith-1.5-35B-A3B MXFP4 (after the USB backup ALLDONE).
#   0 device probe  1 golden check OLD (temp GoldenRoot = tests/golden of b896844)  2 klgate r9700 NEW
#   3 golden record+check gfx1201 NEW  4 seqtest x3  5 speed: 3 interleaved rounds MTP + 3 plain, 2k/32k (bench97.ps1)
# Stops at the first gate failure. Usage: bash r97_ornith.sh [START_STEP]
set -uo pipefail
START=${1:-0}
R=/c/ClaudeWorkspace/models/requant1; L=$R/logs/r97o; mkdir -p $L $R/klgate $R/logs/bench97
T=/c/ClaudeWorkspace/whirl-work/wt_requant1/tools/requant1; WT='C:\ClaudeWorkspace\whirl-work\wt_requant1'
EXE='C:\ClaudeWorkspace\whirl-work\wt_rel020f\build\Release\whirl.exe'
NEW=${NEW:-'C:\ClaudeWorkspace\models\requant1\staging\ornith\Ornith-1.5-35B-A3B-MXFP4.gguf'}
OLD=${OLD:-'C:\Users\tsaip\.lmstudio\models\ornith-ai\Ornith-1.5-35B-A3B-MXFP4\Ornith-1.5-35B-A3B-MXFP4.gguf'}
OLDROOT='C:\Users\tsaip\AppData\Local\Temp\claude\C--ClaudeWorkspace-Qwen3-8-27b-Research\10821dde-6d6e-4a78-bcb1-55026a5afaf2\scratchpad\golden_old_orn\tests\golden'
PY=/c/ClaudeWorkspace/tools/pyref/Scripts/python.exe
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $L/chain.log; }
devchk(){ local d; d=$(grep -o -m1 -E 'gfx1(201|151)' "$1"); echo "${d:-NONE}"; }
"$EXE" golden "$NEW" --suite ngram --balance --device gfx1201 > $L/devprobe.log 2>&1
say "0 device probe rc=$? dev=$(devchk $L/devprobe.log)"
[[ $(devchk $L/devprobe.log) == gfx1201 ]] || { say "STOP: device probe did not report gfx1201"; exit 1; }
if (( START <= 1 )); then
  WHIRL_GOLDEN_MOE="$OLD" pwsh -NoProfile -File "$WT\tools\golden\golden.ps1" check -Arch gfx1201 -Models moe -Exe "$EXE" \
    -GoldenRoot "$OLDROOT" -Out "$(cygpath -w $L/old_check)" > $L/golden_old_check.log 2>&1; rc=$?
  say "1 golden check OLD rc=$rc $(tail -1 $L/golden_old_check.log)"; (( rc == 0 )) || { say "STOP: old-file golden check failed"; exit 1; }
fi
if (( START <= 2 )); then
  $PY "/c/ClaudeWorkspace/whirl-work/wt_requant1/tools/klgate/klgate.py" --exe "$EXE" --corpus 'C:\ClaudeWorkspace\whirl-work\klgate\corpus' \
    --model "moe=$NEW" --device r9700 --report "$R/klgate/kl_ornith_r9700.json" --work "$R/klgate/work_ornith_r97" > $L/klgate_ornith_r97.log 2>&1; rc=$?
  say "2 klgate rc=$rc $(tail -1 $L/klgate_ornith_r97.log)"; (( rc == 0 )) || { say "STOP: klgate failed"; exit 1; }
fi
if (( START <= 3 )); then
  WHIRL_GOLDEN_MOE="$NEW" pwsh -NoProfile -File "$WT\tools\golden\golden.ps1" record -Arch gfx1201 -Models moe -Exe "$EXE" \
    -KlReport "$(cygpath -w $R/klgate/kl_ornith_r9700.json)" -Out "$(cygpath -w $L/new_record)" > $L/golden_new_record.log 2>&1; rc=$?
  say "3 golden record rc=$rc $(tail -1 $L/golden_new_record.log)"; (( rc == 0 )) || { say "STOP: golden record failed"; exit 1; }
  WHIRL_GOLDEN_MOE="$NEW" pwsh -NoProfile -File "$WT\tools\golden\golden.ps1" check -Arch gfx1201 -Models moe -Exe "$EXE" \
    -Out "$(cygpath -w $L/new_check)" > $L/golden_new_check.log 2>&1; rc=$?
  say "3 golden check NEW rc=$rc $(tail -1 $L/golden_new_check.log)"; (( rc == 0 )) || { say "STOP: golden check after record failed"; exit 1; }
fi
if (( START <= 4 )); then
  for m in precise balance fast; do
    "$EXE" seqtest "$NEW" --decode 24 --$m --device r9700 > $L/seqtest_$m.log 2>&1; rc=$?
    say "4 seqtest $m rc=$rc $(grep -E '^seqtest:' $L/seqtest_$m.log | head -1)"
    (( rc == 0 )) || { say "STOP: seqtest failed"; exit 1; }
  done
fi
if (( START <= 5 )); then
  for PL in "" plain; do
    for r in 1 2 3; do
      if (( r % 2 )); then order="old new"; else order="new old"; fi
      for s in $order; do m=$([[ $s == new ]] && echo "$NEW" || echo "$OLD")
        pwsh -NoProfile -File $T/bench97.ps1 -Model "$m" -Label $s -ModelName ornith -Round r$r ${PL:+-Plain} > $R/logs/bench97/orn_${s}${PL:+_plain}_r$r.out 2>&1
        say "5 speed ${PL:-mtp} r$r $s rc=$?"
      done
    done
  done
fi
say "CHAIN DONE"
