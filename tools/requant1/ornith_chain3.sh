#!/usr/bin/env bash
# REQUANT-1 Ornith 8060S after klgate: extra vision (new/old, 10-03 card image), golden record+check (new), speed
# (fresh quiet autotune for the new file, 3 interleaved MTP rounds + 3 interleaved plain rounds, 2k/32k). One GPU job at a time, CPU idle.
set -uo pipefail
R=/c/ClaudeWorkspace/models/requant1; T=/c/ClaudeWorkspace/whirl-work/wt_requant1/tools/requant1; L=$R/logs; WT='C:\ClaudeWorkspace\whirl-work\wt_requant1'
EXE='C:\ClaudeWorkspace\whirl-work\wt_rel020f\build\Release\whirl.exe'
N='C:\ClaudeWorkspace\models\requant1\staging\ornith\Ornith-1.5-35B-A3B-MXFP4.gguf'
O='C:\Users\tsaip\.lmstudio\models\ornith-ai\Ornith-1.5-35B-A3B-MXFP4\Ornith-1.5-35B-A3B-MXFP4.gguf'
MM='C:\Users\tsaip\.lmstudio\models\ornith-ai\Ornith-1.5-35B-A3B-MXFP4\mmproj-Ornith-1.5-35B-A3B-F16.gguf'
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $L/ornith_chain3.log; }
until grep -q "CHAIN2 DONE" $L/ornith_chain2.log; do sleep 20; done
grep -q "klgate ornith rc=0" $L/ornith_chain2.log || { say "KLGATE NOT PASSED - stop"; exit 1; }
pwsh -NoProfile -File $T/serve_test.ps1 -Model "$N" -Tag ornith_new_vis2 -Mmproj "$MM" -VisionOnly > $L/st_ornith_new_vis2.out 2>&1; say "vision2 new rc=$?"
pwsh -NoProfile -File $T/serve_test.ps1 -Model "$O" -Tag ornith_old_vis2 -Mmproj "$MM" -VisionOnly > $L/st_ornith_old_vis2.out 2>&1; say "vision2 old rc=$?"
WHIRL_GOLDEN_MOE="$N" pwsh -NoProfile -File "$WT\tools\golden\golden.ps1" record -Arch gfx1151 -Models moe -Exe "$EXE" \
  -KlReport "C:\ClaudeWorkspace\models\requant1\klgate\kl_ornith_8060s.json" -Out "C:\ClaudeWorkspace\models\requant1\logs\golden\orn_new_record" > $L/golden/orn_new_record.log 2>&1; say "golden record rc=$? $(tail -1 $L/golden/orn_new_record.log)"
WHIRL_GOLDEN_MOE="$N" pwsh -NoProfile -File "$WT\tools\golden\golden.ps1" check -Arch gfx1151 -Models moe -Exe "$EXE" \
  -Out "C:\ClaudeWorkspace\models\requant1\logs\golden\orn_new_check" > $L/golden/orn_new_check.log 2>&1; say "golden check rc=$? $(tail -1 $L/golden/orn_new_check.log)"
TD="$LOCALAPPDATA/whirl"; NT="$TD/tune4-Ornith-1.5-35B-A3B-MXFP4.gguf-19819767584-gfx1151r2.txt"
[[ -f "$NT" ]] && mv "$NT" $L/bench/tune_ornith_new_underload_2340.txt
say "speed start (new tune table moved aside; first new launch re-tunes with the CPU idle)"
for r in 1 2 3; do
  for side in old new; do m=$([[ $side == new ]] && echo "$N" || echo "$O")
    pwsh -NoProfile -File $T/bench8s.ps1 -Model "$m" -Label $side -ModelName ornith -Round $r > $L/bench/orn_${side}_r$r.out 2>&1; say "mtp $side r$r rc=$?"
  done
done
for r in 1 2 3; do
  for side in old new; do m=$([[ $side == new ]] && echo "$N" || echo "$O")
    pwsh -NoProfile -File $T/bench8s.ps1 -Model "$m" -Label $side -ModelName ornith -Round $r -Plain > $L/bench/orn_${side}_plain_r$r.out 2>&1; say "plain $side r$r rc=$?"
  done
done
cp "$NT" $L/bench/tune_ornith_new_quiet.txt 2>/dev/null
say "CHAIN3 DONE"
