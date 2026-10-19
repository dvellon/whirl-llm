#!/usr/bin/env bash
# REQUANT-1 Ornith after the KL base: CPU KL evals (new, old) alone, then 8060S GPU jobs one at a time
# (old-file server/vision control, llama.cpp vision new/old, klgate on the new file). Never CPU KL and GPU at the same time.
set -uo pipefail
R=/c/ClaudeWorkspace/models/requant1; T=/c/ClaudeWorkspace/whirl-work/wt_requant1/tools/requant1; L=$R/logs
N='C:\ClaudeWorkspace\models\requant1\staging\ornith\Ornith-1.5-35B-A3B-MXFP4.gguf'
O='C:\Users\tsaip\.lmstudio\models\ornith-ai\Ornith-1.5-35B-A3B-MXFP4\Ornith-1.5-35B-A3B-MXFP4.gguf'
MM='C:\Users\tsaip\.lmstudio\models\ornith-ai\Ornith-1.5-35B-A3B-MXFP4\mmproj-Ornith-1.5-35B-A3B-F16.gguf'
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $L/ornith_chain2.log; }
until grep -q "^EXIT" $L/kl_ornith_base.log; do sleep 20; done
say "KL base: $(grep -E 'Final estimate|^EXIT' $L/kl_ornith_base.log | tr '\n' ' ')"
grep -q "^EXIT 0" $L/kl_ornith_base.log || { say "KL BASE FAILED"; exit 1; }
for side in new old; do
  m=$([[ $side == new ]] && echo "$N" || echo "$O")
  THREADS=16 bash $T/kl.sh eval "$m" $R/kl/ornith_base_bf16.bin $L/kl_ornith_$side.log
  say "KL $side: $(grep -E '^Mean    KLD|^Same top p|^Mean PPL\(Q\)  |^Mean ln\(PPL\(Q\)/PPL\(base\)\)|^Mean PPL\(Q\)/PPL\(base\)' $L/kl_ornith_$side.log | tr -s ' ' | tr '\n' ';')"
done
pwsh -NoProfile -File $T/serve_test.ps1 -Model "$O" -Tag ornith_old_mtp -Mmproj "$MM" > $L/st_ornith_old_mtp.out 2>&1; say "serve old mtp rc=$?"
pwsh -NoProfile -File $T/llama_vis.ps1 -Model "$N" -Mmproj "$MM" -Tag ornith_new_llama > $L/lv_ornith_new.out 2>&1; say "llama.cpp vision new rc=$?"
pwsh -NoProfile -File $T/llama_vis.ps1 -Model "$O" -Mmproj "$MM" -Tag ornith_old_llama > $L/lv_ornith_old.out 2>&1; say "llama.cpp vision old rc=$?"
say "klgate ornith start"
/c/ClaudeWorkspace/tools/pyref/Scripts/python.exe /c/ClaudeWorkspace/whirl-work/wt_requant1/tools/klgate/klgate.py --exe 'C:\ClaudeWorkspace\whirl-work\wt_rel020f\build\Release\whirl.exe' \
  --corpus 'C:\ClaudeWorkspace\whirl-work\klgate\corpus' --model "moe=$N" --device 8060s \
  --report "$R/klgate/kl_ornith_8060s.json" --work "$R/klgate/work_ornith" > $L/klgate_ornith.log 2>&1; say "klgate ornith rc=$? $(tail -1 $L/klgate_ornith.log)"
say "CHAIN2 DONE"
