#!/usr/bin/env bash
# REQUANT-1 Ornith: download -> verify -> BF16 GGUF -> Q8_0 (imatrix only) -> llama-imatrix on Vulkan0 (8060S)
#   -> quantize (patched quantizer, published tensor types) -> type check vs published -> KL base.
# Stages can be skipped with START=<n> (1 download, 2 convert, 3 q8, 4 imatrix, 5 quantize, 6 klbase).
set -uo pipefail
R=/c/ClaudeWorkspace/models/requant1; T=/c/ClaudeWorkspace/whirl-work/wt_requant1/tools/requant1
PY=/c/ClaudeWorkspace/tools/llamacpp_fn_venv/Scripts/python.exe; L=$R/logs
REV=10fbf86fed7ecee4a061f8b499a618f46001cac1
PUB=/c/Users/tsaip/.lmstudio/models/ornith-ai/Ornith-1.5-35B-A3B-MXFP4
BF=$R/gguf/ornith-1.5-35b-bf16.gguf; Q8=$R/gguf/ornith-1.5-35b-q8_0-for-imatrix.gguf; IMX=$R/imatrix/ornith-1.5-35b-a3b-requant1-imatrix.gguf
START=${START:-1}; STOP=${STOP:-6}
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $L/ornith_pipeline.log; }
freegb(){ df -BG --output=avail /c | tail -1 | tr -dc 0-9; }
need(){ local f=$(freegb); if (( f - $1 < 30 )); then say "BLOCKED (disk): free ${f} GB, next step needs ~$1 GB"; exit 2; fi; }
if (( START <= 1 && STOP >= 1 )); then
  need ${NEED1:-142}   # download 71.9 GB + BF16 GGUF ~70 GB (safetensors deleted only after stage 2); NEED1=72 to download only
  say "download start (free $(freegb) GB)"
  HF_HUB_DISABLE_TELEMETRY=1 /c/myhermes/venv/Scripts/hf.exe download ornith-ai/Ornith-1.5-35B-A3B --revision $REV --exclude "*.png" --local-dir $R/ornith_src > $L/dl_ornith.log 2>&1 || { say "DOWNLOAD FAILED"; exit 1; }
  curl -s "https://huggingface.co/api/models/ornith-ai/Ornith-1.5-35B-A3B/revision/$REV?blobs=true" -o $L/ornith_api_rev.json
  python -I $T/verify_src.py $L/ornith_api_rev.json $R/ornith_src $L/src_sha_ornith.txt > $L/verify_ornith.out 2>&1; VRC=$?
  say "download done; verify: $(cat $L/verify_ornith.out) rc=$VRC"; [[ $VRC == 0 ]] || exit 1
fi
if (( START <= 2 && STOP >= 2 )); then
  need 70
  say "convert start"
  bash $T/convert.sh $R/ornith_src $BF > $L/conv_ornith.log 2>&1 || { say "CONVERT FAILED"; exit 1; }
  say "convert done $(stat -c %s $BF) bytes; $(grep -o 'n_tensors = [0-9]*' $L/conv_ornith.log | head -1)"
fi
if (( START <= 3 && STOP >= 3 )); then
  need 38
  say "Q8_0 (for imatrix) start"
  /c/ClaudeWorkspace/tools/llamacpp_fn_bin/cpu/llama-quantize.exe $BF $Q8 Q8_0 24 > $L/quant_ornith_q8.log 2>&1 || { say "Q8_0 FAILED"; exit 1; }
  say "Q8_0 done $(stat -c %s $Q8) bytes"
fi
if (( START <= 4 && STOP >= 4 )); then
  say "imatrix start (Vulkan0 = 8060S)"
  /c/ClaudeWorkspace/tools/llamacpp_fn_bin/vulkan/llama-imatrix.exe -m $Q8 -f $R/imatrix/orn_calib.txt -o $IMX --process-output --parse-special \
     -c 512 -b 2048 -ub 512 -ngl 99 --device Vulkan0 -t 16 > $L/imatrix_ornith.log 2>&1 || { say "IMATRIX FAILED"; exit 1; }
  grep -q "Vulkan0" $L/imatrix_ornith.log && grep -q "8060S" $L/imatrix_ornith.log || { say "IMATRIX DEVICE CHECK FAILED"; exit 1; }
  say "imatrix done $(stat -c %s $IMX) bytes sha $(sha256sum $IMX | cut -c1-16); $(grep -E 'Final estimate|chunks' $L/imatrix_ornith.log | tail -2 | tr '\n' ' ')"
fi
if (( START <= 5 && STOP >= 5 )); then
  need 21
  f=Ornith-1.5-35B-A3B-MXFP4.gguf
  say "quantize $f"
  THREADS=24 bash $T/quantize.sh $IMX $T/tt_ornith.txt q6_k $BF $R/staging/ornith/$f > $L/quant_ornith.log 2>&1 || { say "QUANT FAILED"; exit 1; }
  "$PY" -E -s $T/cmp_types.py $PUB/$f $R/staging/ornith/$f > $L/cmp_ornith.txt 2>&1
  say "$f $(stat -c %s $R/staging/ornith/$f) bytes; $(tail -1 $L/cmp_ornith.txt)"
fi
if (( START <= 6 && STOP >= 6 )); then
  say "KL base start"
  bash $T/kl.sh base $BF $R/kl/ornith_base_bf16.bin $L/kl_ornith_base.log || { say "KL BASE FAILED"; exit 1; }
  say "KL base done: $(grep -E 'Final estimate' $L/kl_ornith_base.log)"
fi
say "PIPELINE DONE"
