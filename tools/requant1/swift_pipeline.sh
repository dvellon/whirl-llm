#!/usr/bin/env bash
# REQUANT-1 Swift: wait for download -> verify -> BF16 GGUF -> quantize A/B/C (patched quantizer) -> type check -> KL base
set -uo pipefail
R=/c/ClaudeWorkspace/models/requant1; T=/c/ClaudeWorkspace/whirl-work/wt_requant1/tools/requant1
PY=/c/ClaudeWorkspace/tools/llamacpp_fn_venv/Scripts/python.exe
L=$R/logs; PUB=/c/Users/tsaip/.lmstudio/models/ukisai/Swift-1.5-Qwen3.8-27b-MXFP4
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $L/swift_pipeline.log; }
until grep -q -E "swift_src[[:space:]]*$|Error|Traceback" $L/dl_swift.log; do sleep 10; done
if grep -q -E "Error|Traceback" $L/dl_swift.log; then say "DOWNLOAD FAILED"; exit 1; fi
say "download done $(du -sh $R/swift_src | cut -f1)"
curl -s "https://huggingface.co/api/models/ukisai/Swift-1.5-Qwen3.8-27b/revision/b4c84d42903a8646b25857eb2827288d92ed85a4?blobs=true" -o $L/swift_api_rev.json
python -I $T/verify_src.py $L/swift_api_rev.json $R/swift_src $R/logs/src_sha_swift.txt > $L/verify_swift.out 2>&1 &
VPID=$!
say "convert start"
bash $T/convert.sh $R/swift_src $R/gguf/swift-1.5-27b-bf16.gguf > $L/conv_swift.log 2>&1 || { say "CONVERT FAILED"; exit 1; }
say "convert done $(ls -la $R/gguf/swift-1.5-27b-bf16.gguf | awk '{print $5}') bytes"
wait $VPID; VRC=$?; say "verify: $(cat $L/verify_swift.out) rc=$VRC"
[[ $VRC == 0 ]] || { say "SOURCE VERIFY FAILED"; exit 1; }
IMX=$R/imatrix/ukisai_Swift-1.5-Qwen3.8-27b-imatrix.gguf
for v in "A q6_k Q6_K" "B q8_0 Q8_0" "C q4_k Q4_K"; do
  set -- $v; f=Swift-1.5-Qwen3.8-27B-MXFP4-$1-out$3.gguf
  say "quantize $f"
  THREADS=24 bash $T/quantize.sh $IMX $T/tt_swift.txt $2 $R/gguf/swift-1.5-27b-bf16.gguf $R/staging/swift/$f > $L/quant_swift_$1.log 2>&1 || { say "QUANT FAILED $f"; exit 1; }
  "$PY" -E -s $T/cmp_types.py $PUB/$f $R/staging/swift/$f > $L/cmp_swift_$1.txt 2>&1
  say "$f $(ls -la $R/staging/swift/$f | awk '{print $5}') bytes; $(tail -1 $L/cmp_swift_$1.txt)"
done
say "KL base start"
bash $T/kl.sh base $R/gguf/swift-1.5-27b-bf16.gguf $R/kl/swift_base_bf16.bin $L/kl_swift_base.log || { say "KL BASE FAILED"; exit 1; }
say "KL base done: $(grep -E 'Final estimate' $L/kl_swift_base.log)"
say "PIPELINE DONE"
