#!/usr/bin/env bash
# FNMX step 2: BF16 GGUF -> MXFP4 (PLE table iq4_nl or q8_0). llama.cpp b11719 (5aeac37a3) official CPU build.
set -euo pipefail
PLE=$1   # iq4_nl | q8_0
TAG=$(echo "$PLE" | tr a-z A-Z)
Q=${Q:-/c/ClaudeWorkspace/tools/llamacpp_fn_bin/cpu/llama-quantize.exe}
SUF=${SUF:-}
D=/c/ClaudeWorkspace/models/fnmx
"$Q" --imatrix "$D/imatrix/imatrix-swiftfn-v1mix.gguf" \
  --tensor-type-file "$(dirname "$0")/tt_fnmx_ple_$PLE.txt" --output-tensor-type q6_k \
  "$D/gguf/swift-fn-bf16.gguf" "$D/gguf/Swift-1.5-Qwen3.8-Flash-Next-MXFP4${SUF}-ple-$TAG.gguf" MXFP4_MOE 32
