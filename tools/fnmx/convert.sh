#!/usr/bin/env bash
# FNMX step 1: Swift-1.5 Qwen3.8-Flash-Next BF16 safetensors -> mmproj F16 + text BF16 GGUF (MTP + PLE kept).
# llama.cpp b11719 (commit 5aeac37a3) converter; run with the converter venv.
set -euo pipefail
PY=/c/ClaudeWorkspace/tools/llamacpp_fn_venv/Scripts/python.exe
CV=/c/ClaudeWorkspace/tools/llamacpp_fn/convert_hf_to_gguf.py
SRC=/c/ClaudeWorkspace/models/fnmx/bf16
OUT=/c/ClaudeWorkspace/models/fnmx/gguf
mkdir -p "$OUT"
what=${1:-all}
if [[ $what == all || $what == mmproj ]]; then
  "$PY" -E -s "$CV" "$SRC" --mmproj --outtype f16 --outfile "$OUT/mmproj-Swift-1.5-Qwen3.8-Flash-Next-F16.gguf"
fi
if [[ $what == all || $what == text ]]; then
  "$PY" -E -s "$CV" "$SRC" --outtype bf16 --outfile "$OUT/swift-fn-bf16.gguf"
fi
