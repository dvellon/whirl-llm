#!/usr/bin/env bash
# REQUANT-1 step 1: BF16 safetensors -> BF16 GGUF (MTP kept). llama.cpp b11719 (5aeac37a3) converter + venv.
#   convert.sh <src dir> <out.gguf>
set -euo pipefail
PY=/c/ClaudeWorkspace/tools/llamacpp_fn_venv/Scripts/python.exe
CV=/c/ClaudeWorkspace/tools/llamacpp_fn/convert_hf_to_gguf.py
"$PY" -E -s "$CV" "$1" --outtype bf16 --outfile "$2"
