#!/usr/bin/env bash
# FNMX: llama-server on the 8060S only (Vulkan0; R9700 hidden). Args: MODEL NCPUMOE PORT [extra...]
D=/c/ClaudeWorkspace/models/fnmx
M=$1; N=$2; P=$3; shift 3
export GGML_VK_VISIBLE_DEVICES=0
exec /c/ClaudeWorkspace/tools/llamacpp_fn_bin/vulkan/llama-server.exe -m "$D/gguf/$M" --mmproj "$D/gguf/mmproj-Swift-1.5-Qwen3.8-Flash-Next-F16.gguf" \
  --device Vulkan0 -ngl 99 --n-cpu-moe "$N" -c 16384 -np 1 -fa on --host 127.0.0.1 --port "$P" --no-webui --jinja "$@"
