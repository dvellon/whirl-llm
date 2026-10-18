#!/usr/bin/env bash
# FNMX: KL divergence + PPL of a quantized build vs the BF16 reference logits (CPU, same chunks as kl_base.sh).
set -euo pipefail
D=/c/ClaudeWorkspace/models/fnmx
M=$1; TAG=$2
/c/ClaudeWorkspace/tools/llamacpp_fn_bin/cpu/llama-perplexity.exe -m "$D/gguf/$M" \
  -c 512 -b 4096 -ub 4096 --chunks ${CHUNKS:-80} -t 16 --kl-divergence-base "$D/kl/kl_base_bf16.bin" --kl-divergence > "$D/kl/kl_$TAG.log" 2>&1
