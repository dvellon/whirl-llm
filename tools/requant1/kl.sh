#!/usr/bin/env bash
# REQUANT-1: KL/PPL vs BF16 reference logits (FNMX method: official b11719 CPU llama-perplexity, kl_mix.txt, 80 x 512).
#   kl.sh base <bf16.gguf> <base.bin> <log>      |   kl.sh eval <quant.gguf> <base.bin> <log>
set -euo pipefail
P=/c/ClaudeWorkspace/tools/llamacpp_fn_bin/cpu/llama-perplexity.exe
TXT=/c/ClaudeWorkspace/models/requant1/eval/kl_mix.txt
if [[ $1 == base ]]; then
  "$P" -m "$2" -f "$TXT" -c 512 -b 4096 -ub 4096 --chunks 80 -t ${THREADS:-16} --kl-divergence-base "$3" > "$4" 2>&1
else
  "$P" -m "$2" -c 512 -b 4096 -ub 4096 --chunks 80 -t ${THREADS:-16} --kl-divergence-base "$3" --kl-divergence > "$4" 2>&1
fi
