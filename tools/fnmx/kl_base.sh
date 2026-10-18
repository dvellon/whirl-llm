#!/usr/bin/env bash
# FNMX: BF16 reference logits for KL (CPU, mmap from SSD). 80 chunks x 512 tokens of eval/kl_mix.txt.
set -euo pipefail
D=/c/ClaudeWorkspace/models/fnmx
/c/ClaudeWorkspace/tools/llamacpp_fn_bin/cpu/llama-perplexity.exe -m "$D/gguf/swift-fn-bf16.gguf" -f "$D/eval/kl_mix.txt" \
  -c 512 -b 4096 -ub 4096 --chunks 80 -t 16 --kl-divergence-base "$D/kl/kl_base_bf16.bin"
