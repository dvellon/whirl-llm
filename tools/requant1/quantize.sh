#!/usr/bin/env bash
# REQUANT-1 step 2: BF16 GGUF -> MXFP4 with the imatrix-weighted MXFP4 scale search (FNMX patch on llama.cpp 5aeac37a3).
#   quantize.sh <imatrix> <tensor-type file> <output type> <in.gguf> <out.gguf>
set -euo pipefail
Q=${Q:-/c/ClaudeWorkspace/tools/llamacpp_mxq/build/bin/llama-quantize.exe}
"$Q" --imatrix "$1" --tensor-type-file "$2" --output-tensor-type "$3" "$4" "$5" MXFP4_MOE ${THREADS:-24}
