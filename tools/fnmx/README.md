# FNMX: Swift-1.5 Qwen3.8-Flash-Next MXFP4 GGUF build scripts

Release folder: C:\ClaudeWorkspace\hf-release\Swift-1.5-Qwen3.8-Flash-Next-MXFP4-GGUF (RECIPE.md has the full recipe).

- convert.sh: BF16 safetensors -> mmproj F16 + BF16 GGUF (llama.cpp b11719 / 5aeac37 converter)
- quantize.sh [iq4_nl|q8_0]: BF16 GGUF -> MXFP4 (tensor-type files tt_fnmx_ple_*.txt). Q= selects the llama-quantize binary, SUF= adds a name suffix.
- mxfp4_wsearch.patch: imatrix-weighted MXFP4 block-exponent search in quantize_mxfp4 (diff vs llama.cpp 5aeac37); build_mxq.bat builds it, selftest_mxfp4.c / build_selftest.bat test it (selftest_result.txt).
- kl_base.sh: BF16 reference logits; dump_types.py: tensor-type summary; make_eval_inputs.py: test images + KL text; vis_range.py: F16 range check for the projector.
