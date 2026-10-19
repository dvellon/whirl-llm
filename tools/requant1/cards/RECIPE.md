# Reproduction recipe

How the three language-model files in this repository were made (rebuild of 2026-10-19). All sizes are decimal (1 GB = 10^9 bytes). The vision projector `mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf` is unchanged from the first release (2026-10-01).

## Inputs

| Input | Source | Revision | SHA256 |
|---|---|---|---|
| BF16 weights (18 safetensors shards plus config / tokenizer files, 55.6 GB) | [ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) | `b4c84d42903a8646b25857eb2827288d92ed85a4` (weights unchanged since the initial release `bc7a1e10b6`; the later commit only edits the README) | all 34 downloaded files checked: the 19 LFS files against the Hugging Face LFS SHA256, the 15 small text files against their git blob ids (list in `SOURCE_SHA256.txt`) |
| Importance matrix `ukisai_Swift-1.5-Qwen3.8-27b-imatrix.gguf` (13,642,688 bytes; 496 entries, 582 chunks) | [bartowski/ukisai_Swift-1.5-Qwen3.8-27b-GGUF](https://huggingface.co/bartowski/ukisai_Swift-1.5-Qwen3.8-27b-GGUF) | `ce202c7d9a5d71969c60bf62a4f1c3e41747520c` | `125d4e6838a1a62cabf9c48aebdfa47e497e95db590a993ee94e12aaf28a268b` |
| llama.cpp | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) build **b11719** | commit `5aeac37a3` | converter run from the source tree at that commit; `llama-quantize` built from that commit plus [`mxfp4_wsearch.patch`](mxfp4_wsearch.patch); `llama-perplexity` from the official release zip `llama-b11719-bin-win-cpu-x64.zip` |

Converter environment: Python 3.11, `requirements/requirements-convert_hf_to_gguf.txt` of that commit.

## 1. Convert

```
python convert_hf_to_gguf.py <Swift-1.5 BF16 dir> --outtype bf16 --outfile swift-1.5-27b-bf16.gguf
```

866 tensors (54.66 GB): 64 main layers + the MTP layer `blk.64` (`qwen35.nextn_predict_layers = 1`).

## 2. Quantize

### 2a. Patched llama-quantize (imatrix-weighted MXFP4 block scales)

Upstream `quantize_mxfp4()` ignores the importance matrix and always uses the block exponent `floor(log2(amax)) - 2`. The patch [`mxfp4_wsearch.patch`](mxfp4_wsearch.patch) (diff against llama.cpp `5aeac37a3`, `ggml/src/ggml-quants.c` only) changes only the quantize-time function: for each 32-value block it tries that exponent and its two neighbours, rounds every value to the nearest FP4 code at each scale (same code table as upstream), and keeps the exponent with the lowest weighted squared error. Weights are `imatrix[j] * sqrt(sigma2 + x[j]^2)` (the ggml k-quant convention, `sigma2` = mean of `x^2` over the row) when an imatrix entry exists, otherwise `x[j]^2`. The output is the standard `block_mxfp4` layout (GGML type 39); the runtime dequantization code is untouched, so the files load in unmodified llama.cpp and WHIRL.

```
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 5aeac37a3
git apply mxfp4_wsearch.patch
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build build --target llama-quantize
```

(Built with MSVC 2022 Build Tools on Windows.)

### 2b. Tensor types (unchanged from the first release)

`tt_common.txt` (first matching rule wins):

```
blk\.64\.=q8_0
ssm_alpha=q8_0
ssm_beta=q8_0
token_embd=q8_0
attn_gate=mxfp4
attn_qkv=mxfp4
attn_q\.=mxfp4
attn_k\.=mxfp4
attn_v\.=mxfp4
attn_output=mxfp4
ffn_gate=mxfp4
ffn_up=mxfp4
ffn_down=mxfp4
ssm_out=mxfp4
```

```
llama-quantize --imatrix ukisai_Swift-1.5-Qwen3.8-27b-imatrix.gguf --tensor-type-file tt_common.txt \
  --output-tensor-type q6_k swift-1.5-27b-bf16.gguf Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf MXFP4_MOE 24
llama-quantize --imatrix ukisai_Swift-1.5-Qwen3.8-27b-imatrix.gguf --tensor-type-file tt_common.txt \
  --output-tensor-type q8_0 swift-1.5-27b-bf16.gguf Swift-1.5-Qwen3.8-27B-MXFP4-B-outQ8_0.gguf MXFP4_MOE 24
llama-quantize --imatrix ukisai_Swift-1.5-Qwen3.8-27b-imatrix.gguf --tensor-type-file tt_common.txt \
  --output-tensor-type q4_k swift-1.5-27b-bf16.gguf Swift-1.5-Qwen3.8-27B-MXFP4-C-outQ4_K.gguf MXFP4_MOE 24
```

(`llama-quantize` = the patched build from 2a.) Result, checked tensor by tensor against the first-release files: the same 866 tensor names, types and shapes (400 MXFP4, 105 Q8_0, 360 F32, 1 output head); the files differ only in the MXFP4 block scales / codes and 96 bytes of metadata.

Notes:

- `MXFP4_MOE` is only the base ftype (there is no dense-model MXFP4 ftype); the rules above decide every tensor.
- The importance matrix has entries for every matrix of the 64 main layers (attention, linear attention incl. `ssm_alpha` / `ssm_beta`, FFN) and none for `output`, `token_embd` or the MTP layer `blk.64`. llama.cpp does not use the imatrix for Q8_0, so in these files it acts through the patched MXFP4 search on all 400 MXFP4 matrices. The output heads (Q6_K / Q8_0 / Q4_K), `token_embd` and the MTP layer are quantized without imatrix data.

## 3. Quality check

```
llama-perplexity -m swift-1.5-27b-bf16.gguf -f kl_mix.txt -c 512 -b 4096 -ub 4096 --chunks 80 -t 16 --kl-divergence-base kl_base_bf16.bin
llama-perplexity -m <quantized file> -c 512 -b 4096 -ub 4096 --chunks 80 -t 16 --kl-divergence-base kl_base_bf16.bin --kl-divergence
```

`kl_mix.txt` = the first 70,000 characters of a C++/HIP source file with Chinese comments plus the first 40,000 characters of Chinese technical prose (80 chunks of 512 tokens). Official b11719 CPU build. Results: model card, "Quality".
