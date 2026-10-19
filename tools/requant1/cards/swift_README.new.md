---
license: other
license_name: swift-open-license-1.0
license_link: https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b/blob/main/LICENSE
base_model: ukisai/Swift-1.5-Qwen3.8-27b
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- mxfp4
- qwen3.8
- mtp
- whirl
- amd
- rdna4
- rdna3.5
- llama.cpp
---

# Swift-1.5-Qwen3.8-27b MXFP4 GGUF

MXFP4 GGUF quantizations of [ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b), with the **MTP (multi-token prediction) layer kept** for self-speculative decoding. Tuned for [WHIRL](https://github.com/tsaipifong/whirl-llm) 0.2.0, now on AMD Ryzen AI Max+ 395 (Radeon 8060S) and Radeon AI PRO R9700.

**Rebuilt 2026-10-19:** the three language-model files were re-quantized with an **imatrix-weighted MXFP4 quantizer** (same tensor layout, same file names). Against the BF16 model their KL divergence is 17–19% lower than the first release (2026-10-01); see [Quality](#quality). The vision projector is unchanged.

> **Unofficial community quantization.** This repository is not affiliated with, endorsed by, or supported by UkisAI or Alibaba Cloud. The names "Swift" and "UkisAI" are trademarks of UkisAI and are used here only to describe the origin of the weights. For the official model, official quantizations, evaluations and enterprise licensing, see the [upstream repository](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b).

Swift 1.5 Qwen3.8-27B is UkisAI's reasoning-efficient post-trained derivative of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Please read the upstream model card for its training approach and benchmark results; nothing in this card re-evaluates model quality.

Vision is supported through the separate projector file `mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf` (F16, converted from the Swift-1.5 BF16 weights with `convert_hf_to_gguf.py --mmproj --outtype f16`). It works with any of the three variants.

> ### Best performance with WHIRL, now on AMD Ryzen AI Max+ 395 (Radeon 8060S) and Radeon AI PRO R9700
> These files were tuned for **[WHIRL](https://github.com/tsaipifong/whirl-llm)** (Windows HIP Inference for RDNA LLMs), our open-source (Apache-2.0) native-Windows C++/HIP inference engine for the **Radeon 8060S** of the **AMD Ryzen AI Max+ 395** (new in 0.2.0) and the **AMD Radeon AI PRO R9700** — download [**release v0.2.0**](https://github.com/tsaipifong/whirl-llm/releases/tag/v0.2.0) (`whirl-0.2.0-windows-x64.zip`): unzip and run, only the AMD driver needed (AMD Software: Adrenalin Edition 26.8.1 (driver 32.0.31041.1004) or newer, both GPUs; no HIP SDK, ROCm or Visual C++ runtime). They are standard GGUF and also run in llama.cpp (first release tested with b11214; the 2026-10-19 rebuild with b11719, see below).
>
> Quick start with WHIRL (PowerShell, after unzipping): `.\whirl-server.exe C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf --port 8080` starts an OpenAI-compatible server in WHIRL's default mode (balance, near-lossless); `--precise` gives bitwise-lossless output, `--fast` maximum speed within gated KL limits. In precise and balance, MTP + n-gram drafts, batching and cache restores give the same tokens as plain greedy decoding of that mode.
>
> Also from us: [**Ornith-1.5-35B-A3B MXFP4 GGUF**](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF) — a mixture-of-experts model (~3B active of 35B) quantized the same way, with MTP layer and vision projector.

## Files

The three language-model files contain the same weights except the output head (`output.weight`).

| Variant | File | Output head | Size (bytes) | SHA256 | Recommendation |
|---|---|---|---|---|---|
| **A** | `Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf` | Q6_K | 15,815,469,376 | `c765544dee05261e4161281cfc298c493087d623b89c3753675d43255b42e09f` | **Recommended** default |
| C | `Swift-1.5-Qwen3.8-27B-MXFP4-C-outQ4_K.gguf` | Q4_K | 15,487,686,976 | `d24f9692a4db9dda2fe84a6fdcde9f5007e4bc93e9e905e67d1cbfc6ee85edbc` | Slightly faster on llama.cpp and in plain (non-speculative) decoding; lower-precision head |
| B | `Swift-1.5-Qwen3.8-27B-MXFP4-B-outQ8_0.gguf` | Q8_0 | 16,123,386,176 | `4b5f70adf8b462a79920e76c821e76c3875967b0925d78fa59fce01ec287e933` | Not recommended for speed (slowest with MTP); highest-precision head |
| vision | `mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf` | — (vision projector, F16) | 927,607,584 | `71988a54d379191e1abc26c2b79363b87ef8b3a584ebee882c28a1682ddbfe62` | Needed only for image input |

`SHA256SUMS` lists the same hashes. The three language-model files are the 2026-10-19 rebuild; the first-release files (2026-10-01) had the same names, were 96 bytes smaller (A `f12d8dee…`, B `bb075d1c…`, C `221aa39e…`) and remain in this repository's commit history.

## Quantization recipe

**Tensor types** (verified by dumping the quantized files):

| Tensor group | Type |
|---|---|
| All large matrices of the 64 main layers: `attn_qkv`, `attn_gate`, `attn_q/k/v`, `attn_output`, `ssm_out`, `ffn_gate/up/down` | MXFP4 |
| `token_embd` | Q8_0 |
| `ssm_alpha`, `ssm_beta` (48 linear-attention layers × 2) | Q8_0 |
| MTP layer `blk.64` (all of `attn_q/k/v/output`, `ffn_gate/up/down`, `nextn.eh_proj`) | Q8_0 |
| Norms, `ssm_a`, `ssm_conv1d`, `ssm_dt.bias`, `nextn.enorm/hnorm/shared_head_norm` | F32 |
| `output` (LM head) | Q6_K (A) / Q8_0 (B) / Q4_K (C) |

**Steps**

1. BF16 safetensors ([ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) revision `b4c84d42`, every file checked against its Hugging Face hash, list in `SOURCE_SHA256.txt`) → BF16 GGUF with llama.cpp `convert_hf_to_gguf.py <Swift-1.5 BF16 dir> --outtype bf16` (866 tensors; `qwen35.nextn_predict_layers = 1`, i.e. the MTP layer is kept). (llama.cpp build **b11719**, commit 5aeac37a3.)
2. Quantize with `llama-quantize` built from llama.cpp **b11719 (commit 5aeac37a3) plus [`mxfp4_wsearch.patch`](mxfp4_wsearch.patch)**, an imatrix-weighted MXFP4 block-scale search. Upstream `llama-quantize` ignores the importance matrix for MXFP4 and always takes the block exponent from the block maximum; the patch tries that exponent and its two neighbours for every 32-value block and keeps the one with the lowest imatrix-weighted squared error. Only quantize-time code changes: the output is the standard MXFP4 block format, so the files load in unmodified llama.cpp and WHIRL. Full build and run commands: [`RECIPE.md`](RECIPE.md).

(The first release, 2026-10-01, used the same tensor-type file with upstream `llama-quantize` from build 11214, commit 2ebd9ae62.)

`tt_common.txt` (passed with `--tensor-type-file`; the first matching rule wins):

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

Exact command for variant A (B used `--output-tensor-type q8_0`, C used `--output-tensor-type q4_k`):

```
llama-quantize --imatrix ukisai_Swift-1.5-Qwen3.8-27b-imatrix.gguf \
  --tensor-type-file tt_common.txt --output-tensor-type q6_k \
  swift-1.5-27b-bf16.gguf Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf MXFP4_MOE 24
```

(`llama-quantize` = the patched build from step 2.)

Notes:

- **ftype `MXFP4_MOE`**: this llama.cpp build has no dense-model MXFP4 ftype, so `MXFP4_MOE` is used as the base type and the per-tensor rules above decide every tensor. The GGUF metadata therefore reports `general.file_type = 38` (MXFP4_MOE) even though the model is dense.
- **Importance matrix**: the imatrix was made and published by **bartowski** — [bartowski/ukisai_Swift-1.5-Qwen3.8-27b-GGUF](https://huggingface.co/bartowski/ukisai_Swift-1.5-Qwen3.8-27b-GGUF) (`ukisai_Swift-1.5-Qwen3.8-27b-imatrix.gguf`, revision `ce202c7d`, SHA256 `125d4e68…a268b`, 496 entries, 582 chunks). Thank you! It has entries for every matrix of the 64 main layers and none for `output`, `token_embd` or the MTP layer. llama.cpp does not use the imatrix for Q8_0, so in the rebuilt files it acts through the weighted MXFP4 search on all 400 MXFP4 matrices; the output heads, `token_embd` and the MTP layer are quantized without imatrix data. (The first-release card said the imatrix affected the K-quant output heads; that was not correct, since the imatrix has no `output` entry.)
- **Why `ssm_alpha` / `ssm_beta` are Q8_0 and not F32**: a first build kept them in F32. With F32 weights for these small matrices, WHIRL computes single-token decode and multi-token MTP verification through different kernels, so MTP output was no longer bit-identical to plain greedy decoding, and the automatic draft length collapsed. Q8_0 has a fused exact path there, restoring bit-identical MTP output. This choice was made for WHIRL; llama.cpp loads and runs the files normally (see below).
- **Differences from the reference Qwen3.8-27B MXFP4 GGUF we compared against** ([FreedomAISVR/Qwen3.8-27B-MXFP4-GGUF](https://huggingface.co/FreedomAISVR/Qwen3.8-27B-MXFP4-GGUF)): that file is MXFP4 everywhere, including `token_embd`, `ssm_alpha/beta` and the MTP layer, with a Q6_K head. Here those tensors are Q8_0, which costs about 1–2% prefill speed in our measurements.
- **Quality**: KL divergence and perplexity against BF16 are in [Quality](#quality) below. (The first release's perplexity table, measured on the R9700 with another corpus and llama.cpp b11214 and compared with a Qwen3.8-27B MXFP4 file of a different model, was replaced by this BF16 comparison.)

## Quality

KL divergence and perplexity against the **BF16 model** (the unquantized Swift-1.5 weights converted to a BF16 GGUF), measured with llama.cpp b11719 `llama-perplexity` on the CPU (official release build), `-c 512`, 80 chunks of 512 tokens of a mixed text: a C++/HIP source file with Chinese comments (70,000 characters) plus Chinese technical prose (40,000 characters). BF16 perplexity on this text: 4.5053 ± 0.0783. Measured 2026-10-19; "first release" = the 2026-10-01 files with the same names. Lower KLD and higher same-top are better.

| File | Mean KLD vs BF16 | Same top token as BF16 | PPL | PPL / PPL(BF16) |
|---|---|---|---|---|
| **A (Q6_K head), this rebuild** | **0.0495 ± 0.0010** | **91.62%** | 4.6671 | 1.0359 ± 0.0028 |
| A, first release | 0.0606 ± 0.0011 | 91.01% | 4.7458 | 1.0534 ± 0.0033 |
| **B (Q8_0 head), this rebuild** | **0.0492 ± 0.0010** | **91.69%** | 4.6670 | 1.0359 ± 0.0028 |
| B, first release | 0.0604 ± 0.0011 | 90.98% | 4.7459 | 1.0534 ± 0.0032 |
| **C (Q4_K head), this rebuild** | **0.0527 ± 0.0010** | **90.91%** | 4.6810 | 1.0390 ± 0.0028 |
| C, first release | 0.0638 ± 0.0011 | 90.30% | 4.7586 | 1.0562 ± 0.0033 |

- The rebuild has the same tensor types as the first release; only the MXFP4 block scales and codes differ. KLD is 17–19% lower for all three variants, and the perplexity gap to BF16 shrinks from 5.3–5.6% to 3.6–3.9% (paired ratio ± its standard error).
- Both builds are 4-bit and stay measurably above BF16 perplexity on this text. One text and one tokenizer; this is not a downstream task evaluation.
- The three heads differ little; C (Q4_K) is slightly behind A and B.

Other checks of the rebuilt files (2026-10-19, Radeon 8060S, WHIRL 0.2.0):

- `whirl selftest` (precise / balance / fast for A, balance for B and C) and `whirl seqtest` (multi-request paths vs single request; precise / balance / fast for A): all passed.
- WHIRL MTP + n-gram output was identical to plain greedy output on a Chinese and an English coding prompt (370 and 403 tokens, balance mode). WHIRL's golden-hash regression references for this GPU were re-recorded for A after its KL gate (balance and fast vs precise, 8k–128k tokens, 16 of 16 cases) passed.
- Image input with the F16 projector (WHIRL server, and llama.cpp b11719 Vulkan `llama-server`): a Traditional Chinese / English sign was read verbatim ("營業時間：週一至週五", "上午 9:30 – 下午 6:00", "Closed on Sundays", "Tel: 02-2345-6789"); a synthetic image with a red square, a blue circle and the text "WHIRL 42" was described correctly except that the text was read as "WIRL 42". The first-release A file gave the same reading in the same WHIRL test.
- Speed (Radeon 8060S, WHIRL 0.2.0 balance, server, 2k- and 32k-token document prompts, rounds interleaved with the first-release A file): plain decode (no speculation) was identical, 14.8 tok/s after 2k and 13.7 tok/s after 32k tokens for both files. Prefill was also equal when both files used the same WHIRL kernel-tuning table (2k: 483.5 vs 484.2 tok/s, 32k: 424.3 vs 424.1). WHIRL tunes its prefill kernels once per new model file; on this machine a fresh tuning of the rebuilt file gave 2–3% slower prefill than the older cached tuning of the first-release file (2k: 471.8 vs 486.1 tok/s), which comes from the tuning, not from the file.

## Usage with llama.cpp (MTP)

The MTP layer is kept, so llama.cpp can use it for self-speculative decoding with `--spec-type draft-mtp`. This is the exact `llama-server` command used for our llama.cpp check (b11214, ROCm):

```
llama-server -m Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf --device ROCm0 -ngl 99 -c 131072 -np 1 -fa on \
  --port 1234 --no-webui --load-mode none --jinja \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 0 --spec-draft-p-min 0.3
```

(Our machine has two GPUs, so the run also set `HIP_VISIBLE_DEVICES=1` and `GGML_CUDA_NO_PINNED=1`; adjust `--device` / `-c` to your hardware.) Without the `--spec-*` flags the model runs as a plain model; llama.cpp then logs the MTP-layer tensors as "unused", which is expected. The loader also prints `unknown type mxfp4` as a warning; this appears with other MXFP4 GGUFs as well and loading succeeds.

Image input (first release, tested with `llama-mtmd-cli` b11214 on the R9700: a synthetic image with a red square, a blue circle and the text "WHIRL 42" was described correctly, including the exact text; for the 2026-10-19 rebuild see [Quality](#quality)):

```
llama-mtmd-cli -m Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf --mmproj mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf   --image photo.png -ngl 99 -c 8192 -p "Describe this image."
```

`llama-server` accepts the same `--mmproj` flag for image input over the OpenAI-compatible API.

Sampling recommended by the upstream card: temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0. Thinking is on by default in the chat template; our llama.cpp check disabled it per request with `"chat_template_kwargs": {"enable_thinking": false}`.

## Measured performance (WHIRL 0.2.0: Radeon 8060S and AMD Radeon AI PRO R9700; llama.cpp-only rows: R9700, 32 GB)

Single-machine measurements; speed only, not quality evaluations. The tables below were measured with the first-release files (2026-10-01 to 2026-10-17). The 2026-10-19 rebuild has the same format and tensor types; its speed was checked on the Radeon 8060S only (see [Quality](#quality)) and not re-measured on the R9700. The R9700 in this machine is connected as a USB4 eGPU (this slows model loading and host-RAM / SSD KV restores, not prefill or decode).

### llama.cpp b11214 (ROCm, `llama-server`, flags as above; 2026-10-01, single request, short prompts)

Greedy-equivalent sampling (top_k 1), thinking off, 4 short prompts (2 Traditional Chinese, 2 English), max 400 output tokens; prefill measured on a 2,022-token prompt.

| llama.cpp b11214 | A (Q6_K head) | B (Q8_0 head) | C (Q4_K head) |
|---|---|---|---|
| Plain decode, tok/s (mean of 4 prompts) | 33.3 | 32.8 | 33.8 |
| MTP decode, tok/s (`--spec-draft-n-max 3`) | 58.1 | 55.9 | 60.7 |
| MTP draft acceptance | 73.6% (846/1150) | 72.2% (840/1164) | 75.0% (856/1141) |
| Prefill, 2k prompt, tok/s | 1215 | 1212 | 1206 |

- Peak dedicated VRAM of `llama-server` at `-c 131072 -np 1`: 23.65 GiB.
- On the English prompts MTP output matched plain output; on the two Chinese prompts the outputs diverged part-way (llama.cpp's batched numerics are not bit-identical between plain and speculative decoding; this is normal).

### WHIRL 0.2.0 vs llama.cpp b11214 (variant A, first-release file)

WHIRL 0.2.0 in all three modes vs the **official llama.cpp b11214**, its **ROCm** and its **Vulkan** build in separate columns (lead columns for balance, the default; precise and fast lead in every cell too). Same GGUF file (`Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`), same prompts (a long-document question with a unique first line: explain a function quoted in the document and rewrite it, 256 generated tokens, so part of the answer can be copied; [exact prompts](https://github.com/tsaipifong/whirl-llm/blob/main/docs/benchmarks.md#prompts)), greedy, thinking off, one user, tok/s from each server's own timings. WHIRL measured 2026-10-17. llama.cpp decodes with MTP + n-gram (`--spec-type draft-mtp,ngram-mod --spec-draft-n-max 3`), WHIRL with its MTP + n-gram drafts. "~250k" is a prompt of 249,510 tokens. Full methodology: [WHIRL benchmarks](https://github.com/tsaipifong/whirl-llm/blob/main/docs/benchmarks.md).

Radeon 8060S (max power (80 W), ASUS ROG Flow Z13), WHIRL balance vs llama.cpp ROCm / Vulkan: prefill at 8k 470.8 vs 307.5 / 234.4 tok/s; decode after a 128k-token document 23.8 vs 16.2 / 11.6. R9700: prefill 2,805 vs 1,065 / 787.5 at 32k (+163% / +256%), 1,715 vs 718.7 / 557.2 at 128k; decode after a 2k-token document 87.6 vs 50.5 / 50.4; coding chat (new text) 109.6 vs 57.8 / 57.6.

**Radeon 8060S**, max power (80 W), ASUS ROG Flow Z13 (WHIRL single run per cell; llama.cpp median of 1 to 3 runs):

#### Swift-1.5 27B MXFP4 (dense): Prefill tok/s

| Prompt | WHIRL precise | WHIRL balance (default) | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 479.1 | **498.4** | 502.6 | 305.7 | 162.8ᶜ | +63% | +206%ᶜ |
| 8k | 446.5 | **470.8** | 471.8 | 307.5 | 234.4 | +53% | +101% |
| 32k | 401.4 | **426.7** | 428.9 | 266.2 | 221.6 | +60% | +93% |
| 64k | 353.4 | **374.4** | 375.9 | 230.9 | 177.3 | +62% | +111% |
| 96k | 315.7 | **332.9** | 334.1 | 203.3 | 149.0 | +64% | +123% |
| 128k | 282.9 | **298.9** | 297.4 | 180.8 | 127.8 | +65% | +134% |
| ~250k | 197.0 | **221.3** | 222.2 | 135.1 | 78.1 | +64% | +183% |

#### Swift-1.5 27B MXFP4 (dense): Decode tok/s, document QA (256-token answer: explain a function quoted in the document and rewrite it, so part of the answer can be copied)

| Prompt | WHIRL precise | WHIRL balance (default) | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 30.0 | **31.2** | 37.1 | 24.4 (MTP+n-gram) | 25.1 (MTP+n-gram) | +28% | +24% |
| 8k | 30.8 | **32.5** | 37.7 | 24.6 (MTP+n-gram) | 25.2 (MTP+n-gram) | +32% | +29% |
| 32k | 25.5 | **29.5** | 34.7 | 22.7 (MTP+n-gram) | 17.0 (MTP+n-gram) | +30% | +74% |
| 64k | 23.9 | **26.0** | 31.2 | 18.4 (MTP+n-gram) | 18.0 (MTP+n-gram) | +41% | +44% |
| 96k | 21.8 | **24.1** | 29.6 | 17.9 (MTP+n-gram) | 16.6 (MTP+n-gram) | +35% | +45% |
| 128k | 22.4 | **23.8** | 26.1 | 16.2 (MTP+n-gram) | 11.6 (MTP+n-gram) | +46% | +105% |
| ~250k | 14.6 | **16.8** | 20.7 | 12.2 (MTP+n-gram) | 10.9 (MTP+n-gram) | +38% | +54% |

ᶜ 2k prefill on the 8060S, llama.cpp Vulkan: Vulkan prefill is much slower on short prompts on this GPU (well below its own trend at 8k+), so the 2k lead over Vulkan is not used as a headline; headline leads over Vulkan use 8k and longer.

**AMD Radeon AI PRO R9700** (single run per cell):

#### Swift-1.5 27B MXFP4 (dense): Prefill tok/s

| Prompt | WHIRL precise | WHIRL balance (default) | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 1,702 | **3,273** | 3,180 | 1,124 | 829.4 | +191% | +295% |
| 8k | 1,764 | **3,296** | 3,269 | 867.5ᵇ | 705.5ᵇ | +280%ᵇ | +367%ᵇ |
| 32k | 1,626 | **2,805** | 2,790 | 1,065 | 787.5 | +163% | +256% |
| 64k | 1,444 | **2,332** | 2,304 | 913.6 | 691.9 | +155% | +237% |
| 96k | 1,301 | **1,971** | 1,952 | 805.2 | 618.1 | +145% | +219% |
| 128k | 1,185 | **1,715** | 1,697 | 718.7 | 557.2 | +139% | +208% |
| ~250k | not measured ᵈ | **1,153** | 1,153 | 342.0ᵃ | 259.9ᵃ | +237%ᵃ | +344%ᵃ |

#### Swift-1.5 27B MXFP4 (dense): Decode tok/s, document QA (256-token answer: explain a function quoted in the document and rewrite it, so part of the answer can be copied)

| Prompt | WHIRL precise | WHIRL balance (default) | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 70.1 | **87.6** | 115.5 | 50.5 (MTP+n-gram) | 50.4 (MTP+n-gram) | +74% | +74% |
| 8k | 74.3 | **90.7** | 102.3 | 28.2ᵇ (MTP+n-gram) | 44.4ᵇ (MTP+n-gram) | +222%ᵇ | +104%ᵇ |
| 32k | 56.4 | **79.2** | 79.3 | 44.0 (MTP+n-gram) | 43.1 (MTP+n-gram) | +80% | +84% |
| 64k | 56.7 | **60.4** | 67.8 | 38.6 (MTP+n-gram) | 39.5 (MTP+n-gram) | +56% | +53% |
| 96k | 49.8 | **56.3** | 63.4 | 36.1 (MTP+n-gram) | 40.7 (MTP+n-gram) | +56% | +38% |
| 128k | 49.4 | **53.9** | 67.0 | 33.2 (MTP+n-gram) | 34.8 (MTP+n-gram) | +62% | +55% |
| ~250k | not measured ᵈ | **37.9** | 38.8 | 2.3ᵃ (MTP+n-gram) | 6.4ᵃ (MTP+n-gram) | +1548%ᵃ | +492%ᵃ |

ᵃ ~250k on the R9700 with Swift 27B: official llama.cpp with f16 KV does not fit in 32 GB at this length (its log: "failed to fit params to free device memory") and spills into shared memory, which collapses its decode. The numbers are shown as measured, but this cell is not used as a lead anywhere.

ᵇ 8k on the R9700 with Swift 27B: the llama.cpp 8k request was the first long request on a fresh server (cold), and it came out below its own 2k and 32k values. This cell is not used as a lead anywhere.

ᵈ Swift 27B precise at ~250k on the R9700: not measured. precise keeps the KV cache in f16, and f16 KV for this length does not fit next to the weights in 32 GB (the f16 pool holds about 190k tokens).

**Coding chat (new text)**: 512 tokens; the answer (explanation and code) is all newly written, with nothing in the prompt to copy, a hard case for speculative drafts.

| GPU | WHIRL precise | WHIRL balance (default) | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| Radeon 8060S (max power) | 36.3 | **34.2** | 42.1 | 27.6 (MTP+n-gram) | 28.3 (MTP+n-gram) | +24% | +21% |
| Radeon AI PRO R9700 | 83.4 | **109.6** | 130.4 | 57.8 (MTP+n-gram) | 57.6 (MTP+n-gram) | +90% | +90% |

8060S coding chat: one run per WHIRL mode. How many drafts are accepted depends on the text each mode writes, so in this run Swift 27B precise (36.3) came out above balance (34.2).

Why this file is fast in WHIRL: fp8 prefill on the R9700 (the MXFP4 matrix multiplies run on RDNA 4's fp8 matrix units in WHIRL's balance and fast modes), Q8_0 `ssm_alpha` / `ssm_beta` so MTP output stays bitwise identical to plain greedy decoding, and the Q6_K head, the fastest head for WHIRL's draft step. These are speed measurements only. In WHIRL's precise and balance modes the speculative output is identical to plain greedy output of that mode, so the speed-up does not change the generated text.

### Variant comparison in WHIRL (pre-release build, 2026-10-01, mixed 19-prompt protocol)

Used to choose the recommended variant. Server defaults, **19 prompts mixing zh thinking off, zh thinking on and code-edit prompts**, 2 interleaved rounds, greedy-equivalent sampling. Because the code-edit prompts are included, the MTP + n-gram row is an average over generation *and* editing prompts and is not comparable with single-scenario numbers (the zh thinking-off group alone was 106.0 tok/s for A in this run). "Qwen MXFP4" is the reference Qwen3.8-27B MXFP4 GGUF mentioned above, measured the same way.

| WHIRL (pre-release), decode tok/s | A (Q6_K head) | B (Q8_0 head) | C (Q4_K head) | Qwen3.8-27B MXFP4 |
|---|---|---|---|---|
| Plain (no MTP, no speculation) | 36.8 | 36.1 | 37.5 | 36.9 |
| MTP | 117.0 | 98.0 | 112.0 | 118.1 |
| MTP + n-gram — **19-prompt mixed average incl. code-edit prompts** | 157.2 | 143.6 | 157.5 | 156.9 |
| MTP draft acceptance (MTP / MTP + n-gram) | 65.4% / 66.3% | 73.2% / 72.9% | 68.3% / 69.1% | 60.6% / 62.2% |
| Prefill tok/s, 2k / 8k / 32k | 3164 / 3302 / 2570 | 3181 / 3291 / 2563 | 3138 / 3307 / 2572 | 3223 / 3338 / 2602 |
| Plain decode (no MTP) after 16k context | 34.6 | 34.1 | 35.2 | 34.7 |

Why A is recommended: B's 8-bit head is the most expensive part of every draft step, so it is the slowest with MTP despite the highest acceptance rate. C is fastest without speculation and on llama.cpp, but in WHIRL A gets a dedicated Q6_K-head path that made it 4.5% faster than C with MTP (and equal with MTP + n-gram) in this comparison, while keeping a more precise head.

## Thinking length vs Qwen3.8-27B MXFP4 (small sample)

Swift A vs the reference Qwen3.8-27B MXFP4 GGUF, both served by WHIRL (MTP + n-gram) on the R9700. Settings: temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0 (official sampling for both), thinking on (template default), **max_tokens 16,384**, 9 prompts × seeds 1 and 2 (n = 18 runs per model). Thinking tokens = `reasoning_content` counted with the model's tokenizer; wall = full non-streaming request time. `*` = hit the 16,384-token cap (value is a lower bound).

| Prompt | Swift think tokens (s1 / s2) | Swift wall s | Qwen think tokens (s1 / s2) | Qwen wall s |
|---|---|---|---|---|
| 你好 | 24 / 24 | 0.5 / 0.6 | 21 / 21 | 0.5 / 0.5 |
| Hi | 16 / 16 | 0.4 / 0.4 | 46 / 43 | 1.1 / 0.8 |
| zh: Python LRU | 12639* / 10619 | 161.5 / 135.2 | 16382* / 16383* | 189.8 / 199.2 |
| zh: TypeScript hook | 2148 / 7695 | 47.4 / 109.5 | 11795 / 10961 | 179.8 / 166.6 |
| zh: SQL report | 8961 / 7658 | 174.5 / 138.6 | 12893* / 13540* | 210.9 / 210.4 |
| zh: bug fix | 2842 / 2386 | 51.0 / 40.4 | 9598 / 5944 | 140.7 / 97.7 |
| en: ISO-8601 duration parser | 16384* / 16310* | 206.5 / 203.6 | 16382* / 16384* | 212.7 / 132.3 |
| en: Go bounded queue | 16384* / 16384* | 236.1 / 230.1 | 16383* / 16384* | 245.6 / 240.8 |
| en: JS debounce bug | 4020 / 4789 | 57.6 / 65.5 | 1855 / 4495 | 30.5 / 68.2 |

- Totals (18 runs): Swift 129,299 thinking tokens / 1,859 s / 5 capped; Qwen 169,510 / 2,328 s / 8 capped. Because Qwen hit the cap more often, its totals are lower bounds.
- 4 Chinese coding prompts: Swift 54,948 vs Qwen 97,496 thinking tokens (−44%), wall 858 s vs 1,395 s (−38%), capped 1 vs 4.
- 3 English prompts: similar (Swift 74,271 vs Qwen 71,883 tokens); on two of them both models were still thinking at 16k.
- A capped Swift run (en ISO-8601, seed 1) was re-run and inspected: repeated 32-gram ratio 0.5% (first half) / 1.3% (second half) — genuine long deliberation, not a degenerate repetition loop.
- This is a small, single-machine sample with a 16k cap; it does not measure answer correctness. See the upstream card for UkisAI's controlled evaluation.

## License

These files are a quantized (Object-form) Derivative Work of Swift 1.5 Qwen3.8-27B and are distributed under the same terms as upstream:

- **Swift Open License v1.0** for UkisAI's Swift Contribution — see [`LICENSE`](LICENSE) (unmodified copy of the [upstream LICENSE](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b/blob/main/LICENSE)).
- **Apache License 2.0** for the Qwen3.8-27B base model — see [`LICENSE-APACHE-2.0`](LICENSE-APACHE-2.0).
- [`NOTICE`](NOTICE): the upstream NOTICE verbatim, plus a "Changes in this redistribution" section describing the conversion and quantization.

Notice for redistributors, reproduced verbatim from the appendix of the Swift Open License v1.0:

```
   Copyright 2026 UkisAI. Swift Contribution licensed under the Swift Open
   License v1.0 (https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b/blob/main/LICENSE).
   Derivative of Qwen3.8-27B, Copyright 2026 Alibaba Cloud, Apache License 2.0.
```

**Commercial-use threshold (summary, not legal advice — the LICENSE text governs):** personal, research, educational, evaluation and commercial use are permitted for individuals and organizations whose gross revenue, together with all affiliates (entities controlling, controlled by, or under common control with them), is below US$1,000,000 over the most recently completed fiscal year. Commercial use by an entity at or above that threshold is not licensed under the Swift Open License and requires a separate Swift Enterprise License from UkisAI ([contact](https://ukisai.com/contact)). The threshold does not apply to qualified non-profit organizations' non-commercial or research use. The license terminates automatically on non-compliance.

**Base model:** Qwen3.8-27B is Copyright 2026 Alibaba Cloud and licensed under the Apache License 2.0. Per Section 6 of the Swift Open License, nothing in it limits your rights in the Qwen3.8-27B base model under Apache 2.0; the commercial limitation applies only to the Swift Contribution.

## Credits

- **Alibaba Cloud / Qwen team** — [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), the base model.
- **UkisAI** — [Swift 1.5 Qwen3.8-27B](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b), the post-trained model these files are made from.
- **bartowski** — the [importance matrix](https://huggingface.co/bartowski/ukisai_Swift-1.5-Qwen3.8-27b-GGUF) used by the weighted MXFP4 quantizer.
- **llama.cpp / ggml** contributors — conversion, quantization and inference tooling.

Upstream citation:

```bibtex
@misc{swift-1.5-qwen3.8-27b,
  title  = {Swift 1.5 Qwen3.8-27B},
  author = {UkisAI},
  year   = {2026},
  url    = {https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b}
}
```

## 繁體中文摘要

- 這是 [ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) 的**非官方社群 MXFP4 GGUF 量化**，與 UkisAI 無關；「Swift」「UkisAI」商標僅用來說明權重來源。另附視覺投影檔 `mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf`（F16），搭配任一版本即可看圖（llama.cpp `--mmproj`）。
- 保留 MTP 層（Q8_0），llama.cpp 可用 `--spec-type draft-mtp` 做自我推測解碼。
- 三個變體只差輸出頭：**A = Q6_K（推薦）**、C = Q4_K（在 llama.cpp 與不開推測解碼時稍快）、B = Q8_0（開 MTP 時最慢，不建議）。
- **2026-10-19 重新量化**：改用 imatrix 加權的 MXFP4 量化器（[`mxfp4_wsearch.patch`](mxfp4_wsearch.patch)，llama.cpp b11719），張量配置與檔名不變；對 BF16 的 KL 散度比第一版（2026-10-01）低 17～19%（A：0.0495 對 0.0606，同 top-1 91.62% 對 91.01%）。視覺投影檔沒有變。
- 大矩陣 MXFP4；token_embd、ssm_alpha / beta、MTP 層 Q8_0；imatrix 來自 bartowski（涵蓋 64 個主層的所有矩陣，沒有 output／token_embd／MTP 層條目），透過加權 MXFP4 搜尋作用在 400 個 MXFP4 矩陣上；輸出頭沒有用到 imatrix（第一版卡片寫 imatrix 只影響輸出頭，這是錯的）。
- AMD Radeon AI PRO R9700、llama.cpp b11214：A 的 plain decode 33.3 tok/s、MTP 58.1 tok/s、2k prefill 1215 tok/s。
- 最佳效能請用 **[WHIRL](https://github.com/tsaipifong/whirl-llm)**（Apache-2.0 開源，支援 AMD Ryzen AI Max+ 395 的 Radeon 8060S（0.2.0 新增）與 AMD Radeon AI PRO R9700，[v0.2.0 下載](https://github.com/tsaipifong/whirl-llm/releases/tag/v0.2.0)：`whirl-0.2.0-windows-x64.zip` 解壓縮即可執行，只需 AMD 驅動 Adrenalin 26.8.1 或更新版）。本檔已用 WHIRL 0.2.0 三種模式對官方 llama.cpp b11214（ROCm／Vulkan 分開列，llama.cpp 用 MTP + n-gram）量測，完整表格見上方：
Radeon 8060S（最大功率（80 W），ASUS ROG Flow Z13），WHIRL balance 對 llama.cpp ROCm／Vulkan：8k prefill 470.8 對 307.5／234.4 tok/s；128k token 文件之後的 decode 23.8 對 16.2／11.6。R9700：32k prefill 2,805 對 1,065／787.5（+163%／+256%），128k 1,715 對 718.7／557.2；2k token 文件之後的 decode 87.6 對 50.5／50.4；寫程式對話（全新撰寫）109.6 對 57.8／57.6。
- 另有 MoE 模型 [Ornith-1.5-35B-A3B MXFP4 GGUF](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF)（每 token 約 3B 啟用）。
- 對 BF16 的品質（llama.cpp b11719 CPU，程式碼＋中文 80×512 token）：BF16 困惑度 4.5053；本版 A／B／C 的 KLD 0.0495／0.0492／0.0527，困惑度比 BF16 高 3.6%／3.6%／3.9%；第一版 0.0606／0.0604／0.0638，高 5.3%／5.3%／5.6%。
- 速度（Radeon 8060S、WHIRL 0.2.0 balance、server、2k／32k token 文件，與第一版 A 交錯量測）：不開推測解碼的 decode 兩者相同（2k 之後 14.8、32k 之後 13.7 tok/s）；兩個檔案用同一份 WHIRL 核心調校表時 prefill 也相同（2k 483.5 對 484.2、32k 424.3 對 424.1 tok/s）。WHIRL 對每個新模型檔會調校一次 prefill 核心，本機重新調校新檔的結果比第一版檔案的舊調校表慢 2～3%（2k 471.8 對 486.1），差異來自調校而不是檔案。
- 小樣本思考長度比較（16k 上限、9 題 × 2 seed）：中文寫程式 4 題 Swift 思考 token 比 Qwen3.8-27B MXFP4 少 44%、時間少 38%。
- 授權：Swift Open License v1.0（年營收含關係企業達 100 萬美元以上的商業使用需另取得授權）＋ 基底模型 Apache 2.0；詳見 LICENSE / NOTICE。
