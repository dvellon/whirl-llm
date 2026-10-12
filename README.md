**English** | [繁體中文](docs/README_zh-TW.md)

# WHIRL

**Windows HIP Inference for RDNA LLMs**

WHIRL is an LLM inference engine for the AMD Radeon AI PRO R9700 (RDNA 4) and the Radeon 8060S (Ryzen AI Max+ 395, RDNA 3.5) on Windows.

**Design principles**
1. **Built for AMD GPUs on Windows.** A native Windows C++/HIP inference engine with hand-written, hand-tuned kernels for RDNA. No WSL, no Docker, no Linux VM: install the AMD driver and run.
2. **Fully open, kernels included.** Everything from host-side scheduling to every GPU kernel is open source under Apache-2.0, with no closed binary components. Bug reports, benchmark results and code contributions are all welcome. Let's make AMD inference on Windows as good as it can be.
3. **Accuracy before speed.** Three numerics modes ship in the box: **precise**, **balance** (default) and **fast** ([below](#three-modes)). Within precise and balance, every speedup (speculative decoding, batching, prefix caching, cache restore) produces output bit-identical to plain greedy decoding in that mode, enforced by standing tests. precise adds nothing lossy beyond the GGUF file itself; balance stays within KL <= ~0.005 of precise; fast is opt-in and its measured KL is published.
4. **Few models, pushed to the limit.** WHIRL doesn't try to run everything. It targets specific models on specific hardware and pushes them close to the hardware limit, adding architectures one at a time.
5. **Transparent, verifiable numbers.** Every figure is measured, with methods, commands and raw data published. For content-dependent features such as speculative decoding, we report the best case, the typical case and the worst case, so you know what to expect.
6. **Designed for real workflows.** Coding agents and long contexts come first: fast long-prompt processing, a responsive main conversation while others share the GPU, and a VRAM → RAM → SSD cache that survives server restarts.

The architectures supported today are **`qwen35` (dense)** and **`qwen35moe` (mixture of experts)**; more will be added one by one. WHIRL follows the design philosophy of [NInfer](https://github.com/Neroued/ninfer) (Apache-2.0) — build for selected models and selected GPUs, as close to the hardware limit as possible; [zynfer](https://github.com/thanos/zynfer) carried the same idea to RDNA 4.

## WHIRL vs llama.cpp at a glance

Radeon AI PRO R9700, WHIRL 0.2.0 in its default mode (**balance**) against llama.cpp b11214 (ROCm). Same GGUF file, same prompts, greedy decoding, one user unless noted. llama.cpp was started as `llama-server --device ROCm0 -ngl 99 -fa on -b 2048 -ub 1024 --jinja --reasoning off --top-k 1 -np 1` (`-ub 2048` for the MoE model) with f16 KV, and each cell uses the faster of its plain and MTP + n-gram modes. "% faster" is WHIRL / llama.cpp - 1. Every WHIRL speedup keeps the output identical to plain greedy decoding of the same mode ([why](#speed-without-changing-the-answer)).

| R9700, balance (default) | Qwen3.8-27B UD-Q4_K_M (dense) | Ornith-1.5-35B-A3B MXFP4 (MoE, ~3B active) |
|---|---|---|
| Prefill tok/s, 8k-token prompt | 1,914 vs 1,201 (**+59%**) | 11,199 vs 4,426 (**+153%**) |
| Prefill tok/s, 32k-token prompt | 1,721 vs 1,082 (**+59%**) | 8,585 vs 3,853 (**+123%**) |
| Prefill tok/s, 128k-token prompt | 1,230 vs 756 (**+63%**) | 4,273 vs 2,285 (**+87%**) |
| Prefill tok/s, 249k-token prompt ("256k") | 912 vs 536 (**+70%**)¹ | 2,659 vs 1,534 (**+73%**) |
| Decode tok/s, question about an 8k-token document | 84.9 vs 41.8 (**+103%**) | 229.3 vs 103.5 (**+122%**) |
| Decode tok/s, question about a 128k-token document | 49.3 vs 27.1 (**+82%**) | 132.9 vs 69.7 (**+91%**) |
| Decode tok/s, new writing after 128k tokens (worst case for drafts) | 33.8 vs 21.2 (**+59%**) | 113.8 vs 69.9 (**+63%**) |
| Decode tok/s, Chinese coding chat | 100.9 vs 58.1 (**+74%**) | 242.5 vs 108.8 (**+123%**) |
| Server, 4 users at once, aggregate tok/s | 251.4 vs 90.6 (**+177%**) | not measured |

<!-- src: audit/REL3.md, section "R9700": prefill tables (2k..256k rows, balance column and llama.cpp best), decode QA tables, worst-case tables, Chinese chat table (dense llama.cpp 58.1 = MTP+n-gram run 1; REL3 notes it was primed by a 64-token warm-up, which slightly favours llama.cpp), 4-user table (80.4 plain / 90.6 MTP+n-gram; the better one is used). Percentages are REL3's own: WHIRL / llama.cpp - 1. -->

¹ llama.cpp could not hold f16 KV at 256k on the dense model (same as in the 0.1.3 runs) and used q8_0 KV; WHIRL used its balance KV (q8h). The length sweeps are single runs (chat: median of 3), so treat differences under about 5% as noise. Qwen3.8-27B UD-Q4_K_M is the unsloth file; Ornith-1.5 is our MXFP4 quantization. The other two modes, every length, and the cases where the lead is small: [Performance](#performance). Swift-1.5 27B MXFP4 was not re-measured for 0.2.0; its 0.1.3 figures stay in [Performance](#performance).

[8060S: TBD — the Radeon 8060S comparison goes here when REL3 has finished measuring it; see the [8060S section](#radeon-8060s)]

## <a id="three-modes"></a>Three numerics modes

Chosen at start-up with `--precise`, `--balance` (the default, also what you get with no flag) or `--fast`; the same on both GPUs and in `whirl chat`, `whirl bench` and `whirl-server`. Details and the per-item switches: [usage.md](docs/usage.md#modes).

| | **precise** (`--precise`) | **balance** (default) | **fast** (`--fast`, opt-in) |
|---|---|---|---|
| KV cache, dense models | f16 | q8h (int8) | q4 (4-bit) |
| KV cache, MoE models | f16 | f16 | q4 (4-bit) |
| Arithmetic | GGUF weights read as f16; f16 activations in prefill and in decode / verify; no fp8, no int8 activations | the 0.1.x speed path: fp8 / f16 intermediates in prefill, int8 activations in decode | balance, plus relaxed acceptance of MTP drafts (dense models) |
| Output vs the reference | nothing quantized beyond the GGUF file | near-lossless: KL <= ~0.005 against precise (q8h KV vs f16: 0.0002 to 0.002) | within KL <= 0.03; measured KL of q4 KV vs f16: 0.0016 to 0.0100 (R9700) |
| Speedups (MTP, n-gram, batching, cache) | output bit-identical to plain greedy decoding | output bit-identical to plain greedy decoding | not bit-identical (relaxed acceptance); the same request alone and among other users can give different text |
| Speed vs balance, R9700 | decode -25% to -29%, dense prefill -5% to -10% (MoE short prompts -30% to -36%) | reference | dense decode +17% (chat), +15% (4 users), +18% (249k context); MoE faster only above ~64k |
| Pick it when | you need output you can reproduce and compare exactly, or you are checking another engine against WHIRL | almost always | long contexts, or you want the most tok/s and can accept small output changes |

<!-- src: KV per mode: CHANGELOG.md Unreleased "KV format fixed per mode (BAL-Q8)". balance KL <= ~0.005 and fast gate KL <= 0.03: internal_docs/v013_plan.md (owner decision, 2026-10-08) and docs/usage.md Modes. q8h KL 0.0002-0.002: CHANGELOG BAL-Q8. q4 KV KL 0.0016-0.0100 on the R9700: CHANGELOG FAST-1c (code 64k 0.0100, zh 64k 0.0016, code 128k 0.0049, zh 128k 0.0071). Precise speed cost: CHANGELOG "precise decode / verify no longer quantizes activations" (plain decode -26..-29%), audit/REL3.md (chat 75.7 vs 100.9 = -25%; prefill -5..-10% dense, -30..-36% MoE short). Fast vs balance: audit/REL3.md dense chat 118.1 vs 100.9, 4 users 288.3 vs 251.4, 249k QA decode 40.1 vs 33.9; MoE note "Fast on the MoE is slower than balance below ~64k". -->

What the guarantees mean in practice:

- In **precise** and **balance**, a request gives the same tokens alone or among other users, with or without MTP / n-gram / DFlash2 drafts, from a cold prompt or from a cached prefix. Balance differs from precise only through its documented lossy items (fp8 / f16 intermediates, int8 activations, q8h KV), bounded by the KL above.
- **fast** may change output. Its quality is bounded by a gate (KL <= 0.03 against the f16-KV reference, top-1 agreement, and an execution-graded coding check), not by bit-identity. Relaxed MTP acceptance (dense models only) keeps a draft that is not the verify row's argmax when it is among the 4 most likely tokens with p >= 0.1 of the argmax, and refuses an acceptance that would repeat a 6-gram already in the last 4096 tokens. In the gate run (Qwen3.8-27B Q4_K_M) 12 of 12 coding tasks executed correctly, 0 of 24 generations looped, the repetition rate was 0.167 against 0.157 for precise, and about 7% of the accepted drafts came from the relaxed rule. A request run alone and the same request run among other users can differ in this mode; precise and balance guarantee equality.
- The KL numbers compare logits against the same model with f16 KV. They are not an accuracy benchmark of the model.

<!-- src: relaxed acceptance rule and 6-gram guard: docs/usage.md Modes (relaxacc row). Gate results (exec 12/12, loops 0/24, repetition .167 vs precise .157, 6.9% relaxed): progress/FAST-2.log 2026-10-07 12:49 and 13:01 (measured on the 8060S, Q4_K_M; not re-run on the R9700). Solo vs concurrent in fast: audit/CONC-8S.md "Limits" and CHANGELOG conc8s. -->

## Speed without changing the answer

In **precise** and **balance**, every acceleration in WHIRL is checked to produce **the same tokens, bit for bit, as plain greedy decoding** of the same model file in the same mode (**fast** is the exception, see [above](#three-modes)):

| Acceleration | Guarantee | How it is checked |
|---|---|---|
| MTP, n-gram and DFlash2 speculative decoding | the verified output is bit-identical to plain greedy decoding | built-in self-check `whirl selftest` (multi-row kernels vs one row, on the model's own weights), kernel tests, server gate |
| Several users batched together | each request's output is bit-identical to running it alone | kernel tests (row-invariant GEMMs and attention), server gate |
| Prefix-cache hits on shared system prompts and shared prefixes | bit-identical to processing the prompt from scratch (checkpoints sit on the prompt's own chunk schedule) | server gate |
| Restoring a session from the RAM or SSD tier | byte-identical to the state that was saved | byte comparison on spill (`TIER_VERIFY`), server gate |

What WHIRL does **not** do to go faster in precise and balance: no KV cache below 8 bits, no requantizing the model's weights to fewer bits (only the draft-only MTP and draft-head copies, which cannot change the output), no fp8 attention. `--fast` uses a 4-bit KV cache and relaxed draft acceptance; that is why it is a separate, opt-in mode.

To be precise about what this means:

- **Weights** are whatever GGUF you choose; their quantization is the file author's choice, and WHIRL computes with it as stored.
- **KV cache** is fixed per mode: precise f16, balance q8h on dense models and f16 on MoE models, fast q4. A format that does not fit lowers the context (with a warning) or stops the program with a message; it never switches to a lower-precision format on its own. `WHIRL_KV` is a debug override. q8h and q4 are numerically lossy compared with f16, like any quantized KV; WHIRL does not claim "zero loss" for them and publishes the KL.
- **Continuing a conversation** reuses the KV and DeltaNet state computed for the previous turn, including the tokens generated by decoding. They come from the decode kernels (and from a different split of the prompt into chunks), so they are numerically equivalent but not bit-identical to re-processing the whole history as one prompt; the difference is documented and measured ([kv-and-caching.md](docs/guide/en/kv-and-caching.md)).
- **MXFP4 files in balance** use fp8 activations in the MXFP4 prefill matrix multiplies and f16 in parts of the DeltaNet prefill (the 0.1.x speed path); `--precise` turns both off (prefill is slower, see [Performance](#performance)). Decode and verification use the same kernels for every row of a batch, so the guarantees above hold in each mode.

## Requirements

| Item | Details |
|---|---|
| GPU | **AMD Radeon AI PRO R9700** (RDNA 4, gfx1201), or **Radeon 8060S** (Ryzen AI Max+ 395 iGPU, RDNA 3.5, gfx1151), supported from 0.2.0 ([below](#radeon-8060s)); one GPU per process |
| OS | **Windows 11**, 64-bit |
| Driver | **AMD Software: Adrenalin Edition 26.8.1** (driver 32.0.31041.1004) or newer |
| To run | only the driver (it installs `amdhip64_7.dll` and `amd_comgr_3.dll`). No HIP SDK, no ROCm, no Visual C++ runtime |
| To build from source | HIP SDK 7.2 + Visual Studio 2022 Build Tools (MSVC) + CMake + Ninja — see [building.md](docs/building.md) |
| Memory / disk | R9700: the server pins about 1/4 of host RAM (8–32 GiB; 16 GiB on a 64 GB PC) for the RAM cache tier. 8060S: the RAM tier is 1/8 of the **OS-visible** RAM (8 GiB on a 64 GB machine; off under 32 GB). The SSD tier holds at most 64 GiB on disk and by default writes only when the server is stopped normally ([SSD cache](#ssd-cache)). Adjust or turn off with `--kv-ram-mb`, `--kv-ssd-gb`, `--kv-ssd-mode` |

Our R9700 test machine connects the card as a USB4 / Thunderbolt eGPU; that works. Numbers on a card in a direct PCIe slot may differ slightly (mostly where data crosses the host link, such as model loading and the RAM / SSD tiers).

Not supported: other AMD GPUs. The RX 9070 series has the same chip (gfx1201) but 16 GB, most likely too little for these 27B / 35B models (untested).

## Quick start

1. Download `whirl-0.2.0-windows-x64.zip` from [Releases](https://github.com/tsaipifong/whirl-llm/releases) and a [supported model](#supported-models).
2. In PowerShell:

```powershell
Unblock-File .\whirl-0.2.0-windows-x64.zip
Expand-Archive .\whirl-0.2.0-windows-x64.zip -DestinationPath C:\whirl
cd C:\whirl\whirl-0.2.0-windows-x64
.\whirl.exe devices                                    # lists AMD Radeon AI PRO R9700 (gfx1201) and / or Radeon 8060S (gfx1151)
.\whirl.exe chat C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf "Explain TCP slow start in two sentences." --max-tokens 400
.\whirl-server.exe C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf --port 8080
```

The server speaks the OpenAI API at `http://127.0.0.1:8080/v1` (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`), so any OpenAI-compatible client works. Stop it with Ctrl+C; cached sessions are then written to the SSD tier (up to 60 s).

Add `--precise` or `--fast` to pick another [numerics mode](#three-modes); with no flag you get balance.

For long documents, give a slot a longer context, for example `--ctx-per-slot 262144` (256k tokens per request); see [recipes.md](docs/recipes.md#long-context).

The first run with a model tunes the GPU kernels once — about 1.5 minutes for a 27B Q4_K_M file, a few seconds for MXFP4 files — and caches the result in `%LOCALAPPDATA%\whirl`. Full walkthrough: [quick start](docs/quickstart.md). Every option: [usage reference](docs/usage.md).

## Supported models

WHIRL reads `general.architecture` from the GGUF and accepts only `qwen35` and `qwen35moe` (the Gated DeltaNet + gated attention hybrid, with an optional MTP head) and fine-tunes that keep that architecture. Other architectures (Llama, Gemma, Mistral, DeepSeek, `qwen3`, `qwen2`, …) are refused at load time with a clear message, and so are tensor types WHIRL has no kernels for (for example NVFP4); there is no slow generic fallback.

> **Recommended: our own MXFP4 quantizations give the best performance.**
> - Dense 27B: [**Swift-1.5-Qwen3.8-27b-MXFP4-GGUF**](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF), variant A (`Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`). Swift-1.5 also thinks much less than the base model (44% fewer thinking tokens on our mixed Chinese/English coding prompts).
> - MoE 35B (~3B active): **Ornith-1.5-35B-A3B MXFP4**: [tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF).

Tested GGUF files (loaded, greedy output checked, MTP output identical to plain greedy, benchmarked on the R9700):

| Model file (Hugging Face) | Type | Arch | Quantization | Notes |
|---|---|---|---|---|
| [tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF) `…-A-outQ6_K.gguf` (recommended), `…-B-outQ8_0.gguf`, `…-C-outQ4_K.gguf` | dense 27B | qwen35 | MXFP4 (+ Q8_0 MTP / embeddings) | balance uses fp8 MXFP4 prefill (`--precise` does not); quantized by us for WHIRL |
| [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `Qwen3.8-27B-UD-Q4_K_M.gguf` | dense 27B | qwen35 | UD-Q4_K_M | no fp8 prefill in any mode |
| [FreedomAISVR/Qwen3.8-27B-MXFP4-GGUF](https://huggingface.co/FreedomAISVR/Qwen3.8-27B-MXFP4-GGUF) `qwen3.8-27b-mxfp4.gguf` | dense 27B | qwen35 | MXFP4 | balance uses fp8 MXFP4 prefill |
| [`Ornith-1.5-35B-A3B-MXFP4.gguf`](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF) (quantized by us) | MoE 35B, ~3B active | qwen35moe | MXFP4 experts and dense (+ Q8_0 MTP / embeddings, Q6_K head) | balance uses fp8 expert prefill |
| [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) `Ornith-1.5-35B-Q4_K_M.gguf` | MoE 35B, ~3B active | qwen35moe | Q4_K_M | no fp8 prefill in any mode |

Image input works with a Qwen3-VL-style mmproj (F16 / BF16) via `--mmproj`; it was tested with the 27B files above. Other `qwen35` / `qwen35moe` GGUFs should load but are untested. Choosing and making GGUFs: [quantization.md](docs/guide/en/quantization.md).

## Performance

### Radeon AI PRO R9700, WHIRL 0.2.0 (precise / balance / fast) vs llama.cpp

Measured 2026-10-12 on the 0.2.0 release candidate; nothing is reused from older builds. Method: one server start per (engine, model, mode); requests one after another, each with a unique first line so no prefix cache can hit. The prompt is the first N tokens of a non-repeating document, cut at a line boundary (2,050 / 8,155 / 32,774 / 65,541 / 98,310 / 131,084 / 249,506 tokens), followed by a question about it ("QA": the answer quotes the document, so drafts help). Prefill and decode tok/s are the server's own timings (256 generated tokens). The "new writing" rows ask for a brand-new short story after the document, the worst case for drafts. Chinese coding chat: 512 tokens, median of 3 runs after a warm-up. 4 users: four different short Chinese coding prompts at once, 512 tokens each, aggregate tok/s over the wall time including prefill. Length sweeps are n = 1 per cell: treat differences under about 5% as noise; single QA cells move by 5–15% with how well the drafts hit.

WHIRL: draft = MTP + n-gram (default), greedy, thinking off, `--kv-ssd-mode off`; KV precise f16, dense balance q8h, dense fast q4, MoE balance f16, MoE fast q4. llama.cpp: b11214 (`version: 0.5.0-dev (build 11214, commit 2ebd9ae62)`), `llama-server --device ROCm0 -ngl 99 -fa on -b 2048 -ub 1024 (MoE 2048) --jinja --reasoning off --top-k 1 -np 1 -c 133120 (dense, up to 128k) / 262144 (MoE)`, `HIP_VISIBLE_DEVICES=1 GGML_CUDA_NO_PINNED=1`, f16 KV. Dense: run plain and with `--spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-n-min 0 --spec-draft-p-min 0.3`. MoE: plain for the sweeps (the 0.1.3 probes found plain fastest there), and chat also with `draft-mtp --spec-draft-n-max 2`. "llama.cpp best" is the faster mode for that cell. Dense 256k: f16 KV does not fit in llama.cpp, so that row used `-ctk q8_0 -ctv q8_0 -c 262144`.

<!-- src for this subsection: audit/REL3.md, section "R9700" (WHIRL v0.2.0-rc4 tip 100a9a8). Raw data: whirl-work\rel3\r97\results.tsv. -->

**Prefill tok/s**

| Length | Dense: precise | balance | fast | llama.cpp best | balance vs llama.cpp | MoE: precise | balance | fast | llama.cpp best | balance vs llama.cpp |
|---|---|---|---|---|---|---|---|---|---|---|
| 2k | 1,559 | 1,701 | 1,657 | 1,120 | +52% | 6,854 | 10,708 | 10,026 | 3,661 | +192% |
| 8k | 1,737 | 1,914 | 1,885 | 1,201 | +59% | 7,238 | 11,199 | 11,120 | 4,426 | +153% |
| 32k | 1,575 | 1,721 | 1,703 | 1,082 | +59% | 5,988 | 8,585 | 8,496 | 3,853 | +123% |
| 64k | 1,404 | 1,514 | 1,506 | 943 | +60% | 4,882 | 6,515 | 6,425 | 3,160 | +106% |
| 96k | 1,271 | 1,357 | 1,350 | 840 | +61% | 4,090 | 5,170 | 5,140 | 2,647 | +95% |
| 128k | 1,172 | 1,230 | 1,222 | 756 | +63% | 3,520 | 4,273 | 4,266 | 2,285 | +87% |
| 249k ("256k") | n/a² | 912 | 913 | 536 | +70% | 2,347 | 2,659 | 2,662 | 1,534 | +73% |

**Decode tok/s after a long document, question about the document (the answer quotes it; drafts help)**

| Length | Dense: precise | balance | fast | llama.cpp best | balance vs llama.cpp | MoE: precise | balance | fast | llama.cpp best | balance vs llama.cpp |
|---|---|---|---|---|---|---|---|---|---|---|
| 2k | 60.7 | 78.9 | 100.7 | 44.3 | +78% | 187.6 | 235.6 | 220.2 | 107.5 | +119% |
| 8k | 56.5 | 84.9 | 95.3 | 41.8 | +103% | 179.0 | 229.3 | 212.1 | 103.5 | +122% |
| 32k | 57.3 | 72.0 | 81.3 | 38.3 | +88% | 146.8 | 193.6 | 184.2 | 93.7 | +107% |
| 64k | 47.8 | 59.6 | 68.5 | 37.1 | +61% | 135.1 | 162.3 | 169.8 | 83.9 | +93% |
| 96k | 44.3 | 55.0 | 55.2 | 29.1 | +89% | 124.6 | 147.3 | 159.6 | 75.7 | +95% |
| 128k | 40.9 | 49.3 | 51.6 | 27.1 | +82% | 111.8 | 132.9 | 152.7 | 69.7 | +91% |
| 249k ("256k") | n/a² | 33.9 | 40.1 | 15.5 | +119% | 88.3 | 97.2 | 120.8 | 53.7 | +81% |

**Decode tok/s, brand-new writing after the document (worst case for drafts)**

| Length | Dense: precise | balance | fast | llama.cpp best | balance vs llama.cpp | MoE: precise | balance | fast | llama.cpp best | balance vs llama.cpp |
|---|---|---|---|---|---|---|---|---|---|---|
| 8k | 36.3 | 51.1 | 65.3 | 30.2 | +69% | 143.2 | 182.3 | 183.5 | 104.5 | +74% |
| 128k | 29.2 | 33.8 | 40.4 | 21.2 | +59% | 100.1 | 113.8 | 134.7 | 69.9 | +63% |

**Chinese coding chat, decode tok/s** (512 tokens, median of 3 runs; llama.cpp: run 1)

| Model | precise | balance | fast | llama.cpp plain | llama.cpp MTP + n-gram |
|---|---|---|---|---|---|
| Qwen3.8-27B UD-Q4_K_M (dense) | 75.7 | 100.9 | 118.1 | 30.4 | 58.1 |
| Ornith-1.5-35B-A3B MXFP4 (MoE) | 185.2 | 242.5 | 220.0 | 108.8 | 101.4 |

**4 users at once, dense model, aggregate tok/s** (4 x 512 tokens, wall time including prefill, run 1): WHIRL precise 200.9, balance 251.4, fast 288.3; llama.cpp plain 80.4, MTP + n-gram 90.6. The second run of each WHIRL mode agreed within 3%. The MoE model and DFlash2 were not measured with 4 users.

² Dense precise cannot hold 256k: f16 KV is 68 KiB per token and the pool holds 176,128 tokens, so `--ctx-per-slot 262144` is refused at start-up with a message. The 128k row used `--ctx-per-slot 163840`, because a 131,084-token prompt does not fit in 131,072.

Where WHIRL does **not** lead by much, or a mode costs speed:

- The lead over llama.cpp is smallest on the dense model when drafts miss: balance +59% on new writing after 128k tokens, and the same +59% at 8k prefill.
- precise costs speed: dense chat decode 75.7 vs 100.9 tok/s (-25%), dense prefill -5% to -10%, MoE prefill -30% to -36% at short lengths.
- fast on the MoE model is slower than balance below about 64k tokens (chat 220.0 vs 242.5, 8k QA 212.1 vs 229.3) and faster above (128k 152.7 vs 132.9). On the MoE, fast is balance plus 4-bit KV only (relaxed acceptance is skipped there), and the 4-bit dequantization costs more than it saves while the KV is small.
- 2k prefill is lower than 8k in every WHIRL row (ramp-up of a short prompt). Otherwise prefill and decode fall smoothly with length, with no cliff up to 249k tokens.
- llama.cpp's MTP + n-gram learns from a repeated prompt: its 2nd and 3rd identical chat request ran at 168 and 304 tok/s. The tables use the first run (primed by a 64-token warm-up of the same prompt, which slightly favours llama.cpp). WHIRL's n-gram drafts look only at the current request and gave the same speed in all 3 runs.
- Not measured: llama.cpp `draft-mtp` sweeps on the MoE model (plain sweeps and one chat run only), 4-user runs for the MoE model, a llama.cpp agent-session replay, repeats of the length sweeps.

Our R9700 is a USB4 eGPU; prefill and decode stay in VRAM, but model loading and RAM / SSD restores cross the host link.

### Earlier measurements: WHIRL 0.1.3 on the R9700, including Swift-1.5 27B MXFP4

Swift-1.5 MXFP4 was not part of the 0.2.0 run, so these are the 0.1.3 numbers (same llama.cpp b11214, the fastest llama.cpp flags found for each row). Decode numbers lead with WHIRL's default mode, **MTP + n-gram**; rows marked **no MTP** compare plain decoding without speculation. The KV cache format is shown per row.

Our MXFP4 files (Swift-1.5 27B and Ornith-1.5-35B-A3B) are the main comparison; Qwen3.8-27B UD-Q4_K_M is shown for reference only. From 96k tokens on, each cell gives the KV format each engine actually used, as `KV WHIRL / llama.cpp`.

![Decode after a 16k and a 256k-token context](docs/images/bench_long_decode.png)

![Server aggregate throughput with 4 users at once](docs/images/bench_server_concurrency.png)

![Time to first token with a cached prefix](docs/images/bench_ttft_cache.png)

| R9700, greedy — WHIRL vs llama.cpp (ratio) | Swift-1.5 27B MXFP4-A (dense) | Ornith-1.5-35B-A3B MXFP4 (MoE, ~3B active) | Qwen3.8-27B UD-Q4_K_M (dense, reference) |
|---|---|---|---|
| Decode tok/s, 7 coding prompts (800 tokens) — WHIRL MTP+n-gram vs llama.cpp fastest¹ | 112.4 vs 60.8 (1.85×) | 257.6 vs 118.8 (2.17×) | 97.3 vs 56.0 (1.74×) |
| Decode tok/s after a 16k-token context — WHIRL MTP+n-gram vs llama.cpp fastest¹ | 71.4 vs 60.9⁴ (1.17×) | 316.4 vs 106.9⁴ (2.96×) | 80.7 vs 60.7⁴ (1.33×) |
| Decode tok/s, same 7 prompts — WHIRL **no MTP** vs llama.cpp **no MTP** | 38.1 vs 33.4 (1.14×) | 177.4 vs 118.8 (1.49×) | 34.8 vs 30.9 (1.13×) |
| Server, 4 concurrent users, aggregate tok/s (incl. prefill) — WHIRL MTP+n-gram vs llama.cpp fastest¹ | 182.1 vs 64.0⁴ (2.84×) | 381.1 vs 176.8⁴ (2.16×) | 122.4 vs 57.1⁴ (2.14×) |
| Prefill tok/s, 8k-token prompt (CLI bench tools) | 3,469 vs 1,338 (2.59×) | 11,258 vs 4,637 (2.43×) | 1,731 vs 1,223 (1.42×) |
| Prefill tok/s, 32k-token prompt | 2,907 vs 1,174 (2.48×) | 8,633 vs 3,778 (2.29×) | 1,570 vs 1,086 (1.45×) |
| Prefill tok/s, 96k-token prompt | 2,021.1 vs 886.7 (2.28×); KV f16 / f16 | 5,183.5 vs 2,602.6 (1.99×); KV f16 / f16 | 1,266.4 vs 837.3 (1.51×); KV f16 / f16 |
| Prefill tok/s, 128k-token prompt | 1,757 vs 791.3 (2.22×); KV f16 / f16 | 4,311 vs 2,239 (1.93×); KV f16 / f16 | 1,158 vs 752.0 (1.54×); KV f16 / f16 |
| **Prefill tok/s, 256k-token prompt** | 969.8 vs 556.1 (1.74×); KV q8h / f16 | 2,135.2 vs 1,442.2 (1.48×); KV q8h / f16 | 742.2 vs 516.4 (1.44×); KV q8h / q8_0 |
| **Decode tok/s after a 256k-token prompt** — WHIRL MTP+n-gram vs llama.cpp plain decoding³ | 35.1 vs 16.8 (2.09×); KV q8h / f16 | 115.6 vs 53.7 (2.15×); KV q8h / f16 | 33.5 vs 10.5 (3.19×); KV q8h / q8_0 |
| Time to first token, new conversation reusing a 26k-token system prompt (s, lower is better) | 0.108 vs 0.384⁴ (3.6×) | 0.098 vs 0.182⁴ (1.9×) | 0.121 vs 0.369⁴ (3.0×) |
| Time to first token, 26k-token session after a server restart (restored from the SSD tier) | 0.613 s vs N/A² | 0.318 s vs N/A² | 0.672 s vs N/A² |

¹ llama.cpp's best of plain / MTP / MTP + n-gram for that row.

² llama.cpp has no automatic persistent KV cache.

³ llama-bench `tg64@d262144`: 64 tokens generated after a 262,144-token context, plain decoding without speculative decoding. llama.cpp ran Swift-1.5 MXFP4 at 256k with f16 KV (it fits, 31.3 GB); Ornith-1.5 at 96k and 256k also used f16 KV; Qwen3.8 Q4_K_M at 256k needed q8_0 KV, because f16 spilled into shared memory.

⁴ llama.cpp re-measured for v0.1.3 with the same prompts as WHIRL; the mode used (the fastest measured): decode after 16k: Swift-1.5 MTP + n-gram, Ornith-1.5 plain, Qwen3.8 MTP + n-gram; 4 concurrent users: Swift-1.5 plain, Ornith-1.5 plain, Qwen3.8 plain; 26k system prompt TTFT: plain only (a 1-token reply; speculative decoding does not apply).

**Speculative decoding: best, typical and worst case** — Swift-1.5 27B MXFP4-A, WHIRL's default MTP + n-gram against decoding without speculation; the output is the same bit for bit:

| | Best case: file editing (repeated content) | Typical: coding-agent session | Long-document QA (heavy quoting, after 128k) | Worst case: new writing after 128k (nothing to copy) |
|---|---|---|---|---|
| Decode tok/s, MTP + n-gram | 174.4 | 92.9 | 87.6 | 39.4 |
| Decode tok/s, no speculation | 39.1 | 32.8 | 25.3 | 25.2 |
| Speedup | 4.47× | 2.83× | 3.47× | 1.56× |
| Tokens per cycle | 6.19 | — | 6.45 | 1.76 |

**Long context with int8 KV (q8h)** — WHIRL server, one user, `--ctx-per-slot 262144`, q8h KV (int8 keys and values; queries and keys Hadamard-rotated first), the setting used for 256k tokens on the dense models (set with `WHIRL_KV=q8h` for the MoE model):

| WHIRL server, q8h KV, one user | Swift-1.5 27B MXFP4-A | Ornith-1.5-35B-A3B MXFP4 | Qwen3.8-27B UD-Q4_K_M (reference) |
|---|---|---|---|
| Prefill tok/s, 64k-token prompt | 2,162.0 | 5,671.3 | 1,289.9 |
| Prefill tok/s, 128k-token prompt | 1,533.0 | 3,673.6 | 1,032.2 |
| Prefill tok/s, 192k-token prompt | 1,212.2 | 2,770.7 | 876.6 |
| Prefill tok/s, 256k-token prompt | 969.8 | 2,135.2 | 742.2 |
| Decode tok/s (MTP+n-gram) after 128k | 62.5 | 214.0 | 58.3 |
| Decode tok/s (MTP+n-gram) after 256k | 35.1 | 115.6 | 33.5 |
| Long-document Q&A spot check (answer names the function asked about; not a formal needle test) | 128k ✓ / 256k ✓ | 128k ✓ / 256k ✓ | 128k ✓ / 256k ✓ |

Where WHIRL does **not** lead by much: plain decoding (no MTP) of the dense models is memory-bandwidth bound in both engines (1.13–1.14× llama.cpp); decode after a 16k-token context leads by only 1.17× (Swift-1.5) and 1.33× (Qwen3.8) on the dense models; prefill of Qwen3.8 Q4_K_M leads by less (1.51× at 96k and 1.44× at 256k tokens, against 2.28× and 1.74× for Swift-1.5 MXFP4); Ornith-1.5 prefill at 256k leads by 1.48×; and in the worst case for speculative decoding (new writing after 128k tokens, nothing to copy) it gains only 1.56×. Our R9700 is a USB4 eGPU; prefill and decode stay in VRAM, but restores and model loading cross the host link.

Full methodology and all numbers: [benchmarks.md](docs/benchmarks.md).

## <a id="dflash2"></a>DFlash2 draft (opt-in)

By default WHIRL drafts tokens with the model's own MTP head plus n-gram lookup. For the dense Qwen3.8-27B you can switch to a separate drafter, **DFlash2**, which proposes up to 7 tokens per cycle from the target's hidden states:

```powershell
.\whirl-server.exe C:\models\Qwen3.8-27B-UD-Q4_K_M.gguf --draft dflash --draft-model C:\models\dflash2-q8_0.gguf
```

`--draft dflash --draft-model PATH` also works for `whirl chat` and `whirl bench`; n-gram drafts stay on. The default is still `--draft mtp`. DFlash2 drafts are verified like n-gram drafts, so the output equals plain greedy decoding of the same mode: it changes speed only. Without a fitting drafter file, WHIRL warns and uses MTP.

**The drafter is not bundled.** Download it from Hugging Face: [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (Apache-2.0); we tested the Q8_0 GGUF. It takes about 1.9 GiB of VRAM, and the KV pool is about 12% smaller (4 slots, balance: 257,280 vs 292,352 tokens).

Whether it helps depends on the workload. Measured on the R9700, Qwen3.8-27B UD-Q4_K_M, balance, tok/s, DFlash2 + n-gram against the default MTP + n-gram:

| Workload | MTP + n-gram (default) | DFlash2 + n-gram | Change |
|---|---|---|---|
| **Helps** (single user, English-heavy or predictable output) | | | |
| Java API code, chat | 105.7 | 125.6 | +19% |
| SQL report, chat | 83.7 | 123.6 | +48% |
| Java API code, thinking on | 93.8 | 120.0 | +28% |
| Coding-agent session replay (91 steps, 12k to 65k tokens), mean | 83.4 | 86.9 | +4% |
| Same session, last step (65k-token prompt) | 66.2 | 74.4 | +12% |
| **Does not help** | | | |
| Chinese coding chat | 100.9 | 91.3 | -10% |
| 4 users at once, aggregate | 211.3 | 129.7 | -39% |
| Question after a 37k-token prompt | 54.6 | 39.1 | -28% |
| Question after a 108k-token prompt | 40.3 | 25.8 | -36% |
| New writing after 128k tokens | 38.4 | 31.7 | -17% |
| Editing a file (1,536 tokens, lots of copying) | 154.8 | 150.1 | -3% |

<!-- src: Java and SQL rows: audit/DFLASH-1.md section 7, "step 7" table (chat, balance, greedy, 512 tokens; java-api 105.7 / 125.6, sql-report 83.7 / 123.6, java-api (think) 93.8 / 120.0). Agent replay mean 83.4 -> 86.9 (+4%) and last step 66.2 -> 74.4 (+12%): audit/REL3.md DFlash2 table (DFLASH-1 step 8 measured 83.05 -> 85.73, +3%, last step 67.4 -> 73.4, +9%). Chinese chat 100.9 -> 91.3: audit/REL3.md. 4 users 211.3 -> 129.7, 37k 54.6 -> 39.1, 108k 40.3 -> 25.8, new writing after 128k 38.4 -> 31.7, edit_file bench 154.8 -> 150.1: audit/DFLASH-1.md section 7 "step 8" (first-round values for the 4-user row). VRAM +1.9 GiB net and KV pool 257,280 vs 292,352 tokens at -np 4 balance (-12%): audit/DFLASH-1.md section 7 step 3. Correlation -0.89 between Chinese share of the output and tokens per cycle: audit/DFLASH-2a.md table row 1. -->

Use it with a single user, short or medium context, and output that is easy to predict: API or SQL boilerplate, code in English, many similar edits. There we measured +19% to +48%. Leave it off with several users at once (-39%), with contexts of 37k tokens or more (-28% to -36%), and for mostly Chinese output (about -10% in our Chinese coding chat; the share of drafts accepted falls with the share of Chinese in the output, correlation -0.89 over our prompts). A task that mostly copies existing text, such as editing a file, showed no gain, because the n-gram drafts already cover it. Measured only on the R9700 with Qwen3.8-27B Q4_K_M; Swift-1.5, the MoE model and the Radeon 8060S have not been measured with DFlash2.

## <a id="ssd-cache"></a>SSD cache: what changed and how to clear it

The server keeps idle sessions in a RAM tier and, optionally, on an SSD tier, so a long prompt is restored instead of recomputed (also after a restart). The 0.1.x default wrote every session to the SSD about 2 seconds after it went idle. One user running a benchmark of many different long prompts saw about 400 GB written in 23 minutes. We checked the write path and found no hidden amplification (a new session is written once, at 1.004 to 1.012 times its file size); the volume was the old default writing every distinct prompt, 0.3 to 1.5 GiB each. 0.2.0 changes the defaults:

- **Default `--kv-ssd-mode shutdown`: no SSD writes while the server runs.** Sessions stay in the RAM tier. A normal stop (Ctrl+C) writes them, most recently used first, for up to 60 seconds, and the next start restores them. A crash or a kill loses what was not written yet.
- **`--kv-ssd-mode always`** is still there for those who want sessions on disk while the server runs. It writes a session only after it has been unchanged for 5 minutes (`--kv-ssd-delay SEC`, default 300), and only sessions that were reused at least once or continued by a later turn, so a stream of one-shot prompts writes about nothing.
- **At most 64 GiB on disk** (`--kv-ssd-gb`, default 64). The cap counts real allocated bytes, includes files left by other versions, models or modes (deleted first), then removes the least recently used sessions. It is enforced at start-up, not only on writes; half-written or corrupt cache files are deleted.
- **Files not used for 7 days are deleted** (at start-up and hourly). Restoring a session counts as use.
- `--kv-ssd-mode off` or `--kv-ssd-gb 0` writes no SSD files at all; the RAM tier still works.

Cache files written by other versions (0.1.x included) are never restored; they count against the cap and are deleted first.

**To delete the cache by hand:** stop the server and delete the `*.wkv` files in `%LOCALAPPDATA%\whirl\kvcache` (or the folder you gave with `--kv-ssd-dir`); the cache starts empty. Nothing else is stored there. The files contain your prompts as token ids and their KV data, so delete them if the PC is shared. Details: [usage.md](docs/usage.md#ssd-tier).

<!-- src: 400 GB / 23 min report, 1.004-1.012 write ratio, 0.3-1.5 GiB per prompt, 60 s flush, most-recently-used-first flush, 5-minute delay, seen-once filter, 64 GiB cap including foreign files, 7-day expiry refreshed by restores, start-up cleanup: audit/SSD-QUIET.md (+ SSD-QUIET-2) and CHANGELOG.md Unreleased "SSD tier: far fewer disk writes". Other versions' files never restored: docs/usage.md "Safety" paragraph; SSD-QUIET.md "valid file of another model / build / KV format". -->

## <a id="radeon-8060s"></a>Radeon 8060S (Ryzen AI Max+ 395), supported from 0.2.0

0.2.0 is the first release with official support for the Radeon 8060S (RDNA 3.5, gfx1151). Both GPU code sets are in the same zip. `whirl devices` lists the GPUs; the R9700 is the default when present, and `--device 8060s` (or `WHIRL_DEVICE=8060s`) selects the 8060S. All three numerics modes work, and the kernel tests, golden outputs and server gates (basic, concurrent requests, streaming, tier restore) pass on it. Not available on the 8060S yet: image input.

**RAM and SSD cache tiers are on by default.** The 8060S prefills slowly (about 390 tok/s at 8k and 240 tok/s at 128k for Qwen3.8-27B Q4_K_M), so restoring a session pays off even more than on the R9700. On Windows the BIOS splits memory into a VRAM carve-out (where the KV pool lives) and ordinary OS RAM, so the RAM tier is a separate pool of pinned OS memory and a restore is a real copy. Qwen3.8-27B UD-Q4_K_M, balance, time to first token when a session returns:

| Prompt | Re-prefill | From the RAM tier | From the SSD after a restart |
|---|---|---|---|
| 8k tokens | 20.9 s | 0.11 s | 0.21 s |
| 32k tokens | 91.5 s | 0.15 s | 0.40 s |
| 65k tokens | 214.1 s | 0.22 s | 0.76 s |
| 98k tokens | 374.6 s | 0.30 s | 0.78 s |
| 128k tokens | 536.7 s (about 9 minutes) | 0.34 s | 2.00 s |

<!-- src: audit/TIER-8S.md results table (single runs for 32k-128k; the 8k row was measured three times, noise a few percent; SSD column = file cache warm, flushed file cache 0.24-1.82 s; output identical in all five requests per length). -->

The tier does not change decode speed (19.7 vs 19.4 tok/s in a 300-token test). It takes its size out of OS RAM, so it is sized from the **OS-visible** RAM and not from the 128 GB on the box: `clamp(OS-visible RAM / 8, one maximum-context session, 16 GiB)`, at most 1/4 of the RAM free at start-up, and **off under 32 GB**. That is 8 GiB on a 64 GB machine; the start-up log states the size and the reason. `--kv-ram-mb N` sets it yourself, `--kv-ram-mb 0` turns both host tiers off, `--kv-ssd-gb 0` only the SSD tier. One 128k session needs about 5 GB, so an 8 GiB tier holds one such session at a time.

**Precise mode decodes about twice as fast as before.** The precise decode matmul has a new kernel (`gemvh`) that produces the same bits as the older paths. Qwen3.8-27B UD-Q4_K_M, `--precise`, tok/s, output identical in every pair:

| Context | Plain decode: before | now | MTP + n-gram: before | now |
|---|---|---|---|---|
| short (144 tokens) | 5.34 | 11.19 | 20.2 | 40.2 |
| 32k | 4.9 | 9.27 | 15.1 | 23.0 |

<!-- src: "before" = WHIRL_PREC_GEMVH=0 runs in audit/GEMVH-1151.md section 3 (short plain 5.34/5.34, short MTP 20.16/20.28, 32k plain 4.97/4.83, 32k MTP 15.41/14.78); "now" = audit/GEMVH-1151.md GEMVH-2 end-to-end table, "after" column (short plain 11.19/11.17, short MTP 40.23/39.91, 32k plain 9.27, 32k MTP 23.01; single runs at 32k, drift +-3-4%). The two columns come from two measurement sessions on the same 8060S. -->

The gain shrinks with context length: at 128k the second kernel step alone gave plain 6.44 vs 5.35 tok/s and MTP 10.16 vs 9.25 tok/s, because attention time at depth is unchanged. Balance and fast already used a faster decode path and are not affected by this kernel.

**Fast mode on the 8060S** uses 4-bit KV: Ornith-1.5-35B-A3B MXFP4 decodes at 58.5 tok/s at 128k (f16 KV: 32.7) and 67.0 tok/s at 64k (f16 KV: 46.4); the KL against f16 KV is 0.0137 (128k code prompt) and 0.0177 (64k). With several users, fast requests draft with the same caps as on the R9700, so a request usually gives the same text alone or among others; in fast mode this is best effort, not a guarantee.

<!-- src: 4-bit KV numbers and KL: CHANGELOG.md Unreleased "fast kvq4 on the Radeon 8060S". Concurrency: audit/CONC-8S.md "Limits". -->

**Against llama.cpp on the 8060S** (b11214 ROCm, same files and prompts, greedy):

| Radeon 8060S, 0.2.0: WHIRL vs llama.cpp | Qwen3.8-27B UD-Q4_K_M (dense) | Ornith-1.5-35B-A3B MXFP4 (MoE) |
|---|---|---|
| Prefill tok/s, 8k-token prompt | [8060S: TBD] | [8060S: TBD] |
| Prefill tok/s, 32k-token prompt | [8060S: TBD] | [8060S: TBD] |
| Prefill tok/s, 128k-token prompt | [8060S: TBD] | [8060S: TBD] |
| Decode tok/s, balance (default), Chinese coding chat | [8060S: TBD] | [8060S: TBD] |
| Decode tok/s after a long document | [8060S: TBD] | [8060S: TBD] |
| Decode tok/s, precise | [8060S: TBD] | [8060S: TBD] |
| Decode tok/s, fast | [8060S: TBD] | [8060S: TBD] |
| Test machine, power limit, llama.cpp flags | [8060S: TBD] | [8060S: TBD] |

<!-- src: to be filled from audit/REL3.md, section "8060S" (still being measured when this draft was written). The 0.1.x bring-up figures in docs/benchmarks.md section 14 predate the three modes, gemvh and the tier changes; do not reuse them. -->

Our 8060S numbers come from a thermally limited laptop; a desktop or mini PC with the same chip may differ.

## What's new in 0.2.0

- **The Radeon 8060S is supported**, with RAM and SSD cache tiers on by default and a faster precise decode ([above](#radeon-8060s)).
- **Three numerics modes**: precise, balance (default, the 0.1.x behaviour), fast ([above](#three-modes)). New: `--precise` (f16 KV, no fp8 or int8 activations, bit-identical speedups) and `--fast` (4-bit KV, relaxed MTP acceptance).
- **DFlash2 drafter, opt-in** (`--draft dflash --draft-model PATH`) for single-user, predictable output ([above](#dflash2)).
- **The SSD cache writes only at shutdown by default**, with a 64 GiB cap on disk and a 7-day expiry ([above](#ssd-cache)).
- **Faster prefill for dense models of any weight format** in balance and fast: the f16 DeltaNet and GEMM-output paths that only MXFP4 files used before now apply to Q4_K_M, Q5_K, Q6_K, IQ4_XS and others. Qwen3.8-27B UD-Q4_K_M on the R9700: prefill 8k 1,849 to 2,042 tok/s (+10.4%), 32k 1,649 to 1,812 (+9.9%), 128k 1,201 to 1,285 (+7.0%); decode unchanged.
- **Sampling with MTP stays exact at temperature > 0** (`specsample`): the text for a seed is the same with MTP on or off.
- **Fixes from an external code review**, outputs unchanged: the MTP draft window is restored on error paths; the prefill batch can no longer shrink below the scheduler's chunk size; a cache file that is shorter than recorded is deleted and a short read falls back to a cold prefill instead of restoring stale bytes; a failed restore copy is treated as failed; a failed SSD directory lock leaves the SSD tier off instead of running unlocked; models whose DeltaNet shape the kernels do not implement are refused at load.

<!-- src: CHANGELOG.md Unreleased: KG-1 numbers (prefill 1849 -> 2042 etc.), specsample, FIX-REVIEW2 items; audit/FIX-REVIEW2.md table rows 1-7. -->

Earlier releases, measured on the same machine with the same GGUF file and prompts; the greedy output of the two versions is bit-identical.

### v0.1.0 to v0.1.3

| Swift-1.5 27B MXFP4-A, R9700 | v0.1.0 | v0.1.3 | Change |
|---|---|---|---|
| Prefill tok/s, 32k-token prompt (f16 KV) | 2,605.3 | 2,907.4 | +11.6% |
| Prefill tok/s, 128k-token prompt (f16 KV) | 1,407.3 | 1,756.7 | +24.8% |
| Prefill tok/s, 128k-token prompt (q8h KV, server) | 1,352.7 | 1,533.0 | +13.3% |
| Prefill tok/s, 256k-token prompt (q8h KV, server) | 832.5 | 969.8 | +16.5% |
| Decode tok/s, Q&A over a 128k-token document (answer quotes the document a lot) | 63.7 | 87.6 | +37.5% |
| Decode tok/s, coding prompts | 110.7 | 120.7 | +9.1% |
| KV pool, default 4 slots (tokens) | 210,688 | 226,304 | +7.4% |
| 4 users, short prompts sent together: whole batch done (s, lower is better) | 9.57 | 8.62 | −9.9% |
| Decode tok/s, long-context coding-agent request (agent session 3, step 143, 127.9k tokens) | 53.7 | 75.2 | +40.0% |
| Decode tok/s, agent session 3 up to that request (mean of 144 requests) | 64.9 | 81.7 | +25.9% |

v0.1.2 for reference (same machine, same output): 53.5 tok/s for the 127.9k request, 65.0 tok/s for the session; the gain is v0.1.3's.

## Windows security prompts

`whirl.exe` and `whirl-server.exe` are not code-signed. Windows SmartScreen may show "Windows protected your PC" for a freshly downloaded copy: click **More info → Run anyway**, or run `Unblock-File` on the zip before extracting it (only for a zip from the official releases page, after checking its SHA-256). With **Smart App Control** set to *On*, Windows may block the programs without a "Run anyway" option. Details: [windows_security.md](docs/windows_security.md).

## Documentation

| | English | 繁體中文 |
|---|---|---|
| Quick start | [quickstart.md](docs/quickstart.md) | [quickstart_zh-TW.md](docs/quickstart_zh-TW.md) |
| Recipes: connect a client, long agent sessions, long context, several users, images, reading the log, troubleshooting | [recipes.md](docs/recipes.md) | [recipes_zh-TW.md](docs/recipes_zh-TW.md) |
| Usage reference (all commands, options, environment variables, exit codes) | [usage.md](docs/usage.md) | [usage.md](docs/guide/zh-TW/usage.md) |
| Server (API, sampling, tool calls, batching, decode floor, logs) | [server.md](docs/guide/en/server.md) | [server.md](docs/guide/zh-TW/server.md) |
| Benchmarks vs llama.cpp | [benchmarks.md](docs/benchmarks.md) | [benchmarks.md](docs/guide/zh-TW/benchmarks.md) |
| Windows security prompts | [windows_security.md](docs/windows_security.md) | [windows_security_zh-TW.md](docs/windows_security_zh-TW.md) |
| Building from source | [building.md](docs/building.md) | [building_zh-TW.md](docs/building_zh-TW.md) |
| Guide index | [index.md](docs/guide/en/index.md) | [index.md](docs/guide/zh-TW/index.md) |
| Architecture | [architecture.md](docs/guide/en/architecture.md) | [architecture.md](docs/guide/zh-TW/architecture.md) |
| Kernels | [kernels.md](docs/guide/en/kernels.md) | [kernels.md](docs/guide/zh-TW/kernels.md) |
| Speculative decoding (MTP + n-gram) | [speculative-decoding.md](docs/guide/en/speculative-decoding.md) | [speculative-decoding.md](docs/guide/zh-TW/speculative-decoding.md) |
| KV cache, prefix caching, tiers | [kv-and-caching.md](docs/guide/en/kv-and-caching.md) | [kv-and-caching.md](docs/guide/zh-TW/kv-and-caching.md) |
| Quantization and model files | [quantization.md](docs/guide/en/quantization.md) | [quantization.md](docs/guide/zh-TW/quantization.md) |
| Image input | [vision.md](docs/guide/en/vision.md) | [vision.md](docs/guide/zh-TW/vision.md) |
| HIP on Windows | [windows-hip.md](docs/guide/en/windows-hip.md) | [windows-hip.md](docs/guide/zh-TW/windows-hip.md) |
| Benchmarking methodology | [benchmarking.md](docs/guide/en/benchmarking.md) | [benchmarking.md](docs/guide/zh-TW/benchmarking.md) |
| **Pitfalls** (125, each with symptom → root cause → fix → check) | [pitfalls.md](docs/guide/en/pitfalls.md) | [pitfalls.md](docs/guide/zh-TW/pitfalls.md) |

## Development and AI assistance

WHIRL was designed, implemented, optimized and benchmarked with the participation of **Claude Opus 5.5**, an AI model by Anthropic, working under the direction of the project owner ([@tsaipifong](https://github.com/tsaipifong)). Anthropic is not affiliated with and does not endorse this project.

## License and acknowledgements

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party material is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), the origin of every source file in [PROVENANCE.md](PROVENANCE.md).

- [NInfer](https://github.com/Neroued/ninfer) — the design philosophy WHIRL follows (ideas only).
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) — used as the comparison baseline in our benchmarks and as the reference for GGUF and quantization-format compatibility; the few adapted pieces (tokenizer pre-split / BPE loop, lookup tables, image preprocessing) are MIT-licensed and listed in THIRD_PARTY_NOTICES.md.
- [gufo](https://github.com/gufo-org/gufo) and [r9700-stack](https://github.com/bkvargyas/r9700-stack) — measurement methodology and optimization ideas.
- Model authors: the Qwen team (Qwen3.8), [unsloth](https://huggingface.co/unsloth), [FreedomAISVR](https://huggingface.co/FreedomAISVR), [ornith-ai](https://huggingface.co/ornith-ai) (Ornith-1.5), and [ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) (Swift-1.5, the fine-tune our MXFP4 files are quantized from).
