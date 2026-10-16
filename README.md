**English** | [繁體中文](docs/README_zh-TW.md)

# WHIRL

**Windows HIP Inference for RDNA LLMs**

WHIRL is an open-source (Apache-2.0) LLM inference engine for AMD GPUs on Windows: native C++/HIP, every kernel included, no WSL, Docker or ROCm install. Download the zip, unzip, run; the AMD driver is all it needs. Version 0.2.0 runs on the **AMD Radeon AI PRO R9700** (RDNA 4, 32 GB) and the **Radeon 8060S** (Ryzen AI Max+ 395), and it is faster than llama.cpp on both, at every prompt length we measured, from 2k to 128k tokens.

## WHIRL vs llama.cpp at a glance

WHIRL 0.2.0 in its default mode (balance) against the official llama.cpp release **b11214** (ggml-org), shown separately for its **ROCm** and its **Vulkan** build. Same GGUF files, same prompts, greedy decoding, one user. The percentage in each llama.cpp cell is WHIRL's lead over that cell.

**Radeon AI PRO R9700** (gfx1201, 32 GB). Dense = Qwen3.8-27B UD-Q4_K_M, MoE = Ornith-1.5-35B-A3B MXFP4.

<!-- gen:glance.r97.en -->
| tok/s, higher is better | Dense 27B: WHIRL | llama.cpp ROCm | llama.cpp Vulkan | MoE 35B-A3B: WHIRL | llama.cpp ROCm | llama.cpp Vulkan |
|---|---|---|---|---|---|---|
| Prefill, 2k prompt | **1,782** | 1,140 (WHIRL +56%) | 804.4 (WHIRL +122%) | **8,624** | 4,014 (WHIRL +115%) | 3,551 (WHIRL +143%) |
| Prefill, 32k prompt | **1,719** | 1,089 (WHIRL +58%) | 761.2 (WHIRL +126%) | **6,651** | 3,872 (WHIRL +72%) | 3,395 (WHIRL +96%) |
| Prefill, 128k prompt | **1,233** | 759.2 (WHIRL +62%) | 553.9 (WHIRL +123%) | **3,758** | 2,307 (WHIRL +63%) | 1,922 (WHIRL +95%) |
| Decode, QA about a 2k-token document | **82.1** | 46.8 (WHIRL +75%) | 49.8 (WHIRL +65%) | **241.9** | 115.4 (WHIRL +110%) | 115.9 (WHIRL +109%) |
| Decode, QA about a 32k-token document | **75.8** | 40.7 (WHIRL +86%) | 42.7 (WHIRL +78%) | **203.2** | 101.4 (WHIRL +100%) | 100.6 (WHIRL +102%) |
| Decode, QA about a 128k-token document | **48.8** | 27.6 (WHIRL +77%) | 33.9 (WHIRL +44%) | **141.0** | 75.3 (WHIRL +87%) | 73.8 (WHIRL +91%) |
| Decode, Chinese coding chat (all new writing) | **103.1** | 61.4 (WHIRL +68%) | 61.2 (WHIRL +68%) | **250.1** | 118.9 (WHIRL +110%) | 115.1 (WHIRL +117%) |
<!-- /gen -->

**Radeon 8060S** (gfx1151, Ryzen AI Max+ 395). **Max power (80 W), ASUS ROG Flow Z13.** Same models.

<!-- gen:glance.s8.en -->
| tok/s, higher is better | Dense 27B: WHIRL | llama.cpp ROCm | llama.cpp Vulkan | MoE 35B-A3B: WHIRL | llama.cpp ROCm | llama.cpp Vulkan |
|---|---|---|---|---|---|---|
| Prefill, 2k prompt | **472.9** | 302.2 (WHIRL +56%) | 149.6 (WHIRL +216%) | **2,355** | 1,262 (WHIRL +87%) | 447.7¹ (WHIRL +426%) |
| Prefill, 32k prompt | **396.4** | 259.1 (WHIRL +53%) | 201.9 (WHIRL +96%) | **1,794** | 1,049¹ (WHIRL +71%) | 756.2¹ (WHIRL +137%) |
| Prefill, 128k prompt | **288.4** | 176.7¹ (WHIRL +63%) | 118.4¹ (WHIRL +144%) | **1,056** | 636.6¹ (WHIRL +66%) | 428.4¹ (WHIRL +146%) |
| Decode, QA about a 2k-token document | **27.0** | 24.7 (WHIRL +9%) | 25.5 (WHIRL +6%) | **101.8** | 70.2 (WHIRL +45%) | 70.7¹ (WHIRL +44%) |
| Decode, QA about a 32k-token document | **25.0** | 18.9 (WHIRL +32%) | 20.2 (WHIRL +24%) | **84.2** | 57.9¹ (WHIRL +46%) | 57.1¹ (WHIRL +48%) |
| Decode, QA about a 128k-token document | **20.2** | 13.7¹ (WHIRL +48%) | 14.1¹ (WHIRL +44%) | **55.5** | 38.0¹ (WHIRL +46%) | 37.3¹ (WHIRL +49%) |
| Decode, Chinese coding chat (all new writing) | **33.3** | 25.9 (WHIRL +29%) | 27.9 (WHIRL +20%) | **106.3** | 70.6¹ (WHIRL +51%) | 72.9¹ (WHIRL +46%) |
<!-- /gen -->

¹ single run (n = 1); read it as "about". Every other 8060S cell is the median of 2 or 3 runs; every R9700 length cell is a single run, measured the same day for all three engines.

How llama.cpp decodes in these tables: on the dense model with its fastest speculative mode (MTP + n-gram, `--spec-type draft-mtp,ngram-mod`; on the R9700 plain decoding was measured too and was slower in every dense cell). On the MoE model llama.cpp decodes **without** speculation, because its MTP was slower than plain decoding on this model in our measurements, while WHIRL uses its MTP drafts. The MoE decode rows therefore compare both engines **as shipped**, not like for like. Full tables, with WHIRL's precise and fast modes, are under [Performance](#performance).

What the decode rows measure: every length uses the same question about the document (explain one function quoted near the start of the document and write a corrected version of it, 256 tokens), so part of the reply can copy code from the prompt, which helps speculative drafts in every engine that uses them. Exact prompts and commands: [benchmarks.md](docs/benchmarks.md#prompts). The coding-chat row is the opposite, a hard case: everything in the reply is newly written.

**A real coding-agent session on the Radeon 8060S** (max power as above): a 16k-token skills / MCP system prompt, a tool call that reads a 3.6k-token source file, four user turns (explain the file, find the most complex function, rewrite it, review an edited version), full history resent on every call, dense Qwen3.8-27B Q4_K_M:

<!-- gen:s8.agent.en -->
| Session | WHIRL | llama.cpp ROCm | llama.cpp Vulkan |
|---|---|---|---|
| cold (20k-token system + tools prompt not cached yet) | **101.6 s** | 137.7 s | 243.5 s |
| second session (shares that prefix) | **45.7 s** | 52.5 s | 166.4 s |
<!-- /gen -->

WHIRL finishes the cold session in 26% less time than llama.cpp ROCm and 58% less than llama.cpp Vulkan (median of 2 rounds each). The code-rewrite turn decodes at about 36–44 tok/s on WHIRL; that turn is mostly copied code, so n-gram drafts help every engine there (llama.cpp ROCm: 42.2 tok/s in the cold session, 111.1 in the second session, which rewrites a different file), and it is not a WHIRL lead. This session was measured with the v0.2.0-rc7 pre-release build; the later release candidates changed only the 8060S prefill forward size, and 0.2.0 uses rc7's size for dense models.

**The same session on the R9700** (dense Qwen3.8-27B Q4_K_M, same fixture, median of 2 rounds, cold start = a fresh server per round):

<!-- gen:r97.agent.en -->
| Session | WHIRL | llama.cpp ROCm | llama.cpp Vulkan |
|---|---|---|---|
| cold (20k-token system + tools prompt not cached yet) | **26.5 s** | 48.5 s | 61.9 s |
| second session (shares that prefix) | **13.1 s** | 25.9 s | 29.0 s |
<!-- /gen -->

On the R9700 WHIRL finishes the cold session in 45% less time than llama.cpp ROCm and 57% less than llama.cpp Vulkan. The code-rewrite turn decodes at about 121–145 tok/s on WHIRL; as on the 8060S it is mostly copied code (llama.cpp ROCm: 120.5 tok/s in the cold session, 183.7 in the second), so it is not a WHIRL lead.

## Speed without changing the answer

WHIRL 0.2.0 has three numerics modes. You pick one per process (`--precise`, `--balance`, `--fast`); with no flag it runs **balance**.

| Mode | What it is | KV cache | Output guarantee |
|---|---|---|---|
| **precise** | bitwise lossless: nothing is quantized beyond the GGUF's own weights; f16 / f32 activations everywhere | f16 | the same bits with or without speculative drafts, alone or batched with other users, and from a prefix-cache hit or a restored session |
| **balance** (default) | near-lossless: int8 decode activations (the same q8_1 scheme llama.cpp decodes with); on the R9700 also f16-WMMA DeltaNet prefill and, for MXFP4 dense files, fp8 prefill activations | dense: int8 (`q8h`); MoE: f16 | MTP / n-gram drafts, batching and cache restores give the same tokens as plain greedy decoding in balance mode. Against precise: KL about 1e-4 to 1e-3 on the dense models; on the MoE model about 1.9× the model's own noise floor |
| **fast** (opt-in) | maximum speed within an accepted, gated loss: balance plus 4-bit KV and relaxed MTP acceptance (dense models) | 4-bit (`q4`) | lossy within published KL limits; the text can depend on the drafts and on other users |

What this means in practice:

- In **precise and balance**, every acceleration is checked to give the same tokens as plain greedy decoding of that mode: MTP and n-gram speculative decoding, several users batched together, prefix-cache hits on shared system prompts, a second question about the same long document, and sessions restored from the RAM or SSD tier. Standing tests enforce it (`whirl selftest`, kernel tests, server gates, golden outputs on both GPUs).
- **precise and balance never use 4-bit KV, 3-bit weights or fp8 attention.** fast is the only mode with 4-bit KV, and only when you ask for it.
- **precise** is the mode with the bitwise guarantee: bitwise lossless, output identical to plain greedy decoding. **balance** is near-lossless and the default. precise costs speed (R9700, Chinese coding chat, dense model: 77.8 vs 103.1 tok/s in balance).
- **Weights** are whatever GGUF you choose; WHIRL computes with them as stored. Continuing a conversation reuses the state computed for the previous turn, which is numerically equivalent but not bit-identical to re-processing the whole history as one prompt ([kv-and-caching.md](docs/guide/en/kv-and-caching.md)).

Every mode, item and measured KL: [usage.md, numerics modes](docs/usage.md#modes).

## Requirements

| | |
|---|---|
| **Download and run** | Download **`whirl-0.2.0-windows-x64.zip`** from [Releases](https://github.com/tsaipifong/whirl-llm/releases), unzip, run. **Only the AMD driver is needed**: no HIP SDK, no ROCm, no Visual C++ runtime |
| GPU | **AMD Radeon AI PRO R9700** (RDNA 4, gfx1201, 32 GB) or **Radeon 8060S** (RDNA 3.5, gfx1151, the Ryzen AI Max+ 395 iGPU); one GPU per process |
| OS | Windows 11, 64-bit |
| Driver | **AMD Software: Adrenalin Edition 26.8.1 (driver 32.0.31041.1004) or newer**, for both GPUs. Older drivers are untested |
| Memory / disk | the server pins host RAM for its RAM cache tier (R9700 host: about 1/4 of RAM, 16 GiB on a 64 GB PC; Radeon 8060S: 1/8 of the OS-visible RAM, 8 GiB on a 64 GB machine, off under 32 GB) and may use up to 64 GiB of SSD for the SSD tier, written only when the server stops; both adjustable or off (`--kv-ram-mb`, `--kv-ssd-gb`) |
| Advanced: build from source | for users who want to modify and tune WHIRL themselves: HIP SDK 7.2 + Visual Studio 2022 Build Tools (MSVC) + CMake + Ninja; see [building.md](docs/building.md) |

Support scope: WHIRL officially supports the R9700 and the Radeon 8060S and targets cards with **16 GB or more**. Other AMD GPUs are untested (the RX 9070 series has the R9700's chip with 16 GB; untested). Cards under 16 GB are not officially supported; you are welcome to fork the source and adapt it.

Our R9700 is connected as a USB4 / Thunderbolt eGPU; that works. A card in a direct PCIe slot may differ slightly where data crosses the host link (model loading, RAM / SSD tier restores), not in prefill or decode.

## Quick start

1. Download `whirl-0.2.0-windows-x64.zip` from [Releases](https://github.com/tsaipifong/whirl-llm/releases) and a [supported model](#supported-models).
2. Unzip and run, in PowerShell:

```powershell
Unblock-File .\whirl-0.2.0-windows-x64.zip
Expand-Archive .\whirl-0.2.0-windows-x64.zip -DestinationPath C:\whirl
cd C:\whirl\whirl-0.2.0-windows-x64
.\whirl.exe devices                                    # lists the R9700 (gfx1201) and/or the Radeon 8060S (gfx1151)
.\whirl.exe chat C:\models\Qwen3.8-27B-UD-Q4_K_M.gguf "Explain TCP slow start in two sentences." --max-tokens 400
.\whirl-server.exe C:\models\Qwen3.8-27B-UD-Q4_K_M.gguf --port 8080      # default mode: balance
```

The server speaks the OpenAI API at `http://127.0.0.1:8080/v1` (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`), so any OpenAI-compatible client works. Add `--precise` for bitwise-lossless output or `--fast` for maximum speed. Stop the server with Ctrl+C; reused sessions are written to the SSD tier first.

With both GPUs present the R9700 is the default; `--device 8060s` (or `WHIRL_DEVICE=8060s`) selects the 8060S. For long documents give a slot a longer context, for example `--ctx-per-slot 262144`; see [recipes.md](docs/recipes.md#long-context).

The first run with a model tunes the GPU kernels once and caches the result in `%LOCALAPPDATA%\whirl` (on the R9700 about 1.5 minutes for a 27B Q4_K_M file, a few seconds for MXFP4 files; longer on the 8060S). Full walkthrough: [quick start](docs/quickstart.md). Every option: [usage reference](docs/usage.md).

**Advanced: build from source** — for users who want to modify and tune WHIRL themselves: [building.md](docs/building.md).

## Supported models

WHIRL reads `general.architecture` from the GGUF and accepts only `qwen35` and `qwen35moe` (the Gated DeltaNet + gated attention hybrid, with an optional MTP head) and fine-tunes that keep that architecture. Other architectures (Llama, Gemma, Mistral, DeepSeek, `qwen3`, `qwen2`, …) are refused at load time with a clear message, and so are tensor types WHIRL has no kernels for (for example NVFP4); there is no slow generic fallback.

> **Our own MXFP4 quantizations, made for WHIRL:**
> - Dense 27B: [**Swift-1.5-Qwen3.8-27b-MXFP4-GGUF**](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF), variant A (`Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`). Swift-1.5 also thinks much less than the base model (44% fewer thinking tokens than Qwen3.8-27B MXFP4 on 4 Chinese coding prompts, a small sample; see the model card).
> - MoE 35B (~3B active): [**Ornith-1.5-35B-A3B-MXFP4-GGUF**](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF).

Tested GGUF files (loaded, greedy output checked, MTP output identical to plain greedy in precise and balance):

| Model file (Hugging Face) | Type | Arch | Quantization |
|---|---|---|---|
| [tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF) `…-A-outQ6_K.gguf` (recommended), `…-B-outQ8_0.gguf`, `…-C-outQ4_K.gguf` | dense 27B | qwen35 | MXFP4 (+ Q8_0 MTP / embeddings); quantized by us for WHIRL |
| [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `Qwen3.8-27B-UD-Q4_K_M.gguf` | dense 27B | qwen35 | UD-Q4_K_M (the dense model in the 0.2.0 benchmark) |
| [FreedomAISVR/Qwen3.8-27B-MXFP4-GGUF](https://huggingface.co/FreedomAISVR/Qwen3.8-27B-MXFP4-GGUF) `qwen3.8-27b-mxfp4.gguf` | dense 27B | qwen35 | MXFP4 |
| [`Ornith-1.5-35B-A3B-MXFP4.gguf`](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF) (quantized by us) | MoE 35B, ~3B active | qwen35moe | MXFP4 experts and dense (+ Q8_0 MTP / embeddings, Q6_K head) |
| [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) `Ornith-1.5-35B-Q4_K_M.gguf` | MoE 35B, ~3B active | qwen35moe | Q4_K_M |

Image input works with a Qwen3-VL-style mmproj (F16 / BF16) via `--mmproj`, on both GPUs. An opt-in DFlash2 drafter is available for the dense 27B model (`--draft dflash`, [below](#whats-new-in-020)). Other `qwen35` / `qwen35moe` GGUFs should load but are untested. Choosing and making GGUFs: [quantization.md](docs/guide/en/quantization.md).

## Performance

WHIRL 0.2.0 against the official llama.cpp b11214 (ggml-org Windows binaries), ROCm and Vulkan builds shown side by side. Same GGUF files, same prompts (a document-QA prompt with a unique first line, so no prefix-cache hits; 256 generated tokens), greedy decoding, thinking off, one user, tok/s from each server's own timings. Methods, flags and raw data: [benchmarks.md](docs/benchmarks.md).

### Radeon AI PRO R9700

All three WHIRL modes are shown; the leads are for balance, the default. Every cell is a single run, all engines measured on the same day (2026-10-16); treat differences under about 5% as noise. llama.cpp prefill shows the faster of its plain and speculative runs.

<!-- gen:r97.qwen.pp.en -->
#### Qwen3.8-27B UD-Q4_K_M (dense): Prefill tok/s

| Prompt | WHIRL balance | WHIRL precise | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | **1,782** | 1,615 | 1,768 | 1,140 | 804.4 | +56% | +122% |
| 8k | **1,872** | 1,684 | 1,843 | 1,210 | 834.6 | +55% | +124% |
| 32k | **1,719** | 1,565 | 1,701 | 1,089 | 761.2 | +58% | +126% |
| 64k | **1,516** | 1,404 | 1,507 | 947.0 | 675.7 | +60% | +124% |
| 96k | **1,359** | 1,272 | 1,352 | 843.9 | 608.8 | +61% | +123% |
| 128k | **1,233** | 1,163 | 1,226 | 759.2 | 553.9 | +62% | +123% |
<!-- /gen -->

<!-- gen:r97.qwen.tg.en -->
#### Qwen3.8-27B UD-Q4_K_M (dense): Decode tok/s, document QA (256-token answer: explain a function from the document and rewrite it)

| Prompt | WHIRL balance | WHIRL precise | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | **82.1** | 60.6 | 92.2 | 46.8 (MTP+n-gram) | 49.8 (MTP+n-gram) | +75% | +65% |
| 8k | **84.1** | 64.4 | 96.9 | 46.5 (MTP+n-gram) | 47.4 (MTP+n-gram) | +81% | +77% |
| 32k | **75.8** | 58.1 | 78.5 | 40.7 (MTP+n-gram) | 42.7 (MTP+n-gram) | +86% | +78% |
| 64k | **58.8** | 48.5 | 63.8 | 36.4 (MTP+n-gram) | 36.0 (MTP+n-gram) | +62% | +63% |
| 96k | **56.1** | 45.2 | 54.6 | 34.2 (MTP+n-gram) | 39.2 (MTP+n-gram) | +64% | +43% |
| 128k | **48.8** | 42.1 | 58.1 | 27.6 (MTP+n-gram) | 33.9 (MTP+n-gram) | +77% | +44% |
<!-- /gen -->

<!-- gen:r97.ornith.pp.en -->
#### Ornith-1.5-35B-A3B MXFP4 (MoE, ~3B active): Prefill tok/s

| Prompt | WHIRL balance | WHIRL precise | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | **8,624** | 7,502 | 11,477 | 4,014 | 3,551 | +115% | +143% |
| 8k | **7,733** | 6,756 | 10,794 | 4,487 | 3,912 | +72% | +98% |
| 32k | **6,651** | 5,948 | 8,435 | 3,872 | 3,395 | +72% | +96% |
| 64k | **5,333** | 4,892 | 6,440 | 3,173 | 2,735 | +68% | +95% |
| 96k | **4,409** | 4,098 | 5,158 | 2,667 | 2,280 | +65% | +93% |
| 128k | **3,758** | 3,528 | 4,283 | 2,307 | 1,922 | +63% | +95% |
<!-- /gen -->

<!-- gen:r97.ornith.tg.en -->
#### Ornith-1.5-35B-A3B MXFP4 (MoE, ~3B active): Decode tok/s, document QA (256-token answer: explain a function from the document and rewrite it)

| Prompt | WHIRL balance | WHIRL precise | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | **241.9** | 182.0 | 222.7 | 115.4 | 115.9 | +110% | +109% |
| 8k | **218.1** | 188.1 | 207.1 | 112.3 | 111.9 | +94% | +95% |
| 32k | **203.2** | 161.1 | 186.2 | 101.4 | 100.6 | +100% | +102% |
| 64k | **174.2** | 142.8 | 182.9 | 90.7 | 89.6 | +92% | +94% |
| 96k | **156.6** | 132.2 | 166.4 | 81.9 | 81.1 | +91% | +93% |
| 128k | **141.0** | 119.4 | 158.4 | 75.3 | 73.8 | +87% | +91% |
<!-- /gen -->

MoE decode: llama.cpp plain decoding (as shipped; its MTP was slower than plain in 11 of 12 MoE cells and 0.4 tok/s faster in the twelfth), WHIRL MTP.

**Chinese coding chat** (512 tokens; WHIRL median of 3 runs, llama.cpp first run, because its n-gram drafts learn a repeated prompt). This row is a hard case for speculative drafts: the Chinese question, the Chinese explanation and the English code are all newly written, and the prompt holds nothing for the reply to copy beyond a few option names (draft-based decoding is fastest when replies are predictable, such as copying or reciting known text; this row has none of that).

<!-- gen:r97.chat.en -->
| Model | WHIRL precise | WHIRL balance | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B UD-Q4_K_M (dense) | 77.8 | **103.1** | 119.3 | 61.4 (MTP+n-gram) | 61.2 (MTP+n-gram) | +68% | +68% |
| Ornith-1.5-35B-A3B MXFP4 (MoE, ~3B active) | 196.8 | **250.1** | 229.0 | 118.9 (plain) | 115.1 (plain) | +110% | +117% |
<!-- /gen -->

### <a id="radeon-8060s"></a>Radeon 8060S

**Max power (80 W), ASUS ROG Flow Z13** (Ryzen AI Max+ 395, 128 GB, 63.6 GB visible to Windows). WHIRL balance; median of 2–3 runs per cell unless marked.

<!-- gen:s8.qwen.pp.en -->
#### Qwen3.8-27B UD-Q4_K_M (dense): Prefill tok/s

| Prompt | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL vs ROCm | WHIRL vs Vulkan |
|---|---|---|---|---|---|
| 2k | **472.9** | 302.2 | 149.6 | +56% | +216% |
| 32k | **396.4** | 259.1 | 201.9 | +53% | +96% |
| 64k | **346.9** | 221.7 | 161.3 | +57% | +115% |
| 96k | **310.4** | 195.5 | 137.0 | +59% | +127% |
| 128k | **288.4** | 176.7¹ | 118.4¹ | +63% | +144% |
<!-- /gen -->

<!-- gen:s8.qwen.tg.en -->
#### Qwen3.8-27B UD-Q4_K_M (dense): Decode tok/s, document QA (256-token answer: explain a function from the document and rewrite it)

| Prompt | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL vs ROCm | WHIRL vs Vulkan |
|---|---|---|---|---|---|
| 2k | **27.0** | 24.7 | 25.5 | +9% | +6% |
| 32k | **25.0** | 18.9 | 20.2 | +32% | +24% |
| 64k | **22.6** | 17.9 | 18.2 | +27% | +24% |
| 96k | **21.4** | ~17.7² | 16.7 | ~+21%² | +28% |
| 128k | **20.2** | 13.7¹ | 14.1¹ | +48% | +44% |
<!-- /gen -->

<!-- gen:s8.ornith.pp.en -->
#### Ornith-1.5-35B-A3B MXFP4 (MoE, ~3B active): Prefill tok/s

| Prompt | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL vs ROCm | WHIRL vs Vulkan |
|---|---|---|---|---|---|
| 2k | **2,355** | 1,262 | 447.7¹ | +87% | +426% |
| 32k | **1,794** | 1,049¹ | 756.2¹ | +71% | +137% |
| 64k | **1,417** | 848.8 | 603.6¹ | +67% | +135% |
| 96k | **1,178** | 710.4 | 500.5¹ | +66% | +135% |
| 128k | **1,056** | 636.6¹ | 428.4¹ | +66% | +146% |
<!-- /gen -->

<!-- gen:s8.ornith.tg.en -->
#### Ornith-1.5-35B-A3B MXFP4 (MoE, ~3B active): Decode tok/s, document QA (256-token answer: explain a function from the document and rewrite it)

| Prompt | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL vs ROCm | WHIRL vs Vulkan |
|---|---|---|---|---|---|
| 2k | **101.8** | 70.2 | 70.7¹ | +45% | +44% |
| 32k | **84.2** | 57.9¹ | 57.1¹ | +46% | +48% |
| 64k | **72.9** | 49.0 | 48.4¹ | +49% | +51% |
| 96k | **62.0** | 42.7 | 42.2¹ | +45% | +47% |
| 128k | **55.5** | 38.0¹ | 37.3¹ | +46% | +49% |
<!-- /gen -->

¹ single run; read it as "about". ² unreliable: two runs that differ by 31% (20.1 and 15.3 tok/s; llama.cpp's MTP + n-gram acceptance varied), so this lead is not quoted anywhere else. Dense decode: llama.cpp MTP + n-gram, WHIRL MTP. MoE decode: llama.cpp plain (as shipped), WHIRL MTP.

**Chinese coding chat on the 8060S** (same prompt as on the R9700, the hard case described above; WHIRL measured with the v0.2.0-rc7 pre-release build, whose decode path is unchanged in 0.2.0):

<!-- gen:s8.chat.en -->
| Model | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL vs ROCm | WHIRL vs Vulkan |
|---|---|---|---|---|---|
| Qwen3.8-27B UD-Q4_K_M (dense) | **33.3** | 25.9 | 27.9 | +29% | +20% |
| Ornith-1.5-35B-A3B MXFP4 (MoE, ~3B active) | **106.3** | 70.6¹ | 72.9¹ | +51% | +46% |
<!-- /gen -->

### <a id="swift-mxfp4"></a>Swift-1.5 27B MXFP4 (our dense quant)

Our own MXFP4 quantization of the dense 27B (`Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`, variant A), same prompts and flags as above, measured with the 0.2.0 release candidate later the same day. On the R9700 WHIRL's MXFP4 prefill path is much faster than its Q4_K_M path, so the prefill lead is larger than on Q4_K_M: +115% to +166% over llama.cpp ROCm and +194% to +296% over Vulkan; decode +50% to +75% / +43% to +78%. On the 8060S: prefill +56% to +62% / +87% to +207%, decode +27% to +48% / +24% to +107%.

**R9700** (single run per cell; llama.cpp prefill shows the faster of its plain and MTP + n-gram runs):

<!-- gen:r97.swift.pp.en -->
#### Swift-1.5-Qwen3.8-27B MXFP4-A (dense 27B): Prefill tok/s

| Prompt | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL vs ROCm | WHIRL vs Vulkan |
|---|---|---|---|---|---|
| 2k | **3,263** | 1,227 | 823.2 | +166% | +296% |
| 32k | **2,822** | 1,174 | 815.4 | +140% | +246% |
| 64k | **2,334** | 1,010 | 718.8 | +131% | +225% |
| 96k | **1,974** | 892.5 | 645.3 | +121% | +206% |
| 128k | **1,716** | 798.5 | 584.0 | +115% | +194% |
<!-- /gen -->

<!-- gen:r97.swift.tg.en -->
#### Swift-1.5-Qwen3.8-27B MXFP4-A (dense 27B): Decode tok/s, document QA (256-token answer: explain a function from the document and rewrite it)

| Prompt | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL vs ROCm | WHIRL vs Vulkan |
|---|---|---|---|---|---|
| 2k | **87.3** | 50.9 (MTP+n-gram) | 50.2 (MTP+n-gram) | +72% | +74% |
| 32k | **76.9** | 44.0 (MTP+n-gram) | 43.1 (MTP+n-gram) | +75% | +78% |
| 64k | **64.0** | 38.6 (MTP+n-gram) | 39.5 (MTP+n-gram) | +66% | +62% |
| 96k | **60.2** | 36.1 (MTP+n-gram) | 40.7 (MTP+n-gram) | +67% | +48% |
| 128k | **49.8** | 33.2 (MTP+n-gram) | 34.8 (MTP+n-gram) | +50% | +43% |
<!-- /gen -->

**Radeon 8060S**, max power (80 W), ASUS ROG Flow Z13 (each cell median of 2 or 3 runs; llama.cpp MTP + n-gram):

<!-- gen:s8.swift.pp.en -->
#### Swift-1.5-Qwen3.8-27B MXFP4-A (dense 27B): Prefill tok/s

| Prompt | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL vs ROCm | WHIRL vs Vulkan |
|---|---|---|---|---|---|
| 2k | **500.9** | 310.1 | 162.9 | +62% | +207% |
| 32k | **414.9** | 266.2 | 221.6 | +56% | +87% |
| 64k | **368.1** | 230.9 | 177.3 | +59% | +108% |
| 96k | **324.2** | 203.3 | 149.0 | +59% | +118% |
| 128k | **290.6** | 180.8 | 127.8 | +61% | +128% |
<!-- /gen -->

<!-- gen:s8.swift.tg.en -->
#### Swift-1.5-Qwen3.8-27B MXFP4-A (dense 27B): Decode tok/s, document QA (256-token answer: explain a function from the document and rewrite it)

| Prompt | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL vs ROCm | WHIRL vs Vulkan |
|---|---|---|---|---|---|
| 2k | **31.0** | 24.5 | 25.1 | +27% | +24% |
| 32k | **29.0** | 22.7 | 17.0 | +28% | +71% |
| 64k | **24.9** | 18.4 | 18.0 | +35% | +38% |
| 96k | **23.6** | 17.9 | 16.6 | +32% | +42% |
| 128k | **24.0** | 16.2 | 11.6 | +48% | +107% |
<!-- /gen -->

**Chinese coding chat** (512 tokens, the hard case for drafts: all newly written, nothing to copy):

<!-- gen:swift.chat.en -->
| GPU | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL vs ROCm | WHIRL vs Vulkan |
|---|---|---|---|---|---|
| Radeon AI PRO R9700 | **110.7** | 57.8 (MTP+n-gram) | 57.6 (MTP+n-gram) | +92% | +92% |
| Radeon 8060S (max power) | **33.6** | 27.6 (MTP+n-gram) | 28.3 (MTP+n-gram) | +22% | +19% |
<!-- /gen -->

### Where WHIRL does not lead by much

- **Dense decode on the 8060S, document QA at 2k tokens**: +9% over llama.cpp ROCm and +6% over Vulkan at 2k, within run-to-run noise. Dense decode on the 8060S is memory-bound (about 256 GB/s for 15+ GB of weights), so the lead comes from speculative decoding and grows with context (+24% to +48% from 32k on).
- **Plain decoding without speculation** is memory-bandwidth bound in every engine; most of WHIRL's decode lead comes from MTP + n-gram speculative decoding.
- **fast is not always faster than balance**: on the MoE model at short and medium context (R9700), balance decodes faster than fast; fast helps mostly in prefill and in dense decode.
- **precise costs speed**: 14–26% of decode and 6–13% of prefill against balance on the R9700 (tables above).

## What's new in 0.2.0

- **Radeon 8060S (Ryzen AI Max+ 395) support**, tuned: faster than llama.cpp ROCm and Vulkan at every measured prompt length on both models (tables above). Image input works on the 8060S, and the RAM and SSD cache tiers are on by default there (on Windows the BIOS VRAM carve-out and the OS RAM are separate pools, so a 128k-token session restores in 0.34 s from RAM and 2 s from SSD instead of 537 s of prefill).
- **Three numerics modes**: precise (bitwise lossless), balance (default, near-lossless), fast (opt-in, gated loss). See [above](#speed-without-changing-the-answer).
- **DFlash2 drafts, opt-in** (`--draft dflash --draft-model Qwen3.8-27B-DFlash2-Q8_0.gguf`, drafter from Hugging Face `incoai/Qwen3.8-27B-DFlash2-GGUF`, not bundled): for one user with short or medium context and predictable output, +19% to +48% over MTP on the R9700; slower with 4 users (about −39%), after 37k+ tokens of context and on mostly-Chinese output (about −10%); costs about 1.9 GiB of VRAM. Greedy output equals plain greedy decoding of the same mode.
- **SSD cache writes far less**: by default it writes only reused sessions, when the server stops (or after about 5 minutes idle, or when RAM must evict them); a one-shot prompt is never written. The cache never exceeds `--kv-ssd-gb` (default 64 GiB) on disk, and files unused for 7 days are deleted. (Before, a benchmark of many different long prompts could write hundreds of GB.)
- **A second question about the same long document reuses it**: the server keeps a tail checkpoint, so it prefills only the new question (about 0.3k–1.3k tokens) instead of the whole document.
- **CLI and server give bit-identical output** for the same prompt (same prefill schedule).

Full list: [CHANGELOG.md](CHANGELOG.md).

## Windows security prompts

`whirl.exe` and `whirl-server.exe` are not code-signed. Windows SmartScreen may show "Windows protected your PC" for a freshly downloaded copy: click **More info → Run anyway**, or run `Unblock-File` on the zip before extracting it (only for a zip from the official releases page, after checking its SHA-256). With **Smart App Control** set to *On*, Windows may block the programs without a "Run anyway" option. Details: [windows_security.md](docs/windows_security.md).

## Documentation

| | English | 繁體中文 |
|---|---|---|
| Quick start | [quickstart.md](docs/quickstart.md) | [quickstart_zh-TW.md](docs/quickstart_zh-TW.md) |
| Recipes: connect a client, long agent sessions, long context, several users, images, reading the log, troubleshooting | [recipes.md](docs/recipes.md) | [recipes_zh-TW.md](docs/recipes_zh-TW.md) |
| Usage reference (all commands, options, numerics modes, environment variables, exit codes) | [usage.md](docs/usage.md) | [usage.md](docs/guide/zh-TW/usage.md) |
| Server (API, sampling, tool calls, batching, decode floor, logs) | [server.md](docs/guide/en/server.md) | [server.md](docs/guide/zh-TW/server.md) |
| Benchmarks vs llama.cpp | [benchmarks.md](docs/benchmarks.md) | [benchmarks.md](docs/guide/zh-TW/benchmarks.md) |
| Windows security prompts | [windows_security.md](docs/windows_security.md) | [windows_security_zh-TW.md](docs/windows_security_zh-TW.md) |
| Advanced: building from source | [building.md](docs/building.md) | [building_zh-TW.md](docs/building_zh-TW.md) |
| Guide index | [index.md](docs/guide/en/index.md) | [index.md](docs/guide/zh-TW/index.md) |
| Architecture | [architecture.md](docs/guide/en/architecture.md) | [architecture.md](docs/guide/zh-TW/architecture.md) |
| Kernels | [kernels.md](docs/guide/en/kernels.md) | [kernels.md](docs/guide/zh-TW/kernels.md) |
| Speculative decoding (MTP + n-gram, DFlash2) | [speculative-decoding.md](docs/guide/en/speculative-decoding.md) | [speculative-decoding.md](docs/guide/zh-TW/speculative-decoding.md) |
| KV cache, prefix caching, tiers | [kv-and-caching.md](docs/guide/en/kv-and-caching.md) | [kv-and-caching.md](docs/guide/zh-TW/kv-and-caching.md) |
| Quantization and model files | [quantization.md](docs/guide/en/quantization.md) | [quantization.md](docs/guide/zh-TW/quantization.md) |
| Image input | [vision.md](docs/guide/en/vision.md) | [vision.md](docs/guide/zh-TW/vision.md) |
| HIP on Windows | [windows-hip.md](docs/guide/en/windows-hip.md) | [windows-hip.md](docs/guide/zh-TW/windows-hip.md) |
| Benchmarking methodology | [benchmarking.md](docs/guide/en/benchmarking.md) | [benchmarking.md](docs/guide/zh-TW/benchmarking.md) |
| **Pitfalls** (each with symptom → root cause → fix → check) | [pitfalls.md](docs/guide/en/pitfalls.md) | [pitfalls.md](docs/guide/zh-TW/pitfalls.md) |

## Design principles

1. **Built for AMD GPUs on Windows.** A native Windows C++/HIP inference engine with hand-written, hand-tuned kernels for RDNA. No WSL, no Docker, no Linux VM: install the AMD driver and run.
2. **Fully open, kernels included.** Everything from host-side scheduling to every GPU kernel is open source under Apache-2.0, with no closed binary components. Bug reports, benchmark results and code contributions are all welcome.
3. **Accuracy before speed.** In the precise and balance modes every speedup (speculative decoding, batching, prefix caching, cache restore) gives the same tokens as plain greedy decoding of that mode, enforced by standing tests. Anything lossier is opt-in (`--fast`) and gated by KL and accuracy limits.
4. **Few models, pushed to the limit.** WHIRL targets specific models on specific hardware and pushes them close to the hardware limit, adding architectures one at a time. The architectures supported today are `qwen35` (dense) and `qwen35moe` (mixture of experts).
5. **Transparent, verifiable numbers.** Every figure is measured, with methods, commands and raw data published, including where WHIRL does not lead by much.
6. **Designed for real workflows.** Coding agents and long contexts come first: fast long-prompt processing, a responsive main conversation while others share the GPU, and a VRAM → RAM → SSD cache that survives server restarts.

WHIRL follows the design philosophy of [NInfer](https://github.com/Neroued/ninfer) (Apache-2.0) — build for selected models and selected GPUs, as close to the hardware limit as possible; [zynfer](https://github.com/thanos/zynfer) carried the same idea to RDNA 4.

## Development and AI assistance

WHIRL was designed, implemented, optimized and benchmarked with the participation of **Claude Opus 5.5**, an AI model by Anthropic, working under the direction of the project owner ([@tsaipifong](https://github.com/tsaipifong)). Anthropic is not affiliated with and does not endorse this project.

## License and acknowledgements

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party material is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), the origin of every source file in [PROVENANCE.md](PROVENANCE.md).

- [NInfer](https://github.com/Neroued/ninfer) — the design philosophy WHIRL follows (ideas only).
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) — used as the comparison baseline in our benchmarks and as the reference for GGUF and quantization-format compatibility; the few adapted pieces (tokenizer pre-split / BPE loop, lookup tables, image preprocessing) are MIT-licensed and listed in THIRD_PARTY_NOTICES.md.
- [gufo](https://github.com/gufo-org/gufo) and [r9700-stack](https://github.com/bkvargyas/r9700-stack) — measurement methodology and optimization ideas.
- Model authors: the Qwen team (Qwen3.8), [unsloth](https://huggingface.co/unsloth), [FreedomAISVR](https://huggingface.co/FreedomAISVR), [ornith-ai](https://huggingface.co/ornith-ai) (Ornith-1.5), and [ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) (Swift-1.5, the fine-tune our MXFP4 files are quantized from).
