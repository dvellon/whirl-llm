**English** | [繁體中文](../zh-TW/vision.md)

# Image input (vision)

> **Status.** Image input is in WHIRL 0.1.0 for gfx1201 (`whirl chat --mmproj … --image …`, server
> `--mmproj` with OpenAI `image_url` parts). On our test images the C++ encoder's embeddings are
> bit-identical to the research build's. Since 0.2.0 also on gfx1151 (Radeon 8060S), see
> [§7](#gfx1151). Options:
> [usage.md](../../usage.md#serve).

**Who this helps:** anyone adding a vision encoder to a memory-constrained text server — how to add
image input without taking VRAM away from text-only users — and anyone debugging outputs that
depend on what the server did *before* the current request.

## 1. Goal and constraint

The default 27B server leaves the KV pool only 2,816 tokens above the 128k + 64k floor
([kv-and-caching.md](kv-and-caching.md#floor)). A vision encoder that stays resident would push the
default KV format from q8v to the slower q8h for *every* user, including those who never send an
image. So the requirement was: **loading an mmproj must change neither the KV pool size nor
resident VRAM.** Measured on the candidate: pool with `--mmproj` 199,936 tokens (unchanged), VRAM
change at mmproj load 0.0 MiB.

## 2. Design

| Part | Design |
|---|---|
| Projector weights | loaded into **pinned host RAM** at startup; nothing of the encoder is in VRAM until an image arrives |
| Encoder lifetime | kernels, resident weights and a copy stream are created **on demand** with the first image and **released after an idle timeout** (`--vis-idle-s`, default 60 s); VRAM returns to the text-only footprint |
| Weight mode | `--vis-mode auto` (default): weights resident if that much VRAM is free, otherwise **streamed layer by layer per image**; `resident` / `stream` force a mode. The default 27B configuration leaves 0.76 GiB free, so weights stream: ~260 ms of H2D per image on this USB4 eGPU (environment-dependent) |
| Activations | borrow the prefill scratch buffers (no new allocation); the log reports how much one 4096-token image needs |
| Preprocessing | resize/normalize bit-identical to PIL's bicubic path |
| Embedding cache | projected embeddings cached in host memory by **content hash** (LRU, `--vis-cache-mb`, default 1024 MiB): the same image in a later request is not encoded again |
| Prompt integration | image placeholder positions in the prompt receive the image embedding rows (copied into the embedding output before the first layer) |
| Positions | **multi-section RoPE (M-RoPE):** each row gets a multi-component position; the rotary dimensions are split into sections (from the GGUF's RoPE section metadata), each rotated by one position component. A dedicated `attn_prep` variant applies it; per-row positions are uploaded for the forward. Text after an image keeps using the multi-section positions |
| Prefix cache | images are identified by content hash, so a conversation that repeats an image reuses its KV (one test: 238 of 238 prompt tokens cached); a shared `prefix` checkpoint is saved at the end of an image so later turns resume after it |
| API | OpenAI `image_url` content parts with `data:` URLs; several images per request |
| Unsupported GPU | a code object without the multi-section RoPE kernel refuses `--mmproj` at startup (`MropeUnsupported`) |

**Encoder speed.** A second attention version for the encoder (32 queries per wave, 64-key softmax
blocks, lazy rescale, Vᵀ in global memory, padded LDS) took a warm 1080p encode from 422 to 255 ms.

## 3. Numerics

| Comparison | Result |
|---|---|
| WHIRL encoder vs an f32 numpy reference, 4 images | relative L2 0.12–0.30%, minimum cosine ≥ 0.9995 |
| llama.cpp encoder (CPU and 8060S GPU) vs the same reference | relative L2 1.6–12%, minimum cosine 0.89–0.998 |
| Language model with image, vs `llama-mtmd-cli` b11214 on identical prompt tokens (Q4_K_M) | shape description identical for the first 524 characters (one word differs), code transcription exact in both |
| MTP greedy vs plain greedy on an image prompt | identical |

BF16 projectors (as in the MoE model's mmproj) are supported; a MoE smoke test with images passed.

## 4. Bugs found

### <a id="vis-pad"></a>4.1 Uninitialized padding made embeddings depend on server history

**Symptom.** Image embeddings computed inside the server or CLI differed from the standalone encoder
tool, and differed between runs depending on what the server had processed earlier. The standalone
tool was always clean.

**Root cause.** The encoder's Q/K preparation kernel pads the head dimension; it left padding lanes
76–79 unwritten. Attention reads the full padded width, so whatever was in those bytes entered the
dot products. The standalone tool allocated a fresh (clean) arena; inside the server, activations
borrow the **prefill scratch**, which still held data from earlier requests.

**Fix.** Write the padding (zeros) in the prep kernel.

**How we detect it now.** Server embeddings must equal standalone embeddings bit for bit after
arbitrary prior traffic. General rule: when buffers are borrowed or reused, padded lanes must be
written explicitly, and tests must run with *dirty* buffers — a fresh allocation hides this class of
bug.

### 4.2 MTP acceptance dropped after images

Text generated on a slot after an image prompt (resuming from the image's shared checkpoint) showed
lower MTP acceptance than on a fresh slot. After the padding fix the candidate build measured normal
acceptance again (47.2%, equal to a fresh slot). TODO: confirm whether the padding bug was the whole
cause.

### 4.3 Not a vision bug: F32 alpha/beta

The MTP mismatch caused by F32 `ssm_alpha` / `ssm_beta` in one quantization is a separate,
model-file issue ([quantization.md](quantization.md#f32-alpha-beta)); it was found during the MXFP4
quantization work, not in vision.

### 4.4 A gate bug, again

The vision gate's MXFP4 reference servers auto-selected f16 KV (more free VRAM without an mmproj
attached) while the tested server selected q8v, so every comparison "failed". Same lesson as the
system-prompt gate: pin the KV format in gates that compare servers.

## 5. Gate

`vis_gate` (dense and MXFP4): pool size equal with and without `--mmproj`; VRAM change at mmproj load
0.0 MiB; image prompts equal cold references; checkpoint reuse after an image (2,045 tokens in one
case); restore from SSD after a restart; idle release returns VRAM exactly (delta 0.0 MiB; the 8060S
limits are in §7). On the
candidate build the full chain passed: last-token logits bit-identical to the installed build for
text prompts, both server suites 50/50, segmented prefill, concurrency, pool, multi-turn, tier,
system-checkpoint and restore gates, vision gate dense + MXFP4.

## 6. Remaining work

- Done: the encoder, the scheduler integration and the gate are in the C++ engine (`src/vision/`,
  `kernels/vision/`, `whirl-server-gate --suite vis`).
- Confirm the cause of the MTP acceptance drop (§4.2).
- Done (0.2.0): gfx1151 multi-section RoPE kernels and encoder kernels (§7).
- gfx1151: encoder speed (§7: attention and GEMM run at a fraction of the WMMA peak).

## <a id="gfx1151"></a>7. Radeon 8060S (gfx1151)

Since 0.2.0 image input also runs on the 8060S (`--device 8060s`, same flags). What was missing:
the gfx1151 kernel set had no multi-section RoPE attention prep (`attn_prep_m*`, `set_rpos`), so the
server refused `--mmproj`, and the encoder code object existed for gfx1201 only. The encoder is now
one source (`kernels/vision/vision.hip`) built per architecture. Only the two WMMA kernels differ:
on gfx11 a lane holds all 16 k values of an A / B fragment (gfx12: 8) and accumulator element v is row
`2v + lane / 16` (gfx12: `v + 8 * (lane / 16)`); attention runs 8 waves x 16 queries per block (32
queries per wave spilled on gfx11). The arithmetic is unchanged: f16 inputs, f32 accumulation, no
fp8. The R9700 code object disassembles identically and its embeddings are bit-identical to before.

| Check (Qwen3.8-27B mmproj F16) | Result |
|---|---|
| `whirl-kernel-test --only vision` (all 8 encoder kernels vs CPU references), `attn_prep_m*` / `set_rpos` checks | pass on both GPUs |
| 8060S embeddings vs f32 numpy reference, 4 images | relative L2 0.16–0.30%, minimum cosine ≥ 0.9995 (R9700: 0.16–0.31%) |
| 8060S vs R9700 embeddings, 7 images (220–2,040 tokens) | relative L2 0.09–0.23%, minimum cosine ≥ 0.9995 |
| Server, `--precise`, temperature 0, 8 image requests (one or two images, English / Chinese questions, up to 1080p) | answers text-identical on both GPUs; same image again fully cached on both |

Encode time on the 8060S (`whirl-vis vis-encode`, weights resident, third run):

| Image (tokens) | 8060S | R9700 |
|---|---|---|
| 640x352 (220) | 54 ms | 17 ms |
| 512x512 (256) | 59 ms | 18 ms |
| 1024x672 (672) | 219 ms | 52 ms |
| 1280x736 (920) | 314 ms | 78 ms |
| 1024x1024 (1,024) | 332 ms | 92 ms |
| 1920x1088 (2,040) | 934 ms (attention 532, GEMM 303) | 253 ms |

`whirl-server-gate --suite vis` passes on the 8060S and the R9700. The VRAM checks use a counter per
device class (`include/whirl/device_mem.h`, measurements in VIS-8S-2): on unified memory
`hipMemGetInfo` drops by 1,775 MiB at mmproj load, twice the 887 MiB pinned mmproj copy, although
those pages are system RAM (GPU "Shared Usage") and not the BIOS carve-out. On the 8060S the server
therefore reports the change of its carve-out (GPU "Dedicated Usage", the Task Manager number): 1.9
MiB, which is the GPU page tables of the pinned copy (8 bytes per 4 KiB page); the limit is 2 MiB plus
that. The idle release returns to within 8.8 MiB of the level before the first image: the HIP runtime
grows its own memory once at the first long image prompt with the prefix cache on (8.3–8.8 MiB, flat
over 20 load / unload cycles, no leak per cycle). The gate allows 16 MiB there on an integrated GPU
(8 MiB on a discrete one; the R9700 measures 0.0) and checks on both that two more load / unload cycles
do not grow it (at most 2 MiB).

Streamed weights cost about the same on the 8060S (949 ms for 1080p): host memory and the GPU share
the same DRAM. The encoder reaches roughly 10–17 TFLOPS on the 8060S, a fraction of its WMMA peak;
tuning is open (§6).

#### Bug found during the port: a cross-lane read of inactive lanes

The first gfx11 attention returned 0 for head dim 71 only. Its epilogue exchanged accumulator values
between the two half-waves (`permlanex16`) before the per-lane `d < 72` / `q < N` checks, but the
compiler placed the exchange after them; for dims 72–79 the partner lanes were inactive, and
`permlanex16` reads an inactive lane as 0. Fix: per-element stores without a cross-lane step in
epilogues. The kernel test caught it (exactly one dim per head wrong).
