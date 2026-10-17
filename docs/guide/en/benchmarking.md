**English** | [繁體中文](../zh-TW/benchmarking.md)

# Benchmarking methodology and current numbers

> **Status.** This page describes the methodology. The published numbers of the current release
> (Swift-1.5 27B MXFP4 and Ornith-1.5-35B-A3B MXFP4 against the official llama.cpp builds) are in
> [benchmarks.md](../../benchmarks.md).

**Who this helps:** anyone comparing inference engines on AMD GPUs, or measuring small (1–5%)
kernel improvements without fooling themselves; laptop/APU users whose numbers drift with heat.

## 1. Principles

1. **Measure against measured ceilings**, not spec sheets ([kernels.md](kernels.md#ceilings)).
2. **One model process per GPU**, enforced by a named mutex in the binary and a file lock in every
   launcher ([windows-hip.md](windows-hip.md#wddm-demote)). Two incidents of two 15 GiB processes on
   one card invalidated hours of numbers.
3. **Dedicated VRAM only.** Every run is monitored; a run whose process shows more than 256 MiB of
   Shared Usage (baseline ~89 MiB; declared pinned host buffers subtracted) is discarded and rerun.
4. **Interleave A and B, report min–max.** Never compare a new build against an old number.
5. **Report percentages, and keep small wins.** Once the core gets faster, a 2–3% gain (or loss)
   elsewhere is magnified; ideas rejected as noise are re-measured later. (The Q4_K copy of the MTP
   block was rejected at first and adopted after later speedups made its gain measurable.)
6. **Correctness gates before speed.** A faster build that fails any gate is not measured further.

## 2. Interleaved A/B

- Old and new binaries (or the same binary with a setting toggled) alternate run by run. Round 0
  runs prompts in order, round 1 in reverse. Usually 2 rounds; 3+ when ranges overlap.
- Report every result as **min–max over rounds** (or mean with min–max), with the number of rounds.
- Record each output's hash. With greedy decoding the same engine/mode/prompt gives identical output
  across rounds (190/190 in one unified run), so min–max is real run-to-run variation, not content.
- **If the outputs differ between A and B, do not compare MTP tok/s directly.** Speculative speed
  depends on content. Example: a GEMV change altered an English summary's text; the new text was
  harder to predict (fewer tokens per cycle) and the prompt got "slower" while each cycle was
  actually faster. Compare ms per cycle, multi-prompt
  means, and acceptance separately.
- **Noise floors we measured:** plain decode ms/token on the R9700 is stable to ~0.2% across
  processes; the MoE model showed a ±3.5% bimodal shift between time slots (the same binary
  interleaved agrees, but a different hour differs); the automatic draft policy adds ±2–4% per run.
  Machine drift between sessions can look like a 2–6% gain or loss; only interleaved runs count.
- Disable profiling for A/B runs (per-op events halve decode speed; even three events per cycle cost
  ~2.5% on the MoE model).

## <a id="unified"></a>3. The unified protocol (WHIRL vs llama.cpp)

Both engines are measured through their own OpenAI-compatible servers by **one client script**, one
model process on the GPU at a time, the GPU lock held throughout.

| Setting | Value |
|---|---|
| llama.cpp | b11214 ROCm `llama-server`: `-ngl 99 -fa on -c 131072 -np 1 --jinja --load-mode none`, `HIP_VISIBLE_DEVICES=1`, `GGML_CUDA_NO_PINNED=1` (otherwise ~1 GB of pinned host buffers appear as shared GPU memory) |
| llama.cpp MTP | `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 0 --spec-draft-p-min 0.3` (no automatic n-gram) |
| WHIRL | `whirl serve MODEL --port 1234`, default settings (4 slots, automatic context and KV format) |
| Prompts | 7 mixed Chinese/English coding prompts (think off and on, `max_tokens` 800) + 5 file-editing prompts (think off, `max_tokens` 2500, all end with `stop`) = 19 per mode |
| Sampling | greedy-equivalent `top_k = 1` (temperature 1.0, top_p 1.0, min_p 0, seed 42 — **not** temperature 0), `cache_prompt = false` |
| Modes | plain (no speculation), MTP, MTP + n-gram (WHIRL only) |
| Sweeps (plain) | fixed input files: prefill 2,022 / 8,178 / 32,751 tokens (`max_tokens` 1); decode 128 tokens after a 16,354-token context (`ignore_eos`). Token counts calibrated once with llama-server's `/tokenize` and frozen |
| Recorded | prompt tokens, `timings.prompt_per_second`, `timings.predicted_per_second`, MTP acceptance, output SHA-1 |
| Rounds | 2, interleaved (round 0 in order, round 1 reversed) |

Using `top_k = 1` instead of temperature 0 exercises the same sampler path in both engines.
llama.cpp's MXFP4 kernels have no folded-scale fp8 path, so its MXFP4 numbers reflect its current
kernels, not a ceiling.

## <a id="probes"></a>4. Microbenchmarks: Infinity Cache and other traps

- The R9700 has a 64 MB Infinity Cache, the 8060S a 32 MB MALL. Probes rotate through copies of real
  weights totalling > 512 MB (R9700) or > 2 GB (8060S), with warm-up.
- Use real weights extracted from the GGUF, not random data, for decode kernels (branching and LUT
  behavior depend on values).
- Interleave variants for 6–12 rounds; report min/median/max.
- Compare every variant's output byte for byte with the production kernel in the same probe.
- Autotuning: time a batch of repetitions with one final sync; per-repetition syncs measure launch
  overhead and bias selection.

## <a id="thermal"></a>5. Thermal rules for the 8060S (ROG Flow Z13)

The Z13 is a tablet-style laptop: about 80 W sustained power limit (mini-PCs with the same Ryzen AI
Max+ 395 run 120–160 W), and the performance mode holds full power for only about 20–30 s. Numbers
from mini-PCs (including published llama.cpp and gufo results) are not comparable, especially for
compute-bound prefill.

| Observation | Effect |
|---|---|
| Long prefill (4.9k tokens, 512-token chunks), cold | the first ~5 s run at boost speed, then prefill drops by about 15% |
| Same, right after a long decode | starts at the lower speed |
| Plain decode of 1,800 tokens after 2 min idle | about −4% over 139 s (partly context growth); no cliff |
| Back-to-back processes without cooling | decode more than halves after 3–5 minutes, both binaries alike; prefill also drops |

Rules: tests shorter than the boost window measure boost speed — report which one you measured, or
both. Compare versions with interleaved runs and 60–90 s idle before each run. A 40-minute benchmark
suite's later part is not evidence of a regression. Nothing may run on the other GPU while timing
the 8060S — CPU load from the R9700 side lowers the APU too (one run lost 11–18%). A server-vs-CLI
speed check that measured the CLI cold at the start and the server after minutes of load reported a
false −7…−14% "regression"; the check now measures the CLI before *and* after the server (median of
3 each side), allows 5% + the CLI's own drift, fails if that drift exceeds 8%, and waits 15 s
between server requests.

## <a id="vram"></a>6. The VRAM-only rule and its monitor

- A background sampler records Dedicated and Shared Usage per engine process per GPU adapter every
  2 s; a report flags any process above 256 MiB shared (after subtracting declared pinned memory)
  and any overlap of two engine processes on one adapter.
- The adapter is identified by LUID, **which changes after a reboot**; a report filtering for a stale
  LUID once found no samples and printed "all runs in dedicated VRAM". It now fails on empty data.
- Declaration files keyed by PID produced false alarms when a PID was reused; key them by PID + start
  time.
- llama.cpp's `llama-perplexity` peaked at 306 MiB shared in two runs, which by this rule makes those
  runs invalid for speed (values are unaffected).

## <a id="egpu"></a>7. eGPU caveats

The development R9700 sits behind USB4: host link ~3.8 GB/s each way, and kernel dispatch slows while
H2D copies saturate the link ([windows-hip.md](windows-hip.md#egpu)). Affected: model load time,
RAM/SSD KV restore time (~14–16 µs per token), other slots' decode during a restore (+17%), vision
weight streaming (~260 ms per image when weights are not resident). Not affected: everything that
stays in VRAM — prefill, decode, MTP, batching. We did not optimize for the eGPU; direct-PCIe users
should see faster restores.

## <a id="quality"></a>8. Measuring quality

- **KL against path noise.** For a lossy change, compare last-token distributions with the reference,
  and first measure the KL between two equally correct reference paths (chunked vs sequential
  DeltaNet, naive vs WMMA attention). A change is judged relative to that noise (dense 27B path noise
  ~1e-7; the MoE model's is up to 1e-2 because of routing near-ties).
- **Paired QA** (350 items: 150 Python output prediction verified by execution, 100 multi-step
  integer arithmetic, 100 long-context key/value retrievals in 17k–70k-token C++ sources with decoy
  keys), greedy, reasoning before `ANSWER:`, McNemar exact test on the discordant pairs. An
  answer-only version was useless (5–7% correct).
- **Needles** at 10/50/90% depth up to 256k.
- **Perplexity** (llama.cpp `llama-perplexity`) on a 219 KB mixed Chinese/English code corpus, 51
  chunks × 2048 tokens, with per-chunk pairing.

## <a id="gates"></a>9. Correctness gates (run on every change)

| Gate | What must hold |
|---|---|
| Dense check vs numpy f32 reference | batched, step-2, step-q8: KL < 1e-3, top-1 equal, top-10 overlap ≥ 9; step-f32 (exact path): KL < 1e-5; long-attn (WMMA vs naive), long-gdn (chunked vs sequential DeltaNet) |
| MoE checks | vs MoE numpy reference KL < 5e-3; long-context path comparisons KL < 1e-2, top-1 equal, every token with p ≥ 1e-3 in both top-10s |
| MTP exactness | MTP greedy == plain greedy, both models; draft counts 1, 2, 5, 10; n-gram forced every cycle == plain (MXFP4 smoke: plain == MTP == MTP + n-gram == forced n-gram) |
| Kernel self-checks | multi-row GEMV == 1-row (all types, 2–16 rows, all variants, output head); grouped verify attention == per-query; every GEMM option and MoE tile row-invariant |
| Logits vs previous build | last-token logits bit-identical for 4–5 prompts × both quantizations (when the change is meant to be exact) |
| Server suites (dense, MoE) | 50 checks each: endpoints, server == CLI, concurrent == sequential, multi-turn, **default environment** |
| Segmented prefill | 8 simultaneous requests (160–2,489 tokens) each == solo; 27B, MoE and MXFP4 × MTP on/off; MXFP4 C=4 concurrent == solo |
| Pool / cache | eviction and recompute, multi-turn hit ratios, tier restore (RAM, SSD after restart), system-prompt checkpoints, restore under load |
| Vision | encoder vs f32 numpy reference; VRAM unchanged at mmproj load; image prompts == cold references |

The full pipeline takes 55–60 minutes. Hygiene learned the hard way: delete expected output files
before each run (a failed binary once left old files that a check read as "ok"); set
`PYTHONIOENCODING=utf-8`; give every test server its own SSD cache directory; pin the KV format in
gates that compare servers (a reference server with more free VRAM auto-picked f16 while the tested
one picked q8v, and every comparison "failed"); compare tool calls by name + arguments (ids are
random).

## <a id="current-numbers"></a>10. Current numbers

The published numbers are in [benchmarks.md](../../benchmarks.md): Swift-1.5 27B MXFP4 and
Ornith-1.5-35B-A3B MXFP4 on the R9700 and the Radeon 8060S, all three WHIRL modes, against the
official llama.cpp ROCm and Vulkan builds, 2k to ~250k tokens. Earlier development measurements on
other model files are not published.

## 11. Benchmark sets

| Set | Content |
|---|---|
| Mixed Chinese/English coding (primary) | 7 prompts — Python LRU cache + pytest, Spring Boot order API, React hook + component, PostgreSQL schema + report, fixing Python with Chinese identifiers, Node.js refactor, reading 4k tokens of source — 800 tokens each, think off and on |
| File editing | 5 prompts with 1.6–2.2k-token source files (rename, refactor, translate comments; one CRLF file), output the whole file, `max_tokens` 2500 |
| Long context | fixed 2k / 8k / 32k / 128k / 256k files; 16k–128k needle and coding-continuation prompts built from real source code |
| Concurrency | 16 non-overlapping ~1.1k-token prompts, 256 tokens each, C = 1/2/4 (8/16 historically) |
| Agent scenarios | multi-turn tool use (`read_file`, `write_file`), shared 12–30k-token system prompts, eviction pressure, restarts |
| Quality | 350-item paired QA, KL prompt set (arch 1k, zh-short, 4k, 12k, 24k, code 16k/32k/64k, long 64k/128k), PPL corpus |

The primary benchmark reflects the project's main use: questions in Chinese, answers with Chinese
explanations and comments and English code. Mixed text is harder to predict than pure English code
(in the early MTP version pure English coding decoded clearly faster than mixed text), so it is the conservative
choice.
