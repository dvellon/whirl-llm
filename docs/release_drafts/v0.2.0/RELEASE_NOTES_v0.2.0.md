# WHIRL v0.2.0

**English** · [繁體中文](RELEASE_NOTES_v0.2.0.zh-TW.md)

> DRAFT for owner review. Nothing here is published. Items marked `[8060S: TBD]` or `[TBD]` wait for measurements or release-build checks that are not finished yet. Every number carries its source in an HTML comment.

Native Windows LLM inference for the AMD Radeon AI PRO R9700 (gfx1201) and, from this release, the Radeon 8060S (Ryzen AI Max+ 395, gfx1151).

On the R9700, in the default mode, against llama.cpp b11214 with the same GGUF files and prompts (greedy, one user): prefill from 8k to 249k tokens is 59% to 70% faster on the dense Qwen3.8-27B Q4_K_M and 73% to 153% faster on the MoE Ornith-1.5-35B-A3B MXFP4; decode after a 128k-token document is 82% (dense) and 91% (MoE) faster; four users at once on the dense model reach 251 tok/s against 91. The table and the llama.cpp flags are below. The speedups do not change the output: in precise and balance it stays identical to plain greedy decoding of the same mode.

<!-- src for the paragraph: audit/REL3.md, R9700 section, balance column vs "llama.cpp best": dense prefill 8k +59%, 32k +59%, 128k +63%, 249k +70%; MoE prefill 8k +153%, 32k +123%, 128k +87%, 249k +73%; QA decode at 128k dense +82%, MoE +91%; 4 users 251.4 vs 90.6. -->

Download: `whirl-0.2.0-windows-x64.zip` (SHA-256: `[TBD]`). Requirements: Windows 11, Adrenalin 26.8.1 or newer, one R9700 or one Radeon 8060S.

## Highlights

- **Three numerics modes in the box: precise, balance (default), fast.** precise gives bit-reproducible output from a plain f16 reference path; balance is the 0.1.x behaviour (near-lossless, KL <= ~0.005 against precise); fast is opt-in, trades a bounded, published amount of quality for speed (4-bit KV, relaxed draft acceptance). See [the table](#numerics-modes).
- **Radeon 8060S is officially supported.** RAM and SSD cache tiers are on by default there: a 128k-token session comes back in 0.34 s from RAM (2 s from SSD after a restart) instead of 9 minutes of prefill. Precise mode decodes about twice as fast as before. [8060S: TBD — comparison against llama.cpp]
- **The SSD cache no longer writes while the server runs.** The new default writes only on a normal stop, caps the cache at 64 GiB on disk and deletes files unused for 7 days. A user reported about 400 GB written in 23 minutes with the old default.
- **DFlash2 drafter, opt-in.** For one user with short or medium context and predictable output it gave +19% to +48%; with several users or long contexts it is slower, so it is not the default. The drafter file is not bundled.
- **Faster prefill for dense models of any weight format** in balance and fast (+7% to +10% on Qwen3.8-27B Q4_K_M), and fixes from an external code review.

<!-- src: modes: CHANGELOG.md Unreleased; 8060S numbers: audit/TIER-8S.md (128k: 536.7 s cold, 0.34 s RAM, 2.00 s SSD) and audit/GEMVH-1151.md; 400 GB / 23 min: audit/SSD-QUIET.md; DFlash2 +19..+48%: audit/DFLASH-1.md section 7; prefill +7..+10%: CHANGELOG KG-1 (8k +10.4%, 32k +9.9%, 128k +7.0%). -->

## Speed against llama.cpp (R9700)

Balance mode (default) against llama.cpp b11214 (ROCm), `llama-server --device ROCm0 -ngl 99 -fa on -b 2048 -ub 1024 (MoE 2048) --jinja --reasoning off --top-k 1 -np 1`, f16 KV, each cell the faster of llama.cpp's plain and MTP + n-gram modes. Measured 2026-10-12. "% faster" is WHIRL / llama.cpp - 1.

| R9700, balance | Qwen3.8-27B UD-Q4_K_M (dense) | Ornith-1.5-35B-A3B MXFP4 (MoE) |
|---|---|---|
| Prefill tok/s, 8k prompt | 1,914 vs 1,201 (+59%) | 11,199 vs 4,426 (+153%) |
| Prefill tok/s, 32k prompt | 1,721 vs 1,082 (+59%) | 8,585 vs 3,853 (+123%) |
| Prefill tok/s, 128k prompt | 1,230 vs 756 (+63%) | 4,273 vs 2,285 (+87%) |
| Prefill tok/s, 249k prompt | 912 vs 536 (+70%)¹ | 2,659 vs 1,534 (+73%) |
| Decode tok/s, question about an 8k document | 84.9 vs 41.8 (+103%) | 229.3 vs 103.5 (+122%) |
| Decode tok/s, question about a 128k document | 49.3 vs 27.1 (+82%) | 132.9 vs 69.7 (+91%) |
| Decode tok/s, new writing after 128k (worst case for drafts) | 33.8 vs 21.2 (+59%) | 113.8 vs 69.9 (+63%) |
| Decode tok/s, Chinese coding chat | 100.9 vs 58.1 (+74%) | 242.5 vs 108.8 (+123%) |
| 4 users at once, aggregate tok/s | 251.4 vs 90.6 (+177%) | not measured |

<!-- src: audit/REL3.md, R9700 section (rc4 tip 100a9a8), balance column vs "llama.cpp best". -->

¹ llama.cpp could not hold f16 KV at 249k tokens on the dense model and used q8_0 KV. Length sweeps are single runs; treat differences under about 5% as noise. All three modes at every length, the method, and where the lead is small (for example precise costs 25% of dense chat decode speed): README, [Performance](https://github.com/tsaipifong/whirl-llm#performance). Swift-1.5 MXFP4 was not re-measured for this release; its 0.1.3 figures remain in the README.

[8060S: TBD — table: Radeon 8060S, WHIRL vs llama.cpp, prefill / decode, three modes, machine and power limit]

## <a id="numerics-modes"></a>Numerics modes

Pick one at start-up: `--precise`, `--balance` (default), `--fast`.

| | precise | balance (default) | fast |
|---|---|---|---|
| KV cache, dense / MoE | f16 / f16 | q8h (int8) / f16 | q4 (4-bit) / q4 |
| Output | nothing quantized beyond the GGUF file; speedups bit-identical to plain greedy | near-lossless, KL <= ~0.005 against precise; speedups bit-identical to plain greedy | within KL <= 0.03; measured KL of q4 KV vs f16: 0.0016 to 0.0100 (R9700); not bit-identical |
| Speed vs balance (R9700) | decode -25% to -29%, dense prefill -5% to -10% | reference | dense decode +15% to +18%; MoE faster only above ~64k |
| Pick it for | output you need to reproduce exactly, or to check another engine against WHIRL | almost everything | long contexts, or maximum tok/s when small output changes are acceptable |

<!-- src: CHANGELOG.md Unreleased (BAL-Q8, precise, FAST-1c); audit/REL3.md; internal_docs/v013_plan.md (gates: balance KL <= ~0.005, fast KL <= 0.03, owner 2026-10-08). -->

What the guarantees mean:

- precise and balance: a request gives the same tokens alone or among other users, with or without MTP / n-gram / DFlash2 drafts, cold or from a cached prefix. This is enforced by `whirl selftest`, kernel tests, golden outputs and the server gates, in each mode.
- fast: quality is bounded by a gate, not by bit-identity. A request run alone and the same request among other users can give different text. The gate run (Qwen3.8-27B Q4_K_M) had 12 of 12 coding tasks executing correctly, 0 of 24 generations looping and a repetition rate of 0.167 against 0.157 for precise; about 7% of accepted drafts came from the relaxed rule. We did not re-run that gate on the R9700 for this release. [TBD: owner decides whether to re-run it]
- The KL values compare logits with the same model's f16-KV logits. They say how far the output distribution moves, not how accurate the model is.

<!-- src: audit/CONC-8S.md (solo vs concurrent in fast is best effort); progress/FAST-2.log 2026-10-07 12:49 and 13:01 (gate figures; measured on the 8060S). -->

Changes you may notice in balance on dense models: the KV format no longer depends on what fits. 0.1.x chose f16, then q8v, then q8h; now balance always uses q8h on dense models and f16 on MoE models, and a context that does not fit in the chosen format is lowered with a warning (or the program stops with a message if you set `--ctx` / `--ctx-per-slot`). It never switches to a lower-precision format by itself.

## DFlash2 draft (opt-in)

`whirl-server MODEL.gguf --draft dflash --draft-model PATH` (also `whirl chat`, `whirl bench`). The default is still MTP + n-gram. The drafter is not bundled: [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) on Hugging Face (Apache-2.0; we tested the Q8_0 GGUF). It takes about 1.9 GiB of VRAM and shrinks the KV pool by about 12%. Output equals plain greedy decoding of the same mode.

| Qwen3.8-27B Q4_K_M, R9700, balance | MTP + n-gram | DFlash2 + n-gram | Change |
|---|---|---|---|
| Java API code / SQL report, one user (chat) | 105.7 / 83.7 | 125.6 / 123.6 | +19% / +48% |
| Coding-agent session replay, mean (last step at 65k) | 83.4 (66.2) | 86.9 (74.4) | +4% (+12%) |
| Chinese coding chat | 100.9 | 91.3 | -10% |
| 4 users at once | 211.3 | 129.7 | -39% |
| Question after a 37k / 108k prompt | 54.6 / 40.3 | 39.1 / 25.8 | -28% / -36% |

<!-- src: audit/DFLASH-1.md section 7 (steps 3, 7, 8); audit/REL3.md DFlash2 table (agent replay, Chinese chat). Fuller table in the README. -->

Use it for one user, short or medium context, English-heavy or predictable output (API / SQL boilerplate). Leave it off for several users, contexts of 37k tokens or more, and mostly Chinese output. Not measured with DFlash2: Swift-1.5, the MoE model, the Radeon 8060S.

## SSD cache

- Default `--kv-ssd-mode shutdown`: no SSD writes while the server runs; a normal stop (Ctrl+C) writes the sessions, newest first, for up to 60 s.
- `--kv-ssd-mode always` is still available; it writes a session after 5 minutes unchanged (`--kv-ssd-delay SEC`) and only if the session was reused.
- At most 64 GiB on disk (`--kv-ssd-gb`), counting stale files from other versions or models, enforced at start-up; least recently used sessions are removed first. Files unused for 7 days are deleted. Broken or half-written cache files are deleted at start-up.
- The report that led to this (about 400 GB in 23 minutes) was the old default writing every distinct long prompt, 0.3 to 1.5 GiB each. We found no hidden write amplification: a new session is written once, 1.004 to 1.012 times its file size.
- Users of 0.1.x who do not want to wait for the upgrade: `--kv-ssd-gb 0` turns the SSD tier off.
- To delete the cache by hand: stop the server, delete the `*.wkv` files in `%LOCALAPPDATA%\whirl\kvcache` (or your `--kv-ssd-dir`).

<!-- src: audit/SSD-QUIET.md; CHANGELOG.md Unreleased; internal_docs/v013_plan.md 2026-10-11 ("No v0.1.5 hotfix; users on v0.1.4 can use --kv-ssd-gb 0"). -->

## Radeon 8060S

- Both GPU code sets ship in one zip. `--device 8060s` selects the 8060S when an R9700 is also present. All three modes work; kernel tests, golden outputs and server gates pass. Not available there yet: image input.
- **RAM and SSD tiers are on by default.** On Windows the BIOS carve-out (VRAM) and the OS RAM are separate pools, so the tier is real, separate memory. It is sized from the OS-visible RAM: 1/8, within one maximum-context session and 16 GiB, at most 1/4 of the free RAM, off under 32 GB (8 GiB on a 64 GB machine; the start-up log says so). Qwen3.8-27B Q4_K_M, balance, time to first token when a session returns: 8k tokens 20.9 s re-prefill vs 0.11 s (RAM) / 0.21 s (SSD after a restart); 128k tokens 536.7 s vs 0.34 s / 2.00 s.
- **Precise decode about twice as fast**: Qwen3.8-27B Q4_K_M, short context, 5.34 to 11.19 tok/s (MTP + n-gram 20.2 to 40.2); at 32k 4.9 to 9.27 (MTP 15.1 to 23.0). Bit-identical output.
- Fast mode uses 4-bit KV (decode at 128k on Ornith-1.5-35B-A3B: 58.5 tok/s against 32.7 with f16 KV).
- [8060S: TBD — llama.cpp comparison and per-mode speeds]

<!-- src: audit/TIER-8S.md (table and sizing rule), CHANGELOG.md TIER-8S-DEF; audit/GEMVH-1151.md (before = WHIRL_PREC_GEMVH=0 runs in section 3, now = GEMVH-2 end-to-end table; two sessions on the same machine); CHANGELOG fast kvq4 on the Radeon 8060S. -->

## Other changes

- Dense models of any weight format get the f16 DeltaNet and GEMM-output paths in balance and fast (before: MXFP4 files only): Qwen3.8-27B Q4_K_M prefill 8k 1,849 to 2,042 tok/s (+10.4%), 32k +9.9%, 128k +7.0%, decode unchanged.
- Sampling with MTP at temperature > 0 is exact in balance and fast (`specsample`): the same text for a seed whether MTP is on or off.
- Fixes from an external review, outputs unchanged: the MTP draft window is restored when a draft step throws; the prefill batch cannot shrink below the scheduler's chunk size; a cache file shorter than recorded is deleted and a short read falls back to a cold prefill; a failed restore copy counts as failed; a failed SSD directory lock leaves the SSD tier off; DeltaNet shapes the kernels do not implement are refused at load.
- Fast mode: a repeated request gives the same text streamed or not, cold or resumed from its own checkpoints.

<!-- src: CHANGELOG.md Unreleased (KG-1, specsample, FIX-FS); audit/FIX-REVIEW2.md. -->

## Exactness

Gates on the release candidate (R9700): CPU suites 2239 + 212 + 86 + 40 + 960 + 537 checks, 0 failed; kernel tests (Q4_K_M and MXFP4) 0 failing; `whirl selftest` ok on the dense and the MoE model in all three modes; golden outputs 18 of 18 identical to the recorded references; server gates basic + mt_cache 27/27 in fast and in balance; fast_stream 31/31 dense and MoE; SSD tier restore after a normal stop reproduces the output. On the 8060S: kernel tests, selftest (6), golden outputs (18 of 18), server gates basic + mt_cache (balance, fast) 27/27 after the concurrency fix, fast_stream 31/31. [TBD: repeat on the exact files in the release zip]

<!-- src: audit/RC4-GATES.md (R9700 and 8060S tables, "8b0dab5 R9700" re-pass), audit/CONC-8S.md and audit/TIER-8S.md (8060S gates after the fixes). DFlash2: output hash equal to plain greedy in precise (RC4-GATES). -->

## Known limitations

- fast mode is not bit-identical to plain greedy decoding, and a request alone can differ from the same request among other users.
- A conversation that continues reuses the KV and DeltaNet state of the previous turn: numerically equivalent to re-processing the history, not bit-identical (unchanged from 0.1.x).
- Dense precise cannot hold a 256k context on a 32 GB card (f16 KV: pool of 176,128 tokens); the program refuses to start with that setting and says so.
- In `--kv-ssd-mode shutdown` a crash or a kill loses the sessions that were not yet written.
- Image input is not available on the 8060S.
- DFlash2 is measured only on the R9700 with Qwen3.8-27B Q4_K_M.
- The executables are not code-signed (SmartScreen may ask; see the README).
- Only the R9700 and the Radeon 8060S are supported. The RX 9070 series (16 GB) is untested.

## Upgrading

- Nothing to configure. The kernel tuning cache in `%LOCALAPPDATA%\whirl` is reused where the model, GPU and build match.
- Cache files written by 0.1.x are not restored; they count against the 64 GiB cap and are deleted first (or delete `%LOCALAPPDATA%\whirl\kvcache` yourself).
- Dense models in the default mode now always use q8h KV (0.1.x picked f16, q8v or q8h by what fitted). For f16 KV use `--precise`.
- 8060S users: the RAM tier now takes memory from the OS (8 GiB on 64 GB). Use `--kv-ram-mb 0` to turn it off.

Full list of changes: [CHANGELOG.md](https://github.com/tsaipifong/whirl-llm/blob/main/CHANGELOG.md).
