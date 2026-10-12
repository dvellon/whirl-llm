DRAFT for owner review, not posted. Title suggestion: **WHIRL 0.2.0: native Windows LLM inference for the R9700 and now the Radeon 8060S (open source, Apache-2.0)**

---

WHIRL is an open-source inference engine for AMD GPUs on Windows (C++ and HIP, no WSL, no ROCm install, only the Adrenalin driver). 0.2.0 adds official Radeon 8060S (Ryzen AI Max+ 395) support, three numerics modes, an opt-in DFlash2 drafter, and a change to how the SSD cache writes.

- GitHub: https://github.com/tsaipifong/whirl-llm
- Release: https://github.com/tsaipifong/whirl-llm/releases/tag/v0.2.0

**Speed against llama.cpp (R9700, default mode)**

Same GGUF files and prompts, greedy, one user, llama.cpp b11214 ROCm (`llama-server -ngl 99 -fa on -b 2048 -ub 1024 --jinja --reasoning off --top-k 1 -np 1`, f16 KV, faster of plain and MTP + n-gram per cell). Single runs for the length sweeps; I treat differences under about 5% as noise.

| | Qwen3.8-27B Q4_K_M (dense) | Ornith-1.5-35B-A3B MXFP4 (MoE) |
|---|---|---|
| Prefill, 8k prompt (tok/s) | 1,914 vs 1,201 (+59%) | 11,199 vs 4,426 (+153%) |
| Prefill, 128k prompt | 1,230 vs 756 (+63%) | 4,273 vs 2,285 (+87%) |
| Decode after a 128k document (question about it) | 49.3 vs 27.1 (+82%) | 132.9 vs 69.7 (+91%) |
| Decode, new writing after 128k (worst case for drafts) | 33.8 vs 21.2 (+59%) | 113.8 vs 69.9 (+63%) |
| Decode, Chinese coding chat | 100.9 vs 58.1 (+74%) | 242.5 vs 108.8 (+123%) |
| 4 users at once, aggregate | 251.4 vs 90.6 (+177%) | not measured |

[8060S: TBD — one or two lines with the 8060S numbers against llama.cpp, plus machine and power limit]

Where it does not lead by much: plain decode without speculation is memory-bound in both engines, and the lead shrinks when the drafts miss (the +59% rows). My R9700 is a USB4 eGPU, which affects model loading and cache restores, not prefill or decode.

**Precision**

- precise / balance / fast are chosen with a flag. Default is balance, the same behaviour as 0.1.x.
- In precise and balance, MTP, n-gram drafts, batching and cache restores give the same tokens as plain greedy decoding of that mode. This is checked by tests, not assumed.
- precise: f16 KV and no fp8/int8 activations, so there is nothing lossy beyond the GGUF itself. It costs speed (dense chat decode 75.7 vs 100.9 tok/s).
- balance: near-lossless, KL <= about 0.005 against precise (q8h KV vs f16: 0.0002 to 0.002).
- fast (opt-in): 4-bit KV and relaxed draft acceptance. KL of the 4-bit KV against f16 measured 0.0016 to 0.0100. Output is not bit-identical, and the same request can differ between running alone and running among other users.

**DFlash2 (opt-in, `--draft dflash --draft-model PATH`)**

It helps one user with short or medium context and predictable output: Java API code +19%, SQL report +48%. It hurts with 4 users (-39%), after 37k+ tokens of context (-28% to -36%) and on mostly Chinese output (-10% in my Chinese coding chat). It costs about 1.9 GiB of VRAM. The drafter is not bundled: https://huggingface.co/incoai/Qwen3.8-27B-DFlash2. So it stays off by default.

**SSD cache**

A user reported about 400 GB written in 23 minutes while benchmarking many different long prompts. That was the old default writing every distinct prompt. The new default writes only on a normal server stop, the cache is capped at 64 GiB on disk, and files unused for 7 days are deleted. `--kv-ssd-mode always` still exists (writes after 5 minutes unchanged, only sessions that were reused). To clear the cache, stop the server and delete `%LOCALAPPDATA%\whirl\kvcache`.

**Radeon 8060S**

RAM and SSD cache tiers are on by default there (sized from the OS-visible RAM, off under 32 GB). On Windows the BIOS carve-out and OS RAM are separate pools, so a 128k-token session restores in 0.34 s from RAM (2 s from SSD after a restart) where re-prefill takes 537 s. Precise-mode decode is about twice as fast as before (Q4_K_M, short context: 5.3 to 11.2 tok/s). Image input is not on the 8060S yet.

**Limits**

The R9700 and the 8060S only. The exe is unsigned, so SmartScreen may ask. Swift-1.5 MXFP4 was not re-measured for this release. DFlash2 was measured on the R9700 with the dense Q4_K_M model only.

WHIRL was designed and implemented with Claude (Anthropic) under my direction. Bug reports, benchmark results and measurements on desktop 8060S machines are welcome; the README has the full tables, methods and commands.
