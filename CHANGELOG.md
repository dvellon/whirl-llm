# Changelog

All notable changes to WHIRL are listed here. Versions follow `project(whirl VERSION ...)` in
`CMakeLists.txt`.

## Unreleased (0.2.0)

### Radeon 8060S: prefill forward size per model, dense back to 2048 rows (SRV-PP-DENSE)

- On the Radeon 8060S (gfx1151) the default prefill forward is now 4096 rows for MoE models only; dense models keep 2048 rows (`prefillExecDefault(gfx1151, moe)`, `include/whirl/prefill_schedule.h`; `WHIRL_PREFILL_CHUNK` still overrides). The rc8 change to 4096 for every model slowed dense Qwen3.8-27B prefill on the 8060S by 3.5% / 5% / 8% / 13% at 32k / 64k / 96k / 128k (server, 2 rounds; 128k 248.0 -> 284.6 tok/s back with 2048), while Ornith MoE gains 3.0% / 4.3% / 4.2% at 64k / 96k / 128k with 4096. Same output bits (forward size is a schedule-point grouping; seqtest and golden unchanged), no memory change, R9700 unchanged.

### MoE balance keeps int8 decode activations; KL gate MoE balance top-1 tolerance 3.0 points (MOE-BAL-3)

- **MoE balance keeps int8 decode activations** (`q8dec`, one f32 scale per 32 values): the same q8_1 activations llama.cpp itself uses for decode, i.e. the same numeric level as llama.cpp. MoE balance = `gdnwmma` + `h16` + `q8dec`, f16 KV (no `fp8` / `moefp8`, MOE-BAL). A short-lived table without `q8dec` (MOE-BAL-2, never released) cost about 20% of MoE balance decode on the R9700 and was reverted. Dense balance and every fast table are unchanged; `WHIRL_Q8DEC=0` gives f16 decode activations.
- **KL gate (`tools/klgate/klgate.py`), MoE balance: top-1 of each length's 3-position average >= noise top-1 - 3.0 points** (was 2.4); the KL limits (mean over all windows <= 2 N, each length's average <= 3 N), MoE fast and the dense absolute limits are unchanged. Owner decision 2026-10-14.
- KL gate, exe c97326ff, R9700 MoE balance: code 1.91 N mean (largest length 2.48 N, top-1 >= 98.24%), zh 1.91 N (largest 2.35 N), zh 8k top-1 93.88% vs floor 93.40% (+0.48 pt); MoE fast and dense balance / fast unchanged (bit-identical numbers); 36 / 36 pass. Radeon 8060S MoE balance: code 1.97 N mean (largest 2.45 N, top-1 >= 98.05%), zh 1.85 N (largest 2.17 N), lowest top-1 zh 8k 95.05% vs floor 93.24%; MoE fast 6.17 N / 5.20 N; dense zh balance / fast pass. Negative check: MoE balance with fp8 + moefp8 (judged as balance) still fails (zh 3.03-3.57 N per length, 2.92 N mean).
- Speed (REL3-RC5 scripts, Ornith MXFP4 balance), back to the rc5 level: R9700 decode 2k / 8k / 32k / 64k / 96k / 128k 242.0 / 218.3 / 201.8 / 173.4 / 157.5 / 141.9 tok/s, Chinese coding chat 251.6 (llama.cpp b11214 108.8); Radeon 8060S 100.4 / 90.1 / 81.8 / 69.7 / 58.9 / 53.4 tok/s, chat 98.8 (llama.cpp 62.5); prefill unchanged.
- Golden references: only `moe/balance_single` and `moe/balance_pair` re-recorded (gfx1201 and gfx1151), after the KL gate passed on the same exe.

### KL gate measures 3 window positions per length and judges their average (KL-ROBUST-2)

- KL gate measures 3 window positions per length and judges their average: `tools/klgate/klgate.py` dumps 3 windows of 512 rows per length (ending at L - 1216, L - 592, L) and applies the per-window limits to each length's 3-position average; the MoE top-1 floor is the noise run's top-1 averaged over all windows; limit values unchanged (a single window's KL moved +-20-35% with the forward split before it; audit/KL-ROBUST.md, audit/KL-ROBUST-2.md).

### CLI and server give identical output for the same prompt (CLI-SCHED)

- `whirl chat` now prefills on the server's chunk schedule (same forced splits and forwards, `include/whirl/prefill_schedule.h`), so the CLI and the server give bit-identical output for the same prompt; before, prompts of 4096·k plus a few tokens or with a 2048+ token system message could differ in the last bits (`whirl seqtest` CLI-SCHED gate; audit/CLI-SCHED.md).

### SSD tier: shutdown mode never writes a one-shot prompt (CLOSE-REUSED)

- `--kv-ssd-mode shutdown` (default) now writes only reused entries, in every case: when WHIRL is closed (Ctrl+C, console window closed, logoff, OS shutdown), on RAM eviction, and after about 5 minutes idle; a one-shot prompt is never written to the SSD (the close flush used to write it too). `always` mode is unchanged (audit/CLOSE-REUSED.md).

### SSD tier: shutdown mode also writes reused entries after about 5 minutes idle (TIER-IDLE)

- `--kv-ssd-mode shutdown` (default) writes when WHIRL is closed (Ctrl+C, console window closed, logoff, OS shutdown), plus reused entries that RAM must evict, plus, after about 5 minutes with no request, reused entries in the background (one at a time; a new request stops it; one-shot prompts are never written while running); console close, logoff and OS shutdown allow only about 5 s, so Ctrl+C is the way to guarantee a full write (audit/TIER-IDLE.md).

### SSD tier: shutdown mode keeps reused entries that RAM has to evict (TIER-EVICT)

- `--kv-ssd-mode shutdown` (default) writes when WHIRL is closed (Ctrl+C, console window closed, logoff,
  OS shutdown; the last three allow only about 5 s, Ctrl+C a full flush), and now reused entries that RAM
  must evict are also written: a reused entry (prefix restored, session resumed, or a shared system-message checkpoint that
  requests attached) of at least 2048 tokens pushed out of a full RAM tier is written to the SSD in the
  background instead of dropped (cap, dedup and the crash-safe write protocol as for any write; one-shot
  prompts are still never written). Fixes the system checkpoint lost across a restart on the Radeon 8060S
  with an 8 GiB RAM tier and precise f16 KV (audit/INT-RC4F.md, audit/TIER-EVICT.md).

### MoE balance without fp8 activations; KL gate judges MoE against its noise floor (MOE-BAL)

- **MoE models (`qwen35moe`, e.g. Ornith-1.5-35B-A3B MXFP4): balance = `gdnwmma` + `h16` + `q8dec`
  with f16 KV**; `fp8` and `moefp8` are fast-only there (owner decision 2026-10-13, audit/MOE-KL.md:
  fp8 activations in front of the 256-expert router flipped routing from layer 0 and put MoE balance
  at 3.3-3.6x the model's KL noise floor). Fast is unchanged (fp8 + moefp8 + gdnwmma + h16 + q8dec +
  q4 KV); dense models are unchanged; the Radeon 8060S MoE table is unchanged (no fp8 WMMA there).
  The start-up log / `GET /props` list them as `skipped: fp8 (MoE: fast only ...)`. An explicit list
  is used as given: `--balance=fp8,moefp8,gdnwmma,h16,specsample` is the old MoE balance;
  `WHIRL_FP8=1` / `WHIRL_MOE_FP8=1` in MoE balance are logged as lossy user overrides (`balance+overrides`).
- Cost (R9700, Ornith MXFP4, `whirl bench`, 2 runs, same exe old vs new table): prefill 8k
  12,701 -> 9,516 tok/s (-25.1%), 32k 9,360 -> 7,511 (-19.8%), 128k 4,486 -> 4,003 (-10.8%); decode
  unchanged (MTP 264.5 / plain 192.2 tok/s). Use `--fast` for the fp8 prefill speed.
- KL gate (`tools/klgate/klgate.py`): MoE models are judged against the noise floor N measured in
  the same run (precise vs precise with `WHIRL_GDN_SEQ=1`): balance mean KL <= 2 N, each window <= 3 N,
  top-1 >= N's - 2.4 points; fast 8 N / 12 N / -8 points. Dense keeps the absolute limits. Windows
  are 512 rows (was 128; `--rows`), which fixes the dense fast zh 8k one-row top-1 miss without
  relaxing a limit. R9700 (exe a88b0515): **36 / 36 pass**. Dense balance KL 1.1e-4 to 9.8e-4,
  top-1 >= 98.24%; dense fast KL <= 0.0072, top-1 >= 96.09%; Ornith balance 1.88 N (code, N 0.0019) /
  1.90 N (zh, N 0.0051), largest window 2.84 N; Ornith fast 6.09 N / 6.09 N, largest window 11.41 N.
  The MoE margins are thin, and a first 256-row run failed (MoE zh balance 2.04 N, dense zh 8k
  balance 97.27% / fast 92.97%): the window start moves where the prefill is split, so window numbers
  carry chunk-schedule jitter besides sampling noise (audit/MOE-BAL.md).
  Radeon 8060S (same exe, 512 rows): 36 / 36 pass; Ornith balance 1.75 N (code) / 1.82 N (zh), fast
  5.42 N / 5.39 N; the code 96k window that failed the old absolute limit (KL 0.0121 at 128 rows) is
  0.0047 = 2.86 N; dense balance top-1 >= 98.05%, dense fast top-1 >= 94.73%.
- gfx1201 golden references: only `moe/balance_single` and `moe/balance_pair` changed (re-recorded
  after the passing gate on the same exe); dense and MoE precise / fast references are bit-identical,
  `moe/balance_ngram` unchanged. gfx1151 references unchanged (its MoE balance table did not change).

### Image input on the Radeon 8060S (VIS-8S)

- `--mmproj` / `image_url` parts now work on gfx1151 (Radeon 8060S); the server used to refuse with `MropeUnsupported`. The gfx1151 code object gains the multi-section RoPE attention prep (`attn_prep_m`, `_q8`, `_q8h`, `_q8v`, `_q4`) and `set_rpos`; the vision encoder (`kernels/vision/vision.hip`) is built per architecture, with gfx11 WMMA fragment layouts for its GEMM and attention kernels (f16 inputs, f32 accumulation as on the R9700; no fp8 is involved).
- Accuracy: 8060S embeddings vs an f32 numpy reference, relative L2 0.16-0.30%, minimum cosine >= 0.9995 on 4 images (R9700: 0.16-0.31%); vs the R9700, relative L2 0.09-0.23%. Server end to end (`--precise`, temperature 0, 8 image requests incl. two images and 1080p): answers text-identical to the R9700. R9700 vision output unchanged (embeddings bit-identical to before; its vision code object disassembles identically).
- New `whirl-kernel-test --only vision`: every encoder kernel against a CPU reference (both GPUs). `audit/VIS-8S.md`.

### DFlash2 speculative decoding (`--draft dflash`, opt-in)

- `--draft mtp|dflash` for `whirl chat` / `whirl bench` / `whirl-server` (default `mtp`). With
  `--draft dflash --draft-model FILE` (else `WHIRL_DRAFT_MODEL`; the command line wins) a DFlash2
  drafter GGUF that fits the model (Hugging Face `incoai/Qwen3.8-27B-DFlash2-GGUF`,
  `Qwen3.8-27B-DFlash2-Q8_0.gguf`, Apache-2.0) proposes up to block - 1 (7) tokens per cycle from the
  target's hidden states; the MTP block is then not loaded. n-gram drafts stay on and the cost model
  picks n-gram or DFlash2 per cycle. DFlash2 drafts are verified like n-gram drafts with exact
  acceptance in every numerics mode (relaxacc in `--fast` relaxes MTP drafts only), so greedy output
  equals plain greedy decoding of the same mode and a seeded sampling reply does not depend on the
  drafts. Checked on the R9700 in `--fast` (4-bit KV): the 8- and 16-row literal-draft verify gives logits bit-identical to single-row decode (`whirl seqtest --fast`), and `whirl bench --fast --draft dflash` reproduces the plain-decode token stream.
- When it helps (R9700, vs MTP): one user, short / medium context, English-heavy or predictable
  output: up to +19..48%. Slower: 4 concurrent users (about -39%), long contexts (37k+ tokens),
  mostly-Chinese output (about -10%). Costs about 1.9 GiB of VRAM (weights, capture, an 80 MiB context
  ring per slot), so the KV pool is smaller; the VRAM fit check counts it. A Q4_K_M drafter is untested.
  See docs/usage.md (DFlash2 drafts).
- Without a fitting drafter (a MoE model, a drafter whose tensors do not match its metadata, an MXFP4
  `fc.weight`, a file that cannot be opened) or without the `dflash_*` kernels in the code object: a
  warning and the MTP drafts. The kernels are in both kernel sets (gfx1201, gfx1151).

### Long-document Q&A: the second question reuses the document (TAIL-CK)

- Prompts of 4096+ tokens keep a per-slot **tail checkpoint** at the start of their last 1024-token chunk, on the cold-run chunk schedule (precise / balance output stays bit identical to a cold run). Another question about the same document (`document + question` in one message) now resumes there and prefills about 0.3k-1.3k tokens instead of the whole document; before, only the third question was served from the cache (REL3: 131k document, second question 107 s on the R9700, 589 s on the 8060S). `-np 2..4` now default to 3 checkpoints per slot (was 2; +150.6 MiB VRAM per slot for Qwen3.8-27B, +63.8 MiB for Ornith-1.5-35B-A3B, counted in the VRAM fit, which drops back to 2 when VRAM is tight); a full set keeps its newest tail checkpoint, so regenerate (prompt-end) and next-turn (gen-end) reuse are unchanged. With fewer than 3 checkpoints per slot no tail is taken. `WHIRL_TAIL_CK=0` turns it off (debug). `audit/TAIL-CK.md`.

### RAM and SSD tier on by default on the Radeon 8060S (TIER-8S-DEF)

- On integrated GPUs the host RAM tier of the prefix cache (and with it the SSD tier, mode `shutdown`) is **on by default**; it used to be off because the KV pool "already lives in system memory". On Windows the BIOS splits the VRAM carve-out (the KV pool) from the OS RAM, so the tier is separate pinned OS memory and a restore is a real copy. Measured on the 8060S (Qwen3.8-27B Q4_K_M, balance): a 128k session restores in 0.34 s from the RAM tier and 2 s from the SSD tier after a restart, against 537 s of prefill; decode speed unchanged (`audit/TIER-8S.md`).
- Size rule (OS-visible RAM only, the carve-out does not count): `clamp(OS-visible RAM / 8, one max-context entry (about 6 GiB for 27B at 262k), 16 GiB)`, at most 1/4 of the RAM available at startup; off under 32 GB of OS-visible RAM. 8 GiB on a 64 GB machine. The startup log shows the size and the reason. `--kv-ram-mb` / `WHIRL_KV_RAM_MB` still override, `--kv-ram-mb 0` turns both host tiers off, `--kv-ssd-gb 0` / `--kv-ssd-mode off` only the SSD tier. Discrete GPUs are unchanged.

### Fast-mode concurrency (conc8s)

- GPUs without replay kernels (Radeon 8060S) now use the same relaxacc draft caps as the R9700 ({8, 7, 4, 3} for 1-4 slots, 14 snapshot sets, +0.9 GiB), so concurrent fast requests match solo ones in the gate; fast concurrent == solo is best effort, not guaranteed (documented). The server gate's concurrency check runs fast mode with `WHIRL_RELAX=0` and only reports the relax-on result.

### SSD tier: far fewer disk writes, hard size cap, expiry (SSD-QUIET)

- **Default `--kv-ssd-mode` is now `shutdown`**: no SSD writes while the server runs; sessions stay in
  the RAM tier and a graceful stop (Ctrl+C) writes them (bounded by the RAM tier and `--kv-ssd-gb`;
  the stop waits up to 60 s for it instead of 10 s). Reason: a user's benchmark of many different
  long prompts wrote about 400 GB in 23 minutes with the old default (`always`, 2 s). `always` is
  still available (`--kv-ssd-mode always`).
- **`always` mode writes after 5 minutes, not 2 s** (new `--kv-ssd-delay SEC`, default 300; env
  `WHIRL_KV_SSD_DELAY_MS` still works). A session whose next request comes sooner is served from RAM
  and never written (the "unchanged for" clock restarts on every change and on every RAM restore).
  Under RAM pressure an entry that is not due yet is written (>= 2048 tokens) or dropped; it used to be
  dropped or kept for the delay regardless. While entries only wait for their delay the engine sleeps
  until the first one is due instead of polling every 5 ms.
- **The cache directory never exceeds `--kv-ssd-gb` (default 64 GiB) on disk**: startup now enforces the
  cap (it used to be enforced on writes only), counts files of other models / builds against it and
  deletes them first, and deletes unusable `*.wkv` files (not a cache file, unreadable header, half
  written, cut short, corrupt) instead of leaving them uncounted. Dedup (sparse) files are counted by
  allocated bytes. When full the least recently used session goes, together with the files that borrow
  its blocks.
- **Files not used for 7 days are deleted** (start-up, then hourly; a restore from the SSD refreshes the
  file's last-write time and the owners of its shared blocks). Fixed default; `WHIRL_KV_SSD_EXPIRE_S`
  exists for tests. One start-up log line reports files, GiB on disk, the cap and what was removed and why.
- **`always` mode skips one-shot prompts; the graceful stop flushes most recently used first** (SSD-QUIET-2): an
  entry is written (delay expired, RAM pressure) only if its prefix was restored at least once or the session
  was resumed by a later turn, so a stream of distinct prompts writes about nothing; `shutdown` mode still
  writes everything in RAM, and when the 60 s bound runs out the oldest sessions are the ones left out.
- Write amplification was measured with the mock tier (Ornith-35B layout, 2k to 16k-token distinct
  prompts): a new entry writes 1.004 to 1.012 times its file size, once; a continued conversation writes
  only new pages and the changed checkpoint. The volume is real data (about 0.3 to 1.5 GiB per prompt,
  more than a third of it checkpoints), not repeated writes; see audit/SSD-QUIET.md.

### Numerics modes: balance (default), precise, fast

- New `--precise` / `--balance` / `--fast` (or `--mode M`, env `WHIRL_MODE`) for `whirl` and
  `whirl-server`; one mode per process, reported in the start-up log and in `GET /props`
  (`"numerics"`). See [docs/usage.md](docs/usage.md#modes).
- **balance stays the default** when no mode is given (owner decision for 0.2.0); `--precise` and
  `--fast` are opt-in, `--balance` is still accepted, and `GET /props` `"numerics"` reports
  `"source"` (`default`, `command line` or `WHIRL_MODE`).
- What each mode is, in one place (the entries below give the details):

  | mode | KV format | activations / kernels | output guarantee |
  |---|---|---|---|
  | precise | f16 (every model, every GPU) | no fp8, f32 DeltaNet, no h16; f16 decode / verify activations | bitwise reproducible: the same bits with or without drafts, solo or batched; on the server the same bits for a prompt whatever the load, forward size or schedule-point resume; CLI == server except a CLI prompt of 4096·k + fewer than 64 tokens and a server system message of 2048+ tokens (kv-and-caching.md section 7) |
  | balance (default) | q8h on dense models, f16 on MoE models | dense: items `fp8`, `gdnwmma`, `h16`, `kvq8`; MoE: `gdnwmma`, `h16` (no `fp8` / `moefp8` since MOE-BAL); int8 decode / verify activations | near-lossless on dense models (KL vs precise about 1e-4 to 1e-3); on the MoE model about 1.9x the model's own noise floor (KL 0.002-0.014; MoE balance top-1 within 3.0 points of the noise floor's, MOE-BAL-3), MTP / n-gram == plain greedy; **not** bit for bit the 0.1.x output (q8v KV gone, KG-1 kernels, specsample sampler) |
  | fast | q4 (R9700 since FAST-1c, Radeon 8060S) | balance + `kvq4`, `relaxacc` (dense models with an MTP head) | lossy within the published KL; the text may depend on drafts |

- **precise** (`--precise`) on every GPU: no fp8 MXFP4 prefill activations (and no rounding of
  folded MXFP4 exponents), no fp8 MoE expert prefill, f32 DeltaNet prefill chunks instead of
  f16-WMMA, no f16 GEMM intermediates (h16), and **f16 KV everywhere** (no automatic q8v / q8h / q8,
  also on cards under 20 GiB and on the Radeon 8060S). When f16 KV does not fit, the context is
  lowered with a warning, or the program stops with a message when `--ctx` / `--ctx-per-slot` was
  given. MXFP4 prefill is slower in precise mode than in 0.1.x.
- **balance** uses the items `fp8`, `moefp8`, `gdnwmma`, `h16` and `kvq8` (`--balance=ITEMS` picks
  a subset). It is near-lossless, not bit for bit the 0.1.x output: the KV format is q8h on dense
  models (BAL-Q8, no q8v), dense models get the KG-1 kernels, and sampling uses specsample (see
  those entries). **fast** is balance plus the lossy items `kvq4` (4-bit KV) and `relaxacc`
  (relaxed MTP acceptance, dense models with an MTP head); where neither applies (e.g. MoE models
  on a GPU without q4 KV kernels) the log says "fast runs as balance".
- `whirl selftest`: the precise decode checks turn fp8 prefill, MoE fp8 and h16 off themselves
  (FIX-PS). Since balance became the default, selftest loaded in balance and its "== prefill GEMM
  rows" (MXFP4) and precise MoE n=17..32 checks compared against the fp8 prefill paths and
  failed; precise itself was unchanged (`--precise` selftest and the precise goldens passed). The
  log now prints the load mode.
- **KV format fixed per mode (BAL-Q8)**: precise f16; balance q8h on dense models (KL vs f16
  0.0002–0.002) and f16 on MoE models; fast q4 (Radeon 8060S, and the R9700 since FAST-1c). The same on every card, CLI and server (one function,
  `numerics::chooseKv`, decided before the kernels load). The automatic f16 → q8v → q8h fallback and
  the q8v preference on cards under 20 GiB are gone: a format that does not fit lowers the context
  (when not given) or stops with a message, never switches to a lower-precision format.
  `WHIRL_KV` remains as a debug override ([usage 7.7](docs/usage.md#env-debug)).
- **balance / fast on the R9700: WMMA DeltaNet and f16 GEMM outputs for every dense model (KG-1)**:
  `gdnwmma` and `h16` were MXFP4-only; they now also apply to dense models of any weight format
  (Q4_K_M, Q5_K, Q6_K, IQ4_XS, ...). Qwen3.8-27B Q4_K_M, balance, prefill 8k 1849 → 2042 tok/s
  (+10.4%), 32k 1649 → 1812 (+9.9%), 128k 1201 → 1285 (+7.0%); decode unchanged (96.98 → 97.15).
  KL vs precise over the last 128 rows of 128k prompts: code 3.1e-4 → 2.7e-4 (top-1 100% → 100%),
  zh 3.7e-4 → 4.1e-4 (99.22% → 98.44%, one near-tie row); balance before vs after 1.9e-4 / 4.6e-4.
  A fused quantized GEMM choice of the autotuner now runs as its bitwise twin (dequant + f16-out
  GEMM) when an f16 output is wanted, so which GEMMs write f16 never depends on tuning timings.
  Greedy output of balance / fast changes for these models (goldens re-recorded); precise, MoE
  models without MXFP4 experts and the Radeon 8060S are unchanged.
- **MoE verify batches over 16 rows on the decode experts (KG-2)**: `moeBlock` sends every batch up
  to the current verify width (32 inside a wide verify / MTP step) through the int8 decode experts,
  never the prefill grouped GEMM, so a 17–32-row verify computes each row exactly as its 1-token
  decode (selftest: balance MoE block, n = 2..32 bitwise == n = 1). New opt-in
  `WHIRL_MOE_WIDE=1` lets MoE models (balance / fast, R9700) use wide verify; it stays **off by
  default**: MoE verifies were already capped at 16 rows (one user never exceeds 16), and with it
  on, Ornith-1.5-35B-A3B MXFP4 at four users was slower (server_c4_nothink 386.0 → 379.4 tok/s,
  −1.7%; batch_c4 498.7 → 486.9, −2.4%; one user unchanged, 7 prompts 255.7 → 255.8) because each
  extra row reads its own experts. Output is the same either way (Ornith balance / fast, MTP +
  n-gram == plain, solo and concurrent, also with forced n-gram drafts); goldens unchanged.
- The per-item variables (`WHIRL_FP8`, `WHIRL_MOE_FP8`, `WHIRL_GDN_WMMA`, `WHIRL_FFN_H16`,
  `WHIRL_Q4_RELAXED`, `WHIRL_KV`) still work and override the mode (logged as user-requested).
- New balance / fast item `specsample` (server, MTP models): with temperature > 0 the MTP drafts are
  drawn from the draft head's distribution (same temperature / top-k / top-p / min-p), coupled to the
  target draw by a Gumbel-max / exponential race keyed by (seed, output position, token): the target
  token is argmin E(v) / p(v), the draft argmin E(v) / q(v) with the same E, and a draft is accepted
  iff it equals the target token. Sampling stays exact (same distribution as plain sampling), the
  acceptance is close to the optimum sum min(p, q), and the text for a seed depends only on the seed
  and the model's logits: the same with MTP on or off and whatever the draft counts, prefix cache or
  earlier requests. With specsample on (balance / fast, temperature > 0) the token sequence for a
  given seed differs from 0.2.0-rc's inverse-CDF sampler; precise (specsample off) keeps the
  0.2.0-rc sampler and its sequences exactly. Greedy decoding is unchanged in every mode.
  `WHIRL_SPEC_SAMPLE=0|1` overrides (0 = 0.2.0-rc sampler and greedy drafts). (An earlier rc2 build
  used the accept-min(1, p/q) / residual rule; its emitted tokens depended on draft counts and the
  draft head's state, so the same seed could give different text between requests.)
- precise decode / verify no longer quantizes activations to int8: dense matmuls take f16
  activations on the f16 GEMM's numerics (bitwise == the prefill GEMM rows, for every batch size up
  to 32), MoE experts take f32 activations with f32 accumulation. Every decode / verify batch up to
  32 rows (also a long n-gram draft) runs the same kernels, so MTP + n-gram == plain token for token
  on MoE models too. Cost vs balance on R9700 (plain decode): Ornith-1.5-35B-A3B about -29%,
  Swift-1.5 27B about -26%, Qwen3.8-27B Q4_K_M about -29%.
- Server KV tier (RAM / SSD prefix cache): the cache fingerprint now includes the numerics mode and
  its items. Before, a `--precise` server could restore a prompt's KV / DeltaNet state saved by a
  `--balance` run of the same exe and model when both kept f16 KV (MoE models), so its output
  depended on what an earlier process had cached.
- fast `kvq4` on the Radeon 8060S: new q4 decode / verify attention kernel `attn_dq4` (int8 q x 4-bit K
  integer dot, one scale multiply per 32 values, waves on independent key tiles, packed f16 P.V; ~180
  GB/s vs ~55 GB/s); Ornith-1.5-35B-A3B MXFP4 decode with q4 KV at 128k 36.5 -> 58.5 tok/s, at 64k
  48.0 -> 67.0 tok/s (f16 KV: 32.7 / 46.4); KL vs f16 KV the same as the previous kernel (code 128k 0.0137 vs 0.0133, 64k 0.0177 vs 0.0183). `WHIRL_ATTN_DQ4=0` restores
  the previous kernel. precise / balance unchanged.
- fast `kvq4` on the R9700 (FAST-1c): the gfx1201 code object has the q4 KV kernels (the 8060S
  format: 4-bit K / V, one f16 scale per 32 values, Hadamard-rotated q / k): `attn_prep_q4` /
  `kv_store_q4`, `attn_dq4` for decode / verify / MTP rows (ported; blocks ordered so the verify
  rows of a sequence read the same K / V tiles together through L2; ~430 GB/s at 128k for the 27B
  24/4-head shape), `attn_split_q4` / `attn_wsplit1/2_q4` / `attn_decode_q4` /
  `attn_prefill_wmma_q4` as generic fallbacks, and prefill through the Q8P path (keys dequantized
  into f16 rows, f16 `attn_kg`). `--fast` on the R9700 now uses q4 KV. Swift-1.5 27B MXFP4-A,
  `whirl bench --fast`, q4 vs `WHIRL_KV=f16`: decode at a 131k prompt plain 25.08 → 30.98 tok/s
  (+23.5%), MTP 63.12 → 68.79 (per verify cycle only −2.5%: each MTP row reads K / V, the f16
  verify reads it once for 5 rows; a block shared by the rows measured no better, attn_dq4 is
  ALU-bound per row); at a 67k prompt plain 30.42 → 34.08 (+12%), MTP 92.88 → 92.75; prefill 8k
  −1.1%, 128k −1.9%. KL of the prompt logits vs f16 KV (128 rows each): code 128k 0.0049 (top-1
  99.2%), zh 128k 0.0071 (93.0%; the 9 differing rows are near ties, top-1 p ≤ 0.53), code 64k
  0.0100 (95.3%), zh 64k 0.0016 (95.3%). R9700 fast goldens re-recorded; precise / balance and every
  8060S golden unchanged.
- fast: a repeated request gives the same text streamed or not (FIX-FS). The server gate's
  "stream == non-stream zh_think" failed in fast mode (also with f16 KV): the stream request came
  second and resumed from the prompt's think-open checkpoint, and the resume ran the MTP row that
  pairs the checkpoint's last hidden with the next token as a one-row MTP forward, while the cold
  prefill had run that row inside the previous chunk's batch. One-row MTP forwards (q8
  activations, fused kernels) are not bit-identical to the same row in a larger batch, so the MTP
  KV and then the drafts differed; exact acceptance (precise / balance, fast with `WHIRL_RELAX=0`)
  never shows drafts in the text, `relaxacc` keeps them. Every MTP prefill (CLI and server, solo
  and grouped chunks) now runs that chunk-boundary row alone at the start of the next chunk, as a
  resume does, so a cold prefill and a resume from its own checkpoints are bit-identical. Cost:
  one one-row MTP forward per prefill chunk boundary. Text unchanged for precise / balance (exact
  acceptance) and for every golden (gfx1201 and gfx1151, fast included). A prompt resumed from
  another prompt's shared prefix is still prefilled in other chunks than cold and can give
  another fast text (documented, like concurrency). New server-gate suite `fast_stream` (fast,
  relaxacc forced on, also MoE): cold reference server without the prefix cache vs stream /
  non-stream repeats.
- q8 / q8h (and q4) prefill past the f16 scratch (FAST-1c): a prefill whose keys exceed what
  `ffn_g` / `ffn_u` hold (~139k at batch 4096 on the 27B) fell back to `attn_kg_q8` (+29% at 256k
  keys); now the keys go in ranges of at most that size, each dequantized (`kv_dq_rows_r`) and
  walked by `attn_kgs` with the softmax state (m, l, unnormalized output) carried from range to
  range — bit-identical to one f16 `attn_kg` over all keys, so q8h output is unchanged (it was
  already bit-identical to `attn_kg_q8`). Kernel bench, 4096 queries at 256k keys: f16 290.1 ms,
  ranges 294.5 ms (+1.5%), `attn_kg_q8` 373 ms. End to end, Ornith-1.5-9B balance q8h vs f16: 192k
  −1.4%, 256k −1.4% (same output hash); Swift-1.5 27B q8h 256k 1,129 tok/s (f16 KV at 256k does
  not fit 32 GB next to its weights). Prefills within the scratch are unchanged.

### Fixes from the external review (FIX-REVIEW2)

Error-path, guard and dispatch-condition fixes; outputs are bit-identical (golden gfx1201 12/12 and
selftest in every mode unchanged).

- MTP draft attention window (`Model::attn_win`) is restored on every exit path; a throwing draft
  step no longer leaves the window on for the next request's trunk attention (long contexts only,
  OVH-C1).
- `planFit` never shrinks the prefill batch below 1280 rows (one 1024 chunk + the 256-row merged
  tail); 1024 / 512 started the server and then failed long prompts with `BatchTooLarge` (VRAM-01).
- SSD tier: a file shorter than its extent is deleted at start-up, and a read job that gets EOF or
  a short read fails (cold prefill) instead of restoring stale bytes as valid KV (MAIN-001). A
  failed directory lock now leaves the SSD tier off instead of running unlocked (any error other
  than another server holding it). Block dedup can no longer create an A <-> B reference cycle
  between two entries (rewriting a parent to a fork's content), and a cycle already on disk no
  longer recurses until the stack overflows (CACHE1-002).
- An exception while copying a restore (`pumpRestore` / `pumpTail`) marks the restore failed
  instead of treating it as done and valid (RESTORE-EXC).
- Models with DeltaNet layers must have `ssm.conv_kernel` 4, `ssm.state_size` 128 and a value head
  size that is a multiple of 32, otherwise `UnsupportedConfig` before any GPU work (the kernels
  do not implement other shapes, GDN-CFG).
- The pre-load VRAM estimate counts the 256 attention splits of q4 KV, `dq_st_ml`, the dq page
  table and `moe_tiles64`; `Model::load` notes when the estimate is below its buffer allocations
  (KV-FIT-ESTIMATE).
- f32 / f16 weight matrices keep the bitwise-equal-to-n=1 GEMV family in a 17..32-row wide verify
  and without the multi-token GEMV instances (RC-R1 / PREFILL-R2; none of the shipped models has
  such matrices).
- n-gram draft timing: a single measured point no longer extrapolates to a zero or negative time
  (policy only, MTP-01).
- Golden gate: empty model / mode / suite lists, runs without `G` lines, empty references, an
  unrecorded model file and a suite `ngram` that never drafts are errors; every `WHIRL_*`
  variable is cleared for the runs; new suite `ngram` (in the default list for both gfx1201 and
  gfx1151). `fast_stream` of the server gate now requires MTP on and `cached_tokens > 0` for the
  repeats (FS-001). A reference or run with no step line, and a `record` in which a case failed
  (nothing is written then, so references and `models.txt` never disagree), are errors too.
- Server gate: no `--suite`, or a suite that ran no check, is an error (exit 2); every "A == B"
  check requires HTTP 200 and a non-empty reply on both sides (two failed requests no longer compare
  equal).
- `WHIRL_PREFILL_BATCH` below 1280 is raised to 1280 with a warning (a smaller batch failed every
  long prompt with BatchTooLarge).
- SSD / RAM tier review fixes (FIX-RC4-A, FIX-HDR0): in `always` mode with both RAM and SSD
  tiers full, restoring or prefetching an SSD-only session could delete that same session while it
  was being restored (use after free); the entry is now held for the whole restore. An exception in
  the first copy of a restore no longer leaks the restore and pins the entry. Sessions used from the
  RAM tier record that use on their SSD file (a graceful stop, hourly in `always` mode), so they no
  longer expire at the next start. `shutdown` mode really writes nothing while running. A file that
  cannot be deleted still counts against `--kv-ssd-gb`. Rewriting an existing SSD file now flushes
  the invalidated header before overwriting the data, so a power loss can leave a cold entry but
  never an old valid header over half-new data. Asynchronous GPU copy errors fail the restore
  (cold prefill) instead of counting as done. `--help` and the docs now give the real defaults
  (`shutdown`, `--kv-ssd-delay`).

### Fixes from the numerics and cache reviews (FIX-NUM, FIX-CACHE)

- One head-norm reduction order for every logits row: the plain decode head and the prompt-end head
  used `rmsnormQ8` (1024-thread sum) while verify / all-logits rows used `rmsnorm`; all now use
  `rmsnorm`, so CLI plain, CLI MTP / n-gram verify and server batched rows give the same logits
  bits (`whirl seqtest` checks it). Tokens are unchanged in every golden case; only the logits
  hashes (`lh=`) of the affected steps changed and the golden references were re-recorded for that
  reason only. Server decode rows are unchanged; the first token after a prompt can change only
  on a near tie.
- The SSD tier fingerprint includes the GPU (arch, device name, kernel code object), so one exe
  on the R9700 and on the Radeon 8060S never reuses the other's KV files; KV format, batch and
  exe id are covered by tests. The 12 numerics-changing A/B variables are reported as overrides
  (`MODE+overrides`), and `relaxacc` is reported as skipped without an MTP head or with
  `--draft dflash`.
- SSD tier: with a cap smaller than the RAM entries the tier no longer rewrites forever or
  overshoots `--kv-ssd-gb` (a write that cannot fit is skipped until space is freed). Files written
  with another per-slot context (`tok_cap`) are kept and counted instead of deleted as corrupt.
  `WHIRL_SERVE_CKPTS` is limited to 8 (above it the server crashed). A new conversation that
  shares only the system prompt (a compact, or a subagent) no longer cuts the idle long session
  with the same system prompt: it is demoted (evicted first under pressure) instead. Restoring
  part of an entry and then branching from an earlier turn makes a new entry instead of
  overwriting the other branch in place. On integrated GPUs the RAM tier is at least one
  max-context entry (or off, with a log line, when that is more than half the available RAM).

### Test gates for the chunking and KL claims (FIX-TEST-B)

- The chunking guarantee is now stated as measured (docs/guide/en/kv-and-caching.md section 7): on
  the server a prompt always gets the same bits, whatever the forward size (decode floor,
  `WHIRL_PREFILL_CHUNK`, `WHIRL_PREFILL_BATCH`), the tail checkpoint, or a resume / tier restore at a
  schedule point. A split that is not a schedule point (not a multiple of 64 rows, or leaving a
  forward of fewer than 64 rows) changes the last bits, so the CLI equals the server except when its
  last 4096-row forward is shorter than 64 rows (and for server system messages of 2048+ tokens).
  The earlier "batch 1024 / 2048 / 4096 bit-identical" was true only for the server schedule.
- `whirl seqtest` gates it (part of `ok`): a 9,500-token golden prompt run through the server's real
  schedule (`include/whirl/prefill_schedule.h`, now shared by the engine and seqtest) in 6 executions
  the server can choose must give the cold run's bits; R9700 dense + MoE x precise / balance / fast
  pass. whirl-server-tests `prefill_sched` checks the schedule functions on the CPU.
- KL gate `tools/klgate/klgate.py`: balance and fast against precise on a fixed code / Chinese corpus
  at 8k, 32k, 96k and 128k; limits balance KL <= 0.005 and top-1 >= 98%, fast KL <= 0.03 and top-1
  >= 93%; non-zero exit on a failure or 0 cases. `tools/golden/golden.ps1 record` refuses to write
  references without a passing KL report of the same exe (`-KlReport`). `WHIRL_DUMP_AT` (debug)
  dumps logits windows before several positions in one run.
- First KL gate run (R9700, Qwen3.8-27B Q4_K_M + Ornith-1.5-35B-A3B MXFP4): **fails, 22 of 32 cases
  pass**. Dense balance passes everywhere (KL 7e-5 to 4.9e-4, top-1 >= 98.4%); dense fast passes
  except zh at 8k (top-1 92.97%, KL 0.0051). Ornith balance exceeds the balance limit in 6 of 8 cases
  (KL up to 0.020, top-1 down to 91.4%) and Ornith fast in 4 of 8 (zh KL up to 0.037, top-1 88.3%):
  the published balance KL range was a dense-model number. The limits stay as published until a
  decision (per-model limits or a numerics change); until then golden references cannot be
  re-recorded for these models (record requires a passing report).

### Fixes from the DFlash2 review (FIX-DFL)

- The drafter GGUF is checked tensor by tensor against its metadata before anything is loaded
  (q/k/v/o, gate/up/down, conv_proj / conv_base, selector_hidden, norms, 1..64 layers): a malformed
  drafter could make the GPU write past its scratch buffers; it now falls back to MTP with the reason.
  An MXFP4 `fc.weight` is refused (the drafter splits fc into column blocks).
- `whirl chat` / `bench` with `--draft dflash`: `WHIRL_DFLASH_CAPTURE_ROWS` below the verify rows
  made the drafter read far outside the capture buffer (GPU page fault); the ingest now goes through
  the same plan as the server and values below 16 are ignored with a warning.
- `--draft-model` that cannot be opened or parsed now warns and keeps MTP, as documented (it stopped
  the program).
- whirl-server: the VRAM fit check counts the DFlash2 drafter (and no longer the MTP block it does not
  load); after prefix reuse the drafter context keeps only rows its ring still holds (a slot that had
  run far past the reuse point drafted against an empty context and the cold-cycle count missed it);
  relaxacc's "MTP drafts only" rule has one implementation, and the regression test now fails if
  relaxacc is applied to DFlash2 drafts; the replay-less snapshot-set decision is a tested function.

### Fixes from the server review (FIX-SRV)

- **Behaviour change:** cross-origin requests are refused. A non-GET request with an Origin that is
  not allowed (including `null` and `text/plain` "simple" requests) gets 403 before the body is
  read, and the `Host` header must be `localhost`, `127.0.0.1`, `[::1]`, the `--host` address
  (any IP literal when bound to `0.0.0.0` / `::`) or a name given with `--allow-host NAME`
  (repeatable, `*` = any); requests without a Host header are allowed. Clients that reach the
  server by a host name (for example `mypc.lan`) need `--allow-host`. `--allow-local-images`
  reads local files only for requests from a localhost page.
- A tool call cut off by `max_tokens` or a stop string is no longer returned as a complete call
  (`finish_reason` stays `length`, the text is returned as is).
- A client that disconnects frees its slot while queued, prefilling or in a non-streaming
  request. Pressing the KV pool with parallel prefills queues the prompt instead of failing with
  500; the remaining pool-full case is 503 with `Retry-After`. A decode stopped by a full pool
  keeps `finish_reason: "length"` and adds `"finish_cause": "kv_pool_exhausted"`.
- `--draft dflash` on GPUs without DeltaNet replay kernels (Radeon 8060S) allocates snapshot sets
  so DFlash2 / n-gram drafts are made (before: no drafts); the batch log reports literal drafts
  per cycle.
- Smaller fixes: requests during shutdown get 503; an invalid `--kv-ssd-mode` is a usage error;
  streamed text is always a prefix of the final UTF-8-sanitized text; temperature below 1e-3 is
  greedy; `/props` reads only atomic counters; at most 16 stop strings of up to 256 bytes;
  `completion_tokens` no longer counts tokens after a stop string; CRLF n-gram normalization also
  under `--draft dflash`; bad numeric environment variables fall back to the default with a
  warning; control characters in logged request lines are escaped; a short follow-up prompt is
  no longer held behind a long prefill.

## 0.1.4 — 2026-10-06

Routine bug-fix release: no new features, no speed changes. The changes harden input handling (GGUF
files, HTTP requests, tokenizer vocabularies); inference code paths are unchanged, so outputs are
bit-identical to 0.1.3.

### Fixes

- Server (HTTP): a streaming client that stops reading no longer stalls the other slots. While a
  request runs, the engine writes its response through a non-blocking socket; bytes the client does
  not accept stay buffered and are retried on the next flush. With no progress for 30 s, or more than
  64 MiB unsent, the connection counts as gone and the request is dropped (its slot is freed). The
  rest of a finished response is sent by the connection thread (a10ac5b).
- Server: shutdown no longer waits up to 30 s for connections still reading a request; `stop()`
  shuts them down and cancels the pending reads. Connections close with a linger so clients receive
  error responses instead of a reset (a10ac5b).
- Tokenizer: `piece()` decodes `<0xNN>` byte tokens with a small hex parser instead of `std::stoi`,
  which threw on `<0xZZ>` and silently mis-decoded `<0x-1>` / `<0x1Z>` (24475c4).
- zh-TW guide: bold runs broken by CJK punctuation render correctly again (no text change, 8b2c58d).

### Security hardening

- GGUF parser: tensor extent checks are overflow-safe (subtraction-based `File::inBounds`, at parse
  and in `tensorData()`), so `data_offset + offset + nbytes` can no longer wrap past 2^64; dimensions
  above INT64_MAX and shapes whose row count, element count or byte size overflow are rejected;
  `TensorInfo::nbytes()` / `rowBytes()` are overflow-checked. The header's KV and tensor counts must
  fit in the remaining file bytes, so their `reserve()` is bounded by the file size (b6af30a).
- Model load: `Config::fromGguf` rejects inconsistent or out-of-range hyper-parameters (for example
  `nextn >= block_count`, zero dimensions, `head_dim > 256`, GQA group > 8,
  `value_length != key_length`, interval 0) with `UnsupportedConfig`; `Config::validateTensors`
  checks the type and shape of every tensor `Model::load` reads (including MoE experts and the MTP
  block) before any GPU allocation, with the new error codes `MissingTensor`, `UnsupportedTensorType`
  and `UnsupportedTensorShape` (b6af30a).
- HTTP: 30 s idle timeout per read and a 60 s deadline for the request line and headers (`408`); at
  most 64 open connections (`503`, then close); headers capped at 64 KiB / 100 lines (`431`);
  `Content-Length` above 64 MiB is refused (`413`) instead of wrapping, conflicting duplicate
  `Content-Length` headers get `400`; the body buffer grows as data arrives instead of reserving the
  claimed length (a10ac5b).
- CORS: `Access-Control-Allow-Origin` is sent only to pages from `http(s)://localhost`, `127.0.0.1`
  or `[::1]` (any port), echoing the `Origin` with `Vary: Origin`; other origins (including `null`,
  i.e. `file://` pages) get no CORS headers, and preflight answers the same way. New option
  `--cors-origin ORIGIN` (repeatable; `*` restores the previous allow-all behaviour) (a10ac5b,
  docs 765f49a).
- Tokenizer: a byte-type token whose text is not exactly `<0xNN>` (two hex digits) fails loading with
  `TokenizerError("malformed byte token at id N")` (24475c4).

### Behaviour changes

- CORS default tightened (see above). Browser pages on other origins, or opened from `file://`, can no
  longer read the server's responses unless started with `--cors-origin <origin>` (or
  `--cors-origin "*"`). SDKs, curl, IDE agents and other non-browser clients are not affected.
- HTTP limits: 64 connections, 30 s read idle / 60 s header deadline, 30 s send stall or 64 MiB
  unsent on a streaming response, 64 KiB / 100 header lines, 64 MiB request body.
- Malformed GGUF files that 0.1.3 might have loaded (or crashed on) are now refused at load with one
  of the error codes above; the release front-end reports them as "cannot be loaded".

### Internal

- Version 0.1.4 (`project(whirl VERSION 0.1.4)`).
- MXFP4: the loader and the tests share one row-repack implementation (`kernels::repackMxfp4Row`,
  new scratch-buffer overload; scratch still allocated once per tensor); model tests pin the bytes
  against frozen copies of both previous implementations (ee87049).
- `build.bat` finds `vcvars64.bat` with `vswhere`, restricted to Visual Studio 2022
  (`-version [17.0,18.0)`; HIP clang with the VS 2026 STL is unverified), any edition with the C++ x64
  tools; it falls back to the default Build Tools path and prints a clear error if neither exists
  (f54e2da).
- Tests: crafted malformed GGUFs, synthetic tiny models with mutations (optional
  `WHIRL_TEST_GGUF="a.gguf;b.gguf"` runs the load validation on real files without a GPU), HTTP limits,
  connection cap, stuck reader, CORS matrix, byte-token validation.

## 0.1.3 — 2026-10-05

Every change below keeps the output bit-identical to plain greedy decoding, unless it says otherwise.

### Performance

- Prefill attention: new GQA-grouped kernel (`attn_kg`) for f16, q8v, q8 and q8h KV. One block holds the
  query heads of one KV head over 16 queries; K fragments are loaded straight into WMMA registers,
  f16 Vᵀ fragments with RDNA 4's transposing load (`global_load_tr_b128`), q8v V is dequantized once
  per block into a double-buffered LDS stage, and there is one barrier per 32-key tile. Outputs are
  bit-identical to the previous kernel (kernel-test invariants; last-token logits, MTP == plain and
  server outputs identical to 0.1.2). Kernel throughput at long context: f16 ~65 → ~93 TFLOPS, q8v
  ~61 → ~84 TFLOPS. `whirl bench` prefill, Swift-1.5 27B MXFP4-A (f16 KV, R9700): 32k +10%, 64k
  +17%, 96k +21%, 128k +24%; Ornith MXFP4 64k +11%, 128k +14%; Qwen3.8-27B UD-Q4_K_M (f16 KV) 32k
  +5%, 128k +16%. Long prompts on the server (q8v KV for the dense models): Swift 96k +19%, 128k +21%.
  q8 and q8h KV use `attn_kg` too (`attn_kg6_q8` / `attn_kg4_q8`; K dequantized from int8 with a
  magic-number f16 conversion, bit-identical to `attn_kx_q8`; probe at 61k 57.8 → 72.1 TFLOPS). Server
  prefill, Swift-1.5 27B MXFP4-A, one user, `--ctx-per-slot 262144`, q8h KV: 128k 1,342 → 1,522
  (+13.4%), 192k 1,021 → 1,177 (+15.3%), 256k 824 → 961 (+16.6%). `WHIRL_ATTN_KG=0` restores the
  previous kernel. See [kernels.md](docs/guide/en/kernels.md#flash).
- Radeon 8060S: q8 / q8h / q4 KV prefill attention through dequantized f16 rows (KG-3, the gfx1151
  port of Q8P / FC-1c). `kv_dq_rows_r` / `kv_dq_rows_r_q4` dequantize the sequence's keys once per
  layer and chunk into `ffn_g` / `ffn_u` and the f16 `attn_prefill_wmma` runs on them; keys past the
  scratch (139k at batch 4096 on the 27B) go in ranges through the new `attn_prefill_wmma_s` with the
  softmax state carried. Bit-identical to `attn_prefill_wmma_q8` / `_q4` (kernel test: 180 new
  invariants; 16 greedy tokens after 64k / 128k prompts identical; bench hash identical), so goldens
  are unchanged. Speed, Qwen3.8-27B UD-Q4_K_M, end to end: within the run-to-run drift of the 8060S
  (about ±3% between consecutive 4-minute runs). balance q8h vs `WHIRL_KV=f16`: prefill 8k 374.2 vs
  377.5 tok/s (−0.9%; old loader 369.5, −2.1%); 64k, two runs in opposite orders, 279.6 / 279.7 vs
  288.3 / 277.0 (mean −1.1%; old loader 272.8 / 280.2, −2.2%); 128k, two runs in opposite orders,
  212.1 / 221.9 vs 221.0 / 213.3 (mean −0.1%; whichever ran first was ~4% faster). fast q4 at 128k 211.9 vs the old loader 214.6 (−1.3%, noise). The kernel
  bench (4096 queries, 24 / 4 heads, random K / V) disagrees with the end-to-end runs: there the f16
  kernel on f16 data was slower than the int8 loader (64k 490–523 vs 388–425 ms) and key ranges of
  16k–32k keys cut f16 by 26–35%, but `WHIRL_ATTN_SEG=32768` end to end changed nothing (64k q8h
  279.9, f16 279.4, same text), so the bench is not representative on this GPU. `WHIRL_ATTN_DQF=0`
  restores the old loaders; `WHIRL_ATTN_SEG=N` (new, test knob) forces key ranges, f16 KV included.
- **q8 / q8h KV prefill attention at f16 speed (Q8P)**, so balance can default to q8h on dense
  models: per layer and prefill chunk, `kv_dq_rows` dequantizes the sequence's keys once into f16 rows
  (in `ffn_g` / `ffn_u`, idle during attention; up to 139k keys at batch 4096) and the f16 `attn_kg`
  runs on them, bit-identical to `attn_kg_q8` (which converts K once per query head and was 29%
  slower than f16 at 128k keys). R9700, balance, q8h vs `WHIRL_KV=f16`: Swift-1.5 27B MXFP4-A prefill
  8k −1.5%, 32k −0.9%, 128k −1.0% (before −3.3 / −7.7 / −15.1%), decode_7p +1.7%; Qwen3.8-27B
  UD-Q4_K_M 8k −0.5%, 32k −0.4%, 128k −0.7%, decode_7p +0.4%. Radeon 8060S (unchanged kernels):
  Q4_K_M prefill 8k −0.8%, 64k −2.7%, decode at 64k −1.9%. `WHIRL_ATTN_DQF=0` restores the int8
  kernel. Goldens unchanged on gfx1201; gfx1151 dense balance re-recorded for the q8h default.
- Speculative verify attention: a sequence's verify rows share one K/V pass in groups of up to 32
  query columns (`attn_wsplit2`; 27B: 5 rows × 6 GQA heads) instead of 16 (2 rows). The second column
  group runs on the block's otherwise idle second wave over the same staged Vᵀ tile; each column's
  arithmetic is unchanged (checked by `checkAttnGroups`, the kernel test and server A/B against
  0.1.2). Attention kernel time per layer at 16k context: one user with 4 drafts 0.286 → 0.141 ms,
  four users with 3 drafts each 0.853 → 0.607 ms; at 32k, 1.784 → 1.162 ms. Single-row decode keeps
  `attn_wsplit1`. `WHIRL_ATTN_WIDE=0` restores the old grouping.
- Wide verify: verify forwards of 17–32 rows use dual-token WMMA GEMV kernels (`gemvx_v6` for MXFP4,
  IQ4_XS, Q4_K, Q5_K, Q6_K) and the output head covers the whole vocabulary per chunk
  (`head_chunk = 248320`; `WHIRL_WIDE_VERIFY=0`, `WHIRL_HEAD_CHUNK=N`). Only several users (or n-gram
  drafts adding rows) take this path; one user's verify has at most 16 rows. 4 users, prose:
  247.1 → 267.8 tok/s. Default draft caps stay 8 / 7 / 4 / 3 for 1–4 decoding slots.
- MTP block stored as Q8_0 (e.g. the Swift-1.5 MXFP4 files) is requantized to Q4_K for drafting, like
  Q6_K blocks already were (`WHIRL_MTP_Q4=0` keeps the file's types). Drafts only.
- Server: Windows timer resolution set to 1 ms while serving (`timeBeginPeriod`), and the main loop's
  idle / burst-gather / restore waits wake on a new request instead of sleeping: a 1 ms sleep took
  11.5 ms before, 1.9 ms after (`WHIRL_TIMER_PROBE=1`). A short request arriving while the
  prefix-cache tier is still writing the previous one: TTFT median 86 → 69 ms. C = 4 burst of ~1.1k
  prompts: mean TTFT 1207 → 1187 ms.
- Server: with the decode floor off (`--decode-min-tps 0`), a prefilling request's chunks merge into
  2048-row forwards even while other slots decode. With the floor on they still do not while any slot
  decodes (merging next to a long prompt made its prefill only ~4% faster but doubled the decoders'
  stalls, 96k: 1.2 → 3.3 s).

### Fixes

- Server KV pool sized under the WDDM budget, MTP KV allocated first (FIX-1). The automatic pool kept
  only 768 MiB of free VRAM, which put the process over its WDDM local budget (31.29 / 31.02 GiB on
  the R9700); Windows then demoted the last allocation, the MTP KV, to system memory and the draft
  head read it across PCIe: a ~128k-token agent request decoded at 29.0 tok/s. The pool is now
  min(free VRAM − reserve, WDDM budget − usage − margin) (`hip::wddmMemInfo()`, DXGI
  `IDXGIAdapter3::QueryVideoMemoryInfo`, adapter matched by LUID; margin 768 MiB, MoE 1.5 GiB,
  `WHIRL_POOL_BUDGET_MARGIN_MB`; `WHIRL_POOL_RESERVE_MB` keeps the old free-VRAM-only rule), and
  `allocKvPool` allocates the MTP KV before the trunk layers. After startup the server logs WDDM local
  usage / budget and non-local growth, and warns when over budget. Swift-1.5 27B MXFP4-A, default 4
  slots: pool 226,304 tokens (v0.1.2: 210,688; +7.4%), local 30.06 / 31.02 GiB; the 128k agent request
  decodes at 67.2 tok/s, output identical. Server tests 340/340; gates basic / tier / restore_conc
  identical to v0.1.2.
- Decode floor (`--decode-min-tps N`): every decoding slot is protected except one that arrived in
  the same burst (within the 30 ms gathering window) as a request that is still prefilling. Requests
  arriving together therefore merge their prefill without false floor throttling (Scenario A, C = 4
  concurrent ~1.1k prompts: 0 floor forwards, batch wall 4.65 s vs 4.91 s in v0.1.2, median of 3),
  while a stream is protected regardless of arrival order. An earlier version of this change protected
  only slots that arrived before every prefilling request, which left a conversation's next turn
  unprotected when it arrived after its subagents' prompts ("reverse order", 25.7k: 8.6 tok/s while
  they prefilled; now 22.8–23.0, v0.1.2 22.4); the server test `decode_floor` now covers that order.
  Scenario B (Swift 27B MXFP4-A, 3 concurrent ~17k subagents) at 25.7k context is unchanged
  (22.7 tok/s, longest pause 0.47 s; v0.1.2 23.1 / 0.47). Outputs are identical to v0.1.2.
- CLI: n-gram drafts are capped at 1 on the unfused DeltaNet decode path (for example F32 / F16
  `ssm_alpha` / `ssm_beta` without `gdn_ab`), like the MTP draft count and the server already were.
  That path writes only snapshot 0, so accepting 1..nd−1 drafts restored a never-written snapshot and
  diverged from greedy. `restoreSnapshot` now throws `SnapshotNotWritten` for out-of-range or
  unwritten sets (42b4cc9).

### Defaults

- The token embedding table is kept in pinned host memory (`WHIRL_EMBD_HOST`, now on by default;
  `WHIRL_EMBD_HOST=0` keeps it in VRAM). The KV pool grows by ~11.9% on the R9700, prefill −0.4–0.9%.
  Models with tied embeddings or an MXFP4 embedding keep it in VRAM. The server adds the embedding
  size to its pinned-memory declaration (`%LOCALAPPDATA%\whirl\pinned\<pid>.txt`), so Shared Usage
  monitoring does not count it as spill.
- The MTP draft head uses a 64k-token vocabulary subset (`WHIRL_DRAFT_VOCAB`, default `64k`),
  embedded in the executable (`data/draft_vocab/subset_64k.bin` via whirl-bin2c), for dense qwen35
  models with the 2-bit draft head and a 248,320-token vocabulary; MoE models and other vocabularies
  keep the full head. Main scenario +3.4–5.0% tok/s, same acceptance. `WHIRL_DRAFT_VOCAB=off|48k|<file>`
  overrides. The subset is only a list of token ids (uint32 LE, ascending), ranked by token frequency
  over permissively licensed code and documentation (llama.cpp, ROCm aiter, hipfire, dflash,
  PaddleNLP / PaddleOCR, ECharts, WHIRL docs) and Wikipedia samples (CC BY-SA 4.0), tokenized with the
  Qwen3.x tokenizer; no corpus text is included. Generated by whirl-cloud `tools/vocab_subset/build.py`
  (sources and licenses in its `sources.tsv`).
- MTP draft attention window on by default (DEF-2, 2ba0713): from a 64k-token context on, the draft
  attention sees only the first 256 and the last 16,384 positions. Trunk and verification always see
  the whole context, so output hashes are identical. Window W = 16,384 against off: typical case
  (real agent sessions, server) 110k +1.1%, 128k +7.6%, 200k +3.7%; worst case (long non-repeating
  text) 128k −0.5%, 256k +17% (single run); best case (repeated editing) 128k +3.1%, 256k +4.2%.
  `WHIRL_DRAFT_WINDOW=0` turns it off; `WHIRL_DRAFT_WINDOW=W` / `WHIRL_DRAFT_WINDOW_MIN=N` change the
  window and the threshold.

### Internal

- Version 0.1.3 (`project(whirl VERSION 0.1.3)`, 8fbee5f).
- `package_release`: `dxgi.dll` (WDDM budget query, FIX-1) and `winmm.dll` (`timeBeginPeriod`) added
  to the allowed system DLLs; both ship with every Windows 10/11 install, so requirements are
  unchanged (3b1bebc).
- `whirl bench`: `WHIRL_PROFILE=1` prints the per-op-class GPU time of each prefill size.

## 0.1.2 — 2026-10-03

### Changed

- Server: the pinned-RAM tier of the prefix cache (`--kv-ram-mb`) is now sized from the machine
  instead of a fixed max(8 GiB, one full-length session + checkpoints) ≈ 9 GiB: **1/4 of physical
  RAM** (nearest GiB), at least that old minimum, at most 32 GiB (the minimum wins over the cap), and
  never more than half of the RAM available at startup (so the pinned arena does not push the machine
  into paging; logged as a warning when it limits the size; under 1 GiB the tier stays off). A 64 GB
  PC now gets 16 GiB. In real agent use (Hermes, 30–40k-token sessions with subagents) the old 9 GiB
  filled within an hour, after which restores came from the SSD. `--kv-ram-mb N` /
  `WHIRL_KV_RAM_MB=N` still override (used as given), `0` still turns the host tiers off. On an
  integrated GPU (shared system memory) the RAM tier is off unless a size is given. The startup log
  prints the chosen size and the reason, and the pinning time.

## 0.1.1 — 2026-10-03

### Added

- Server: read-only compatibility endpoints for clients that auto-detect the server type and context
  length (agent frameworks, chat front ends, IDE plugins):
  - `GET /props` and `GET /v1/props` — llama.cpp-server-style subset: `default_generation_settings`
    (`n_ctx` = context per slot, `model`), `total_slots`, `model_path` (file name only), `model_alias`,
    `modalities`, `build_info`.
  - `GET /version` — `{"version":"0.1.1","name":"whirl"}`.

- Chat requests: the OpenRouter / OpenAI Responses-style `"reasoning": {"effort": "...", "enabled": bool}`
  object, and effort aliases: `max` / `ultra` / `xhigh` / `high` → xhigh, `medium` → medium,
  `low` / `minimal` → low, `none` → thinking off (values are case-insensitive).

- Server: decode floor, `--decode-min-tps N` (env `WHIRL_DECODE_MIN_TPS`, default 20, `0` = off).
  While other requests prefill long prompts, every streaming request keeps at least N tok/s: prefill
  forwards are limited to a row budget (whole chunks; the oldest request always progresses) and decode
  cycles are interleaved, adapted every cycle from the measured decode-cycle time, tokens per cycle and
  prefill rate. Swift-1.5 27B MXFP4 on the R9700, one stream at 25.7k context + three ~17k-token prompts:
  the stream goes from 3.3 to 23.0 tok/s (longest pause 1.22 → 0.47 s), prefill throughput meanwhile
  2791 → 1862 tok/s, mean TTFT of the three 17.8 → 18.6 s. With no slot decoding, prefill is unchanged;
  outputs are bit-identical for any N; short-prompt C = 1 / C = 4 throughput unchanged.

### Changed

- Chat requests: an unknown reasoning effort no longer fails the request with 400; the server logs a
  warning and uses the default effort. Prompts for the values accepted before are unchanged.
- Server: probes of LM Studio / Ollama native paths (`/api/v1/models`, `/api/tags`, `/api/show`,
  `/api/version`, `/api/v0/models`) still answer 404 but are logged once per path at `I` level
  instead of a `W` line per request.

## 0.1.0 — 2026-10-03

- First public release: native Windows C++/HIP inference engine for qwen35 / qwen35moe GGUF models on
  the Radeon AI PRO R9700 (gfx1201) — `whirl.exe` (CLI) and `whirl-server.exe` (OpenAI-compatible
  server with continuous batching, prefix cache, RAM / SSD KV tiers, MTP and n-gram speculative
  decoding, image input).
