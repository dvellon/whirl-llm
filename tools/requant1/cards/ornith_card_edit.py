"""REQUANT-1: apply the 2026-10-19 rebuild edits to the Ornith-1.5-35B-A3B MXFP4 card (input: the LIVE README, LF).
Usage: python -I ornith_card_edit.py README.md CHECKS_EN_FILE CHECKS_ZH_FILE
  CHECKS_EN_FILE / CHECKS_ZH_FILE: bullet lines (UTF-8, each line starts with "- ") for the WHIRL checks and speed
  of the rebuilt file, inserted under "Other checks" / into the zh summary."""
import sys
p = sys.argv[1]
checks_en = open(sys.argv[2], encoding="utf-8").read().strip()
checks_zh = open(sys.argv[3], encoding="utf-8").read().strip()
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


NEW_SIZE, NEW_SHA = "19,819,767,584", "8fa77518492831be0c5487fecbb0a04ca10d60d9c342235c59361af4f8cef07a"

# 1. intro: rebuild note
rep("Radeon 8060S) and Radeon AI PRO R9700.\n\n> **Unofficial community quantization.**",
    "Radeon 8060S) and Radeon AI PRO R9700.\n\n"
    "**Rebuilt 2026-10-19:** the language-model file was re-quantized with an **imatrix-weighted MXFP4 quantizer** and an "
    "importance matrix computed for this exact checkpoint (same tensor layout, same file name). Against the BF16 model its KL "
    "divergence is 19% lower than the first release (2026-10-03) and its perplexity is lower too; see [Quality](#quality). "
    "The vision projector is unchanged.\n\n"
    "> **Unofficial community quantization.**")
rep("(the only quality measurement here is a perplexity comparison between quantizations, see [Quality](#quality)).",
    "(the only quality measurements here are KL divergence and perplexity against the BF16 model, see [Quality](#quality)).")
# 2. llama.cpp tested-with note in the WHIRL box
rep("> The files are standard GGUF and **also run in llama.cpp**: tested with llama.cpp **b11214** (ROCm build) — `llama-bench`, `llama-server` (plain and `--spec-type draft-mtp`) and `llama-mtmd-cli` with the projector. Other llama.cpp builds and other GGUF runtimes were not tested.",
    "> The files are standard GGUF and **also run in llama.cpp**: the first release was tested with llama.cpp **b11214** (ROCm build) — `llama-bench`, `llama-server` (plain and `--spec-type draft-mtp`) and `llama-mtmd-cli` with the projector; the 2026-10-19 rebuild with **b11719** (CPU `llama-perplexity`; Vulkan `llama-server` with `--spec-type draft-mtp` and the projector). Other llama.cpp builds and other GGUF runtimes were not tested.")
# 3. files table
rep("| `Ornith-1.5-35B-A3B-MXFP4.gguf` | language model + MTP layer (MXFP4 experts, see recipe) | 19,819,767,136 | `6735282db893ec925eabe864bca72521563d9f85f7152f0cab7bec5bfaab9960` |",
    f"| `Ornith-1.5-35B-A3B-MXFP4.gguf` | language model + MTP layer (MXFP4, see recipe) | {NEW_SIZE} | `{NEW_SHA}` |")
rep("The language-model file is 4.46 bits per weight on average and takes 18.45 GiB of VRAM when loaded (plus KV cache). The projector is needed only for image input.",
    "The language-model file is 4.46 bits per weight on average and takes 18.45 GiB of VRAM when loaded (plus KV cache). The projector is needed only for image input.\n\n"
    "`SHA256SUMS` lists the same hashes. The language-model file is the 2026-10-19 rebuild; the first-release file (2026-10-03) had "
    "the same name, was 448 bytes smaller (metadata only; SHA256 `6735282d…`) and remains in this repository's commit history.")
# 4. recipe
rep("1. **Convert** the official BF16 checkpoint with llama.cpp `convert_hf_to_gguf.py --outtype bf16`.",
    "1. **Convert** the official BF16 checkpoint ([ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) revision `10fbf86fed`, every file checked against its Hugging Face hash, list in `SOURCE_SHA256.txt`) with llama.cpp `convert_hf_to_gguf.py --outtype bf16` (build **b11719**, commit 5aeac37a3).")
rep("2. **Quantize** with `llama-quantize` from llama.cpp **build 11214 (commit 2ebd9ae62)**, ROCm build, **without an importance matrix** (mainline llama.cpp ignores the imatrix for MXFP4 and Q8_0 anyway), using this tensor-type file (first matching rule wins):",
    "2. **Importance matrix**: computed by us for this exact checkpoint (no published imatrix was made from revision `10fbf86fed`: "
    "the upstream re-uploaded one weight shard on 2026-08-23, after the community imatrices were made). `llama-imatrix "
    "--process-output -c 512` (llama.cpp b11719, Vulkan, Radeon 8060S) on a Q8_0 conversion of the BF16 GGUF, over 982 chunks "
    "of 512 tokens of calibration text: bartowski's calibration_datav6 plus C++/HIP source code with Chinese comments and Chinese "
    "technical prose, with no overlap with the text used for the KL / perplexity measurement in [Quality](#quality). 511 entries: "
    "every matrix of the 40 main layers (all 256 experts covered) and `output`; none for `token_embd` or the MTP layer. "
    "File SHA256 `21216dbe71161194ee85aa0464754a0d7ac913acb935fc44d4d746570438e703`; details in [`RECIPE.md`](RECIPE.md).\n"
    "3. **Quantize** with `llama-quantize` built from llama.cpp **b11719 (commit 5aeac37a3) plus [`mxfp4_wsearch.patch`](mxfp4_wsearch.patch)**, "
    "an imatrix-weighted MXFP4 block-scale search. Upstream `llama-quantize` ignores the importance matrix for MXFP4 and always takes "
    "the block exponent from the block maximum; the patch tries that exponent and its two neighbours for every 32-value block and keeps "
    "the one with the lowest imatrix-weighted squared error. Only quantize-time code changes: the output is the standard MXFP4 block "
    "format, so the file loads in unmodified llama.cpp and WHIRL. Same tensor-type file as the first release (first matching rule wins):")
rep("""llama-quantize --tensor-type-file tt_orn.txt --output-tensor-type q6_k \\
  ornith-1.5-35b-bf16.gguf Ornith-1.5-35B-A3B-MXFP4.gguf MXFP4_MOE 16
```""", """llama-quantize --imatrix ornith-1.5-35b-a3b-requant1-imatrix.gguf --tensor-type-file tt_orn.txt --output-tensor-type q6_k \\
  ornith-1.5-35b-bf16.gguf Ornith-1.5-35B-A3B-MXFP4.gguf MXFP4_MOE 24
```

(`llama-quantize` = the patched build. The first release, 2026-10-03, used the same tensor-type file with upstream `llama-quantize` from build 11214, commit 2ebd9ae62, without an importance matrix. The rebuild has the same 753 tensor names and types; only the MXFP4 block scales / codes and the Q6_K head differ.)""")
rep("3. **Vision projector**: `convert_hf_to_gguf.py --mmproj --outtype f16`",
    "4. **Vision projector** (unchanged from the first release): `convert_hf_to_gguf.py --mmproj --outtype f16`")
# 5. measured performance: which file
rep("WHIRL measured 2026-10-17.",
    "WHIRL measured 2026-10-17 with the first-release file; the 2026-10-19 rebuild has the same format and tensor types, and its speed was re-checked against the first-release file (see [Quality](#quality), other checks).")
# 6. quality section
i = s.index("## Quality\n")
j = s.index("## License\n")
quality = """## Quality

KL divergence and perplexity against the **BF16 model** (the unquantized Ornith-1.5-35B-A3B weights converted to a BF16 GGUF), measured with llama.cpp b11719 `llama-perplexity` on the CPU (official release build), `-c 512`, 80 chunks of 512 tokens of a mixed text: a C++/HIP source file with Chinese comments (70,000 characters) plus Chinese technical prose (40,000 characters), the same text and settings as our Swift MXFP4 card. BF16 perplexity on this text: 6.2537 ± 0.1226. Measured 2026-10-20; "first release" = the 2026-10-03 file with the same name. Lower KLD and higher same-top are better.

| File | Mean KLD vs BF16 | 99th-percentile KLD | Same top token as BF16 | PPL | PPL / PPL(BF16) |
|---|---|---|---|---|---|
| **this rebuild** | **0.1316 ± 0.0019** | **1.206** | **85.81%** | **6.5151** | **1.0426 ± 0.0047** |
| first release | 0.1618 ± 0.0023 | 1.449 | 84.61% | 6.6594 | 1.0657 ± 0.0053 |

- Same tensor types in both files (MXFP4 for every large matrix of the 40 main layers, see the recipe); the rebuild differs only in the MXFP4 block scales and codes (imatrix-weighted search) and an imatrix-weighted Q6_K head. KLD is 19% lower and the perplexity gap to BF16 shrinks from 6.6% to 4.3% (paired ratio ± its standard error).
- This layout puts the attention and shared-expert matrices in MXFP4 too, so the loss against BF16 is larger than in our dense Swift file; both builds are 4-bit and stay measurably above BF16 perplexity. One text and one tokenizer; this is not a downstream task evaluation, and these numbers are not comparable with other models.
- (The first release's card compared perplexity with another 4-bit GGUF format on another corpus; that table was replaced by this BF16 comparison.)

Other checks of the rebuilt file (2026-10-19/20, WHIRL 0.2.0 and llama.cpp b11719):

CHECKS_EN_PLACEHOLDER

For the model's own evaluations, see the upstream card. A 4.46-bpw MXFP4 file is expected to lose some accuracy against BF16; if that matters for your use, compare against the official Q8_0 / BF16 GGUFs.

"""
s = s[:i] + quality.replace("CHECKS_EN_PLACEHOLDER", checks_en) + s[j:]
# 7. credits
rep("- **llama.cpp / ggml** contributors — conversion, quantization and inference tooling.",
    "- **llama.cpp / ggml** contributors — conversion, quantization and inference tooling.\n"
    "- **[bartowski](https://huggingface.co/bartowski)** — the calibration text calibration_datav6, part of our imatrix calibration data.")
# 8. zh summary
rep("- 量化：專家、共享專家與所有稠密矩陣 MXFP4；token_embd、ssm_alpha / beta、整個 MTP 層 Q8_0；router 與 norm 為 F32；輸出頭 Q6_K；不使用 imatrix。平均 4.46 bpw，載入 18.45 GiB。",
    "- **2026-10-19 重新量化**：改用 imatrix 加權的 MXFP4 量化器（[`mxfp4_wsearch.patch`](mxfp4_wsearch.patch)，llama.cpp b11719）與我們針對這個確切版本（`10fbf86fed`）算的 imatrix（llama-imatrix --process-output，982×512 token 校準文字，與品質量測文字不重疊）；張量配置與檔名不變，視覺投影檔沒有變。\n"
    "- 量化：專家、共享專家與所有稠密矩陣 MXFP4；token_embd、ssm_alpha / beta、整個 MTP 層 Q8_0；router 與 norm 為 F32；輸出頭 Q6_K。平均 4.46 bpw，載入 18.45 GiB。")
i = s.index("- 品質：困惑度（llama.cpp b11214")
j = s.index("\n", i)
s = s[:i] + ("- 品質（對 BF16，llama.cpp b11719 CPU，程式碼＋中文 80×512 token，BF16 困惑度 6.2537）：本版 KLD **0.1316**、與 BF16 同 top-1 **85.81%**、困惑度 6.5151（比 BF16 高 4.3%）；"
             "第一版（2026-10-03）KLD 0.1618、84.61%、6.6594（高 6.6%）→ KLD 低 19%。單一語料、非下游任務評估。\n" + checks_zh) + s[j:]
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
