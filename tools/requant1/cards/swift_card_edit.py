"""REQUANT-1: apply the rebuild edits to the Swift-1.5 27B MXFP4 card (input: live README, LF).
Usage: python -I swift_card_edit.py README.md SPEED_EN_FILE SPEED_ZH_FILE   (one-paragraph texts, UTF-8)"""
import sys
p = sys.argv[1]
speed_en = open(sys.argv[2], encoding="utf-8").read().strip()
speed_zh = open(sys.argv[3], encoding="utf-8").read().strip()
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


# 1. intro: rebuild note
rep("Radeon 8060S) and Radeon AI PRO R9700.\n\n> **Unofficial community quantization.**",
    "Radeon 8060S) and Radeon AI PRO R9700.\n\n"
    "**Rebuilt 2026-10-19:** the three language-model files were re-quantized with an **imatrix-weighted MXFP4 quantizer** "
    "(same tensor layout, same file names). Against the BF16 model their KL divergence is 17–19% lower than the first release "
    "(2026-10-01); see [Quality](#quality). The vision projector is unchanged.\n\n"
    "> **Unofficial community quantization.**")
# 2. llama.cpp tested-with note in the WHIRL box
rep("They are standard GGUF and also run in llama.cpp (tested with b11214, see below).",
    "They are standard GGUF and also run in llama.cpp (first release tested with b11214; the 2026-10-19 rebuild with b11719, see below).")
# 3. files table
rep("| **A** | `Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf` | Q6_K | 15,815,469,280 | `f12d8dee7d677a0583b243d65c5e75e853b1839dc61e2c101da090bb6163065f` |",
    "| **A** | `Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf` | Q6_K | 15,815,469,376 | `c765544dee05261e4161281cfc298c493087d623b89c3753675d43255b42e09f` |")
rep("| C | `Swift-1.5-Qwen3.8-27B-MXFP4-C-outQ4_K.gguf` | Q4_K | 15,487,686,880 | `221aa39e7f7c003ae94903ea686d33d6bb187ed8b943955fb21d0790c7cf6b4a` |",
    "| C | `Swift-1.5-Qwen3.8-27B-MXFP4-C-outQ4_K.gguf` | Q4_K | 15,487,686,976 | `d24f9692a4db9dda2fe84a6fdcde9f5007e4bc93e9e905e67d1cbfc6ee85edbc` |")
rep("| B | `Swift-1.5-Qwen3.8-27B-MXFP4-B-outQ8_0.gguf` | Q8_0 | 16,123,386,080 | `bb075d1cf83352d38751a799feffc0e0c91bb8ccb44a14551c0f485ffd65c72a` |",
    "| B | `Swift-1.5-Qwen3.8-27B-MXFP4-B-outQ8_0.gguf` | Q8_0 | 16,123,386,176 | `4b5f70adf8b462a79920e76c821e76c3875967b0925d78fa59fce01ec287e933` |")
rep("| vision | `mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf` | — (vision projector, F16) | 927,607,584 | `71988a54d379191e1abc26c2b79363b87ef8b3a584ebee882c28a1682ddbfe62` | Needed only for image input |\n",
    "| vision | `mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf` | — (vision projector, F16) | 927,607,584 | `71988a54d379191e1abc26c2b79363b87ef8b3a584ebee882c28a1682ddbfe62` | Needed only for image input |\n\n"
    "`SHA256SUMS` lists the same hashes. The three language-model files are the 2026-10-19 rebuild; the first-release files "
    "(2026-10-01) had the same names, were 96 bytes smaller (A `f12d8dee…`, B `bb075d1c…`, C `221aa39e…`) and remain in this "
    "repository's commit history.\n")
# 4. recipe steps
rep("""1. BF16 safetensors → BF16 GGUF with llama.cpp `convert_hf_to_gguf.py <Swift-1.5 BF16 dir> --outtype bf16` (866 tensors; `qwen35.nextn_predict_layers = 1`, i.e. the MTP layer is kept). (llama.cpp `b10645-47-gef6876693`, commit ef6876693.)
2. Quantize with `llama-quantize` from llama.cpp **build 11214 (commit 2ebd9ae62)**, ROCm build.""",
    """1. BF16 safetensors ([ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) revision `b4c84d42`, every file checked against its Hugging Face hash, list in `SOURCE_SHA256.txt`) → BF16 GGUF with llama.cpp `convert_hf_to_gguf.py <Swift-1.5 BF16 dir> --outtype bf16` (866 tensors; `qwen35.nextn_predict_layers = 1`, i.e. the MTP layer is kept). (llama.cpp build **b11719**, commit 5aeac37a3.)
2. Quantize with `llama-quantize` built from llama.cpp **b11719 (commit 5aeac37a3) plus [`mxfp4_wsearch.patch`](mxfp4_wsearch.patch)**, an imatrix-weighted MXFP4 block-scale search. Upstream `llama-quantize` ignores the importance matrix for MXFP4 and always takes the block exponent from the block maximum; the patch tries that exponent and its two neighbours for every 32-value block and keeps the one with the lowest imatrix-weighted squared error. Only quantize-time code changes: the output is the standard MXFP4 block format, so the files load in unmodified llama.cpp and WHIRL. Full build and run commands: [`RECIPE.md`](RECIPE.md).

(The first release, 2026-10-01, used the same tensor-type file with upstream `llama-quantize` from build 11214, commit 2ebd9ae62.)""")
rep("""  swift-1.5-27b-bf16.gguf Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf MXFP4_MOE 16
```""", """  swift-1.5-27b-bf16.gguf Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf MXFP4_MOE 24
```

(`llama-quantize` = the patched build from step 2.)""")
rep("- **Importance matrix**: the imatrix was made and published by **bartowski** — [bartowski/ukisai_Swift-1.5-Qwen3.8-27b-GGUF](https://huggingface.co/bartowski/ukisai_Swift-1.5-Qwen3.8-27b-GGUF) (`ukisai_Swift-1.5-Qwen3.8-27b-imatrix.gguf`, 496 entries, 582 chunks). Thank you! Note that mainline llama.cpp **ignores the imatrix for MXFP4 and for Q8_0**, so in these files it only affects the K-quant output head (Q6_K in A, Q4_K in C). B is effectively imatrix-free.",
    "- **Importance matrix**: the imatrix was made and published by **bartowski** — [bartowski/ukisai_Swift-1.5-Qwen3.8-27b-GGUF](https://huggingface.co/bartowski/ukisai_Swift-1.5-Qwen3.8-27b-GGUF) (`ukisai_Swift-1.5-Qwen3.8-27b-imatrix.gguf`, revision `ce202c7d`, SHA256 `125d4e68…a268b`, 496 entries, 582 chunks). Thank you! It has entries for every matrix of the 64 main layers and none for `output`, `token_embd` or the MTP layer. llama.cpp does not use the imatrix for Q8_0, so in the rebuilt files it acts through the weighted MXFP4 search on all 400 MXFP4 matrices; the output heads, `token_embd` and the MTP layer are quantized without imatrix data. (The first-release card said the imatrix affected the K-quant output heads; that was not correct, since the imatrix has no `output` entry.)")
# 5. perplexity bullet -> pointer; new Quality section before the llama.cpp usage section
i = s.index("- **Perplexity** (llama.cpp b11214 `llama-perplexity`")
j = s.index("## Usage with llama.cpp (MTP)")
quality = """## Quality

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
- SPEED_EN_PLACEHOLDER

"""
quality = quality.replace("SPEED_EN_PLACEHOLDER", speed_en)
s = s[:i] + ("- **Quality**: KL divergence and perplexity against BF16 are in [Quality](#quality) below. (The first release's perplexity "
             "table, measured on the R9700 with another corpus and llama.cpp b11214 and compared with a Qwen3.8-27B MXFP4 file of a "
             "different model, was replaced by this BF16 comparison.)\n\n") + quality + s[j:]
# 6. measured performance: which files
rep("Single-machine measurements; speed only, not quality evaluations.",
    "Single-machine measurements; speed only, not quality evaluations. The tables below were measured with the first-release files "
    "(2026-10-01 to 2026-10-17). The 2026-10-19 rebuild has the same format and tensor types; its speed was checked on the Radeon 8060S "
    "only (see [Quality](#quality)) and not re-measured on the R9700.")
# 7. credits
rep("- **bartowski** — the [importance matrix](https://huggingface.co/bartowski/ukisai_Swift-1.5-Qwen3.8-27b-GGUF) used for the K-quant output heads.",
    "- **bartowski** — the [importance matrix](https://huggingface.co/bartowski/ukisai_Swift-1.5-Qwen3.8-27b-GGUF) used by the weighted MXFP4 quantizer.")
# 8. zh summary
rep("- 大矩陣 MXFP4；token_embd、ssm_alpha / beta、MTP 層 Q8_0；imatrix 來自 bartowski，但 mainline 量 MXFP4 / Q8_0 時不使用 imatrix，只影響 K-quant 輸出頭。",
    "- **2026-10-19 重新量化**：改用 imatrix 加權的 MXFP4 量化器（[`mxfp4_wsearch.patch`](mxfp4_wsearch.patch)，llama.cpp b11719），張量配置與檔名不變；對 BF16 的 KL 散度比第一版（2026-10-01）低 17～19%（A：0.0495 對 0.0606，同 top-1 91.62% 對 91.01%）。視覺投影檔沒有變。\n"
    "- 大矩陣 MXFP4；token_embd、ssm_alpha / beta、MTP 層 Q8_0；imatrix 來自 bartowski（涵蓋 64 個主層的所有矩陣，沒有 output／token_embd／MTP 層條目），透過加權 MXFP4 搜尋作用在 400 個 MXFP4 矩陣上；輸出頭沒有用到 imatrix（第一版卡片寫 imatrix 只影響輸出頭，這是錯的）。")
rep("- 困惑度（中英混合程式碼語料，2048 ctx）：A 3.8271、B 3.8289、C 3.8386，三者在誤差內；Swift 是另一個微調模型，不宜直接和 Qwen 量化版（3.7952）比較。",
    "- 對 BF16 的品質（llama.cpp b11719 CPU，程式碼＋中文 80×512 token）：BF16 困惑度 4.5053；本版 A／B／C 的 KLD 0.0495／0.0492／0.0527，困惑度比 BF16 高 3.6%／3.6%／3.9%；第一版 0.0606／0.0604／0.0638，高 5.3%／5.3%／5.6%。\n"
    "- " + speed_zh)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
