[English](../README.md) | **繁體中文**

# WHIRL

**Windows HIP Inference for RDNA LLMs**

WHIRL 是 AMD Radeon AI PRO R9700（RDNA 4）與 Radeon 8060S（Ryzen AI Max+ 395，RDNA 3.5）在 Windows 上的 LLM 推論引擎。

**WHIRL 的設計理念**
1. **專為 Windows 上的 AMD GPU 打造**：原生 Windows 的 C++／HIP 推論引擎，針對 RDNA 架構逐一手寫、調校 kernel。不需要 WSL、Docker 或 Linux 虛擬機，裝好 AMD 驅動就能跑。
2. **完整開源，包含每一支 kernel**：從 host 端排程到 GPU kernel，全部原始碼以 Apache-2.0 公開，沒有封閉的二進位元件。歡迎回報問題、提供量測數據，或直接貢獻程式碼，一起把 AMD 在 Windows 上的推論做到最好。
3. **精度優先於速度**：隨附三種數值模式：**precise**、**balance**（預設）與 **fast**（[見下](#三種數值模式)）。在 precise 與 balance 裡，所有加速手段（投機解碼、多人合批、prefix cache、快取還原）的輸出，都與該模式下單純 greedy 解碼逐位元相同，並有常駐測試把關。precise 除了 GGUF 檔本身，不加入任何有損項目；balance 與 precise 的 KL 在約 0.005 以內；fast 需自行開啟，並公布實測的 KL。
4. **少數模型，做到極致**：不追求「什麼模型都能跑」，而是針對特定模型與硬體，把效能推到接近硬體極限。支援的架構一個一個加，每加一個就做到最好。
5. **數字透明、可驗證**：所有效能數字都是實測，量測方法、指令和原始數據全部公開。投機解碼這類效果會隨內容變化的功能，同時列出最適用場景、主要場景和最差情況，讓你知道自己的用法會落在哪裡。
6. **為真實工作流程設計**：以 coding agent 和長上下文為核心場景：長 prompt 快速處理、多人同時使用時主對話不卡頓、VRAM → RAM → SSD 三層快取，server 重啟後對話也能接回來。

目前支援的架構是 **`qwen35`**（dense）與 **`qwen35moe`**（混合專家，MoE），之後會逐一加入更多架構。WHIRL 依循 [NInfer](https://github.com/Neroued/ninfer)（Apache-2.0）的設計理念——為選定的模型與選定的 GPU 打造，盡可能逼近硬體極限；[zynfer](https://github.com/thanos/zynfer) 把同樣的想法帶到了 RDNA 4。

## WHIRL 對 llama.cpp：一眼看

Radeon AI PRO R9700，WHIRL 0.2.0 的預設模式（**balance**）對 llama.cpp b11214（ROCm）。同一個 GGUF 檔、同一組提示、greedy 解碼，除非另外註明都是一位使用者。llama.cpp 的啟動方式為 `llama-server --device ROCm0 -ngl 99 -fa on -b 2048 -ub 1024 --jinja --reasoning off --top-k 1 -np 1`（MoE 模型用 `-ub 2048`）、f16 KV，每一格取它的一般模式與 MTP + n-gram 模式中較快的一個。「快 X%」是 WHIRL / llama.cpp - 1。WHIRL 的每一項加速，輸出都與同一模式下單純 greedy 解碼相同（[為什麼](#變快但答案不變)）。

| R9700，balance（預設） | Qwen3.8-27B UD-Q4_K_M（dense） | Ornith-1.5-35B-A3B MXFP4（MoE，約 3B 啟用） |
|---|---|---|
| Prefill tok/s，8k token 提示 | 1,914 對 1,201（快 **59%**） | 11,199 對 4,426（快 **153%**） |
| Prefill tok/s，32k token 提示 | 1,721 對 1,082（快 **59%**） | 8,585 對 3,853（快 **123%**） |
| Prefill tok/s，128k token 提示 | 1,230 對 756（快 **63%**） | 4,273 對 2,285（快 **87%**） |
| Prefill tok/s，249k token 提示（「256k」） | 912 對 536（快 **70%**）¹ | 2,659 對 1,534（快 **73%**） |
| Decode tok/s，針對 8k token 文件發問 | 84.9 對 41.8（快 **103%**） | 229.3 對 103.5（快 **122%**） |
| Decode tok/s，針對 128k token 文件發問 | 49.3 對 27.1（快 **82%**） | 132.9 對 69.7（快 **91%**） |
| Decode tok/s，128k token 之後寫全新內容（草稿最不利的情況） | 33.8 對 21.2（快 **59%**） | 113.8 對 69.9（快 **63%**） |
| Decode tok/s，中文寫程式對話 | 100.9 對 58.1（快 **74%**） | 242.5 對 108.8（快 **123%**） |
| 伺服器，4 人同時使用，總吞吐 tok/s | 251.4 對 90.6（快 **177%**） | 未量測 |

<!-- src: audit/REL3.md「R9700」一節：prefill 表（2k..256k，balance 欄與 llama.cpp best）、decode QA 表、最差情況表、中文對話表（dense 的 llama.cpp 58.1 是 MTP+n-gram 第 1 輪；REL3 註明它先以同一提示做過 64 token 暖機，對 llama.cpp 略有利）、4 人表（plain 80.4／MTP+n-gram 90.6，取較好者）。百分比為 REL3 自己的算法：WHIRL / llama.cpp - 1。 -->

¹ llama.cpp 在 dense 模型的 256k 放不下 f16 KV（與 0.1.3 的量測相同），改用 q8_0 KV；WHIRL 用 balance 的 KV（q8h）。長度掃描每格只量一次（對話為 3 次取中位數），差距小於約 5% 請當成雜訊。Qwen3.8-27B UD-Q4_K_M 是 unsloth 的檔；Ornith-1.5 是我們量化的 MXFP4。另外兩個模式、每個長度，以及領先幅度小的情況：[效能](#效能)。Swift-1.5 27B MXFP4 沒有為 0.2.0 重新量測，它 0.1.3 的數字留在[效能](#效能)。

[8060S: TBD ——Radeon 8060S 的對照表，等 REL3 量完後放在這裡；見 [8060S 一節](#radeon-8060s)]

## <a id="三種數值模式"></a>三種數值模式

啟動時以 `--precise`、`--balance`（預設，不加旗標就是它）或 `--fast` 選擇；兩張 GPU 相同，`whirl chat`、`whirl bench`、`whirl-server` 皆適用。細節與逐項開關：[使用參考](guide/zh-TW/usage.md)。

| | **precise**（`--precise`） | **balance**（預設） | **fast**（`--fast`，需自行開啟） |
|---|---|---|---|
| KV 快取，dense 模型 | f16 | q8h（int8） | q4（4-bit） |
| KV 快取，MoE 模型 | f16 | f16 | q4（4-bit） |
| 運算方式 | GGUF 權重以 f16 讀取；prefill 與 decode／驗證都用 f16 activation；沒有 fp8、沒有 int8 activation | 0.1.x 的速度路徑：prefill 用 fp8／f16 中間值，decode 用 int8 activation | balance，再加上 MTP 草稿的放寬接受（dense 模型） |
| 輸出與基準的差距 | 除了 GGUF 檔本身，沒有再量化任何東西 | 近乎無損：對 precise 的 KL <= 約 0.005（q8h KV 對 f16：0.0002 到 0.002） | KL <= 0.03 以內；q4 KV 對 f16 實測 KL：0.0016 到 0.0100（R9700） |
| 加速手段（MTP、n-gram、合批、快取） | 輸出與單純 greedy 解碼逐位元相同 | 輸出與單純 greedy 解碼逐位元相同 | 不保證逐位元相同（放寬接受）；同一個請求單獨跑與和其他人一起跑，文字可能不同 |
| 對 balance 的速度，R9700 | decode 慢 25% 到 29%，dense prefill 慢 5% 到 10%（MoE 短提示慢 30% 到 36%） | 基準 | dense decode 快 17%（對話）、15%（4 人）、18%（249k context）；MoE 只有約 64k 以上才比較快 |
| 什麼時候選 | 需要可以重現、能逐位元比對的輸出，或要拿別的引擎來對照 WHIRL | 幾乎所有情況 | 長上下文，或想要最高的 tok/s 且能接受輸出有小幅變動 |

<!-- src: 各模式的 KV：CHANGELOG.md Unreleased「KV format fixed per mode (BAL-Q8)」。balance KL <= 約 0.005、fast 門檻 KL <= 0.03：internal_docs/v013_plan.md（owner 決定，2026-10-08）與 docs/usage.md Modes。q8h KL 0.0002-0.002：CHANGELOG BAL-Q8。R9700 上 q4 KV KL 0.0016-0.0100：CHANGELOG FAST-1c（code 64k 0.0100、zh 64k 0.0016、code 128k 0.0049、zh 128k 0.0071）。precise 的速度代價：CHANGELOG「precise decode / verify no longer quantizes activations」（plain decode 慢 26..29%）、audit/REL3.md（對話 75.7 對 100.9 = 慢 25%；prefill dense 慢 5..10%、MoE 短提示慢 30..36%）。fast 對 balance：audit/REL3.md dense 對話 118.1 對 100.9、4 人 288.3 對 251.4、249k QA decode 40.1 對 33.9；MoE 註記「Fast on the MoE is slower than balance below ~64k」。 -->

這些保證實際上代表什麼：

- 在 **precise** 與 **balance**，一個請求單獨跑或和其他人一起跑、開或不開 MTP／n-gram／DFlash2 草稿、從冷的提示開始或從快取的前綴接續，得到的 token 都一樣。balance 與 precise 的差別只來自文件記載的有損項目（fp8／f16 中間值、int8 activation、q8h KV），上限就是上表的 KL。
- **fast** 可能改變輸出。它的品質由門檻把關（對 f16 KV 基準的 KL <= 0.03、top-1 一致率，以及用執行結果評分的寫程式檢查），而不是靠逐位元相同。放寬的 MTP 接受（僅 dense 模型）會在草稿不是驗證列的 argmax、但落在機率前 4 名且 p >= argmax 的 0.1 倍時保留它，並且拒絕會讓最近 4096 個 token 內已出現過的 6-gram 重複的接受。門檻測試（Qwen3.8-27B Q4_K_M）中，12 個寫程式任務有 12 個執行正確、24 次生成有 0 次陷入迴圈，重複率 0.167（precise 為 0.157），被接受的草稿中約 7% 來自放寬規則。這個模式下，同一個請求單獨跑與和其他人一起跑可能不同；precise 與 balance 則保證相同。
- KL 是拿 logits 與同一個模型的 f16 KV 版本比，不是模型本身的準確度評測。

<!-- src: 放寬接受規則與 6-gram 防護：docs/usage.md Modes（relaxacc 列）。門檻結果（exec 12/12、loops 0/24、重複率 .167 對 precise .157、6.9% 放寬）：progress/FAST-2.log 2026-10-07 12:49 與 13:01（在 8060S 上以 Q4_K_M 量測；R9700 上沒有重跑）。fast 的單獨對並行：audit/CONC-8S.md「Limits」與 CHANGELOG conc8s。 -->

## 變快，但答案不變

在 **precise** 與 **balance**，WHIRL 的每一項加速都經過檢查：**產生的 token 與同一個模型檔、同一模式下的單純 greedy 解碼逐位元相同**（**fast** 是例外，見[上面](#三種數值模式)）。

| 加速手段 | 保證 | 怎麼檢查 |
|---|---|---|
| MTP、n-gram 與 DFlash2 投機解碼| 驗證後的輸出與單純 greedy 解碼逐位元相同| 內建自我檢查 `whirl selftest`（用模型自己的權重比對多列 kernel 與單列）、kernel 測試、伺服器 gate |
| 多位使用者合併批次 | 每個請求的輸出與單獨執行時逐位元相同| kernel 測試（GEMM 與 attention 不受批次列數影響）、伺服器 gate |
| 共用系統提示與共用前綴的快取命中 | 與從頭處理這個提示逐位元相同（檢查點放在提示自身的 chunk 排程點上） | 伺服器 gate |
| 從主記憶體或 SSD 層還原 session | 與存入時的狀態逐位元組相同| 寫出時逐位元組比對（`TIER_VERIFY`）、伺服器 gate |

為了變快，WHIRL 在 precise 與 balance **不做**的事：KV 快取不低於 8 bit、不把模型權重重新量化成更少的 bit（只有草稿用的 MTP 與草稿頭副本會這樣做，而草稿不會改變輸出）、不用 fp8 attention。`--fast` 使用 4-bit KV 與放寬的草稿接受；這就是它是獨立、需自行開啟的模式的原因。

精確地說，這代表：

- **權重**就是您選用的 GGUF；量化方式是模型檔作者的選擇，WHIRL 照檔案存的內容計算。
- **KV 快取**依模式固定：precise 為 f16，balance 在 dense 模型為 q8h、在 MoE 模型為 f16，fast 為 q4。放不下的格式會降低 context（並顯示警告），或以訊息停止程式；絕不會自己改用精度較低的格式。`WHIRL_KV` 是除錯用的覆寫。q8h 與 q4 和 f16 相比在數值上有損失，就像任何量化 KV 一樣；WHIRL 不宣稱它們「零損失」，並公布 KL。
- **接續一段對話**時，會重用上一輪算出的 KV 與 DeltaNet 狀態（包含 decode 生成的 token）。它們來自 decode kernel （以及不同的 chunk 切法），與「把整段歷史當成一個提示重新處理」在數值上等價，但不是逐位元相同；差異已記錄並量測（[kv-and-caching.md](guide/zh-TW/kv-and-caching.md)）。
- **balance 下的 MXFP4 檔**在 MXFP4 prefill 矩陣乘法使用 fp8 activation，DeltaNet prefill 的部分步驟使用 f16（0.1.x 的速度路徑）；`--precise` 兩者都關掉（prefill 較慢，見[效能](#效能)）。decode 與驗證對批次中每一列都用同樣的 kernel，所以上述保證在各個模式下都成立。

## 需求

| 項目 | 內容 |
|---|---|
| GPU | **AMD Radeon AI PRO R9700**（RDNA 4，gfx1201），或 **Radeon 8060S**（Ryzen AI Max+ 395 內顯，RDNA 3.5，gfx1151），0.2.0 起正式支援（[見下](#radeon-8060s)）；每個行程用一張 GPU |
| 作業系統 | **Windows 11**，64 位元 |
| 驅動程式 | **AMD Software: Adrenalin Edition 26.8.1**（驅動程式 32.0.31041.1004）或更新版 |
| 執行| 只需要驅動程式（它會安裝 `amdhip64_7.dll` 與 `amd_comgr_3.dll`）。不用 HIP SDK、不用 ROCm、不用 Visual C++ runtime |
| 從原始碼建置 | HIP SDK 7.2 + Visual Studio 2022 Build Tools（MSVC）+ CMake + Ninja——見[從原始碼建置](building_zh-TW.md) |
| 記憶體 / 磁碟 | R9700：伺服器會為主記憶體快取層 pin 住約四分之一的主記憶體（8～32 GiB；64 GB 的電腦為 16 GiB）。8060S：主記憶體快取層是**作業系統可見** RAM 的 1/8（64 GB 的機器為 8 GiB；低於 32 GB 則關閉）。SSD 快取層在磁碟上最多佔 64 GiB，預設只在伺服器正常停止時才寫入（[SSD 快取](#ssd-快取)）。可用 `--kv-ram-mb`、`--kv-ssd-gb`、`--kv-ssd-mode` 調整或關閉 |

我們的 R9700 測試機以 USB4 / Thunderbolt eGPU 連接，可以正常使用。插在主機板 PCIe 插槽上的顯示卡，數字可能略有不同（主要是資料要經過主機連結的部分，例如載入模型、主記憶體 / SSD 快取層）。

不支援其他 AMD GPU。RX 9070 系列的晶片相同（gfx1201），但只有 16 GB，對這些 27B / 35B 模型大概不夠（未測試）。

## 快速上手

1. 從 [Releases](https://github.com/tsaipifong/whirl-llm/releases) 下載 `whirl-0.2.0-windows-x64.zip`，以及一個 [支援的模型](#支援的模型)。
2. 在 PowerShell 執行：

```powershell
Unblock-File .\whirl-0.2.0-windows-x64.zip
Expand-Archive .\whirl-0.2.0-windows-x64.zip -DestinationPath C:\whirl
cd C:\whirl\whirl-0.2.0-windows-x64
.\whirl.exe devices                                    # 列出 AMD Radeon AI PRO R9700 (gfx1201) 和／或 Radeon 8060S (gfx1151)
.\whirl.exe chat C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf "用兩句話說明 TCP 慢啟動（slow start）。" --max-tokens 400
.\whirl-server.exe C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf --port 8080
```

伺服器在 `http://127.0.0.1:8080/v1` 提供 OpenAI API（`/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/health`），任何 OpenAI 相容的用戶端都能用。按 Ctrl+C 停止；快取的 session 接著會寫入 SSD 層（最多 60 秒）。

加上 `--precise` 或 `--fast` 可選擇其他[數值模式](#三種數值模式)；不加旗標就是 balance。

處理長文件時，可以給每個 slot 更長的 context，例如 `--ctx-per-slot 262144`（每個請求 256k token），見 [情境食譜](recipes_zh-TW.md#long-context)。

第一次使用某個模型時會先調校一次 GPU kernel——27B Q4_K_M 約 1.5 分鐘，MXFP4 檔只要幾秒——結果快取在 `%LOCALAPPDATA%\whirl`。完整步驟：[快速上手](quickstart_zh-TW.md)。所有選項：[使用參考](guide/zh-TW/usage.md)。

## 支援的模型

WHIRL 讀取 GGUF 的 `general.architecture`，只接受 `qwen35` 與 `qwen35moe`（Gated DeltaNet + gated attention 的混合架構，可帶 MTP 層），以及保留這個架構的微調模型。其他架構（Llama、Gemma、Mistral、DeepSeek、`qwen3`、`qwen2` 等）會在載入時被拒絕並顯示清楚的訊息；WHIRL 沒有 kernel 的 tensor 型別（例如 NVFP4）也一樣，不會退回某條很慢的泛用路徑。

> **推薦：我們自己量化的 MXFP4 檔效能最好。**
> - Dense 27B：[**Swift-1.5-Qwen3.8-27b-MXFP4-GGUF**](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF)，A 版（`Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`）。Swift-1.5 的思考也比原模型精簡得多（在我們的中英混合程式提示上，思考 token 少 44%）。
> - MoE 35B（約 3B 啟用）：**Ornith-1.5-35B-A3B MXFP4**：[tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF)。

測試過的 GGUF 檔（可載入、greedy 輸出已檢查、MTP 輸出與一般 greedy 相同，並在 R9700 上量測過）：

| 模型檔（Hugging Face） | 類型 | 架構 | 量化 | 備註 |
|---|---|---|---|---|
| [tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF) `…-A-outQ6_K.gguf`（推薦）、`…-B-outQ8_0.gguf`、`…-C-outQ4_K.gguf` | dense 27B | qwen35 | MXFP4（+ Q8_0 MTP / embedding） | balance 的 MXFP4 prefill 走 fp8（`--precise` 不走）；我們為 WHIRL 量化 |
| [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `Qwen3.8-27B-UD-Q4_K_M.gguf` | dense 27B | qwen35 | UD-Q4_K_M | 任何模式都不走 fp8 prefill |
| [FreedomAISVR/Qwen3.8-27B-MXFP4-GGUF](https://huggingface.co/FreedomAISVR/Qwen3.8-27B-MXFP4-GGUF) `qwen3.8-27b-mxfp4.gguf` | dense 27B | qwen35 | MXFP4 | balance 的 MXFP4 prefill 走 fp8 |
| [`Ornith-1.5-35B-A3B-MXFP4.gguf`](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF)（我們量化） | MoE 35B，約 3B 啟用 | qwen35moe | MXFP4 專家與 dense（+ Q8_0 MTP / embedding、Q6_K head） | balance 的專家 prefill 走 fp8 |
| [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) `Ornith-1.5-35B-Q4_K_M.gguf` | MoE 35B，約 3B 啟用 | qwen35moe | Q4_K_M | 任何模式都不走 fp8 prefill |

圖片輸入搭配 Qwen3-VL 形式的 mmproj（F16 / BF16），以 `--mmproj` 指定；已用上表的 27B 檔測試。其他 `qwen35` / `qwen35moe` GGUF 應該可以載入，但未經測試。如何選擇與製作 GGUF：[quantization.md](guide/zh-TW/quantization.md)。

## 效能

### Radeon AI PRO R9700，WHIRL 0.2.0（precise / balance / fast）對 llama.cpp

2026-10-12 在 0.2.0 候選版上量測；沒有沿用舊版的任何數字。方法：每個（引擎、模型、模式）啟動一次伺服器；請求一個接一個送，每個請求的第一行都不同，所以不會命中任何前綴快取。提示是一份不重複文件的前 N 個 token，在行界切開（2,050／8,155／32,774／65,541／98,310／131,084／249,506 token），後面接一個關於這份文件的問題（「QA」：回答會大量引用文件，所以草稿有幫助）。Prefill 與 decode 的 tok/s 取自伺服器自己的計時（生成 256 個 token）。「寫全新內容」列是在文件之後要求寫一篇全新的短篇故事，是草稿最不利的情況。中文寫程式對話：512 個 token，暖機後取 3 次的中位數。4 人：同時送出四個不同的中文寫程式短提示，每個 512 token，總吞吐為總 token 數除以含 prefill 的牆鐘時間。長度掃描每格只量一次：差距小於約 5% 請當成雜訊；單一 QA 格會隨草稿命中率變動 5–15%。

WHIRL：草稿 = MTP + n-gram（預設）、greedy、關閉思考、`--kv-ssd-mode off`；KV 為 precise f16、dense balance q8h、dense fast q4、MoE balance f16、MoE fast q4。llama.cpp：b11214（`version: 0.5.0-dev (build 11214, commit 2ebd9ae62)`），`llama-server --device ROCm0 -ngl 99 -fa on -b 2048 -ub 1024（MoE 2048）--jinja --reasoning off --top-k 1 -np 1 -c 133120（dense，到 128k）／262144（MoE）`，`HIP_VISIBLE_DEVICES=1 GGML_CUDA_NO_PINNED=1`，f16 KV。Dense：一般模式與 `--spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-n-min 0 --spec-draft-p-min 0.3` 都跑。MoE：掃描只跑一般模式（0.1.3 的探測發現 MoE 上一般模式最快），對話另外跑了 `draft-mtp --spec-draft-n-max 2`。「llama.cpp best」是該格較快的模式。Dense 的 256k：llama.cpp 放不下 f16 KV，所以那一列用 `-ctk q8_0 -ctv q8_0 -c 262144`。

<!-- src：本小節全部：audit/REL3.md「R9700」一節（WHIRL v0.2.0-rc4 tip 100a9a8）。原始資料：whirl-work\rel3\r97\results.tsv。 -->

**Prefill tok/s**

| 長度 | Dense：precise | balance | fast | llama.cpp best | balance 對 llama.cpp | MoE：precise | balance | fast | llama.cpp best | balance 對 llama.cpp |
|---|---|---|---|---|---|---|---|---|---|---|
| 2k | 1,559 | 1,701 | 1,657 | 1,120 | +52% | 6,854 | 10,708 | 10,026 | 3,661 | +192% |
| 8k | 1,737 | 1,914 | 1,885 | 1,201 | +59% | 7,238 | 11,199 | 11,120 | 4,426 | +153% |
| 32k | 1,575 | 1,721 | 1,703 | 1,082 | +59% | 5,988 | 8,585 | 8,496 | 3,853 | +123% |
| 64k | 1,404 | 1,514 | 1,506 | 943 | +60% | 4,882 | 6,515 | 6,425 | 3,160 | +106% |
| 96k | 1,271 | 1,357 | 1,350 | 840 | +61% | 4,090 | 5,170 | 5,140 | 2,647 | +95% |
| 128k | 1,172 | 1,230 | 1,222 | 756 | +63% | 3,520 | 4,273 | 4,266 | 2,285 | +87% |
| 249k（「256k」） | 不適用² | 912 | 913 | 536 | +70% | 2,347 | 2,659 | 2,662 | 1,534 | +73% |

**針對長文件發問之後的 decode tok/s（回答會引用文件；草稿有幫助）**

| 長度 | Dense：precise | balance | fast | llama.cpp best | balance 對 llama.cpp | MoE：precise | balance | fast | llama.cpp best | balance 對 llama.cpp |
|---|---|---|---|---|---|---|---|---|---|---|
| 2k | 60.7 | 78.9 | 100.7 | 44.3 | +78% | 187.6 | 235.6 | 220.2 | 107.5 | +119% |
| 8k | 56.5 | 84.9 | 95.3 | 41.8 | +103% | 179.0 | 229.3 | 212.1 | 103.5 | +122% |
| 32k | 57.3 | 72.0 | 81.3 | 38.3 | +88% | 146.8 | 193.6 | 184.2 | 93.7 | +107% |
| 64k | 47.8 | 59.6 | 68.5 | 37.1 | +61% | 135.1 | 162.3 | 169.8 | 83.9 | +93% |
| 96k | 44.3 | 55.0 | 55.2 | 29.1 | +89% | 124.6 | 147.3 | 159.6 | 75.7 | +95% |
| 128k | 40.9 | 49.3 | 51.6 | 27.1 | +82% | 111.8 | 132.9 | 152.7 | 69.7 | +91% |
| 249k（「256k」） | 不適用² | 33.9 | 40.1 | 15.5 | +119% | 88.3 | 97.2 | 120.8 | 53.7 | +81% |

**文件之後寫全新內容的 decode tok/s（草稿最不利的情況）**

| 長度 | Dense：precise | balance | fast | llama.cpp best | balance 對 llama.cpp | MoE：precise | balance | fast | llama.cpp best | balance 對 llama.cpp |
|---|---|---|---|---|---|---|---|---|---|---|
| 8k | 36.3 | 51.1 | 65.3 | 30.2 | +69% | 143.2 | 182.3 | 183.5 | 104.5 | +74% |
| 128k | 29.2 | 33.8 | 40.4 | 21.2 | +59% | 100.1 | 113.8 | 134.7 | 69.9 | +63% |

**中文寫程式對話的 decode tok/s**（512 個 token，3 次取中位數；llama.cpp 為第 1 輪）

| 模型 | precise | balance | fast | llama.cpp 一般 | llama.cpp MTP + n-gram |
|---|---|---|---|---|---|
| Qwen3.8-27B UD-Q4_K_M（dense） | 75.7 | 100.9 | 118.1 | 30.4 | 58.1 |
| Ornith-1.5-35B-A3B MXFP4（MoE） | 185.2 | 242.5 | 220.0 | 108.8 | 101.4 |

**4 人同時使用，dense 模型，總吞吐 tok/s**（4 x 512 token，含 prefill 的牆鐘時間，第 1 輪）：WHIRL precise 200.9、balance 251.4、fast 288.3；llama.cpp 一般 80.4、MTP + n-gram 90.6。WHIRL 每個模式的第二輪與第一輪相差在 3% 以內。MoE 模型與 DFlash2 沒有量 4 人的情況。

² Dense 的 precise 放不下 256k：f16 KV 每個 token 68 KiB，KV 池可容納 176,128 個 token，所以 `--ctx-per-slot 262144` 會在啟動時被拒絕並顯示訊息。128k 那一列用了 `--ctx-per-slot 163840`，因為 131,084 token 的提示放不進 131,072。

WHIRL 領先**不多**、或某個模式要付出速度代價的地方：

- dense 模型在草稿落空時領先最小：128k token 之後寫全新內容，balance 只快 59%；8k prefill 也是快 59%。
- precise 要付出速度：dense 對話 decode 75.7 對 100.9 tok/s（慢 25%），dense prefill 慢 5% 到 10%，MoE 短長度的 prefill 慢 30% 到 36%。
- MoE 的 fast 在約 64k token 以下比 balance 慢（對話 220.0 對 242.5，8k QA 212.1 對 229.3），以上比較快（128k 152.7 對 132.9）。MoE 上 fast 只是 balance 加 4-bit KV（放寬接受在 MoE 上略過），KV 還小的時候，4-bit 反量化花的時間比省下的多。
- 2k 的 prefill 在每一列 WHIRL 都比 8k 低（短提示的暖身）。其餘情況 prefill 與 decode 都隨長度平滑下降，到 249k token 都沒有斷崖。
- llama.cpp 的 MTP + n-gram 會從重複的提示學習：它的第 2、第 3 次相同對話請求跑到 168 與 304 tok/s。表中用的是第 1 輪（已先以同一提示做 64 token 暖機，對 llama.cpp 略有利）。WHIRL 的 n-gram 草稿只看當前請求，3 次的速度相同。
- 沒有量到的：llama.cpp 在 MoE 上的 `draft-mtp` 掃描（只有一般模式的掃描與一次對話）、MoE 的 4 人測試、llama.cpp 的 agent 工作階段重播、長度掃描的重複量測。

我們的 R9700 是 USB4 eGPU；prefill 與 decode 都在 VRAM 內，但載入模型與主記憶體 / SSD 還原要經過主機連結。

### 較早的量測：WHIRL 0.1.3 在 R9700 上，含 Swift-1.5 27B MXFP4

Swift-1.5 MXFP4 沒有納入 0.2.0 的這輪量測，所以這裡是 0.1.3 的數字（同樣的 llama.cpp b11214，每一列都用我們找到最快的旗標）。decode 數字先報 WHIRL 的預設模式 **MTP + n-gram**；標示 **no MTP** 的列比較的是不用投機解碼的一般 decode。每一列都註明 KV 快取格式。

主角是我們的 MXFP4 檔（Swift-1.5 27B 與 Ornith-1.5-35B-A3B）；Qwen3.8-27B UD-Q4_K_M 只列為參考。96k token 起，每一格都標出兩個引擎實際用的 KV 格式，寫法是「KV WHIRL / llama.cpp」。

![16k 與 256k token context 之後的 decode 速度](images/bench_long_decode.png)

![伺服器 4 人同時使用的總吞吐](images/bench_server_concurrency.png)

![前綴已快取時的首 token 時間](images/bench_ttft_cache.png)

| R9700，greedy——WHIRL 對 llama.cpp（倍數） | Swift-1.5 27B MXFP4-A（dense） | Ornith-1.5-35B-A3B MXFP4（MoE，約 3B 啟用） | Qwen3.8-27B UD-Q4_K_M（dense，參考） |
|---|---|---|---|
| Decode tok/s，7 個寫程式提示（800 token）——WHIRL MTP+n-gram 對 llama.cpp 最快¹ | 112.4 vs 60.8（1.85×） | 257.6 vs 118.8（2.17×） | 97.3 vs 56.0（1.74×） |
| 16k token context 之後的 decode tok/s——WHIRL MTP+n-gram 對 llama.cpp 最快¹ | 71.4 vs 60.9⁴（1.17×） | 316.4 vs 106.9⁴（2.96×） | 80.7 vs 60.7⁴（1.33×） |
| Decode tok/s，同樣 7 個提示——WHIRL **no MTP** 對 llama.cpp **no MTP** | 38.1 vs 33.4（1.14×） | 177.4 vs 118.8（1.49×） | 34.8 vs 30.9（1.13×） |
| 伺服器 4 人並發總吞吐 tok/s（含 prefill）——WHIRL MTP+n-gram 對 llama.cpp 最快¹ | 182.1 vs 64.0⁴（2.84×） | 381.1 vs 176.8⁴（2.16×） | 122.4 vs 57.1⁴（2.14×） |
| Prefill tok/s，8k token 提示（各自的 CLI bench 工具） | 3,469 vs 1,338（2.59×） | 11,258 vs 4,637（2.43×） | 1,731 vs 1,223（1.42×） |
| Prefill tok/s，32k token 提示 | 2,907 vs 1,174（2.48×） | 8,633 vs 3,778（2.29×） | 1,570 vs 1,086（1.45×） |
| Prefill tok/s，96k token 提示 | 2,021.1 vs 886.7（2.28×）；KV f16 / f16 | 5,183.5 vs 2,602.6（1.99×）；KV f16 / f16 | 1,266.4 vs 837.3（1.51×）；KV f16 / f16 |
| Prefill tok/s，128k token 提示 | 1,757 vs 791.3（2.22×）；KV f16 / f16 | 4,311 vs 2,239（1.93×）；KV f16 / f16 | 1,158 vs 752.0（1.54×）；KV f16 / f16 |
| **Prefill tok/s，256k token 提示** | 969.8 vs 556.1（1.74×）；KV q8h / f16 | 2,135.2 vs 1,442.2（1.48×）；KV q8h / f16 | 742.2 vs 516.4（1.44×）；KV q8h / q8_0 |
| **256k token 提示之後的 decode tok/s**——WHIRL MTP+n-gram 對 llama.cpp 一般 decode³ | 35.1 vs 16.8（2.09×）；KV q8h / f16 | 115.6 vs 53.7（2.15×）；KV q8h / f16 | 33.5 vs 10.5（3.19×）；KV q8h / q8_0 |
| 首 token 時間：新對話重用 26k token 的系統提示（秒，越低越好） | 0.108 vs 0.384⁴（3.6×） | 0.098 vs 0.182⁴（1.9×） | 0.121 vs 0.369⁴（3.0×） |
| 首 token 時間：伺服器重啟後的 26k token session（從 SSD 層還原） | 0.613 s vs N/A² | 0.318 s vs N/A² | 0.672 s vs N/A² |

¹ 該列 llama.cpp 在 plain / MTP / MTP + n-gram 中最快者。

² llama.cpp 沒有自動的持久化 KV 快取。

³ llama-bench `tg64@d262144`：在 262,144 token 的 context 之後生成 64 個 token，一般 decode，沒有投機解碼。llama.cpp 跑 Swift-1.5 MXFP4 256k 用的是 f16 KV（放得下，31.3 GB）；Ornith-1.5 在 96k 與 256k 也是 f16 KV；Qwen3.8 Q4_K_M 256k 用 f16 會溢到共用記憶體，所以改用 q8_0 KV。

⁴ llama.cpp 為 v0.1.3 用與 WHIRL 相同的提示重量；採用的模式（量到的最快者）：16k 之後的 decode：Swift-1.5 MTP + n-gram、Ornith-1.5 plain、Qwen3.8 MTP + n-gram；4 人並發：Swift-1.5 plain、Ornith-1.5 plain、Qwen3.8 plain；26k 系統提示 TTFT：只量 plain（回答只有 1 個 token，投機解碼用不上）。

**投機解碼：最適用、主要、最差情況**——Swift-1.5 27B MXFP4-A，WHIRL 預設的 MTP + n-gram 對不開投機解碼；輸出逐位元相同：

| | 最適用：檔案編輯（內容重複） | 主要：coding agent 工作階段 | 長文件問答（引用多，128k 之後） | 最差：128k 之後寫全新內容（沒有可照抄的內容） |
|---|---|---|---|---|
| Decode tok/s，MTP + n-gram | 174.4 | 92.9 | 87.6 | 39.4 |
| Decode tok/s，不開投機解碼 | 39.1 | 32.8 | 25.3 | 25.2 |
| 加速倍數 | 4.47× | 2.83× | 3.47× | 1.56× |
| 每個 cycle 的 token 數 | 6.19 | — | 6.45 | 1.76 |

**int8 KV（q8h）的長上下文**——WHIRL 伺服器、單人、`--ctx-per-slot 262144`、q8h KV（key 與 value 存成 int8；query 與 key 先做 Hadamard 旋轉），這是 dense 模型跑 256k token 時使用的設定（MoE 模型以 `WHIRL_KV=q8h` 指定）：

| WHIRL 伺服器，q8h KV，單人 | Swift-1.5 27B MXFP4-A | Ornith-1.5-35B-A3B MXFP4 | Qwen3.8-27B UD-Q4_K_M（參考） |
|---|---|---|---|
| Prefill tok/s，64k token 提示 | 2,162.0 | 5,671.3 | 1,289.9 |
| Prefill tok/s，128k token 提示 | 1,533.0 | 3,673.6 | 1,032.2 |
| Prefill tok/s，192k token 提示 | 1,212.2 | 2,770.7 | 876.6 |
| Prefill tok/s，256k token 提示 | 969.8 | 2,135.2 | 742.2 |
| 128k 之後的 decode tok/s（MTP+n-gram） | 62.5 | 214.0 | 58.3 |
| 256k 之後的 decode tok/s（MTP+n-gram） | 35.1 | 115.6 | 33.5 |
| 長文問答粗略檢查（回答有提到被問的函式；不是正式 needle 測試） | 128k ✓ / 256k ✓ | 128k ✓ / 256k ✓ | 128k ✓ / 256k ✓ |

WHIRL 領先**不多**的地方：dense 模型的一般 decode（不開 MTP）兩邊都受記憶體頻寬限制（llama.cpp 的 1.13–1.14×）；16k token context 之後的 decode，dense 模型只快 1.17×（Swift-1.5）與 1.33×（Qwen3.8）；Qwen3.8 Q4_K_M 的 prefill 領先較少（96k 為 1.51×、256k token 為 1.44×，Swift-1.5 MXFP4 是 2.28× 與 1.74×）；Ornith-1.5 在 256k 的 prefill 領先 1.48×；投機解碼的最差情況（128k token 之後寫全新內容，沒有可照抄的內容）只快 1.56×。我們的 R9700 是 USB4 eGPU；prefill 與 decode 都在 VRAM 內，但還原與載入模型會經過主機連結。

完整方法與所有數字：[benchmarks.md](guide/zh-TW/benchmarks.md)。

## <a id="dflash2"></a>DFlash2 草稿（需自行開啟）

預設情況下，WHIRL 用模型自己的 MTP 層加上 n-gram 查表來產生草稿。對 dense 的 Qwen3.8-27B，可以改用獨立的草稿模型 **DFlash2**，它根據目標模型的 hidden state，每個 cycle 提出最多 7 個 token：

```powershell
.\whirl-server.exe C:\models\Qwen3.8-27B-UD-Q4_K_M.gguf --draft dflash --draft-model C:\models\dflash2-q8_0.gguf
```

`--draft dflash --draft-model PATH` 也適用於 `whirl chat` 與 `whirl bench`；n-gram 草稿仍然開著。預設仍是 `--draft mtp`。DFlash2 的草稿與 n-gram 草稿一樣經過驗證，所以輸出等於同一模式下單純 greedy 解碼：它只改變速度。找不到相符的草稿檔時，WHIRL 會警告並改用 MTP。

**草稿模型不隨發行包附上。** 請從 Hugging Face 下載：[incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)（Apache-2.0）；我們測試的是 Q8_0 的 GGUF。它佔用約 1.9 GiB VRAM，KV 池約縮小 12%（4 個 slot、balance：257,280 對 292,352 個 token）。

有沒有幫助取決於工作內容。在 R9700、Qwen3.8-27B UD-Q4_K_M、balance 上量測，tok/s，DFlash2 + n-gram 對預設的 MTP + n-gram：

| 工作內容 | MTP + n-gram（預設） | DFlash2 + n-gram | 變化 |
|---|---|---|---|
| **有幫助**（一位使用者、英文為主或可預測的輸出） | | | |
| Java API 程式碼，對話 | 105.7 | 125.6 | +19% |
| SQL 報表，對話 | 83.7 | 123.6 | +48% |
| Java API 程式碼，開啟思考 | 93.8 | 120.0 | +28% |
| 寫程式 agent 工作階段重播（91 步，12k 到 65k token），平均 | 83.4 | 86.9 | +4% |
| 同一工作階段，最後一步（65k token 提示） | 66.2 | 74.4 | +12% |
| **沒有幫助** | | | |
| 中文寫程式對話 | 100.9 | 91.3 | -10% |
| 4 人同時使用，總吞吐 | 211.3 | 129.7 | -39% |
| 37k token 提示之後的問答 | 54.6 | 39.1 | -28% |
| 108k token 提示之後的問答 | 40.3 | 25.8 | -36% |
| 128k token 之後寫全新內容 | 38.4 | 31.7 | -17% |
| 編輯檔案（1,536 token，大量複製） | 154.8 | 150.1 | -3% |

<!-- src：Java 與 SQL 列：audit/DFLASH-1.md 第 7 節「步驟 7」表（對話、balance、greedy、512 token；java-api 105.7／125.6、sql-report 83.7／123.6、java-api (think) 93.8／120.0）。agent 重播平均 83.4 -> 86.9（+4%）與最後一步 66.2 -> 74.4（+12%）：audit/REL3.md DFlash2 表（DFLASH-1 步驟 8 量到 83.05 -> 85.73，+3%，最後一步 67.4 -> 73.4，+9%）。中文對話 100.9 -> 91.3：audit/REL3.md。4 人 211.3 -> 129.7、37k 54.6 -> 39.1、108k 40.3 -> 25.8、128k 後寫新內容 38.4 -> 31.7、edit_file bench 154.8 -> 150.1：audit/DFLASH-1.md 第 7 節「步驟 8」（4 人列用第一輪）。VRAM 淨增 1.9 GiB、-np 4 balance 的 KV 池 257,280 對 292,352（-12%）：audit/DFLASH-1.md 第 7 節步驟 3。輸出中文比例與每 cycle token 數相關係數 -0.89：audit/DFLASH-2a.md 表第 1 列。 -->

適合：一位使用者、短到中等長度的 context、輸出容易預測的內容——API 或 SQL 樣板、英文程式碼、大量相似的修改。這些情況我們量到快 19% 到 48%。不建議：同時有多位使用者（慢 39%）、context 達 37k token 以上（慢 28% 到 36%）、以及以中文為主的輸出（我們的中文寫程式對話約慢 10%；草稿被接受的比例會隨輸出中文比例下降，在我們的提示上相關係數為 -0.89）。主要是複製既有文字的工作（例如編輯檔案）沒有增益，因為 n-gram 草稿本來就涵蓋了。只在 R9700、Qwen3.8-27B Q4_K_M 上量測過；Swift-1.5、MoE 模型與 Radeon 8060S 還沒有用 DFlash2 量測。

## <a id="ssd-快取"></a>SSD 快取：改了什麼，以及怎麼清掉

伺服器把閒置的 session 放在主記憶體層，也可以放到 SSD 層，這樣長提示可以直接還原而不必重算（重啟之後也是）。0.1.x 的預設是 session 閒置約 2 秒後就整個寫進 SSD。有一位使用者拿很多不同的長提示做壓測，23 分鐘內被寫入約 400 GB。我們檢查過寫入路徑，沒有發現隱藏的寫入放大（新 session 只寫一次，大小是檔案的 1.004 到 1.012 倍）；寫入量就是舊預設把每一個不同的提示（每個 0.3 到 1.5 GiB）都寫下去。0.2.0 改了預設值：

- **預設 `--kv-ssd-mode shutdown`：伺服器執行期間不寫 SSD。** Session 留在主記憶體層。正常停止（Ctrl+C）時，依最近使用的順序寫入，最多 60 秒，下次啟動會還原。當機或被強制結束，會失去尚未寫入的部分。
- **`--kv-ssd-mode always`** 仍然保留，給想要在執行期間把 session 放到磁碟的人。它只在 session 5 分鐘沒有變動後才寫（`--kv-ssd-delay SEC`，預設 300），而且只寫至少被重用過一次、或被後續回合接續過的 session，所以一連串只用一次的提示幾乎不會寫入。
- **磁碟上最多 64 GiB**（`--kv-ssd-gb`，預設 64）。上限以實際配置的位元組計算，其他版本、模型或模式留下的檔案也算進去（先刪），然後依最久未使用刪除 session。啟動時就會強制執行，不只在寫入時；寫到一半或損毀的快取檔會被刪除。
- **7 天沒有使用的檔案會被刪除**（啟動時與每小時）。還原一個 session 算作使用。
- `--kv-ssd-mode off` 或 `--kv-ssd-gb 0` 完全不寫 SSD 檔案；主記憶體層仍然有效。

其他版本（包含 0.1.x）寫下的快取檔不會被還原，它們計入上限並最先被刪除。

**手動刪除快取：** 停止伺服器，刪除 `%LOCALAPPDATA%\whirl\kvcache`（或您用 `--kv-ssd-dir` 指定的資料夾）中的 `*.wkv` 檔；快取就會從空的開始。那裡沒有存放其他東西。這些檔案含有您的提示（以 token id 存放）與其 KV 資料，電腦若是共用的，請刪除。細節：[使用參考](guide/zh-TW/usage.md)。

<!-- src：400 GB／23 分鐘的回報、寫入比 1.004-1.012、每個提示 0.3-1.5 GiB、60 秒 flush、最近使用優先 flush、5 分鐘延遲、seen-once 過濾、含外來檔的 64 GiB 上限、由還原更新的 7 天到期、啟動清理：audit/SSD-QUIET.md（含 SSD-QUIET-2）與 CHANGELOG.md Unreleased「SSD tier: far fewer disk writes」。其他版本的檔案不會還原：docs/usage.md「Safety」段；SSD-QUIET.md「valid file of another model / build / KV format」。 -->

## <a id="radeon-8060s"></a>Radeon 8060S（Ryzen AI Max+ 395），0.2.0 起正式支援

0.2.0 是第一個正式支援 Radeon 8060S（RDNA 3.5，gfx1151）的版本。兩套 GPU 程式碼在同一個 zip 裡。`whirl devices` 會列出 GPU；有 R9700 時預設用 R9700，`--device 8060s`（或 `WHIRL_DEVICE=8060s`）可指定 8060S。三種數值模式都能用，kernel 測試、golden 輸出與伺服器 gate（basic、並行請求、串流、tier 還原）在它上面都通過。8060S 上尚未提供：圖片輸入。

**主記憶體與 SSD 快取層預設開啟。** 8060S 的 prefill 很慢（Qwen3.8-27B Q4_K_M 在 8k 約 390 tok/s、128k 約 240 tok/s），所以還原 session 比在 R9700 上更划算。在 Windows 上，BIOS 會把記憶體分成 VRAM 保留區（KV 池放在這裡）與一般的作業系統 RAM，所以主記憶體層是另一塊 pin 住的作業系統記憶體，還原是真正的複製。Qwen3.8-27B UD-Q4_K_M、balance，session 回來時的首 token 時間：

| 提示 | 重新 prefill | 從主記憶體層 | 重啟後從 SSD |
|---|---|---|---|
| 8k token | 20.9 秒 | 0.11 秒 | 0.21 秒 |
| 32k token | 91.5 秒 | 0.15 秒 | 0.40 秒 |
| 65k token | 214.1 秒 | 0.22 秒 | 0.76 秒 |
| 98k token | 374.6 秒 | 0.30 秒 | 0.78 秒 |
| 128k token | 536.7 秒（約 9 分鐘） | 0.34 秒 | 2.00 秒 |

<!-- src：audit/TIER-8S.md 結果表（32k–128k 只量一次；8k 量了三次，雜訊約數 %；SSD 欄為檔案快取已暖的情況，清掉檔案快取後 0.24-1.82 秒；每個長度五個請求的輸出皆相同）。 -->

快取層不影響 decode 速度（300 token 的測試中 19.7 對 19.4 tok/s）。它的大小是從作業系統 RAM 扣的，所以依**作業系統可見**的 RAM 決定，而不是機器上的 128 GB：`clamp(作業系統可見 RAM / 8, 一個最大 context 的 session, 16 GiB)`，最多不超過啟動時可用 RAM 的 1/4，**低於 32 GB 則關閉**。64 GB 的機器上是 8 GiB；啟動 log 會寫出大小與原因。可用 `--kv-ram-mb N` 自己指定，`--kv-ram-mb 0` 關閉兩個主機端層，`--kv-ssd-gb 0` 只關 SSD 層。一個 128k 的 session 約需 5 GB，所以 8 GiB 的層一次放得下一個這樣的 session。

**precise 模式的 decode 快了約一倍。** precise 的 decode 矩陣乘法有了新 kernel（`gemvh`），位元與舊路徑完全相同。Qwen3.8-27B UD-Q4_K_M、`--precise`，tok/s，每一組的輸出都相同：

| Context | 一般 decode：之前 | 現在 | MTP + n-gram：之前 | 現在 |
|---|---|---|---|---|
| 短（144 token） | 5.34 | 11.19 | 20.2 | 40.2 |
| 32k | 4.9 | 9.27 | 15.1 | 23.0 |

<!-- src：「之前」= audit/GEMVH-1151.md 第 3 節的 WHIRL_PREC_GEMVH=0（短 plain 5.34/5.34、短 MTP 20.16/20.28、32k plain 4.97/4.83、32k MTP 15.41/14.78）；「現在」= audit/GEMVH-1151.md GEMVH-2 端到端表的「after」欄（短 plain 11.19/11.17、短 MTP 40.23/39.91、32k plain 9.27、32k MTP 23.01；32k 為單次，漂移 +-3-4%）。兩欄來自同一台 8060S 的兩次量測。 -->

增益隨 context 長度縮小：128k 時，第二階段的 kernel 單獨帶來 plain 6.44 對 5.35 tok/s、MTP 10.16 對 9.25 tok/s，因為深處的 attention 時間沒有變。balance 與 fast 本來就走較快的 decode 路徑，不受這個 kernel 影響。

**8060S 上的 fast 模式**使用 4-bit KV：Ornith-1.5-35B-A3B MXFP4 在 128k 的 decode 為 58.5 tok/s（f16 KV：32.7），64k 為 67.0 tok/s（f16 KV：46.4）；對 f16 KV 的 KL 為 0.0137（128k 程式碼提示）與 0.0177（64k）。多人同時使用時，fast 請求的草稿上限與 R9700 相同，所以同一個請求單獨跑或和其他人一起跑通常得到相同的文字；在 fast 模式這是盡力而為，不是保證。

<!-- src：4-bit KV 數字與 KL：CHANGELOG.md Unreleased「fast kvq4 on the Radeon 8060S」。並行：audit/CONC-8S.md「Limits」。 -->

**8060S 對 llama.cpp**（b11214 ROCm，相同的檔案與提示，greedy）：

| Radeon 8060S，0.2.0：WHIRL 對 llama.cpp | Qwen3.8-27B UD-Q4_K_M（dense） | Ornith-1.5-35B-A3B MXFP4（MoE） |
|---|---|---|
| Prefill tok/s，8k token 提示 | [8060S: TBD] | [8060S: TBD] |
| Prefill tok/s，32k token 提示 | [8060S: TBD] | [8060S: TBD] |
| Prefill tok/s，128k token 提示 | [8060S: TBD] | [8060S: TBD] |
| Decode tok/s，balance（預設），中文寫程式對話 | [8060S: TBD] | [8060S: TBD] |
| Decode tok/s，長文件之後 | [8060S: TBD] | [8060S: TBD] |
| Decode tok/s，precise | [8060S: TBD] | [8060S: TBD] |
| Decode tok/s，fast | [8060S: TBD] | [8060S: TBD] |
| 測試機器、功耗限制、llama.cpp 旗標 | [8060S: TBD] | [8060S: TBD] |

<!-- src：待 audit/REL3.md「8060S」一節填入（撰寫此稿時仍在量測）。docs/benchmarks.md 第 14 節的 0.1.x bring-up 數字早於三種模式、gemvh 與 tier 變更，不要再用。 -->

我們的 8060S 數字來自一台受散熱限制的筆電；同一顆晶片的桌機或迷你主機可能不同。

## 0.2.0 有什麼新東西

- **支援 Radeon 8060S**，主記憶體與 SSD 快取層預設開啟，precise 的 decode 也變快（[見上](#radeon-8060s)）。
- **三種數值模式**：precise、balance（預設，即 0.1.x 的行為）、fast（[見上](#三種數值模式)）。新增 `--precise`（f16 KV、沒有 fp8 或 int8 activation、加速逐位元相同）與 `--fast`（4-bit KV、放寬 MTP 接受）。
- **DFlash2 草稿，需自行開啟**（`--draft dflash --draft-model PATH`），適合單人、輸出可預測的情況（[見上](#dflash2)）。
- **SSD 快取預設只在關閉時寫入**，磁碟上限 64 GiB，7 天到期（[見上](#ssd-快取)）。
- **任何權重格式的 dense 模型 prefill 更快**（balance 與 fast）：以前只有 MXFP4 檔使用的 f16 DeltaNet 與 GEMM 輸出路徑，現在也用在 Q4_K_M、Q5_K、Q6_K、IQ4_XS 等。R9700 上的 Qwen3.8-27B UD-Q4_K_M：prefill 8k 1,849 到 2,042 tok/s（+10.4%）、32k 1,649 到 1,812（+9.9%）、128k 1,201 到 1,285（+7.0%）；decode 不變。
- **MTP 搭配 temperature > 0 的取樣仍然精確**（`specsample`）：同一個 seed 的文字在 MTP 開或關時都一樣。
- **外部程式碼審查的修正**，輸出不變：MTP 草稿視窗在錯誤路徑上也會還原；prefill 批次不會再縮到小於排程器的 chunk 大小；比記錄短的快取檔會被刪除，讀取不足會退回冷 prefill，而不是還原過期的位元組；還原複製失敗就視為失敗；SSD 目錄鎖失敗時關閉 SSD 層，而不是沒有鎖地繼續跑；DeltaNet 形狀是 kernel 沒實作的模型，載入時就拒絕。

<!-- src：CHANGELOG.md Unreleased：KG-1 數字（prefill 1849 -> 2042 等）、specsample、FIX-REVIEW2 各項；audit/FIX-REVIEW2.md 表第 1-7 列。 -->

較早的版本，在同一台機器上用同一個 GGUF 檔與同一組提示量測；兩版的 greedy 輸出逐位元相同。

### v0.1.0 → v0.1.3

| Swift-1.5 27B MXFP4-A，R9700 | v0.1.0 | v0.1.3 | 變化 |
|---|---|---|---|
| Prefill tok/s，32k token 提示（f16 KV） | 2,605.3 | 2,907.4 | +11.6% |
| Prefill tok/s，128k token 提示（f16 KV） | 1,407.3 | 1,756.7 | +24.8% |
| Prefill tok/s，128k token 提示（q8h KV，伺服器） | 1,352.7 | 1,533.0 | +13.3% |
| Prefill tok/s，256k token 提示（q8h KV，伺服器） | 832.5 | 969.8 | +16.5% |
| 128k token 文件問答的 decode tok/s（回答大量引用文件） | 63.7 | 87.6 | +37.5% |
| Decode tok/s，寫程式提示 | 110.7 | 120.7 | +9.1% |
| KV 池容量，預設 4 個 slot（token） | 210,688 | 226,304 | +7.4% |
| 4 人短提示同時送出：整批完成時間（秒，越低越好） | 9.57 | 8.62 | −9.9% |
| 長上下文 coding agent 請求的 decode tok/s（agent 工作階段 3 第 143 步，127.9k token） | 53.7 | 75.2 | +40.0% |
| agent 工作階段 3 到該請求為止的 decode tok/s（144 個請求平均） | 64.9 | 81.7 | +25.9% |

v0.1.2 參考值（同一台機器、輸出相同）：127.9k 那個請求 53.5 tok/s，整段 65.0 tok/s；提升來自 v0.1.3。

## Windows 安全性提示

`whirl.exe` 與 `whirl-server.exe` 沒有程式碼簽章。剛下載的檔案可能讓 Windows SmartScreen 顯示「Windows 已保護您的電腦」：按 **其他資訊 → 仍要執行**，或在解壓縮前先對 zip 執行 `Unblock-File`（只對從官方發布頁下載、且已核對 SHA-256 的 zip 這樣做）。**智慧型應用程式控制**設為*開啟*時，Windows 可能直接封鎖程式，而且沒有「仍要執行」選項。詳見 [Windows 安全性提示](windows_security_zh-TW.md)。

## 文件

| | 繁體中文 | English |
|---|---|---|
| 快速上手 | [quickstart_zh-TW.md](quickstart_zh-TW.md) | [quickstart.md](quickstart.md) |
| 情境食譜：接上用戶端、長時間 agent 工作階段、長上下文、多使用者、圖片、看懂 log、疑難排解 | [recipes_zh-TW.md](recipes_zh-TW.md) | [recipes.md](recipes.md) |
| 使用參考（所有指令、選項、環境變數、結束代碼） | [usage.md](guide/zh-TW/usage.md) | [usage.md](usage.md) |
| 伺服器（API、取樣、工具呼叫、批次、decode 保底、log） | [server.md](guide/zh-TW/server.md) | [server.md](guide/en/server.md) |
| 與 llama.cpp 的效能比較 | [benchmarks.md](guide/zh-TW/benchmarks.md) | [benchmarks.md](benchmarks.md) |
| Windows 安全性提示 | [windows_security_zh-TW.md](windows_security_zh-TW.md) | [windows_security.md](windows_security.md) |
| 從原始碼建置 | [building_zh-TW.md](building_zh-TW.md) | [building.md](building.md) |
| 指南首頁| [index.md](guide/zh-TW/index.md) | [index.md](guide/en/index.md) |
| 架構 | [architecture.md](guide/zh-TW/architecture.md) | [architecture.md](guide/en/architecture.md) |
| Kernel | [kernels.md](guide/zh-TW/kernels.md) | [kernels.md](guide/en/kernels.md) |
| 投機解碼（MTP + n-gram） | [speculative-decoding.md](guide/zh-TW/speculative-decoding.md) | [speculative-decoding.md](guide/en/speculative-decoding.md) |
| KV 快取、前綴快取、分層快取 | [kv-and-caching.md](guide/zh-TW/kv-and-caching.md) | [kv-and-caching.md](guide/en/kv-and-caching.md) |
| 量化與模型檔 | [quantization.md](guide/zh-TW/quantization.md) | [quantization.md](guide/en/quantization.md) |
| 圖片輸入 | [vision.md](guide/zh-TW/vision.md) | [vision.md](guide/en/vision.md) |
| Windows 上的 HIP | [windows-hip.md](guide/zh-TW/windows-hip.md) | [windows-hip.md](guide/en/windows-hip.md) |
| 量測方法 | [benchmarking.md](guide/zh-TW/benchmarking.md) | [benchmarking.md](guide/en/benchmarking.md) |
| **踩坑全集**（125 條，每條都有：症狀 → 根本原因 → 修正 → 檢查） | [pitfalls.md](guide/zh-TW/pitfalls.md) | [pitfalls.md](guide/en/pitfalls.md) |

## 開發與 AI 協助

WHIRL 是在專案擁有者（[@tsaipifong](https://github.com/tsaipifong)）的指導下，由 Anthropic 的 AI 模型 **Claude Opus 5.5** 參與設計、實作、最佳化與量測。Anthropic 與本專案沒有隸屬關係，也不為本專案背書。

## 授權與致謝

Apache License 2.0；見 [LICENSE](../LICENSE) 與 [NOTICE](../NOTICE)。第三方素材列在 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)，每個原始檔的來源列在 [PROVENANCE.md](../PROVENANCE.md)。

- [NInfer](https://github.com/Neroued/ninfer)——WHIRL 依循的設計理念（只借理念）。
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)——我們量測時的比較基準，也是 GGUF 與量化格式相容性的參考；少數改寫的部分（tokenizer 預切詞 / BPE 迴圈、查表、圖片前處理）採 MIT 授權，列在 THIRD_PARTY_NOTICES.md。
- [gufo](https://github.com/gufo-org/gufo) 與 [r9700-stack](https://github.com/bkvargyas/r9700-stack)——量測方法與最佳化想法。
- 模型作者：Qwen 團隊（Qwen3.8）、[unsloth](https://huggingface.co/unsloth)、[FreedomAISVR](https://huggingface.co/FreedomAISVR)、[ornith-ai](https://huggingface.co/ornith-ai)（Ornith-1.5），以及 [ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b)（Swift-1.5，我們的 MXFP4 檔就是由它量化而來）。
