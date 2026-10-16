[English](../README.md) | **繁體中文**

# WHIRL

**Windows HIP Inference for RDNA LLMs**

WHIRL 是 Windows 上 AMD GPU 的開源（Apache-2.0）LLM 推論引擎：原生 C++／HIP，每一支 kernel 都公開，不需要 WSL、Docker，也不用安裝 ROCm。下載 zip、解壓縮、執行，只需要 AMD 驅動。0.2.0 支援 **AMD Radeon AI PRO R9700**（RDNA 4，32 GB）與 **Radeon 8060S**（Ryzen AI Max+ 395），兩張卡上都比 llama.cpp 快，我們量過的每個 prompt 長度（2k 到 128k token）都是如此。

## WHIRL 對 llama.cpp：一眼看

WHIRL 0.2.0 預設模式（balance）對 ggml-org 官方 llama.cpp **b11214**，**ROCm** 與 **Vulkan** 兩個版本分開列。同一個 GGUF 檔、同一組提示、greedy 解碼、一個使用者。llama.cpp 每一格括號裡的百分比，是 WHIRL 領先該格的幅度。

**Radeon AI PRO R9700**（gfx1201，32 GB）。稠密＝Qwen3.8-27B UD-Q4_K_M，MoE＝Ornith-1.5-35B-A3B MXFP4。

<!-- gen:glance.r97.zh -->
| tok/s，越高越好 | 稠密 27B：WHIRL | llama.cpp ROCm | llama.cpp Vulkan | MoE 35B-A3B：WHIRL | llama.cpp ROCm | llama.cpp Vulkan |
|---|---|---|---|---|---|---|
| Prefill，2k prompt | **1,782** | 1,140 (WHIRL +56%) | 804.4 (WHIRL +122%) | **8,624** | 4,014 (WHIRL +115%) | 3,551 (WHIRL +143%) |
| Prefill，32k prompt | **1,719** | 1,089 (WHIRL +58%) | 761.2 (WHIRL +126%) | **6,651** | 3,872 (WHIRL +72%) | 3,395 (WHIRL +96%) |
| Prefill，128k prompt | **1,233** | 759.2 (WHIRL +62%) | 553.9 (WHIRL +123%) | **3,758** | 2,307 (WHIRL +63%) | 1,922 (WHIRL +95%) |
| Decode，2k token 文件問答 | **82.1** | 46.8 (WHIRL +75%) | 49.8 (WHIRL +65%) | **241.9** | 115.4 (WHIRL +110%) | 115.9 (WHIRL +109%) |
| Decode，32k token 文件問答 | **75.8** | 40.7 (WHIRL +86%) | 42.7 (WHIRL +78%) | **203.2** | 101.4 (WHIRL +100%) | 100.6 (WHIRL +102%) |
| Decode，128k token 文件問答 | **48.8** | 27.6 (WHIRL +77%) | 33.9 (WHIRL +44%) | **141.0** | 75.3 (WHIRL +87%) | 73.8 (WHIRL +91%) |
| Decode，中文寫程式對話（全新撰寫） | **103.1** | 61.4 (WHIRL +68%) | 61.2 (WHIRL +68%) | **250.1** | 118.9 (WHIRL +110%) | 115.1 (WHIRL +117%) |
<!-- /gen -->

**Radeon 8060S**（gfx1151，Ryzen AI Max+ 395）。**最大功率（80 W），ASUS ROG Flow Z13。** 模型相同。

<!-- gen:glance.s8.zh -->
| tok/s，越高越好 | 稠密 27B：WHIRL | llama.cpp ROCm | llama.cpp Vulkan | MoE 35B-A3B：WHIRL | llama.cpp ROCm | llama.cpp Vulkan |
|---|---|---|---|---|---|---|
| Prefill，2k prompt | **472.9** | 302.2 (WHIRL +56%) | 149.6 (WHIRL +216%) | **2,355** | 1,262 (WHIRL +87%) | 447.7¹ (WHIRL +426%) |
| Prefill，32k prompt | **396.4** | 259.1 (WHIRL +53%) | 201.9 (WHIRL +96%) | **1,794** | 1,049¹ (WHIRL +71%) | 756.2¹ (WHIRL +137%) |
| Prefill，128k prompt | **288.4** | 176.7¹ (WHIRL +63%) | 118.4¹ (WHIRL +144%) | **1,056** | 636.6¹ (WHIRL +66%) | 428.4¹ (WHIRL +146%) |
| Decode，2k token 文件問答 | **27.0** | 24.7 (WHIRL +9%) | 25.5 (WHIRL +6%) | **101.8** | 70.2 (WHIRL +45%) | 70.7¹ (WHIRL +44%) |
| Decode，32k token 文件問答 | **25.0** | 18.9 (WHIRL +32%) | 20.2 (WHIRL +24%) | **84.2** | 57.9¹ (WHIRL +46%) | 57.1¹ (WHIRL +48%) |
| Decode，128k token 文件問答 | **20.2** | 13.7¹ (WHIRL +48%) | 14.1¹ (WHIRL +44%) | **55.5** | 38.0¹ (WHIRL +46%) | 37.3¹ (WHIRL +49%) |
| Decode，中文寫程式對話（全新撰寫） | **33.3** | 25.9 (WHIRL +29%) | 27.9 (WHIRL +20%) | **106.3** | 70.6¹ (WHIRL +51%) | 72.9¹ (WHIRL +46%) |
<!-- /gen -->

¹ 只量一次（n = 1），請當作「約」。8060S 其餘每格都是 2 或 3 次的中位數；R9700 每個長度格都是單次量測，三個引擎同一天量完。

表中 llama.cpp 的解碼方式：稠密模型用它最快的投機解碼（MTP + n-gram，`--spec-type draft-mtp,ngram-mod`；R9700 上也量了一般解碼，每個稠密格都比較慢）。MoE 模型上 llama.cpp **不用**投機解碼，因為我們量到它的 MTP 在這個模型上比一般解碼慢；WHIRL 則用自己的 MTP 草稿。所以 MoE 的 decode 列是兩邊**各自的出廠設定**對比，不是同條件對比。含 WHIRL precise 與 fast 模式的完整表格在[效能](#效能)。

decode 列量的是什麼：每個長度都問同一個關於文件的問題（解釋文件開頭附近的一個函式，並寫出修正後的版本，256 token），所以回答有一部分可以照抄提示中的程式碼，這對所有使用投機草稿的引擎都有幫助。實際提示與指令：[benchmarks.md](guide/zh-TW/benchmarks.md#prompts)。寫程式對話那一列正好相反，是困難情況：回答全部是全新撰寫。

**8060S 上一段真實的 coding agent 工作階段**（最大功率同上）：16k token 的 skills／MCP 系統提示、一次讀取 3.6k token 原始檔的工具呼叫、四輪使用者提問（說明檔案、找出最複雜的函式、改寫它、審閱修改後的版本），每次呼叫都重送完整歷史，稠密 Qwen3.8-27B Q4_K_M：

<!-- gen:s8.agent.zh -->
| 工作階段 | WHIRL | llama.cpp ROCm | llama.cpp Vulkan |
|---|---|---|---|
| 冷啟動（20k token 系統＋工具提示尚未快取） | **101.6 s** | 137.7 s | 243.5 s |
| 第二次（共用同一前綴） | **45.7 s** | 52.5 s | 166.4 s |
<!-- /gen -->

冷啟動那一段，WHIRL 比 llama.cpp ROCm 少花 26% 時間，比 llama.cpp Vulkan 少 58%（各 2 輪的中位數）。改寫程式碼那一輪，WHIRL 解碼約 36–44 tok/s；這一輪大多是照抄檔案裡的程式碼，n-gram 草稿對每個引擎都有幫助（llama.cpp ROCm：冷啟動段 42.2 tok/s，第二段 111.1，第二段改寫的是另一個檔案），所以這一輪不算 WHIRL 領先。這段工作階段用 v0.2.0-rc7 預發布版量測；之後的候選版只改了 8060S 的 prefill forward 大小，0.2.0 的稠密模型用的就是 rc7 的大小。

**同一段工作階段在 R9700 上**（稠密 Qwen3.8-27B Q4_K_M，相同測試資料，各 2 輪中位數，冷啟動＝每輪重開 server）：

<!-- gen:r97.agent.zh -->
| 工作階段 | WHIRL | llama.cpp ROCm | llama.cpp Vulkan |
|---|---|---|---|
| 冷啟動（20k token 系統＋工具提示尚未快取） | **26.5 s** | 48.5 s | 61.9 s |
| 第二次（共用同一前綴） | **13.1 s** | 25.9 s | 29.0 s |
<!-- /gen -->

R9700 上冷啟動那一段，WHIRL 比 llama.cpp ROCm 少花 45% 時間，比 llama.cpp Vulkan 少 57%。改寫程式碼那一輪 WHIRL 解碼約 121–145 tok/s；和 8060S 一樣，這一輪大多是照抄程式碼（llama.cpp ROCm：冷啟動段 120.5 tok/s，第二段 183.7），所以不算 WHIRL 領先。

## 變快，但答案不變

WHIRL 0.2.0 有三種數值模式，每個行程選一種（`--precise`、`--balance`、`--fast`）；不加旗標時是 **balance**。

| 模式 | 是什麼 | KV 快取 | 輸出保證 |
|---|---|---|---|
| **precise** | 逐位元無損：除了 GGUF 本身的權重之外，不再量化任何東西；激活值全程 f16／f32 | f16 | 有沒有投機草稿、單獨跑或和其他使用者合批、命中前綴快取或還原工作階段，輸出位元都相同 |
| **balance**（預設） | 接近無損：解碼激活值用 int8（和 llama.cpp 解碼相同的 q8_1 做法）；R9700 上另有 f16-WMMA DeltaNet prefill，MXFP4 稠密檔另用 fp8 prefill 激活值 | 稠密：int8（`q8h`）；MoE：f16 | MTP／n-gram 草稿、合批、快取還原，輸出的 token 都和 balance 模式下單純 greedy 解碼相同。對 precise 的 KL：稠密模型約 1e-4 到 1e-3；MoE 模型約為模型本身雜訊底的 1.9 倍 |
| **fast**（自選） | 在通過關卡的可接受損失內求最高速度：balance 再加 4-bit KV 與放寬的 MTP 接受規則（稠密模型） | 4-bit（`q4`） | 有損，但在公布的 KL 門檻內；文字可能隨草稿與其他使用者而不同 |

實際上代表：

- 在 **precise 與 balance**，每一項加速都經過檢查，輸出的 token 和該模式下單純 greedy 解碼相同：MTP 與 n-gram 投機解碼、多人合批、共用系統提示的前綴快取命中、對同一份長文件的第二個問題、從 RAM 或 SSD 層還原的工作階段。常駐測試把關（`whirl selftest`、kernel 測試、server 關卡、兩張卡的 golden 輸出）。
- **precise 與 balance 絕不使用 4-bit KV、3-bit 權重或 fp8 attention。** 只有 fast 會用 4-bit KV，而且只在你要求時。
- **precise** 是有逐位元保證的模式：逐位元無損，輸出與單純 greedy 解碼完全相同。**balance** 接近無損，是預設。precise 要付出速度（R9700、中文寫程式對話、稠密模型：77.8 對 balance 的 103.1 tok/s）。
- **權重**就是你選的 GGUF，WHIRL 照檔案原樣計算。接續對話時沿用上一輪算好的狀態，數值上等價，但和把整段歷史當成一個 prompt 重新處理並非逐位元相同（[kv-and-caching.md](guide/zh-TW/kv-and-caching.md)）。

所有模式、項目與量到的 KL：[usage.md 的數值模式](usage.md#modes)。

## 需求

| | |
|---|---|
| **下載就能跑** | 從 [Releases](https://github.com/tsaipifong/whirl-llm/releases) 下載 **`whirl-0.2.0-windows-x64.zip`**，解壓縮、執行。**只需要 AMD 驅動**：不需要 HIP SDK、ROCm 或 Visual C++ 執行階段 |
| GPU | **AMD Radeon AI PRO R9700**（RDNA 4，gfx1201，32 GB）或 **Radeon 8060S**（RDNA 3.5，gfx1151，Ryzen AI Max+ 395 內顯）；一個行程用一張 GPU |
| 作業系統 | Windows 11 64 位元 |
| 驅動 | **AMD Software: Adrenalin Edition 26.8.1（驅動 32.0.31041.1004）或更新版**，兩張卡都一樣。更舊的版本未測試 |
| 記憶體／磁碟 | server 會鎖定一部分主記憶體作 RAM 快取層（R9700 主機：約 RAM 的 1/4，64 GB 電腦為 16 GiB；Radeon 8060S：Windows 看得到的 RAM 的 1/8，64 GB 機器為 8 GiB，低於 32 GB 則關閉），SSD 層最多用 64 GiB，只在 server 停止時寫入；兩者都可調整或關閉（`--kv-ram-mb`、`--kv-ssd-gb`） |
| 進階：從原始碼建置 | 給想自己修改、調校 WHIRL 的使用者：HIP SDK 7.2 + Visual Studio 2022 Build Tools（MSVC）+ CMake + Ninja；見 [building_zh-TW.md](building_zh-TW.md) |

支援範圍：WHIRL 正式支援 R9700 與 Radeon 8060S，目標是 **16 GB 以上**的卡。其他 AMD GPU 未測試（RX 9070 系列和 R9700 同晶片、16 GB，未測試）。16 GB 以下的卡不在正式支援範圍，歡迎 fork 原始碼自行修改。

我們的 R9700 是用 USB4／Thunderbolt 外接，可以正常運作。直接插在 PCIe 插槽的卡，在資料要經過主機連線的地方（載入模型、RAM／SSD 層還原）數字可能略有不同，prefill 與 decode 不受影響。

## 快速上手

1. 從 [Releases](https://github.com/tsaipifong/whirl-llm/releases) 下載 `whirl-0.2.0-windows-x64.zip`，以及一個[支援的模型](#支援的模型)。
2. 解壓縮後執行，在 PowerShell：

```powershell
Unblock-File .\whirl-0.2.0-windows-x64.zip
Expand-Archive .\whirl-0.2.0-windows-x64.zip -DestinationPath C:\whirl
cd C:\whirl\whirl-0.2.0-windows-x64
.\whirl.exe devices                                    # 應列出 R9700（gfx1201）和／或 Radeon 8060S（gfx1151）
.\whirl.exe chat C:\models\Qwen3.8-27B-UD-Q4_K_M.gguf "用兩句話解釋 TCP slow start。" --max-tokens 400
.\whirl-server.exe C:\models\Qwen3.8-27B-UD-Q4_K_M.gguf --port 8080      # 預設模式：balance
```

server 在 `http://127.0.0.1:8080/v1` 提供 OpenAI API（`/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/health`），任何相容 OpenAI 的用戶端都能用。加 `--precise` 得到逐位元無損輸出，加 `--fast` 追求最高速度。用 Ctrl+C 停止 server，會先把重複使用過的工作階段寫進 SSD 層。

兩張卡都在時預設用 R9700；`--device 8060s`（或 `WHIRL_DEVICE=8060s`）選 8060S。長文件可加大每個 slot 的上下文，例如 `--ctx-per-slot 262144`；見 [recipes_zh-TW.md](recipes_zh-TW.md#long-context)。

每個模型第一次執行時會調校一次 GPU kernel，結果快取在 `%LOCALAPPDATA%\whirl`（R9700 上 27B Q4_K_M 約 1.5 分鐘，MXFP4 檔幾秒；8060S 較久）。完整步驟：[快速上手](quickstart_zh-TW.md)。所有選項：[使用參考](guide/zh-TW/usage.md)。

**進階：從原始碼建置**——給想自己修改、調校 WHIRL 的使用者：[building_zh-TW.md](building_zh-TW.md)。

## 支援的模型

WHIRL 讀取 GGUF 的 `general.architecture`，只接受 `qwen35` 與 `qwen35moe`（Gated DeltaNet + gated attention 混合架構，可附 MTP head），以及保留此架構的微調模型。其他架構（Llama、Gemma、Mistral、DeepSeek、`qwen3`、`qwen2`……）載入時會被拒絕並給出清楚訊息；WHIRL 沒有 kernel 的張量型別（例如 NVFP4）也一樣；沒有慢速的通用退路。

> **我們為 WHIRL 製作的 MXFP4 量化：**
> - 稠密 27B：[**Swift-1.5-Qwen3.8-27b-MXFP4-GGUF**](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF)，變體 A（`Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`）。Swift-1.5 的思考也比基底模型短很多（4 題中文寫程式提示的小樣本，思考 token 比 Qwen3.8-27B MXFP4 少 44%；見模型卡）。
> - MoE 35B（約 3B 啟用）：[**Ornith-1.5-35B-A3B-MXFP4-GGUF**](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF)。

測試過的 GGUF 檔（可載入、greedy 輸出檢查過、precise 與 balance 下 MTP 輸出與單純 greedy 相同）：

| 模型檔（Hugging Face） | 類型 | 架構 | 量化 |
|---|---|---|---|
| [tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF) `…-A-outQ6_K.gguf`（推薦）、`…-B-outQ8_0.gguf`、`…-C-outQ4_K.gguf` | 稠密 27B | qwen35 | MXFP4（MTP／embedding 為 Q8_0）；我們為 WHIRL 量化 |
| [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `Qwen3.8-27B-UD-Q4_K_M.gguf` | 稠密 27B | qwen35 | UD-Q4_K_M（0.2.0 評測用的稠密模型） |
| [FreedomAISVR/Qwen3.8-27B-MXFP4-GGUF](https://huggingface.co/FreedomAISVR/Qwen3.8-27B-MXFP4-GGUF) `qwen3.8-27b-mxfp4.gguf` | 稠密 27B | qwen35 | MXFP4 |
| [`Ornith-1.5-35B-A3B-MXFP4.gguf`](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF)（我們量化） | MoE 35B，約 3B 啟用 | qwen35moe | 專家與稠密矩陣 MXFP4（MTP／embedding Q8_0，輸出頭 Q6_K） |
| [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) `Ornith-1.5-35B-Q4_K_M.gguf` | MoE 35B，約 3B 啟用 | qwen35moe | Q4_K_M |

圖片輸入透過 `--mmproj` 使用 Qwen3-VL 式的 mmproj（F16／BF16），兩張卡都可用。稠密 27B 另可選用 DFlash2 草稿模型（`--draft dflash`，見[下方](#020-新功能)）。其他 `qwen35`／`qwen35moe` GGUF 應該能載入，但未測試。選擇與製作 GGUF：[quantization.md](guide/zh-TW/quantization.md)。

## 效能

WHIRL 0.2.0 對 ggml-org 官方 llama.cpp b11214 Windows 執行檔，ROCm 與 Vulkan 兩個版本並列。同一個 GGUF 檔、同一組提示（文件問答提示，第一行各不相同，不會命中前綴快取；生成 256 token）、greedy 解碼、關閉思考、一個使用者，tok/s 取自各 server 自己的 timings。方法、旗標與原始數據：[benchmarks.md](guide/zh-TW/benchmarks.md)。

### Radeon AI PRO R9700

列出 WHIRL 三種模式；領先幅度以預設的 balance 計算。每格都是單次量測，所有引擎在同一天（2026-10-16）量完；差距在 5% 左右以內請視為雜訊。llama.cpp 的 prefill 取它一般與投機兩次量測中較快者。

<!-- gen:r97.qwen.pp.zh -->
#### Qwen3.8-27B UD-Q4_K_M（稠密）: Prefill tok/s

| 長度 | WHIRL balance | WHIRL precise | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | **1,782** | 1,615 | 1,768 | 1,140 | 804.4 | +56% | +122% |
| 8k | **1,872** | 1,684 | 1,843 | 1,210 | 834.6 | +55% | +124% |
| 32k | **1,719** | 1,565 | 1,701 | 1,089 | 761.2 | +58% | +126% |
| 64k | **1,516** | 1,404 | 1,507 | 947.0 | 675.7 | +60% | +124% |
| 96k | **1,359** | 1,272 | 1,352 | 843.9 | 608.8 | +61% | +123% |
| 128k | **1,233** | 1,163 | 1,226 | 759.2 | 553.9 | +62% | +123% |
<!-- /gen -->

<!-- gen:r97.qwen.tg.zh -->
#### Qwen3.8-27B UD-Q4_K_M（稠密）: Decode tok/s，文件問答（256 token 回答：說明文件中的一個函式並改寫）

| 長度 | WHIRL balance | WHIRL precise | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | **82.1** | 60.6 | 92.2 | 46.8 (MTP+n-gram) | 49.8 (MTP+n-gram) | +75% | +65% |
| 8k | **84.1** | 64.4 | 96.9 | 46.5 (MTP+n-gram) | 47.4 (MTP+n-gram) | +81% | +77% |
| 32k | **75.8** | 58.1 | 78.5 | 40.7 (MTP+n-gram) | 42.7 (MTP+n-gram) | +86% | +78% |
| 64k | **58.8** | 48.5 | 63.8 | 36.4 (MTP+n-gram) | 36.0 (MTP+n-gram) | +62% | +63% |
| 96k | **56.1** | 45.2 | 54.6 | 34.2 (MTP+n-gram) | 39.2 (MTP+n-gram) | +64% | +43% |
| 128k | **48.8** | 42.1 | 58.1 | 27.6 (MTP+n-gram) | 33.9 (MTP+n-gram) | +77% | +44% |
<!-- /gen -->

<!-- gen:r97.ornith.pp.zh -->
#### Ornith-1.5-35B-A3B MXFP4（MoE，約 3B 啟用）: Prefill tok/s

| 長度 | WHIRL balance | WHIRL precise | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | **8,624** | 7,502 | 11,477 | 4,014 | 3,551 | +115% | +143% |
| 8k | **7,733** | 6,756 | 10,794 | 4,487 | 3,912 | +72% | +98% |
| 32k | **6,651** | 5,948 | 8,435 | 3,872 | 3,395 | +72% | +96% |
| 64k | **5,333** | 4,892 | 6,440 | 3,173 | 2,735 | +68% | +95% |
| 96k | **4,409** | 4,098 | 5,158 | 2,667 | 2,280 | +65% | +93% |
| 128k | **3,758** | 3,528 | 4,283 | 2,307 | 1,922 | +63% | +95% |
<!-- /gen -->

<!-- gen:r97.ornith.tg.zh -->
#### Ornith-1.5-35B-A3B MXFP4（MoE，約 3B 啟用）: Decode tok/s，文件問答（256 token 回答：說明文件中的一個函式並改寫）

| 長度 | WHIRL balance | WHIRL precise | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | **241.9** | 182.0 | 222.7 | 115.4 | 115.9 | +110% | +109% |
| 8k | **218.1** | 188.1 | 207.1 | 112.3 | 111.9 | +94% | +95% |
| 32k | **203.2** | 161.1 | 186.2 | 101.4 | 100.6 | +100% | +102% |
| 64k | **174.2** | 142.8 | 182.9 | 90.7 | 89.6 | +92% | +94% |
| 96k | **156.6** | 132.2 | 166.4 | 81.9 | 81.1 | +91% | +93% |
| 128k | **141.0** | 119.4 | 158.4 | 75.3 | 73.8 | +87% | +91% |
<!-- /gen -->

MoE decode：llama.cpp 用一般解碼（出廠設定；它的 MTP 在 12 個 MoE 格子中有 11 格比一般解碼慢，第 12 格快 0.4 tok/s），WHIRL 用 MTP。

**中文寫程式對話**（512 token；WHIRL 取 3 次中位數，llama.cpp 取第 1 次，因為它的 n-gram 草稿會學會重複的提示）。這一列對投機草稿是困難情況：中文提問、中文說明與英文程式碼都是全新撰寫，提示裡除了幾個選項名稱之外沒有可讓回答照抄的內容（投機解碼在回答可預測時最快，例如照抄或背誦已知文字；這一列沒有這種情況）。

<!-- gen:r97.chat.zh -->
| 模型 | WHIRL precise | WHIRL balance | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B UD-Q4_K_M（稠密） | 77.8 | **103.1** | 119.3 | 61.4 (MTP+n-gram) | 61.2 (MTP+n-gram) | +68% | +68% |
| Ornith-1.5-35B-A3B MXFP4（MoE，約 3B 啟用） | 196.8 | **250.1** | 229.0 | 118.9 (一般) | 115.1 (一般) | +110% | +117% |
<!-- /gen -->

### <a id="radeon-8060s"></a>Radeon 8060S

**最大功率（80 W），ASUS ROG Flow Z13**（Ryzen AI Max+ 395，128 GB，Windows 看得到 63.6 GB）。WHIRL balance；除非標註，每格是 2–3 次的中位數。

<!-- gen:s8.qwen.pp.zh -->
#### Qwen3.8-27B UD-Q4_K_M（稠密）: Prefill tok/s

| 長度 | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL 對 ROCm | WHIRL 對 Vulkan |
|---|---|---|---|---|---|
| 2k | **472.9** | 302.2 | 149.6 | +56% | +216% |
| 32k | **396.4** | 259.1 | 201.9 | +53% | +96% |
| 64k | **346.9** | 221.7 | 161.3 | +57% | +115% |
| 96k | **310.4** | 195.5 | 137.0 | +59% | +127% |
| 128k | **288.4** | 176.7¹ | 118.4¹ | +63% | +144% |
<!-- /gen -->

<!-- gen:s8.qwen.tg.zh -->
#### Qwen3.8-27B UD-Q4_K_M（稠密）: Decode tok/s，文件問答（256 token 回答：說明文件中的一個函式並改寫）

| 長度 | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL 對 ROCm | WHIRL 對 Vulkan |
|---|---|---|---|---|---|
| 2k | **27.0** | 24.7 | 25.5 | +9% | +6% |
| 32k | **25.0** | 18.9 | 20.2 | +32% | +24% |
| 64k | **22.6** | 17.9 | 18.2 | +27% | +24% |
| 96k | **21.4** | ~17.7² | 16.7 | ~+21%² | +28% |
| 128k | **20.2** | 13.7¹ | 14.1¹ | +48% | +44% |
<!-- /gen -->

<!-- gen:s8.ornith.pp.zh -->
#### Ornith-1.5-35B-A3B MXFP4（MoE，約 3B 啟用）: Prefill tok/s

| 長度 | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL 對 ROCm | WHIRL 對 Vulkan |
|---|---|---|---|---|---|
| 2k | **2,355** | 1,262 | 447.7¹ | +87% | +426% |
| 32k | **1,794** | 1,049¹ | 756.2¹ | +71% | +137% |
| 64k | **1,417** | 848.8 | 603.6¹ | +67% | +135% |
| 96k | **1,178** | 710.4 | 500.5¹ | +66% | +135% |
| 128k | **1,056** | 636.6¹ | 428.4¹ | +66% | +146% |
<!-- /gen -->

<!-- gen:s8.ornith.tg.zh -->
#### Ornith-1.5-35B-A3B MXFP4（MoE，約 3B 啟用）: Decode tok/s，文件問答（256 token 回答：說明文件中的一個函式並改寫）

| 長度 | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL 對 ROCm | WHIRL 對 Vulkan |
|---|---|---|---|---|---|
| 2k | **101.8** | 70.2 | 70.7¹ | +45% | +44% |
| 32k | **84.2** | 57.9¹ | 57.1¹ | +46% | +48% |
| 64k | **72.9** | 49.0 | 48.4¹ | +49% | +51% |
| 96k | **62.0** | 42.7 | 42.2¹ | +45% | +47% |
| 128k | **55.5** | 38.0¹ | 37.3¹ | +46% | +49% |
<!-- /gen -->

¹ 只量一次，請當作「約」。² 不可靠：兩次量測相差 31%（20.1 與 15.3 tok/s；llama.cpp 的 MTP + n-gram 接受率不穩），因此這一格的領先幅度不在其他地方引用。稠密 decode：llama.cpp 用 MTP + n-gram，WHIRL 用 MTP。MoE decode：llama.cpp 用一般解碼（出廠設定），WHIRL 用 MTP。

**8060S 上的中文寫程式對話**（與 R9700 相同提示，即上述困難情況；WHIRL 用 v0.2.0-rc7 預發布版量測，其解碼路徑在 0.2.0 沒有改變）：

<!-- gen:s8.chat.zh -->
| 模型 | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL 對 ROCm | WHIRL 對 Vulkan |
|---|---|---|---|---|---|
| Qwen3.8-27B UD-Q4_K_M（稠密） | **33.3** | 25.9 | 27.9 | +29% | +20% |
| Ornith-1.5-35B-A3B MXFP4（MoE，約 3B 啟用） | **106.3** | 70.6¹ | 72.9¹ | +51% | +46% |
<!-- /gen -->

### <a id="swift-mxfp4"></a>Swift-1.5 27B MXFP4（我們的稠密量化）

我們自己製作的稠密 27B MXFP4 量化（`Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`，變體 A），提示與旗標同上，同一天稍晚用 0.2.0 候選版量測。R9700 上 WHIRL 的 MXFP4 prefill 路徑比 Q4_K_M 快很多，所以 prefill 領先比 Q4_K_M 大：對 llama.cpp ROCm +115% 到 +166%、對 Vulkan +194% 到 +296%；decode +50% 到 +75%／+43% 到 +78%。8060S：prefill +56% 到 +62%／+87% 到 +207%，decode +27% 到 +48%／+24% 到 +107%。

**R9700**（每格單次量測；llama.cpp prefill 取它一般與 MTP + n-gram 兩次量測中較快者）：

<!-- gen:r97.swift.pp.zh -->
#### Swift-1.5-Qwen3.8-27B MXFP4-A（稠密 27B）: Prefill tok/s

| 長度 | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL 對 ROCm | WHIRL 對 Vulkan |
|---|---|---|---|---|---|
| 2k | **3,263** | 1,227 | 823.2 | +166% | +296% |
| 32k | **2,822** | 1,174 | 815.4 | +140% | +246% |
| 64k | **2,334** | 1,010 | 718.8 | +131% | +225% |
| 96k | **1,974** | 892.5 | 645.3 | +121% | +206% |
| 128k | **1,716** | 798.5 | 584.0 | +115% | +194% |
<!-- /gen -->

<!-- gen:r97.swift.tg.zh -->
#### Swift-1.5-Qwen3.8-27B MXFP4-A（稠密 27B）: Decode tok/s，文件問答（256 token 回答：說明文件中的一個函式並改寫）

| 長度 | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL 對 ROCm | WHIRL 對 Vulkan |
|---|---|---|---|---|---|
| 2k | **87.3** | 50.9 (MTP+n-gram) | 50.2 (MTP+n-gram) | +72% | +74% |
| 32k | **76.9** | 44.0 (MTP+n-gram) | 43.1 (MTP+n-gram) | +75% | +78% |
| 64k | **64.0** | 38.6 (MTP+n-gram) | 39.5 (MTP+n-gram) | +66% | +62% |
| 96k | **60.2** | 36.1 (MTP+n-gram) | 40.7 (MTP+n-gram) | +67% | +48% |
| 128k | **49.8** | 33.2 (MTP+n-gram) | 34.8 (MTP+n-gram) | +50% | +43% |
<!-- /gen -->

**Radeon 8060S**，最大功率（80 W），ASUS ROG Flow Z13（每格為 2 或 3 次中位數；llama.cpp 用 MTP + n-gram）：

<!-- gen:s8.swift.pp.zh -->
#### Swift-1.5-Qwen3.8-27B MXFP4-A（稠密 27B）: Prefill tok/s

| 長度 | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL 對 ROCm | WHIRL 對 Vulkan |
|---|---|---|---|---|---|
| 2k | **500.9** | 310.1 | 162.9 | +62% | +207% |
| 32k | **414.9** | 266.2 | 221.6 | +56% | +87% |
| 64k | **368.1** | 230.9 | 177.3 | +59% | +108% |
| 96k | **324.2** | 203.3 | 149.0 | +59% | +118% |
| 128k | **290.6** | 180.8 | 127.8 | +61% | +128% |
<!-- /gen -->

<!-- gen:s8.swift.tg.zh -->
#### Swift-1.5-Qwen3.8-27B MXFP4-A（稠密 27B）: Decode tok/s，文件問答（256 token 回答：說明文件中的一個函式並改寫）

| 長度 | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL 對 ROCm | WHIRL 對 Vulkan |
|---|---|---|---|---|---|
| 2k | **31.0** | 24.5 | 25.1 | +27% | +24% |
| 32k | **29.0** | 22.7 | 17.0 | +28% | +71% |
| 64k | **24.9** | 18.4 | 18.0 | +35% | +38% |
| 96k | **23.6** | 17.9 | 16.6 | +32% | +42% |
| 128k | **24.0** | 16.2 | 11.6 | +48% | +107% |
<!-- /gen -->

**中文寫程式對話**（512 token，對草稿是困難情況：全部全新撰寫，沒有可照抄的內容）：

<!-- gen:swift.chat.zh -->
| GPU | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL 對 ROCm | WHIRL 對 Vulkan |
|---|---|---|---|---|---|
| Radeon AI PRO R9700 | **110.7** | 57.8 (MTP+n-gram) | 57.6 (MTP+n-gram) | +92% | +92% |
| Radeon 8060S（最大功率） | **33.6** | 27.6 (MTP+n-gram) | 28.3 (MTP+n-gram) | +22% | +19% |
<!-- /gen -->

### WHIRL 領先不多的地方

- **8060S 上稠密 decode，2k token 文件問答**：對 llama.cpp ROCm 領先 9%、對 Vulkan 6%，在量測雜訊範圍內。8060S 的稠密 decode 受記憶體頻寬限制（約 256 GB/s，權重 15 GB 以上），領先來自投機解碼，並隨上下文變長而擴大（32k 起 +24% 到 +48%）。
- **不用投機的一般解碼**在每個引擎都受記憶體頻寬限制；WHIRL 大部分的 decode 領先來自 MTP + n-gram 投機解碼。
- **fast 不一定比 balance 快**：R9700 上 MoE 模型在短、中等上下文時，balance 解碼比 fast 快；fast 主要幫助 prefill 與稠密 decode。
- **precise 要付出速度**：R9700 上比 balance 慢 14–26%（decode）與 6–13%（prefill）（見上表）。

## 0.2.0 新功能

- **支援 Radeon 8060S（Ryzen AI Max+ 395）並已調校**：兩個模型在每個量過的 prompt 長度都比 llama.cpp ROCm 與 Vulkan 快（見上表）。8060S 也能圖片輸入，RAM 與 SSD 快取層預設開啟（Windows 上 BIOS 的 VRAM carve-out 與作業系統 RAM 是兩個獨立的池，所以 128k token 的工作階段從 RAM 還原 0.34 秒、從 SSD 2 秒，不必重跑 537 秒的 prefill）。
- **三種數值模式**：precise（逐位元無損）、balance（預設，接近無損）、fast（自選，損失受關卡管控）。見[上方](#變快但答案不變)。
- **DFlash2 草稿，自選**（`--draft dflash --draft-model Qwen3.8-27B-DFlash2-Q8_0.gguf`，草稿模型在 Hugging Face `incoai/Qwen3.8-27B-DFlash2-GGUF`，不隨附）：R9700 上一個使用者、短或中等上下文、可預測的輸出時，比 MTP 快 19% 到 48%；4 個使用者時較慢（約 −39%），37k token 以上上下文與以中文為主的輸出（約 −10%）也較慢；約佔 1.9 GiB VRAM。greedy 輸出與同模式的單純 greedy 解碼相同。
- **SSD 快取寫入大幅減少**：預設只在 server 停止時（或閒置約 5 分鐘、或 RAM 必須淘汰時）寫入被重複使用過的工作階段；只用一次的 prompt 永遠不寫。快取在磁碟上永遠不超過 `--kv-ssd-gb`（預設 64 GiB），7 天沒用的檔案會刪除。（以前對許多不同長 prompt 做評測，可能寫入數百 GB。）
- **對同一份長文件問第二個問題時沿用它**：server 保留尾端檢查點，只需 prefill 新問題（約 0.3k–1.3k token），不必重跑整份文件。
- **CLI 與 server 對同一個 prompt 輸出逐位元相同**（相同的 prefill 排程）。

完整清單：[CHANGELOG.md](../CHANGELOG.md)。

## Windows 安全性提示

`whirl.exe` 與 `whirl-server.exe` 沒有程式碼簽章。剛下載的檔案可能讓 Windows SmartScreen 顯示「Windows 已保護您的電腦」：按 **其他資訊 → 仍要執行**，或在解壓縮前先對 zip 執行 `Unblock-File`（只對從官方發布頁下載、且已核對 SHA-256 的 zip 這樣做）。**智慧型應用程式控制**設為*開啟*時，Windows 可能直接封鎖程式，而且沒有「仍要執行」選項。詳見 [Windows 安全性提示](windows_security_zh-TW.md)。

## 文件

| | 繁體中文 | English |
|---|---|---|
| 快速上手 | [quickstart_zh-TW.md](quickstart_zh-TW.md) | [quickstart.md](quickstart.md) |
| 情境食譜：接上用戶端、長時間 agent 工作階段、長上下文、多使用者、圖片、看懂 log、疑難排解 | [recipes_zh-TW.md](recipes_zh-TW.md) | [recipes.md](recipes.md) |
| 使用參考（所有指令、選項、數值模式、環境變數、結束代碼） | [usage.md](guide/zh-TW/usage.md) | [usage.md](usage.md) |
| 伺服器（API、取樣、工具呼叫、批次、decode 保底、log） | [server.md](guide/zh-TW/server.md) | [server.md](guide/en/server.md) |
| 與 llama.cpp 的效能比較 | [benchmarks.md](guide/zh-TW/benchmarks.md) | [benchmarks.md](benchmarks.md) |
| Windows 安全性提示 | [windows_security_zh-TW.md](windows_security_zh-TW.md) | [windows_security.md](windows_security.md) |
| 進階：從原始碼建置 | [building_zh-TW.md](building_zh-TW.md) | [building.md](building.md) |
| 指南首頁 | [index.md](guide/zh-TW/index.md) | [index.md](guide/en/index.md) |
| 架構 | [architecture.md](guide/zh-TW/architecture.md) | [architecture.md](guide/en/architecture.md) |
| Kernel | [kernels.md](guide/zh-TW/kernels.md) | [kernels.md](guide/en/kernels.md) |
| 投機解碼（MTP + n-gram、DFlash2） | [speculative-decoding.md](guide/zh-TW/speculative-decoding.md) | [speculative-decoding.md](guide/en/speculative-decoding.md) |
| KV 快取、前綴快取、分層快取 | [kv-and-caching.md](guide/zh-TW/kv-and-caching.md) | [kv-and-caching.md](guide/en/kv-and-caching.md) |
| 量化與模型檔 | [quantization.md](guide/zh-TW/quantization.md) | [quantization.md](guide/en/quantization.md) |
| 圖片輸入 | [vision.md](guide/zh-TW/vision.md) | [vision.md](guide/en/vision.md) |
| Windows 上的 HIP | [windows-hip.md](guide/zh-TW/windows-hip.md) | [windows-hip.md](guide/en/windows-hip.md) |
| 量測方法 | [benchmarking.md](guide/zh-TW/benchmarking.md) | [benchmarking.md](guide/en/benchmarking.md) |
| **踩坑全集**（每條都有：症狀 → 根本原因 → 修正 → 檢查） | [pitfalls.md](guide/zh-TW/pitfalls.md) | [pitfalls.md](guide/en/pitfalls.md) |

## 設計理念

1. **專為 Windows 上的 AMD GPU 打造**：原生 Windows 的 C++／HIP 推論引擎，針對 RDNA 架構逐一手寫、調校 kernel。不需要 WSL、Docker 或 Linux 虛擬機，裝好 AMD 驅動就能跑。
2. **完整開源，包含每一支 kernel**：從 host 端排程到 GPU kernel，全部原始碼以 Apache-2.0 公開，沒有封閉的二進位元件。歡迎回報問題、提供量測數據，或直接貢獻程式碼。
3. **精度優先於速度**：在 precise 與 balance 模式下，所有加速手段（投機解碼、多人合批、前綴快取、快取還原）輸出的 token 都和該模式的單純 greedy 解碼相同，有常駐測試把關。任何損失更多的做法都是自選（`--fast`），並受 KL 與準確度門檻管控。
4. **少數模型，做到極致**：針對特定模型與硬體，把效能推到接近硬體極限，支援的架構一個一個加。目前支援 `qwen35`（稠密）與 `qwen35moe`（混合專家）。
5. **數字透明、可驗證**：所有數字都是實測，量測方法、指令和原始數據全部公開，包括 WHIRL 領先不多的地方。
6. **為真實工作流程設計**：以 coding agent 和長上下文為核心場景：長 prompt 快速處理、多人同時使用時主對話不卡頓、VRAM → RAM → SSD 三層快取，server 重啟後對話也能接回來。

WHIRL 依循 [NInfer](https://github.com/Neroued/ninfer)（Apache-2.0）的設計理念——為選定的模型與選定的 GPU 打造，盡可能逼近硬體極限；[zynfer](https://github.com/thanos/zynfer) 把同樣的想法帶到了 RDNA 4。

## 開發與 AI 協助

WHIRL 是在專案擁有者（[@tsaipifong](https://github.com/tsaipifong)）的指導下，由 Anthropic 的 AI 模型 **Claude Opus 5.5** 參與設計、實作、最佳化與量測。Anthropic 與本專案沒有隸屬關係，也不為本專案背書。

## 授權與致謝

Apache License 2.0；見 [LICENSE](../LICENSE) 與 [NOTICE](../NOTICE)。第三方素材列在 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)，每個原始檔的來源列在 [PROVENANCE.md](../PROVENANCE.md)。

- [NInfer](https://github.com/Neroued/ninfer)——WHIRL 依循的設計理念（只借理念）。
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)——我們量測時的比較基準，也是 GGUF 與量化格式相容性的參考；少數改寫的部分（tokenizer 預切詞 / BPE 迴圈、查表、圖片前處理）採 MIT 授權，列在 THIRD_PARTY_NOTICES.md。
- [gufo](https://github.com/gufo-org/gufo) 與 [r9700-stack](https://github.com/bkvargyas/r9700-stack)——量測方法與最佳化想法。
- 模型作者：Qwen 團隊（Qwen3.8）、[unsloth](https://huggingface.co/unsloth)、[FreedomAISVR](https://huggingface.co/FreedomAISVR)、[ornith-ai](https://huggingface.co/ornith-ai)（Ornith-1.5），以及 [ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b)（Swift-1.5，我們的 MXFP4 檔就是由它量化而來）。
