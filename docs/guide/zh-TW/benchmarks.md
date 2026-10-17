[English](../../benchmarks.md) | **繁體中文**

# WHIRL 發行版效能評測（v0.2.0）— Radeon AI PRO R9700 與 Radeon 8060S 上 WHIRL 對 llama.cpp

WHIRL 0.2.0 對 **ggml-org 官方 llama.cpp b11214**，**ROCm** 與 **Vulkan** 兩個 Windows 版本各佔一欄、各自列出領先幅度。本頁不和其他引擎比較。模型是我們的兩個 MXFP4 量化：**Swift-1.5 27B MXFP4**（稠密）與 **Ornith-1.5-35B-A3B MXFP4**（MoE）。下面每張表都由腳本從量測值產生（量測工作區的 `release020/make_fullmx_tables.py` 與 `public_numbers.json`），沒有手打的數字。

**請先看：**

- **WHIRL 在每一個量過的格子都領先兩個 llama.cpp 版本**：兩張卡、兩個模型、三種 WHIRL 模式、2k 到 ~250k token、prefill 與 decode。
- **三類 llama.cpp 格子附註腳，絕不拿來當主打**：R9700＋Swift 27B 的 ~250k（llama.cpp 的 f16 KV 放不進 32 GB 而溢出，ᵃ）、R9700＋Swift 27B 的 8k（llama.cpp 的第一個長請求是冷的，ᵇ）、8060S Vulkan 的 2k prefill（那張卡上 Vulkan 的短提示劣勢，ᶜ）。有一格 WHIRL 未量：R9700 上 Swift 27B precise 的 ~250k（ᵈ，f16 KV 放不下）。
- **Ornith 的 decode 是各自出廠設定的對比**：WHIRL 用 MTP 草稿，llama.cpp 不用投機解碼，因為它的 MTP 在這個 MoE 模型上比一般解碼慢。
- **fast 還沒有完全優化**，在 Ornith 上約 64k token 以下比 balance 慢（§7）。
- 每個投機解碼的數字都標明回答的內容：**文件問答**（部分回答可照抄提示中的程式碼）或**寫程式對話（全新撰寫）**（沒有可照抄的內容）。確切提示見 §2.1。

## 1. 測試環境

| | |
|---|---|
| 主機 | ASUS ROG Flow Z13（GZ302EA）：AMD Ryzen AI Max+ 395（16 核／32 執行緒），128 GB LPDDR5X-8000，其中 63.6 GB 給 Windows |
| GPU 1 | AMD Radeon AI PRO R9700（gfx1201，RDNA 4，32 GB GDDR6），以 **USB4 外接**於上面這台主機 |
| GPU 2 | AMD Radeon 8060S（gfx1151，RDNA 3.5），主機的內顯。**最大功率（80 W），ASUS ROG Flow Z13** |
| 作業系統／驅動 | Windows 11 Home 26H2；AMD Software: Adrenalin Edition 26.8.1（驅動 32.0.31041.1004） |
| WHIRL | v0.2.0 候選版 rc11（`bd9ba0c`）；0.2.0 正式版在它之上只多了 `whirl bench` 指令的修改，這些 server 量測用不到 |
| llama.cpp | 官方 b11214 Windows 執行檔，ROCm 版與 Vulkan 版 |
| 量測日期 | WHIRL：2026-10-17（兩張卡、所有格子）。llama.cpp：8k 與 ~250k 格於 2026-10-17（R9700 Ornith 8k：2026-10-16），其他格於 2026-10-16，執行檔與旗標相同 |

**外接說明：** R9700 經 USB4 連接。這會影響模型載入時間與主記憶體／SSD 的 KV 還原；不影響 prefill 與 decode，那些都在 VRAM 裡進行。

### 模型

| 簡稱 | 檔案（Hugging Face） | 類型 |
|---|---|---|
| Swift 27B | `Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`（[tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF)，變體 A） | 稠密 27B，含 MTP head；MXFP4，輸出頭 Q6_K，MTP 層／embedding／`ssm_alpha`／`ssm_beta` 為 Q8_0 |
| Ornith 35B-A3B | `Ornith-1.5-35B-A3B-MXFP4.gguf`（[tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF)） | MoE，總共 35B、每 token 約 3B 啟用，含 MTP head；專家與稠密矩陣 MXFP4，輸出頭 Q6_K，MTP 層／embedding／`ssm_alpha`／`ssm_beta` 為 Q8_0 |

## 2. 方法

- **長度**：2k、8k、32k、64k、96k、128k 與 ~250k token。「~250k」是 256k 的提示檔，用這個 tokenizer 為 249,510 token，所有地方都標成 ~250k。
- **greedy 解碼、關閉思考、一個使用者**（`-np 1`），tok/s 取自各 server 自己的 `timings`。機器上同時只跑一個 GPU 工作（鎖檔），每個引擎、模型、模式各開一個 server，每輪量測前先送一個短的暖機請求。
- **次數**：R9700 每格一次。Radeon 8060S：WHIRL 每格一次；llama.cpp 為 1 到 3 次的中位數。差距小於約 5% 視為雜訊。
- **寫程式對話**：WHIRL 在 R9700 取 3 次中位數、8060S 一次；llama.cpp 只取第一次，因為它的 n-gram 草稿會學會重複的提示（之後幾次會照抄前一次的回答）。
- **llama.cpp 的 decode 模式**：Swift 27B＝MTP + n-gram（`--spec-type draft-mtp,ngram-mod`），是它在稠密模型上最快的模式；Ornith＝一般解碼（出廠設定），因為它的 MTP 在這個 MoE 模型上比一般解碼慢。llama.cpp 的 prefill 取自同一次量測。
- **WHIRL**：`whirl-server MODEL --precise|--balance|--fast -np 1 --ctx-per-slot 262144 --kv-ssd-mode off`；MTP（+ n-gram）草稿照預設開啟。

### <a id="prompts"></a>2.1 實際提示與重現方法

投機解碼在回答可預測時最快（照抄或背誦文字），所以這裡寫明每個 decode 數字回答的是什麼。我們自己的提示檔放在 repo 的 [`tools/bench/prompts/`](../../../tools/bench/README.md)。

**文件問答**（2k 到 ~250k 的 prefill 與 decode 表）。提示依序是：每個請求都不同的第一行 `【測試編號 <nonce>】`（不會命中前綴快取），接著一份由授權寬鬆的真實來源組成的文件（C++、Python、Rust、TypeScript 原始檔，中文技術文件、中文維基、英文技術文件；每段以 `### 檔案 N：專案/路徑（授權）` 開頭，沒有重複的 4096-token 視窗），依整行截到目標長度，然後是 `---` 和下面這個問題，每個長度都一樣（[`document_qa_question.txt`](../../../tools/bench/prompts/document_qa_question.txt)，原文）：

> 請根據上面第 1 個檔案（llama.cpp/src/models/muse-glimmer.cpp）中的函式 `graph`，用繁體中文說明它的用途、參數與回傳值，指出一個可能出錯的邊界情況，並寫出修改後能正確處理該情況的完整函式程式碼。

第 1 個檔案在每個提示的最前面，所以 256 token 的回答是說明提示裡的一個函式並改寫它：有一部分可以照抄提示中的程式碼，這對所有使用草稿的引擎都有幫助（WHIRL 的 MTP + n-gram，Swift 上 llama.cpp 的 MTP + n-gram）。文件本身不在 repo 裡重新散布，因為它由各自授權的第三方文字組成。

**寫程式對話（全新撰寫）**（512 token；先送一個 64 token 的暖機請求），[`coding_chat_zh.txt`](../../../tools/bench/prompts/coding_chat_zh.txt)，原文：

```text
請用 Python 寫一個命令列工具 `dedup.py`：掃描指定目錄（可遞迴）中的所有檔案，用檔案大小先分組、再用 SHA-256 找出內容完全相同的檔案，最後輸出每一組重複檔案的路徑與大小，並統計可以省下多少空間。要求：
1. 用 argparse，支援 `--min-size`（忽略太小的檔案）與 `--json`（改用 JSON 輸出）。
2. 大檔案要分塊讀取，不能一次讀進記憶體。
3. 遇到沒有權限或讀取失敗的檔案要略過並在最後列出。
4. 附上 pytest 測試，至少涵蓋：空目錄、兩個相同檔案、大小相同但內容不同。
請先簡短說明設計，再給完整程式碼。
```

回答（說明與程式碼）全部是新寫的，提示裡除了幾個選項名稱之外沒有可照抄的內容：對投機草稿是困難的情況。

**Coding agent 工作階段**（§6）：測試資料 [`agent_fixture.json`](../../../tools/bench/prompts/agent_fixture.json)，每一輪的確切流程見 [`tools/bench/README.md`](../../../tools/bench/README.md)。

**請求與重現**（兩種 server 都照 §8 啟動；PowerShell）：

```powershell
$prompt = Get-Content -Raw -Encoding UTF8 .\prompt.txt      # 文件問答提示，或 tools\bench\prompts\coding_chat_zh.txt
$body = @{ model = 'whirl'; messages = @(@{ role = 'user'; content = $prompt }); max_tokens = 256; temperature = 0;   # 對話：512
           chat_template_kwargs = @{ enable_thinking = $false } } | ConvertTo-Json -Depth 6
$r = Invoke-RestMethod http://127.0.0.1:8099/v1/chat/completions -Method Post `
       -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
$r.timings.prompt_per_second, $r.timings.predicted_per_second   # prefill tok/s、decode tok/s（兩種 server 欄位名稱相同）
```

## 3. 領先幅度摘要

<!-- gen:ranges.zh -->
| WHIRL balance 領先幅度（每個量過的長度） | Prefill 對 ROCm | Prefill 對 Vulkan | Decode 對 ROCm | Decode 對 Vulkan |
|---|---|---|---|---|
| R9700 · Swift 27B | +139% 到 +191% | +208% 到 +295% | +56% 到 +80% | +38% 到 +84% |
| R9700 · Ornith 35B-A3B | +75% 到 +188% | +119% 到 +225% | +74% 到 +107% | +78% 到 +106% |
| 8060S · Swift 27B | +53% 到 +65% | +93% 到 +183% | +28% 到 +46% | +24% 到 +105% |
| 8060S · Ornith 35B-A3B | +64% 到 +91% | +142% 到 +199% | +41% 到 +54% | +43% 到 +57% |

有註腳的格子（ᵃ ᵇ ᶜ）不計入這些範圍；那些格子 WHIRL 也領先。
<!-- /gen -->

## 4. Radeon AI PRO R9700

每格一次。Decode：WHIRL MTP（+ n-gram）；llama.cpp 在 Swift 用 MTP + n-gram，在 Ornith 用一般解碼（出廠設定）。

<!-- gen:r97.swift.pp.zh -->
#### Swift-1.5 27B MXFP4（稠密）：Prefill tok/s

| 提示長度 | WHIRL precise | WHIRL balance（預設） | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 1,702 | **3,273** | 3,180 | 1,124 | 829.4 | +191% | +295% |
| 8k | 1,764 | **3,296** | 3,269 | 867.5ᵇ | 705.5ᵇ | +280%ᵇ | +367%ᵇ |
| 32k | 1,626 | **2,805** | 2,790 | 1,065 | 787.5 | +163% | +256% |
| 64k | 1,444 | **2,332** | 2,304 | 913.6 | 691.9 | +155% | +237% |
| 96k | 1,301 | **1,971** | 1,952 | 805.2 | 618.1 | +145% | +219% |
| 128k | 1,185 | **1,715** | 1,697 | 718.7 | 557.2 | +139% | +208% |
| ~250k | 未量 ᵈ | **1,153** | 1,153 | 342.0ᵃ | 259.9ᵃ | +237%ᵃ | +344%ᵃ |
<!-- /gen -->

<!-- gen:r97.swift.tg.zh -->
#### Swift-1.5 27B MXFP4（稠密）：Decode tok/s，文件問答（256 token 回答：說明文件中引用的一個函式並改寫，部分回答可照抄提示）

| 提示長度 | WHIRL precise | WHIRL balance（預設） | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 70.1 | **87.6** | 115.5 | 50.5 (MTP+n-gram) | 50.4 (MTP+n-gram) | +74% | +74% |
| 8k | 74.3 | **90.7** | 102.3 | 28.2ᵇ (MTP+n-gram) | 44.4ᵇ (MTP+n-gram) | +222%ᵇ | +104%ᵇ |
| 32k | 56.4 | **79.2** | 79.3 | 44.0 (MTP+n-gram) | 43.1 (MTP+n-gram) | +80% | +84% |
| 64k | 56.7 | **60.4** | 67.8 | 38.6 (MTP+n-gram) | 39.5 (MTP+n-gram) | +56% | +53% |
| 96k | 49.8 | **56.3** | 63.4 | 36.1 (MTP+n-gram) | 40.7 (MTP+n-gram) | +56% | +38% |
| 128k | 49.4 | **53.9** | 67.0 | 33.2 (MTP+n-gram) | 34.8 (MTP+n-gram) | +62% | +55% |
| ~250k | 未量 ᵈ | **37.9** | 38.8 | 2.3ᵃ (MTP+n-gram) | 6.4ᵃ (MTP+n-gram) | +1548%ᵃ | +492%ᵃ |
<!-- /gen -->

<!-- gen:r97.ornith.pp.zh -->
#### Ornith-1.5-35B-A3B MXFP4（MoE，約 3B 啟用）：Prefill tok/s

| 提示長度 | WHIRL precise | WHIRL balance（預設） | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 7,575 | **11,550** | 11,801 | 4,014 | 3,551 | +188% | +225% |
| 8k | 6,846 | **10,601** | 10,501 | 4,487 | 3,912 | +136% | +171% |
| 32k | 5,959 | **8,469** | 8,450 | 3,872 | 3,395 | +119% | +149% |
| 64k | 4,873 | **6,451** | 6,420 | 3,173 | 2,735 | +103% | +136% |
| 96k | 4,098 | **5,156** | 5,143 | 2,667 | 2,280 | +93% | +126% |
| 128k | 3,520 | **4,275** | 4,270 | 2,307 | 1,922 | +85% | +122% |
| ~250k | 2,351 | **2,664** | 2,666 | 1,519 | 1,218 | +75% | +119% |
<!-- /gen -->

<!-- gen:r97.ornith.tg.zh -->
#### Ornith-1.5-35B-A3B MXFP4（MoE，約 3B 啟用）：Decode tok/s，文件問答（256 token 回答：說明文件中引用的一個函式並改寫，部分回答可照抄提示）

| 提示長度 | WHIRL precise | WHIRL balance（預設） | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 181.8 | **238.4** | 208.3 | 115.4 (一般解碼) | 115.9 (一般解碼) | +107% | +106% |
| 8k | 174.6 | **212.9** | 191.3 | 112.3 (一般解碼) | 111.9 (一般解碼) | +90% | +90% |
| 32k | 154.2 | **199.1** | 183.5 | 101.4 (一般解碼) | 100.6 (一般解碼) | +96% | +98% |
| 64k | 138.6 | **169.1** | 174.9 | 90.7 (一般解碼) | 89.6 (一般解碼) | +86% | +89% |
| 96k | 126.4 | **148.1** | 156.2 | 81.9 (一般解碼) | 81.1 (一般解碼) | +81% | +83% |
| 128k | 121.3 | **131.3** | 152.4 | 75.3 (一般解碼) | 73.8 (一般解碼) | +74% | +78% |
| ~250k | 90.0 | **101.7** | 125.3 | 54.1 (一般解碼) | 53.6 (一般解碼) | +88% | +90% |
<!-- /gen -->

<!-- gen:foot.r97.zh -->
ᵃ R9700＋Swift 27B 的 ~250k：官方 llama.cpp 用 f16 KV 在這個長度放不進 32 GB（記錄檔："failed to fit params to free device memory"），溢到共用記憶體，decode 因此崩落。表中照實列出量到的數字，但這一格不拿來當任何領先的依據。

ᵇ R9700＋Swift 27B 的 8k：llama.cpp 的 8k 請求是新開 server 後的第一個長請求（冷的），數字比它自己的 2k 與 32k 還低。這一格不拿來當任何領先的依據。

ᵈ R9700＋Swift 27B 的 precise ~250k：未量。precise 的 KV 快取用 f16，這個長度的 f16 KV 加上權重放不進 32 GB（f16 池約可容納 19 萬 token）。
<!-- /gen -->

## <a id="radeon-8060s"></a>5. Radeon 8060S，最大功率（80 W），ASUS ROG Flow Z13

WHIRL 每格一次；llama.cpp 為 1 到 3 次的中位數。

<!-- gen:s8.swift.pp.zh -->
#### Swift-1.5 27B MXFP4（稠密）：Prefill tok/s

| 提示長度 | WHIRL precise | WHIRL balance（預設） | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 479.1 | **498.4** | 502.6 | 305.7 | 162.8ᶜ | +63% | +206%ᶜ |
| 8k | 446.5 | **470.8** | 471.8 | 307.5 | 234.4 | +53% | +101% |
| 32k | 401.4 | **426.7** | 428.9 | 266.2 | 221.6 | +60% | +93% |
| 64k | 353.4 | **374.4** | 375.9 | 230.9 | 177.3 | +62% | +111% |
| 96k | 315.7 | **332.9** | 334.1 | 203.3 | 149.0 | +64% | +123% |
| 128k | 282.9 | **298.9** | 297.4 | 180.8 | 127.8 | +65% | +134% |
| ~250k | 197.0 | **221.3** | 222.2 | 135.1 | 78.1 | +64% | +183% |
<!-- /gen -->

<!-- gen:s8.swift.tg.zh -->
#### Swift-1.5 27B MXFP4（稠密）：Decode tok/s，文件問答（256 token 回答：說明文件中引用的一個函式並改寫，部分回答可照抄提示）

| 提示長度 | WHIRL precise | WHIRL balance（預設） | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 30.0 | **31.2** | 37.1 | 24.4 (MTP+n-gram) | 25.1 (MTP+n-gram) | +28% | +24% |
| 8k | 30.8 | **32.5** | 37.7 | 24.6 (MTP+n-gram) | 25.2 (MTP+n-gram) | +32% | +29% |
| 32k | 25.5 | **29.5** | 34.7 | 22.7 (MTP+n-gram) | 17.0 (MTP+n-gram) | +30% | +74% |
| 64k | 23.9 | **26.0** | 31.2 | 18.4 (MTP+n-gram) | 18.0 (MTP+n-gram) | +41% | +44% |
| 96k | 21.8 | **24.1** | 29.6 | 17.9 (MTP+n-gram) | 16.6 (MTP+n-gram) | +35% | +45% |
| 128k | 22.4 | **23.8** | 26.1 | 16.2 (MTP+n-gram) | 11.6 (MTP+n-gram) | +46% | +105% |
| ~250k | 14.6 | **16.8** | 20.7 | 12.2 (MTP+n-gram) | 10.9 (MTP+n-gram) | +38% | +54% |
<!-- /gen -->

<!-- gen:s8.ornith.pp.zh -->
#### Ornith-1.5-35B-A3B MXFP4（MoE，約 3B 啟用）：Prefill tok/s

| 提示長度 | WHIRL precise | WHIRL balance（預設） | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 2,044 | **2,411** | 2,401 | 1,259 | 447.7ᶜ | +91% | +439%ᶜ |
| 8k | 1,867 | **2,244** | 2,232 | 1,282 | 749.2 | +75% | +199% |
| 32k | 1,577 | **1,829** | 1,830 | 1,049 | 756.2 | +74% | +142% |
| 64k | 1,299 | **1,463** | 1,466 | 848.9 | 603.6 | +72% | +142% |
| 96k | 1,108 | **1,227** | 1,229 | 710.4 | 500.5 | +73% | +145% |
| 128k | 966.3 | **1,055** | 1,059 | 636.6 | 428.4 | +66% | +146% |
| ~250k | 644.7 | **685.0** | 697.9 | 418.3 | 278.9 | +64% | +146% |
<!-- /gen -->

<!-- gen:s8.ornith.tg.zh -->
#### Ornith-1.5-35B-A3B MXFP4（MoE，約 3B 啟用）：Decode tok/s，文件問答（256 token 回答：說明文件中引用的一個函式並改寫，部分回答可照抄提示）

| 提示長度 | WHIRL precise | WHIRL balance（預設） | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 91.6 | **101.4** | 93.8 | 70.0 (一般解碼) | 70.7 (一般解碼) | +45% | +43% |
| 8k | 87.5 | **94.9** | 93.6 | 64.3 (一般解碼) | 65.2 (一般解碼) | +48% | +46% |
| 32k | 76.3 | **83.8** | 86.1 | 57.9 (一般解碼) | 57.1 (一般解碼) | +45% | +47% |
| 64k | 67.9 | **69.2** | 76.2 | 49.0 (一般解碼) | 48.4 (一般解碼) | +41% | +43% |
| 96k | 58.5 | **63.9** | 67.0 | 42.7 (一般解碼) | 42.2 (一般解碼) | +50% | +51% |
| 128k | 51.1 | **58.4** | 62.7 | 38.0 (一般解碼) | 37.3 (一般解碼) | +54% | +57% |
| ~250k | 38.9 | **40.4** | 48.9 | 26.3 (一般解碼) | 25.8 (一般解碼) | +54% | +57% |
<!-- /gen -->

<!-- gen:foot.s8.zh -->
ᶜ 8060S 的 2k prefill、llama.cpp Vulkan：這張卡上 Vulkan 處理短提示的 prefill 特別慢（遠低於它在 8k 以上的趨勢），所以對 Vulkan 的 2k 領先不當主打；對 Vulkan 的主打領先一律用 8k 以上。
<!-- /gen -->

### 5.1 寫程式對話（全新撰寫），兩張卡

<!-- gen:chat.zh -->
| 顯示卡 · 模型 | WHIRL precise | WHIRL balance（預設） | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| R9700 · Swift 27B | 83.4 | **109.6** | 130.4 | 57.8 (MTP+n-gram) | 57.6 (MTP+n-gram) | +90% | +90% |
| R9700 · Ornith 35B-A3B | 191.2 | **241.5** | 219.0 | 118.9 (一般解碼) | 115.1 (一般解碼) | +103% | +110% |
| 8060S · Swift 27B | 36.3 | **34.2** | 42.1 | 27.6 (MTP+n-gram) | 28.3 (MTP+n-gram) | +24% | +21% |
| 8060S · Ornith 35B-A3B | 94.5 | **106.0** | 102.8 | 70.6 (一般解碼) | 72.9 (一般解碼) | +50% | +45% |

8060S 寫程式對話：WHIRL 每個模式單次量測。草稿被接受多少取決於各模式寫出的文字，所以這次 Swift 27B 的 precise（36.3）高於 balance（34.2）。
<!-- /gen -->

## <a id="agent-session"></a>6. Coding agent 工作階段（R9700 與 Radeon 8060S）

Swift 27B。每段工作階段：約 20k token 的系統＋工具提示（coding agent 指示加上 skills／MCP 工具手冊），以及回傳約 3.6k token C++ 原始檔的 `read_file` 工具；四輪使用者提問（說明這個檔案、找出最複雜的函式、把它改寫成程式碼、審查重送的修改版），每次呼叫都重送完整歷史，一個用戶端、依序執行、temperature 0，每次呼叫最多 384 token。WHIRL balance 搭配 MTP + n-gram 草稿；llama.cpp MTP + n-gram。第 1 段從冷啟動開始（系統＋工具提示尚未快取），讀 `llama-batch.cpp`；第 2 段共用同一個前綴，讀 `llama-chat.cpp`（每個引擎都是同樣的檔案）。每個引擎一輪，各自重開 server。

**R9700：**

<!-- gen:agent.r97.zh -->
| 工作階段（總時間） | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL 比 ROCm／Vulkan 省下的時間 | 模型呼叫次數 WHIRL／ROCm／Vulkan |
|---|---|---|---|---|---|
| 冷啟動：20k token 的系統＋工具提示尚未快取 | **17.0 秒** | 81.0 秒 | 69.4 秒 | 79% / 76% | 5/6/6 |
| 第二段：共用同一個前綴，處理另一個檔案 | **11.8 秒** | 60.9 秒 | 38.0 秒 | 81% / 69% | 6/5/6 |
<!-- /gen -->

**Radeon 8060S**（最大功率（80 W），ASUS ROG Flow Z13）：

<!-- gen:agent.s8.zh -->
| 工作階段（總時間） | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL 比 ROCm／Vulkan 省下的時間 | 模型呼叫次數 WHIRL／ROCm／Vulkan |
|---|---|---|---|---|---|
| 冷啟動：20k token 的系統＋工具提示尚未快取 | **95.7 秒** | 145.6 秒 | 264.4 秒 | 34% / 64% | 6/6/6 |
| 第二段：共用同一個前綴，處理另一個檔案 | **44.9 秒** | 66.2 秒 | 196.8 秒 | 32% / 77% | 5/5/6 |
<!-- /gen -->

- temperature 0 時每個引擎寫出自己的文字，所以工作量不是逐位元組相同：在第 1 或第 3 輪又呼叫一次 `read_file` 的引擎，會多一次模型呼叫（最後一欄）。R9700 上冷啟動段 WHIRL 比兩個 llama.cpp 版本少一次呼叫，第二段比 llama.cpp ROCm 多一次。
- 每個引擎一輪（n = 1）。

## 7. 模式：precise 與 fast 的代價與收益

<!-- gen:precise.zh -->
precise 會犧牲速度。R9700 上代價最大的是 MXFP4 檔的 prefill，因為 precise 不用 fp8（Swift 27B 2k：1,702 對 balance 3,273 tok/s；Ornith：7,575 對 11,550）；寫程式對話 Swift decode 83.4 對 109.6 tok/s。8060S 沒有 fp8 矩陣單元，所以 precise 與 balance 的 prefill 很接近（Swift 2k：479.1 對 498.4）。precise 在每個量過的格子仍領先兩個 llama.cpp 版本。
<!-- /gen -->

<!-- gen:fast.zh -->
**fast 模式還沒有完全優化，之後的 0.2.x 小版本會再改進。** 目前實測的效果：主要加快稠密 Swift 27B 的 decode（R9700：2k token 文件之後 115.5 對 balance 87.6 tok/s，寫程式對話 130.4 對 109.6；8060S 8k：37.7 對 32.5），以及 Ornith MoE 的長上下文 decode（R9700 ~250k：125.3 對 101.7）。在 Ornith 上，**約 64k token 以下 fast 比 balance 慢**（R9700：2k 208.3 對 238.4、32k 183.5 對 199.1；64k 174.9 對 169.1；8060S 2k：93.8 對 101.4），因為 4-bit KV 快取每個 token 有固定的額外成本，要到長上下文才划算。fast 目前不會加快 prefill（R9700 Swift 32k：2,790 對 2,805 tok/s）。
<!-- /gen -->

R9700 上 Swift 27B 為什麼沒有 precise 的 ~250k：precise 的 KV 快取用 f16，249,510 token 的 f16 KV 加上 27B 權重放不進 32 GB；f16 池約可容納 19 萬 token。balance 與 fast 在那裡用較小的 KV 格式，能跑完整長度。Ornith 的 KV 快取小得多，所以 precise 在兩張卡上都能跑 ~250k。

## 8. 確切指令（0.2.0）

```bat
:: WHIRL server（每個引擎／模型／模式各一個；量測時關閉 SSD 層）
whirl-server.exe MODEL.gguf --port 8099 --balance -np 1 --ctx-per-slot 262144 --kv-ssd-mode off
    （其他模式用 --precise／--fast；兩張卡都在時 Radeon 8060S 要加 --device 8060s）

:: llama.cpp b11214 server，R9700（ROCm 版；Vulkan 版用 --device Vulkan1、不加 --load-mode）
set HIP_VISIBLE_DEVICES=1
set GGML_CUDA_NO_PINNED=1
llama-server -m MODEL.gguf --port 8099 --device ROCm0 -ngl 99 -fa on -b 2048 -ub 1024 -ctk f16 -ctv f16 -c 262144 -np 1 ^
    --jinja --reasoning off --top-k 1 --no-webui --load-mode none
    [Swift：--spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-n-min 0 --spec-draft-p-min 0.3]
    （Ornith：-ub 2048，一般解碼）

:: llama.cpp b11214 server，Radeon 8060S（ROCm 與 Vulkan 版）
llama-server -m MODEL.gguf --port 8099 -ngl 99 -fa on -b 2048 -ub 512 -ctk f16 -ctv f16 -c 262144 -np 1 ^
    --jinja --reasoning off --top-k 1 --no-webui
    [Swift：--spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-n-min 0 --spec-draft-p-min 0.3]
    （Ornith：-ub 2048，一般解碼）
```

2026-10-16 量測的 llama.cpp 格子用 `-c 133120`（足夠 128k）；8k 與 ~250k 格用 `-c 262144`。量測框架（PowerShell 工作排程、只用標準函式庫的 Python 用戶端）不在這個 repo 裡；它記錄每個請求的 server timings，原始結果檔與 server log 留在量測機器上。
