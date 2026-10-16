[English](../../benchmarks.md) | **繁體中文**

# WHIRL 發行版效能評測（v0.2.0）— Radeon AI PRO R9700 與 Radeon 8060S 上 WHIRL 對 llama.cpp

WHIRL 0.2.0 對 **ggml-org 官方 llama.cpp b11214**，**ROCm** 與 **Vulkan** 兩個 Windows 版本各佔一欄（絕不取兩者最佳）。本頁不和其他引擎比較。下面每張表都由腳本從原始結果檔產生（量測工作區的 `release020/make_public_tables.py`），每一格都能追溯到一筆實測紀錄。

**WHIRL 領先不多的地方（請先看）：**

- **Radeon 8060S 上稠密 decode，2k token 文件問答**：對 llama.cpp ROCm 領先 9%、對 Vulkan 6%，在量測雜訊範圍內。那裡的稠密 decode 受記憶體頻寬限制（約 256 GB/s，權重 15 GB 以上）；領先隨上下文擴大（32k 起 +24% 到 +48%）。
- **MoE decode 是各自出廠設定對比，不是同條件對比**：WHIRL 用自己的 MTP 草稿；llama.cpp 在 MoE 模型上不用投機解碼，因為它的 MTP 在這個模型上比一般解碼慢（R9700：12 格中 11 格較慢，1 格快 0.4 tok/s）。
- **precise 要付出速度**：R9700 上 decode 比 balance 慢 14–26%、prefill 慢 6–13%。**fast 不一定比 balance 快**：MoE 模型在短、中等上下文時 balance 解碼較快。
- **不用投機的一般解碼**在每個引擎都受記憶體頻寬限制；WHIRL 大部分的 decode 領先來自 MTP + n-gram 投機解碼。

## 1. 測試環境

| | |
|---|---|
| 主機 | ASUS ROG Flow Z13（GZ302EA）：AMD Ryzen AI Max+ 395（16 核／32 執行緒），128 GB LPDDR5X-8000，Windows 看得到 63.6 GB |
| GPU 1 | AMD Radeon AI PRO R9700（gfx1201，RDNA 4，32 GB GDDR6），以 **USB4 外接**於上述主機 |
| GPU 2 | AMD Radeon 8060S（gfx1151，RDNA 3.5），主機內顯。**最大功率（80 W），ASUS ROG Flow Z13**（未更動任何功率、散熱或驅動設定） |
| 作業系統／驅動 | Windows 11 Home 26H2；AMD Software: Adrenalin Edition 26.8.1（驅動 32.0.31041.1004） |
| WHIRL | R9700：v0.2.0 候選版 rc9（分支 `rel020` `bc78613`；rc10 在 R9700 上完全相同，只改了 8060S 的 prefill forward 大小）。8060S：候選版 rc10（`bd81154`）；8060S 的中文對話與 agent 工作階段：rc7（見 §5）。[TBD：正式 0.2.0 `whirl.exe`／`whirl-server.exe` 的 SHA-256，以及正式版是否由 `bd81154` 建置] |
| llama.cpp | 官方 b11214 Windows 執行檔，ROCm 版與 Vulkan 版 |
| 量測時間 | 2026-10-16（R9700 07:00–10:28；8060S 在 00:30 到 14:44 之間；Swift-1.5 與 R9700 agent 工作階段 15:15–17:24） |

**外接說明：**R9700 透過 USB4（單向約 3.8 GB/s）連接。這會影響模型載入時間與主記憶體／SSD KV 還原；不影響 prefill 與 decode，它們都在 VRAM 內。

### 模型

| 簡稱 | 檔案 | 類型 |
|---|---|---|
| 稠密 27B | `Qwen3.8-27B-UD-Q4_K_M.gguf`（unsloth） | 稠密 27B，含 MTP head，UD-Q4_K_M |
| MoE 35B-A3B | `Ornith-1.5-35B-A3B-MXFP4.gguf`（我們製作） | MoE，總計 35B、每 token 約 3B 啟用，含 MTP head，MXFP4 專家 |
| Swift 27B MXFP4 | `Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`（我們製作，變體 A） | 稠密 27B，含 MTP head，MXFP4（輸出 head Q6_K，MTP 層與 embedding Q8_0）；§4.2 |

Swift-1.5 27B MXFP4（§4.2）與 R9700 的 agent 工作階段（§5）在同一天稍晚用候選版 rc10 量測（在 R9700 上，本頁量到的路徑 rc10 與 rc9 相同）。Swift 的 0.1.3 數字在[附錄](#appendix-v013)。

## 2. 方法

- **提示**：2k、8k、32k、64k、96k、128k token 的長文件問答提示（8k 只在 R9700 量），每個請求第一行都不同，不會命中前綴快取；生成 256 token。中文提問寫程式對話：一題中文寫程式問題（中文說明、英文程式碼），512 token。
- **greedy 解碼、關閉思考、一個使用者**（`-np 1`），tok/s 取自各 server 自己的 `timings`。整台機器一次只跑一個 GPU 工作（鎖檔），每個引擎、模型、模式各開一個 server。
- **R9700**：每格都在 2026-10-16 同一輪量完，每個長度一個請求（n = 1）；WHIRL 對話取 3 次中位數；llama.cpp 對話只取第 1 次，因為它的 n-gram 草稿會學會重複的提示（之後幾次靠照抄達到 180–330 tok/s）。
- **Radeon 8060S**：每輪掃描前至少 25 秒的 2k prompt 暖機；引擎交錯執行，每輪輪換順序；每格取各輪中位數（n = 1 到 3，有標註）；有一格（llama.cpp ROCm、稠密、96k decode）兩次相差 31%，標為不可靠。
- **llama.cpp 解碼模式**：稠密＝每個後端取一般與 MTP + n-gram 中較快者（每格標註；每個稠密格都是 MTP + n-gram 較快）；MoE＝一般解碼（出廠設定）。llama.cpp prefill＝每個後端兩次量測中較快者。
- **WHIRL**：`whirl-server MODEL --balance|--precise|--fast -np 1 --ctx-per-slot 163840 --kv-ssd-mode off`；MTP（+ n-gram）照預設開啟。

### <a id="prompts"></a>2.1 實際提示與重現方法

投機解碼在回答可預測時最快（照抄或背誦文字）。所以這裡寫明每個 decode 數字回答的是什麼。

**文件問答**（2k–128k 的 prefill 與 decode 表）。提示依序是：每個請求都不同的第一行 `【測試編號 <nonce>】`（不會命中前綴快取），接著一份由授權寬鬆的真實來源組成的文件（C++、Python、Rust、TypeScript 原始檔，中文技術文件、中文維基、英文技術文件；每段以 `### 檔案 N：專案/路徑（授權）` 開頭，沒有重複的 4096-token 視窗），依整行截到目標長度，然後是 `---` 和下面這個問題，每個長度都一樣（原文）：

> 請根據上面第 1 個檔案（llama.cpp/src/models/muse-glimmer.cpp）中的函式 `graph`，用繁體中文說明它的用途、參數與回傳值，指出一個可能出錯的邊界情況，並寫出修改後能正確處理該情況的完整函式程式碼。

第 1 個檔案在每個提示的最前面，所以 256 token 的回答是說明提示裡的一個函式並改寫它：有一部分可以照抄提示中的程式碼，這對所有使用草稿的引擎都有幫助（WHIRL 的 MTP + n-gram，稠密模型上 llama.cpp 的 MTP + n-gram）。

**中文寫程式對話**（512 token；先送一個 64 token 的暖機請求），原文：

```text
請用 Python 寫一個命令列工具 `dedup.py`：掃描指定目錄（可遞迴）中的所有檔案，用檔案大小先分組、再用 SHA-256 找出內容完全相同的檔案，最後輸出每一組重複檔案的路徑與大小，並統計可以省下多少空間。要求：
1. 用 argparse，支援 `--min-size`（忽略太小的檔案）與 `--json`（改用 JSON 輸出）。
2. 大檔案要分塊讀取，不能一次讀進記憶體。
3. 遇到沒有權限或讀取失敗的檔案要略過並在最後列出。
4. 附上 pytest 測試，至少涵蓋：空目錄、兩個相同檔案、大小相同但內容不同。
請先簡短說明設計，再給完整程式碼。
```

這一列對投機草稿是困難情況：中文提問、中文說明與英文程式碼都是全新撰寫，提示裡除了幾個選項名稱之外沒有可讓回答照抄的內容（投機解碼在回答可預測時最快，例如照抄或背誦已知文字；這一列沒有這種情況）。

**Coding agent 工作階段**（§5）：16k token 的 skills／MCP 系統提示，`read_file` 工具回傳一份 3.6k token 的 llama.cpp C++ 原始檔，接著四輪使用者提問（說明檔案；找出最複雜的函式；把該函式改寫成程式碼；審閱訊息中附上的修改版檔案），每次呼叫重送完整歷史，每次最多 384 token。

[TBD：把提示檔（含各檔來源與授權的文件語料、提示產生腳本、agent 工作階段測試資料）放進 repository 或作為發布附件，並在此連結；需 owner 決定，因為語料包含各自授權的第三方文字。]

**請求與重現**（兩種 server 都照 §6 啟動；PowerShell）：

```powershell
$prompt = Get-Content -Raw -Encoding UTF8 .\prompt.txt      # document QA prompt (or the chat prompt)
$body = @{ model = 'whirl'; messages = @(@{ role = 'user'; content = $prompt }); max_tokens = 256; temperature = 0;   # chat: 512
           chat_template_kwargs = @{ enable_thinking = $false } } | ConvertTo-Json -Depth 6
$r = Invoke-RestMethod http://127.0.0.1:8099/v1/chat/completions -Method Post `
       -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
$r.timings.prompt_per_second, $r.timings.predicted_per_second   # prefill tok/s, decode tok/s (same field names in both servers)
```

### 2.2 llama.cpp 旗標

| | R9700 | Radeon 8060S |
|---|---|---|
| 共同 | `-ngl 99 -fa on -b 2048 -c 133120 -np 1 --jinja --reasoning off --top-k 1`，KV f16 | `-ngl 99 -fa on -b 2048 -ctk f16 -ctv f16 -np 1 -c 133120 --top-k 1` |
| `-ub` | 1024（稠密）、2048（MoE） | 512（稠密）、2048（MoE） |
| 稠密投機解碼 | `--spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-n-min 0 --spec-draft-p-min 0.3` | `--spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-p-min 0.3` |
| MoE MTP（量過，比一般解碼慢） | `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-p-min 0.3` | 不使用（一般解碼） |
| 裝置 | ROCm：`HIP_VISIBLE_DEVICES=1`；Vulkan：`Vulkan1` | 內顯 |

## 3. Radeon AI PRO R9700

列出 WHIRL 三種模式；領先幅度以 balance（預設）計算。每格單次量測；差距在 5% 左右以內請視為雜訊。

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

MoE decode：llama.cpp 一般解碼（出廠設定），WHIRL 用 MTP。

### 3.1 中文寫程式對話（512 token）

對草稿是困難情況：全部全新撰寫，提示裡沒有可照抄的內容（[提示](#prompts)）。

<!-- gen:r97.chat.zh -->
| 模型 | WHIRL precise | WHIRL balance | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance 對 ROCm | balance 對 Vulkan |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B UD-Q4_K_M（稠密） | 77.8 | **103.1** | 119.3 | 61.4 (MTP+n-gram) | 61.2 (MTP+n-gram) | +68% | +68% |
| Ornith-1.5-35B-A3B MXFP4（MoE，約 3B 啟用） | 196.8 | **250.1** | 229.0 | 118.9 (一般) | 115.1 (一般) | +110% | +117% |
<!-- /gen -->

## 4. Radeon 8060S（最大功率（80 W），ASUS ROG Flow Z13）

WHIRL balance。¹ 只量一次，請當作「約」。² 不可靠（兩次分別 20.1 與 15.3 tok/s，相差 31%）。

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

稠密 decode：llama.cpp 用 MTP + n-gram，WHIRL 用 MTP。MoE decode：llama.cpp 一般解碼（出廠設定），WHIRL 用 MTP。

### 4.1 中文寫程式對話（512 token）

與 §3.1 相同提示（對草稿是困難情況，沒有可照抄的內容）。

WHIRL 用 v0.2.0-rc7 量測（rc8–rc10 只改了 8060S 的 prefill forward 大小；這項測試以解碼為主）。

<!-- gen:s8.chat.zh -->
| 模型 | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL 對 ROCm | WHIRL 對 Vulkan |
|---|---|---|---|---|---|
| Qwen3.8-27B UD-Q4_K_M（稠密） | **33.3** | 25.9 | 27.9 | +29% | +20% |
| Ornith-1.5-35B-A3B MXFP4（MoE，約 3B 啟用） | **106.3** | 70.6¹ | 72.9¹ | +51% | +46% |
<!-- /gen -->

### 4.2 Swift-1.5 27B MXFP4-A（稠密），R9700 與 Radeon 8060S

我們自己製作的稠密 27B MXFP4 量化（`Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`，變體 A），提示與旗標同上，同一天稍晚用 0.2.0 候選版量測。R9700 上 WHIRL 的 MXFP4 prefill 路徑比 Q4_K_M 快很多，所以 prefill 領先比 Q4_K_M 大：對 llama.cpp ROCm +115% 到 +166%、對 Vulkan +194% 到 +296%；decode +50% 到 +75%／+43% 到 +78%。8060S：prefill +56% 到 +62%／+87% 到 +207%，decode +27% 到 +48%／+24% 到 +116%。

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

**Radeon 8060S**，最大功率（80 W），ASUS ROG Flow Z13（只量了 2k、32k、128k 的提示長度；¹ 只量一次，請當作「約」；其他格為 2 或 3 次中位數；llama.cpp 用 MTP + n-gram）：

<!-- gen:s8.swift.pp.zh -->
#### Swift-1.5-Qwen3.8-27B MXFP4-A（稠密 27B）: Prefill tok/s

| 長度 | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL 對 ROCm | WHIRL 對 Vulkan |
|---|---|---|---|---|---|
| 2k | **500.9** | 310.1 | 162.9 | +62% | +207% |
| 32k | **414.9** | 265.1 | 221.6 | +56% | +87% |
| 128k | **289.8¹** | 181.1¹ | 127.9¹ | +60% | +127% |
<!-- /gen -->

<!-- gen:s8.swift.tg.zh -->
#### Swift-1.5-Qwen3.8-27B MXFP4-A（稠密 27B）: Decode tok/s，文件問答（256 token 回答：說明文件中的一個函式並改寫）

| 長度 | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL 對 ROCm | WHIRL 對 Vulkan |
|---|---|---|---|---|---|
| 2k | **31.0** | 24.5 | 25.1 | +27% | +24% |
| 32k | **29.0** | 22.2 | 17.0 | +31% | +71% |
| 128k | **24.2¹** | 16.3¹ | 11.2¹ | +48% | +116% |
<!-- /gen -->

**中文寫程式對話**（512 token，對草稿是困難情況：全部全新撰寫，沒有可照抄的內容）：

<!-- gen:swift.chat.zh -->
| GPU | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL 對 ROCm | WHIRL 對 Vulkan |
|---|---|---|---|---|---|
| Radeon AI PRO R9700 | **110.7** | 57.8 (MTP+n-gram) | 57.6 (MTP+n-gram) | +92% | +92% |
| Radeon 8060S（最大功率） | **33.6** | 27.6 (MTP+n-gram) | 28.3 (MTP+n-gram) | +22% | +19% |
<!-- /gen -->

## 5. Coding agent 工作階段（Radeon 8060S 與 R9700）

### 5.1 Radeon 8060S（最大功率同上）

稠密 Qwen3.8-27B Q4_K_M。每段工作階段：16k token 的共用 skills／MCP 系統提示加上 `read_file` 工具（讀取 3.6k token 的原始檔）、四輪使用者提問（說明檔案、找出最複雜的函式、把它改寫成程式碼、審閱重送的修改版檔案），每次呼叫重送完整歷史，單一用戶端依序執行，temperature 0，每次最多 384 token。所有引擎都用 MTP（llama.cpp 用 MTP + n-gram）。第 0 段冷啟動（約 20k token 的系統＋工具提示尚未快取）；第 1 段共用該前綴。各 2 輪中位數。WHIRL v0.2.0-rc7（8060S 上稠密模型的 prefill forward 大小與 0.2.0 相同）。

<!-- gen:s8.agent.zh -->
| 工作階段 | WHIRL | llama.cpp ROCm | llama.cpp Vulkan |
|---|---|---|---|
| 冷啟動（20k token 系統＋工具提示尚未快取） | **101.6 s** | 137.7 s | 243.5 s |
| 第二次（共用同一前綴） | **45.7 s** | 52.5 s | 166.4 s |
<!-- /gen -->

- 冷啟動段，WHIRL 比 llama.cpp ROCm 少花 26% 時間，比 llama.cpp Vulkan 少 58%。
- 改寫程式碼那一輪（384 token），WHIRL 解碼約 36–44 tok/s。這一輪大多照抄檔案裡的程式碼，n-gram 草稿對每個引擎都有幫助：llama.cpp ROCm 第 0 段 42.2 tok/s、第 1 段 111.1（它抄了自己先前的回答）；這一輪不算 WHIRL 領先。
- temperature 0 時每個引擎寫出自己的文字，所以工作量並非逐位元相同（有些引擎在後面的回合重讀了檔案）。

### 5.2 R9700

相同的測試資料、用戶端與旗標（WHIRL balance；llama.cpp ROCm 與 Vulkan 用 MTP + n-gram、§2.2 的 R9700 旗標），WHIRL 候選版 rc10。各 2 輪中位數；每輪重開 server，所以第 0 段是冷啟動。每個引擎兩輪相差都不到 1%。

<!-- gen:r97.agent.zh -->
| 工作階段 | WHIRL | llama.cpp ROCm | llama.cpp Vulkan |
|---|---|---|---|
| 冷啟動（20k token 系統＋工具提示尚未快取） | **26.5 s** | 48.5 s | 61.9 s |
| 第二次（共用同一前綴） | **13.1 s** | 25.9 s | 29.0 s |
<!-- /gen -->

- 冷啟動段，WHIRL 比 llama.cpp ROCm 少花 45% 時間，比 llama.cpp Vulkan 少 57%。
- 改寫程式碼那一輪，WHIRL 解碼約 121–145 tok/s；llama.cpp ROCm 第 0 段 120.5 tok/s、第 1 段 183.7（它抄了自己先前的回答）；不算 WHIRL 領先。
- 和 8060S 一樣，各引擎寫出的文字不同，有些會多一次工具呼叫重讀檔案，所以工作量並非逐位元相同。

## 6. 實際指令（0.2.0）

```bat
:: WHIRL server（每個引擎／模型／模式各一個；量測時關閉 SSD 層）
whirl-server.exe MODEL.gguf --port 8099 --balance -np 1 --ctx-per-slot 163840 --kv-ssd-mode off
    （其他模式用 --precise / --fast）

:: llama.cpp b11214 server，R9700（ROCm 版；Vulkan 版用 --device Vulkan1）
set HIP_VISIBLE_DEVICES=1
llama-server -m MODEL.gguf --port 8099 -ngl 99 -fa on -b 2048 -ub 1024 -c 133120 -np 1 --jinja --reasoning off --top-k 1
    [稠密：--spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-n-min 0 --spec-draft-p-min 0.3]
    （MoE：-ub 2048，一般解碼）

:: llama.cpp b11214 server，Radeon 8060S（ROCm 版與 Vulkan 版）
llama-server -m MODEL.gguf --port 8099 -ngl 99 -fa on -b 2048 -ub 512 -ctk f16 -ctv f16 -np 1 -c 133120 --top-k 1
    [稠密：--spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-p-min 0.3]
    （MoE：-ub 2048，一般解碼）
```

量測腳本（PowerShell 工作執行器、只用標準函式庫的 Python 用戶端）不在本 repository；它記錄每個請求的 server timings，原始結果檔與 server log 留在量測機上。

## <a id="appendix-v013"></a>附錄：v0.1.3 發行版評測（R9700，留作參考）

以 WHIRL 0.1.3 對 llama.cpp b11214（ROCm 版）於 2026-10-05 量測；0.2.0 沒有重量。0.1.3 沒有數值模式。0.1.3 的 Radeon 8060S bring-up 快照已由上方 §4 取代，不再列出。

本頁所有 WHIRL 數字都用 v0.1.3 發行包（`whirl.exe`／`whirl-server.exe`，候選版 1，其執行檔與正式版逐位元相同；SHA-256 見 §1）量測。版本對照表中的 v0.1.0（代理工作階段那幾列另有 v0.1.2）在同一天、同一套量測腳本下用各自的發行包重量，所以版本之間是同條件比較。llama.cpp 為 **b11214**（ROCm，commit `2ebd9ae62`），與 v0.1.0 評測相同。兩邊用同樣的提示、同樣的 context 長度、greedy 解碼，GPU 上一次只有一個模型程序。

**WHIRL 領先不多的地方（請先看）：**

- **不用推測解碼的純 decode** 兩邊都受記憶體頻寬限制；dense 模型上 WHIRL 只快 1.13–1.14×。WHIRL 大部分的 decode 領先來自 MTP + n-gram 推測解碼。
- **Qwen3.8 Q4_K_M 的 prefill** 領先比 MXFP4 檔少：96k 為 1.51×、256k token 為 1.44×，Swift-1.5 MXFP4 是 2.28× 與 1.74×。Q4_K_M 冷的 26k token 系統提示只快 1.34×（§8）。
- **dense 模型在 16k context 之後的 decode** 只有 llama.cpp 最快模式的 1.17×（Swift-1.5）與 1.33×（Qwen3.8 Q4_K_M）（§6）。
- **推測解碼的最差情況**（128k token 文件之後寫全新內容，沒有可照抄的內容）只比純 decode 快 1.56×（§5）。

### 1. 測試環境

| | |
|---|---|
| GPU | AMD Radeon AI PRO R9700（gfx1201，RDNA 4，32 GB GDDR6），**以 USB4 外接（eGPU）** |
| 主機 | ASUS ROG Flow Z13（GZ302EA）：AMD Ryzen AI MAX+ 395（16 核／32 緒），128 GB LPDDR5X-8000，其中 Windows 可用 63.6 GB |
| 作業系統／GPU 驅動 | Windows 11 Home 26H2（build 26300.9457）；AMD Software: Adrenalin Edition 26.8.1（驅動 32.0.31041.1004） |
| WHIRL | `whirl-0.1.3-windows-x64` 候選版 1（執行檔與正式版逐位元相同）；`whirl.exe` SHA-256 `af7201bcbfab8257c53837b2cecb3d67e191802c1d0b4bf7716ab04844f907af`、`whirl-server.exe` `8c2feba05929496fa81d30c6526e637930ac6f49aa5de7ae5293dad7b7c9e519` |
| 舊版 WHIRL | 0.1.0 與 0.1.2 發行包（同一台機器、同一套腳本、同一天） |
| llama.cpp | b11214 ROCm Windows 版（`llama-server`、`llama-bench`），`HIP_VISIBLE_DEVICES=1`、`GGML_CUDA_NO_PINNED=1` |
| 量測日期 | 2026-10-05 |

**eGPU 說明：** R9700 接在 USB4 後面（主機連線每方向約 3.8 GB/s）。這會影響模型載入時間，以及主機 RAM／SSD 的 KV 還原；prefill 與 decode 都在 VRAM 內，不受影響。直接插 PCIe 的系統，§8 的還原時間應該會更快。

#### 模型

| 簡稱 | 檔案 | 類型 | 量化 |
|---|---|---|---|
| Swift MXFP4-A（dense） | `Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`（14.7 GiB） | dense 27B，含 MTP head | MXFP4，輸出 head Q6_K（我們發佈並推薦的量化） |
| Ornith MXFP4（MoE） | `Ornith-1.5-35B-A3B-MXFP4.gguf`（18.4 GiB） | MoE，總參數 35B，每 token 約 3B active，含 MTP head | MXFP4 專家（WHIRL 自己發佈的量化） |
| Qwen3.8-27B Q4_K_M（dense，參考） | `Qwen3.8-27B-UD-Q4_K_M.gguf`（15.3 GiB，unsloth） | dense 27B，含 MTP head | 標準 Q4_K_M（unsloth UD） |

主要比較對象是 Swift 與 Ornith；Qwen3.8 Q4_K_M 列出來供參考。

### 2. 方法

#### 2.1 規則

- **一次只跑一個量測項目。** 每個項目先確認 R9700 狀態為 OK、取得機器的 R9700 鎖，執行 `whirl bench` 或啟動一個 server 並等 `/health` 回應，量測後正常關閉 server（Ctrl+C，讓 KV 層寫完），釋放鎖、再確認一次 GPU 狀態，最後把結果附加到同一個結果檔。
- **兩邊都是 greedy 解碼。** 請求送 `temperature 0`；llama-server 另加 `--top-k 1`。兩個引擎收到的請求內容完全相同；llama-server 以 `--reasoning off` 執行。7 題寫程式提示、系統提示與還原測試會送 `chat_template_kwargs: {enable_thinking: false}`。WHIRL 的 server 在請求沒指定時預設開 thinking，所以 16k context 之後的 decode（§6）與 4 人並發（§7）對每個 WHIRL 版本都改送同樣的 `enable_thinking: false` 重量；llama.cpp 那邊原本就以 `--reasoning off` 執行。
- **採用哪一次：** 每個 case、模型、執行檔只採用**最後一次**（一次量測的所有列共用同一個時間戳）；只有 4 人並發測試（§7）是刻意重複量，採**所有次數的平均**（每個版本、每個模型 3 次，llama.cpp 1 次）。量測時標為錯誤或警告的列不採用。
- **每次都用新的 SSD 層：** WHIRL server 每次執行都有自己的 `--kv-ssd-dir`，不會還原到前一次的提示（只有還原測試依設計重用同一個目錄）。
- **輸出相同：** 各版 WHIRL 的 greedy 輸出用 hash 比對（`whirl bench` 用它自己印的 hash；server 用生成文字的 SHA-1，tool call 也算在內，但不含每次隨機的 `id`）。本頁所有版本對照的輸出都相同。
- **量測不含 WHIRL 第一次執行時的 GEMM 自動調校**；server 的計時從啟動完成之後開始。
- **倍數 = WHIRL / llama.cpp**，llama.cpp 取該列**最快**的模式（plain、MTP、MTP + n-gram），並寫出是哪一個模式。時間（TTFT）的倍數是 llama.cpp 時間 / WHIRL 時間。
- 「**no MTP**」表示完全不用推測解碼的純 decode。「N/A」= llama.cpp 沒有對應功能。

#### 2.2 每項測試量什麼

| 測試 | 工具 | 說明 |
|---|---|---|
| Prefill（CLI） | `whirl bench --prefill N --decode 8 --modes plain` | 8,192／32,768／65,536／98,304／131,072 token，KV f16。8k、32k：暖機後取 2 次中最好；64k：暖機後 1 次；96k、128k：計時 1 次 |
| Prefill（server，長文件） | `whirl serve`，一個客戶端 | q8h KV，`-np 1 --ctx-per-slot 262144`；不重複的文件 61.4k／123.5k／180.8k／249k token（192k 檔是 256k 檔在文件邊界截斷），`max_tokens 32`；tok/s = `timings.prompt_per_second` |
| 主要 decode 情境 | 兩個 server，一個客戶端 | v0.1.0 的 7 題 `bench_zhcode` 提示（中文提問，中文說明加英文程式碼），thinking 關閉，`max_tokens 800`；tok/s = `timings.predicted_per_second`，每輪取 7 題平均，3 輪取中位數 |
| 推測解碼的各種情況 | `whirl bench`／`whirl serve` | 最適用：回答會重複輸入內容的檔案編輯提示；主要：代理工作階段重播（91 個請求）；長文件問答與最差：123.5k token 文件之後，分別問一個大量引用文件的問題（512 token）、寫一篇全新的短篇小說（沒有可照抄的內容，1,024 token）。每項都和同一個 case 不開推測解碼比 |
| 16k 之後的 decode | 兩個 server，一個客戶端 | v0.1.0 的 16,354 token 程式碼提示之後生成 128 token（`ignore_eos`） |
| 4 人並發 | 兩個 server，一個客戶端 | 4 個 slot × 4,096 context；4 個不同的約 1.1k token 提示同時送出，各 256 token（`ignore_eos`）；總吞吐 = 生成的 token 數 / 牆鐘時間（含 prefill） |
| 系統提示 TTFT | 兩個 server，一個客戶端 | TTFT = `max_tokens 1` 請求的牆鐘時間；26.4k token 系統提示冷啟動，再用同一個系統提示開新對話（warm）。WHIRL `--kv-ram-mb 16384`，llama-server `--cache-ram 16384` |
| SSD 還原 | WHIRL server | 25.7k token 的 session，正常重啟並沿用同一個 `--kv-ssd-dir`，量下一輪的 TTFT |
| 長文問答 | WHIRL server，q8h | 128k／256k 文件之後回答 1,024 token；粗略檢查：回答有提到被問的函式（不是正式 needle 測試） |
| 代理工作階段 | WHIRL server | 錄下來的 coding agent 工作階段逐個請求重播（OpenAI `tools`／`tool_calls`）：decode tok/s、TTFT、prefix cache 命中 |
| 多人 | WHIRL server，`-np 4` | 啟動 log 的 KV 池容量；4 個短提示同時送出；3 個子代理（每個約 17k token 的提示）prefill 期間的 decode 保底 |

#### 2.3 llama.cpp 設定

- `llama-server --device ROCm0 -ngl 99 -fa on -b 2048 -ub 1024`（MoE `-ub 2048`，8k–32k 最快）`--no-webui --load-mode none --jinja --reasoning off --top-k 1`，KV f16。
- case 的 WHIRL server 參數對應：`-np N` → `-np N`；`--ctx-per-slot N` → `-c N×np`；沒有指定時 `-np 1 -c 131072`；`--kv-ram-mb N` → `--cache-ram N`。
- 模式：plain；MTP = `--spec-type draft-mtp`；MTP + n-gram = `--spec-type draft-mtp,ngram-mod`；兩者都加 `--spec-draft-n-min 0 --spec-draft-p-min 0.3`，`--spec-draft-n-max` 為 3（dense）／MTP 2、MTP + n-gram 1（Ornith），取自 v0.1.0 探測的最佳值。
- llama.cpp 欄標「v0.1.0」的列沿用 v0.1.0 評測的 b11214 數字（同一個 build、同樣的旗標，prefill 用 `llama-bench`）；16k 之後的 decode、4 人並發與系統提示這幾列，是 v0.1.3 用和 WHIRL 完全相同的請求重量的（§6–§8）。

WHIRL 用預設值執行：`whirl serve MODEL`（4 個 slot，自動決定 KV 格式與池大小），加上上面各 case 的參數；no MTP = `WHIRL_MTP=0 WHIRL_NGRAM=0`。

### 3. Prefill

v0.1.0 頁面的圖沒有為 v0.1.3 重畫，以表格為準。

#### 3.1 各自的 bench 工具（KV f16）

| 提示 token | Swift MXFP4-A WHIRL | llama.cpp | 倍數 | Ornith MXFP4 WHIRL | llama.cpp | 倍數 | Qwen3.8 Q4_K_M WHIRL | llama.cpp | 倍數 |
|---|---|---|---|---|---|---|---|---|---|
| 8,192 | 3,469 | 1,338 | **2.59×** | 11,258 | 4,637 | **2.43×** | 1,731 | 1,223 | **1.42×** |
| 32,768 | 2,907 | 1,174 | **2.48×** | 8,633 | 3,778 | **2.29×** | 1,570 | 1,086 | **1.45×** |
| 65,536 | 2,388 | 1,002 | **2.38×** | 6,507 | 3,069 | **2.12×** | 1,398 | 939.8 | **1.49×** |
| 98,304 | 2,021 | 886.7 | **2.28×** | 5,184 | —¹ | — | 1,266 | 837.3 | **1.51×** |
| 131,072 | 1,757 | 791.3 | **2.22×** | 4,311 | 2,239 | **1.93×** | 1,158 | 752.0 | **1.54×** |

llama.cpp：`llama-bench` b11214，每個長度取最快的 micro-batch（v0.1.0 評測，見該頁 §2.3）；96k 是 v0.1.3 README 採用的 b11214 數字。¹ Ornith 在 96k 與 256k token 沒有量 llama.cpp。

#### 3.2 長文件經 WHIRL server（q8h KV）

| 提示 | Swift MXFP4-A | Ornith MXFP4 | Qwen3.8 Q4_K_M |
|---|---|---|---|
| 61.4k token | 2,162 | 5,671 | 1,290 |
| 123.5k token | 1,533 | 3,674 | 1,032 |
| 180.8k token | 1,212 | 2,771 | 876.6 |
| 249k token | 969.8 | 2,135 | 742.2 |
| 256k，llama.cpp（llama-bench b11214） | 556.1，KV f16（**1.74×**） | —¹ | 516.4，KV q8_0（**1.44×**） |

llama.cpp 跑 Swift MXFP4 256k 用的是 f16 KV（放得下，31.3 GB）；Qwen3.8 Q4_K_M 用 f16 會溢到共用記憶體，所以改用 q8_0 KV。

#### 3.3 v0.1.0 → v0.1.3（Swift MXFP4-A）

| Prefill | v0.1.0 | v0.1.3 | 變化 |
|---|---|---|---|
| 8k，f16 KV（bench） | 3,273 | 3,469 | +6.0% |
| 32k，f16 KV（bench） | 2,605 | 2,907 | +11.6% |
| 96k，f16 KV（bench） | 1,653 | 2,021 | +22.3% |
| 128k，f16 KV（bench） | 1,407 | 1,757 | +24.8% |
| 128k，q8h KV（server） | 1,353 | 1,533 | +13.3% |
| 192k，q8h KV（server） | 1,054 | 1,212 | +15.1% |
| 256k，q8h KV（server） | 832.5 | 969.8 | +16.5% |

提示越長進步越多，因為 v0.1.3 的 prefill attention kernel（`attn_kg`，[kernels.md](kernels.md#flash)）讓同一組的 query head 共用每一次 key／value 讀取；輸出與舊 kernel 逐位元相同。

### 4. Decode — 主要情境：中文提問、中文說明、英文程式碼

| 模型 | WHIRL MTP+n-gram（預設） | WHIRL **no MTP** | llama.cpp **plain（no MTP）** | llama.cpp 最快（模式） | WHIRL 預設 / llama.cpp 最快 | WHIRL no MTP / llama.cpp plain |
|---|---|---|---|---|---|---|
| Swift MXFP4-A（dense） | **112.4** | 38.1 | 33.4 | 60.8（MTP） | **1.85×** | **1.14×** |
| Ornith MXFP4（MoE） | **257.6** | 177.4 | 118.8 | 118.8（plain） | **2.17×** | **1.49×** |
| Qwen3.8-27B Q4_K_M（dense） | **97.3** | 34.8 | 30.9 | 56.0（MTP + n-gram） | **1.74×** | **1.13×** |

7 題 × 800 token 的平均，3 輪取中位數。llama.cpp 為 v0.1.0 的數字（b11214）。MoE 模型上 llama.cpp 的 MTP 比它的純 decode 慢，所以對照的是 plain。與 v0.1.0 比（同一天、同一套腳本）：Swift 109.2 → 112.4（+2.9%），Qwen3.8 99.7 → 97.3（−2.4%）；兩者輸出都相同。

### 5. 推測解碼：最適用、主要與最差情況（Swift MXFP4-A）

| | 最適用：檔案編輯（內容重複） | 主要：coding agent 工作階段 | 長文件問答（大量引用，128k 之後） | 最差：128k 之後寫全新內容（沒有可照抄的內容） |
|---|---|---|---|---|
| Decode tok/s，MTP + n-gram | 174.4 | 92.9 | 87.6 | 39.4 |
| Decode tok/s，不開推測解碼 | 39.1 | 32.8 | 25.3 | 25.2 |
| 加速倍數 | **4.47×** | **2.83×** | **3.47×** | **1.56×** |
| 每個 cycle 的 token 數 | 6.19 | — | 6.45 | 1.76 |

開與不開推測解碼的輸出逐位元相同。主要 = 代理工作階段重播（91 個請求，12k → 65k token）；重播不記錄每個 cycle 的 token 數。中文寫程式短提示（`whirl bench`，512 token）：120.7 對 39.6 tok/s（3.05×，每個 cycle 4.00 token）。

| v0.1.0 → v0.1.3，MTP + n-gram | v0.1.0 | v0.1.3 | 變化 |
|---|---|---|---|
| 檔案編輯 | 157.3 | 174.4 | +10.9% |
| 中文寫程式短提示 | 110.7 | 120.7 | +9.1% |
| 代理工作階段 1（91 個請求） | 81.5 | 92.9 | +13.9% |
| 128k 之後的長文件問答 | 63.7 | 87.6 | +37.5% |

### 6. 16k token context 之後的 decode

| 模型 | WHIRL MTP+n-gram（預設） | llama.cpp plain | llama.cpp MTP | llama.cpp MTP + n-gram | WHIRL / llama.cpp 最快 |
|---|---|---|---|---|---|
| Swift MXFP4-A（dense） | **71.4** | 31.2 | 60.8 | 60.9 | **1.17×**（MTP + n-gram） |
| Ornith MXFP4（MoE） | **316.4** | 106.9 | 104.7 | 89.7 | **2.96×**（plain） |
| Qwen3.8-27B Q4_K_M（dense） | **80.7** | 29.4 | 60.6 | 60.7 | **1.33×**（MTP + n-gram） |

兩邊都關閉 thinking（§2.1）。同一批請求中 16,354 token 提示的 prefill：Swift 3,156 對 1,285 tok/s（2.46×），Ornith 9,770 對 4,184（2.33×），Qwen3.8 1,596 對 1,179（1.35×），都是對 llama.cpp plain。v0.1.0 → v0.1.3（同一天）：Swift 70.7 → 71.4（+0.9%），Ornith 313.5 → 316.4（+0.9%），Qwen3.8 76.5 → 80.7（+5.5%），輸出相同。v0.1.0 頁面的 16k 數字（Ornith 301.5、Swift 69.5）是用另一套量測腳本量的，不能直接比較。

### 7. Server，4 人並發

| 模型 | WHIRL MTP+n-gram（預設） | llama.cpp plain | llama.cpp MTP | llama.cpp MTP + n-gram | WHIRL / llama.cpp 最快 |
|---|---|---|---|---|---|
| Swift MXFP4-A（dense） | **182.1** | 64.0 | 54.9 | 56.4 | **2.84×**（plain） |
| Ornith MXFP4（MoE） | **381.1** | 176.8 | 107.7 | 129.0 | **2.16×**（plain） |
| Qwen3.8-27B Q4_K_M（dense） | **122.4** | 57.1 | 53.3 | 50.1 | **2.14×**（plain） |

總吞吐 tok/s，含 prefill，兩邊都關閉 thinking（§2.1）。WHIRL：每個模型 3 次的平均；同一天的 v0.1.0（各 3 次）：Swift 173.9（v0.1.3 +4.7%）、Ornith 377.5（+1.0%），輸出相同。llama.cpp：每個模式 1 次。b11214 的推測解碼模式不會隨並發使用者數擴展；WHIRL 每個 slot 都維持 MTP + n-gram（批次 verify）。

### 8. 首 token 時間：系統提示與 SSD 還原

| 模型 | 請求 | WHIRL TTFT（秒） | llama-server TTFT（秒） | llama.cpp / WHIRL |
|---|---|---|---|---|
| Swift MXFP4-A | 26.4k 系統提示，冷 | 9.276 | 21.768 | **2.35×** |
| Swift MXFP4-A | 新對話，同一個系統提示（warm） | 0.108 | 0.384 | **3.56×** |
| Swift MXFP4-A | 伺服器重啟後的 25.7k session（從 SSD 還原） | 0.613 | N/A | N/A |
| Ornith MXFP4 | 26.4k 系統提示，冷 | 3.082 | 6.667 | **2.16×** |
| Ornith MXFP4 | 新對話，同一個系統提示（warm） | 0.098 | 0.182 | **1.86×** |
| Ornith MXFP4 | 伺服器重啟後的 25.7k session（從 SSD 還原） | 0.318 | N/A | N/A |
| Qwen3.8 Q4_K_M | 26.4k 系統提示，冷 | 17.515 | 23.555 | **1.34×** |
| Qwen3.8 Q4_K_M | 新對話，同一個系統提示（warm） | 0.121 | 0.369 | **3.05×** |
| Qwen3.8 Q4_K_M | 伺服器重啟後的 25.7k session（從 SSD 還原） | 0.672 | N/A | N/A |

- *Warm*：WHIRL 重用它的系統提示檢查點，llama-server 重用快取的前綴（兩邊都快取了 26,403 token）。
- *SSD 還原*：server 正常關閉（Ctrl+C）後沿用同一個 `--kv-ssd-dir` 重新啟動，session 的下一輪從 WHIRL 的 SSD 層還原。llama.cpp 沒有自動的持久化 KV 快取，所以這一列是 N/A。
- 這裡 llama.cpp 只跑 plain：回答只有 1 個 token，推測解碼用不上。
- WHIRL 的冷 TTFT 包含對提示跑一次 MTP 區塊。

### 9. 長上下文（q8h KV，WHIRL server，一個使用者）

| | Swift MXFP4-A | Ornith MXFP4 | Qwen3.8 Q4_K_M |
|---|---|---|---|
| 128k 之後的 decode tok/s（回答 1,024 token） | 62.5 | 214.0 | 58.3 |
| 256k 之後的 decode tok/s（回答 1,024 token） | 35.1 | 115.6 | 33.5 |
| llama.cpp 256k 之後，plain（llama-bench `tg64@d262144`） | 16.8，KV f16（**2.09×**） | —¹ | 10.5，KV q8_0（**3.19×**） |
| 長文問答粗略檢查（128k／256k） | ✓／✓ | ✓／✓ | ✓／✓ |

`--ctx-per-slot 262144`，q8h KV（int8 的 key 與 value；query 與 key 先做 Hadamard 旋轉）。粗略檢查是問文件中的某個函式，✓ = 回答有提到它（不是正式的 needle 測試）。llama.cpp 那一列是在 262,144 token 的 context 之後純 decode 64 token，沒有推測解碼。

### 10. Coding agent 工作階段（Swift MXFP4-A，WHIRL server）

| 重播 | v0.1.0 | v0.1.2 | v0.1.3 | 變化（v0.1.3 / v0.1.0） |
|---|---|---|---|---|
| 工作階段 1（scrapy，91 個請求，12k → 65k token），平均 decode tok/s | 81.5 | — | 92.9 | +13.9% |
| 工作階段 4（aiohttp，102 個請求，14k → 102k），平均 decode tok/s | 79.2 | — | 95.2 | +20.1% |
| 工作階段 3 到第 143 個請求（共 144 個），平均 decode tok/s | 64.9 | 65.0 | 81.7 | +25.9% |
| 工作階段 3 第 143 個請求本身（127.9k token） | 53.7 | 53.5 | 75.2 | +40.0% |

錄下來的 coding agent 工作階段，連同 tools 與 tool call 逐個請求重播；每一版的 prefix cache 命中都是提示 token 的 98–99%，每一版的輸出都相同。

### 11. 一個 server 多個使用者（Swift MXFP4-A）

| | v0.1.0 | v0.1.3 | 變化 |
|---|---|---|---|
| KV 池容量，預設 4 個 slot（token） | 210,688 | 226,304 | +7.4% |
| 4 個短提示同時送出：整批完成時間（秒） | 9.57 | 8.62 | −9.9% |
| ……總吞吐 tok/s | 213.9 | 237.6 | +11.1% |
| 3 個子代理 prefill 期間主串流的 tok/s | 1.08 | 9.63 | ×8.9 |
| ……最差 1 秒視窗的 token 數 | 1 | 5 | |
| 3 個子代理先到、主請求 0.5 秒後才到：主請求 TTFT（秒） | 17.256 | 16.205 | −6.1% |

子代理：3 個各約 17k token 的提示。兩版輸出相同。

### 12. 實際指令

```bat
:: WHIRL CLI
whirl bench MODEL.gguf --prefill 8192 --decode 8 --modes plain
whirl bench MODEL.gguf --prefill 512 --decode 512 --prompt @zh_code.txt

:: WHIRL server（每次執行都用自己的 SSD 層目錄）
whirl serve MODEL.gguf --port 8099 --log-file LOG --kv-ssd-dir RUN\kvcache [-np 1 --ctx-per-slot 262144] [--kv-ram-mb 16384]
    （q8h KV：設 WHIRL_KV=q8h；no MTP：設 WHIRL_MTP=0 與 WHIRL_NGRAM=0）

:: llama.cpp server
set HIP_VISIBLE_DEVICES=1& set GGML_CUDA_NO_PINNED=1
llama-server -m MODEL.gguf --port 8099 --device ROCm0 -ngl 99 -fa on -b 2048 -ub 1024 --no-webui --load-mode none ^
    --jinja --reasoning off --top-k 1 -np 1 -c 131072 [--cache-ram 16384]
    [--spec-type draft-mtp --spec-draft-n-max N --spec-draft-n-min 0 --spec-draft-p-min 0.3]
    [--spec-type draft-mtp,ngram-mod --spec-draft-n-max N --spec-draft-n-min 0 --spec-draft-p-min 0.3]
```

量測腳本（PowerShell 的工作執行器，加上只用標準函式庫的 Python 客戶端）不在本 repository；它記錄每個請求的計時，每次執行的 server log 與完整命令列都保存在量測機器上。

### 13. v0.1.3 沒有重量的項目

VRAM 用量、vision 編碼、88 與 2,048 token 的 prefill、同一份文字經 server 的 prefill，以及檔案編輯情境與 llama.cpp 的對照，這次都沒有重量；它們的 v0.1.0 數字在 [v0.1.0 頁面](https://github.com/tsaipifong/whirl-llm/blob/v0.1.0/docs/guide/zh-TW/benchmarks.md)。v0.1.3 把 token embedding 放到 pinned 主記憶體，VRAM 用量應該比 v0.1.0 低。
