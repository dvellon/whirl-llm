# WHIRL v0.2.0

[English](RELEASE_NOTES_v0.2.0.md) · **繁體中文**

> 草稿，供 owner 審閱，尚未發布任何東西。標示 `[8060S: TBD]` 或 `[TBD]` 的項目，等尚未完成的量測或發行版檢查。每個數字的來源都寫在 HTML 註解裡。

AMD Radeon AI PRO R9700（gfx1201）的原生 Windows LLM 推論引擎；從這一版起也支援 Radeon 8060S（Ryzen AI Max+ 395，gfx1151）。

在 R9700 上，用預設模式、相同的 GGUF 檔與相同的提示（greedy、單一使用者）對 llama.cpp b11214：8k 到 249k token 的 prefill，dense 的 Qwen3.8-27B Q4_K_M 快 59% 到 70%，MoE 的 Ornith-1.5-35B-A3B MXFP4 快 73% 到 153%；128k token 文件之後的 decode，dense 快 82%、MoE 快 91%；dense 模型 4 人同時使用，總吞吐 251 tok/s 對 91。表格與 llama.cpp 的旗標在下面。這些加速不改變輸出：在 precise 與 balance，輸出與同一模式下單純 greedy 解碼相同。

<!-- src：audit/REL3.md R9700 一節，balance 欄對「llama.cpp best」：dense prefill 8k +59%、32k +59%、128k +63%、249k +70%；MoE prefill 8k +153%、32k +123%、128k +87%、249k +73%；128k QA decode dense +82%、MoE +91%；4 人 251.4 對 90.6。 -->

下載：`whirl-0.2.0-windows-x64.zip`（SHA-256：`[TBD]`）。需求：Windows 11、Adrenalin 26.8.1 或更新版、一張 R9700 或一顆 Radeon 8060S。

## 重點

- **內建三種數值模式：precise、balance（預設）、fast。** precise 走單純的 f16 參考路徑，輸出可逐位元重現；balance 是 0.1.x 的行為（近乎無損，對 precise 的 KL <= 約 0.005）；fast 需自行開啟，用有上限、且公布數字的品質換速度（4-bit KV、放寬草稿接受）。見[下表](#數值模式)。
- **Radeon 8060S 正式支援。** 那裡的主記憶體與 SSD 快取層預設開啟：128k token 的 session 從主記憶體回來只要 0.34 秒（重啟後從 SSD 2 秒），而不是 9 分鐘的 prefill。precise 模式的 decode 快了約一倍。[8060S: TBD ——對 llama.cpp 的比較]
- **SSD 快取在伺服器執行期間不再寫入。** 新預設只在正常停止時寫入，磁碟上限 64 GiB，7 天沒用的檔案會被刪除。有使用者回報舊預設在 23 分鐘內寫了約 400 GB。
- **DFlash2 草稿模型，需自行開啟。** 單人、短到中等 context、輸出可預測時快 19% 到 48%；多人或長 context 反而變慢，所以不是預設。草稿檔不隨發行包附上。
- **任何權重格式的 dense 模型 prefill 更快**（balance 與 fast；Qwen3.8-27B Q4_K_M 快 7% 到 10%），並包含外部程式碼審查的修正。

<!-- src：模式：CHANGELOG.md Unreleased；8060S 數字：audit/TIER-8S.md（128k：冷 536.7 秒、RAM 0.34 秒、SSD 2.00 秒）與 audit/GEMVH-1151.md；400 GB／23 分鐘：audit/SSD-QUIET.md；DFlash2 +19..+48%：audit/DFLASH-1.md 第 7 節；prefill +7..+10%：CHANGELOG KG-1（8k +10.4%、32k +9.9%、128k +7.0%）。 -->

## 對 llama.cpp 的速度（R9700）

balance 模式（預設）對 llama.cpp b11214（ROCm），`llama-server --device ROCm0 -ngl 99 -fa on -b 2048 -ub 1024（MoE 2048）--jinja --reasoning off --top-k 1 -np 1`，f16 KV，每一格取 llama.cpp 一般模式與 MTP + n-gram 模式中較快的一個。2026-10-12 量測。「快 X%」是 WHIRL / llama.cpp - 1。

| R9700，balance | Qwen3.8-27B UD-Q4_K_M（dense） | Ornith-1.5-35B-A3B MXFP4（MoE） |
|---|---|---|
| Prefill tok/s，8k 提示 | 1,914 對 1,201（快 59%） | 11,199 對 4,426（快 153%） |
| Prefill tok/s，32k 提示 | 1,721 對 1,082（快 59%） | 8,585 對 3,853（快 123%） |
| Prefill tok/s，128k 提示 | 1,230 對 756（快 63%） | 4,273 對 2,285（快 87%） |
| Prefill tok/s，249k 提示 | 912 對 536（快 70%）¹ | 2,659 對 1,534（快 73%） |
| Decode tok/s，針對 8k 文件發問 | 84.9 對 41.8（快 103%） | 229.3 對 103.5（快 122%） |
| Decode tok/s，針對 128k 文件發問 | 49.3 對 27.1（快 82%） | 132.9 對 69.7（快 91%） |
| Decode tok/s，128k 之後寫全新內容（草稿最不利） | 33.8 對 21.2（快 59%） | 113.8 對 69.9（快 63%） |
| Decode tok/s，中文寫程式對話 | 100.9 對 58.1（快 74%） | 242.5 對 108.8（快 123%） |
| 4 人同時使用，總吞吐 tok/s | 251.4 對 90.6（快 177%） | 未量測 |

<!-- src：audit/REL3.md R9700 一節（rc4 tip 100a9a8），balance 欄對 llama.cpp best。 -->

¹ llama.cpp 在 dense 模型的 249k token 放不下 f16 KV，改用 q8_0 KV。長度掃描都是單次量測，差距小於約 5% 請當成雜訊。三種模式在每個長度的數字、方法，以及領先幅度小的地方（例如 precise 讓 dense 對話 decode 慢 25%）：README 的[效能](https://github.com/tsaipifong/whirl-llm/blob/main/docs/README_zh-TW.md#效能)。Swift-1.5 MXFP4 沒有為這個版本重新量測，0.1.3 的數字仍留在 README。

[8060S: TBD ——表格：Radeon 8060S，WHIRL 對 llama.cpp，prefill／decode，三種模式，機器與功耗限制]

## <a id="數值模式"></a>數值模式

啟動時擇一：`--precise`、`--balance`（預設）、`--fast`。

| | precise | balance（預設） | fast |
|---|---|---|---|
| KV 快取，dense／MoE | f16／f16 | q8h（int8）／f16 | q4（4-bit）／q4 |
| 輸出 | 除了 GGUF 檔本身，沒有再量化任何東西；加速與單純 greedy 逐位元相同 | 近乎無損，對 precise 的 KL <= 約 0.005；加速與單純 greedy 逐位元相同 | KL <= 0.03 以內；q4 KV 對 f16 實測 KL：0.0016 到 0.0100（R9700）；不保證逐位元相同 |
| 對 balance 的速度（R9700） | decode 慢 25% 到 29%，dense prefill 慢 5% 到 10% | 基準 | dense decode 快 15% 到 18%；MoE 只有約 64k 以上才比較快 |
| 什麼時候選 | 需要精確重現輸出，或要拿別的引擎對照 WHIRL | 幾乎所有情況 | 長上下文，或能接受輸出小幅變動而想要最高 tok/s |

<!-- src：CHANGELOG.md Unreleased（BAL-Q8、precise、FAST-1c）；audit/REL3.md；internal_docs/v013_plan.md（門檻：balance KL <= 約 0.005、fast KL <= 0.03，owner 2026-10-08）。 -->

這些保證的意思：

- precise 與 balance：一個請求單獨跑或和其他人一起跑、開或不開 MTP／n-gram／DFlash2 草稿、冷啟動或從快取前綴接續，得到的 token 都一樣。每個模式都由 `whirl selftest`、kernel 測試、golden 輸出與伺服器 gate 把關。
- fast：品質由門檻限定，而不是靠逐位元相同。同一個請求單獨跑與和其他人一起跑，文字可能不同。門檻測試（Qwen3.8-27B Q4_K_M）中，12 個寫程式任務有 12 個執行正確、24 次生成有 0 次陷入迴圈，重複率 0.167（precise 為 0.157）；被接受的草稿約 7% 來自放寬規則。這次發行沒有在 R9700 上重跑該門檻。[TBD：owner 決定要不要重跑]
- KL 是拿 logits 與同一個模型的 f16 KV logits 比較，表示輸出分布移動多少，不是模型本身有多準。

<!-- src：audit/CONC-8S.md（fast 單獨對並行是盡力而為）；progress/FAST-2.log 2026-10-07 12:49 與 13:01（門檻數字；在 8060S 上量測）。 -->

balance 下 dense 模型可能注意到的變化：KV 格式不再看放得下什麼。0.1.x 依序選 f16、q8v、q8h；現在 balance 在 dense 模型一律用 q8h、在 MoE 模型用 f16，放不下所選格式時會降低 context 並顯示警告（若您設了 `--ctx`／`--ctx-per-slot`，則停止並顯示訊息）。它絕不會自己改用精度較低的格式。

## DFlash2 草稿（需自行開啟）

`whirl-server MODEL.gguf --draft dflash --draft-model PATH`（`whirl chat`、`whirl bench` 也可以）。預設仍是 MTP + n-gram。草稿檔不隨發行包附上：Hugging Face 上的 [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)（Apache-2.0；我們測試的是 Q8_0 GGUF）。它佔用約 1.9 GiB VRAM，KV 池縮小約 12%。輸出等於同一模式下的單純 greedy 解碼。

| Qwen3.8-27B Q4_K_M，R9700，balance | MTP + n-gram | DFlash2 + n-gram | 變化 |
|---|---|---|---|
| Java API 程式碼／SQL 報表，一位使用者（對話） | 105.7／83.7 | 125.6／123.6 | +19%／+48% |
| 寫程式 agent 工作階段重播，平均（65k 最後一步） | 83.4（66.2） | 86.9（74.4） | +4%（+12%） |
| 中文寫程式對話 | 100.9 | 91.3 | -10% |
| 4 人同時使用 | 211.3 | 129.7 | -39% |
| 37k／108k 提示之後的問答 | 54.6／40.3 | 39.1／25.8 | -28%／-36% |

<!-- src：audit/DFLASH-1.md 第 7 節（步驟 3、7、8）；audit/REL3.md DFlash2 表（agent 重播、中文對話）。完整表格在 README。 -->

適合：一位使用者、短到中等 context、英文為主或可預測的輸出（API／SQL 樣板）。不建議：多位使用者、context 達 37k token 以上、以中文為主的輸出。尚未用 DFlash2 量測：Swift-1.5、MoE 模型、Radeon 8060S。

## SSD 快取

- 預設 `--kv-ssd-mode shutdown`：伺服器執行期間不寫 SSD；正常停止（Ctrl+C）時依最近使用順序寫入 session，最多 60 秒。
- `--kv-ssd-mode always` 仍然可用；session 5 分鐘沒有變動才寫入（`--kv-ssd-delay SEC`），而且只寫被重用過的。
- 磁碟上最多 64 GiB（`--kv-ssd-gb`），其他版本或模型留下的舊檔也計入，啟動時就強制執行；先刪最久沒用的 session。7 天沒用的檔案會被刪除。損毀或寫到一半的快取檔啟動時刪除。
- 促成這項改動的回報（23 分鐘約 400 GB）是舊預設把每一個不同的長提示都寫下去，每個 0.3 到 1.5 GiB。我們沒有找到隱藏的寫入放大：新 session 只寫一次，大小是檔案的 1.004 到 1.012 倍。
- 0.1.x 的使用者若不想等升級：`--kv-ssd-gb 0` 可關閉 SSD 層。
- 手動刪除快取：停止伺服器，刪除 `%LOCALAPPDATA%\whirl\kvcache`（或您的 `--kv-ssd-dir`）裡的 `*.wkv` 檔。

<!-- src：audit/SSD-QUIET.md；CHANGELOG.md Unreleased；internal_docs/v013_plan.md 2026-10-11（「不出 v0.1.5 hotfix；v0.1.4 使用者可用 --kv-ssd-gb 0」）。 -->

## Radeon 8060S

- 兩套 GPU 程式碼在同一個 zip。同時有 R9700 時，`--device 8060s` 可選 8060S。三種模式都能用；kernel 測試、golden 輸出與伺服器 gate 都通過。尚未提供：圖片輸入。
- **主記憶體與 SSD 快取層預設開啟。** 在 Windows 上，BIOS 保留區（VRAM）與作業系統 RAM 是分開的兩塊，所以這一層是真正獨立的記憶體。大小依作業系統可見的 RAM 決定：1/8，介於一個最大 context 的 session 與 16 GiB 之間，不超過可用 RAM 的 1/4，低於 32 GB 關閉（64 GB 的機器上為 8 GiB；啟動 log 會寫出）。Qwen3.8-27B Q4_K_M、balance，session 回來時的首 token 時間：8k token 重新 prefill 20.9 秒對 0.11 秒（RAM）／0.21 秒（重啟後 SSD）；128k token 536.7 秒對 0.34 秒／2.00 秒。
- **precise 的 decode 快了約一倍**：Qwen3.8-27B Q4_K_M，短 context 5.34 到 11.19 tok/s（MTP + n-gram 20.2 到 40.2）；32k 4.9 到 9.27（MTP 15.1 到 23.0）。輸出逐位元相同。
- fast 模式使用 4-bit KV（Ornith-1.5-35B-A3B 在 128k 的 decode：58.5 tok/s，f16 KV 為 32.7）。
- [8060S: TBD ——對 llama.cpp 的比較與各模式速度]

<!-- src：audit/TIER-8S.md（表與大小規則）、CHANGELOG.md TIER-8S-DEF；audit/GEMVH-1151.md（之前 = 第 3 節 WHIRL_PREC_GEMVH=0 的結果，現在 = GEMVH-2 端到端表；同一台機器的兩次量測）；CHANGELOG fast kvq4 on the Radeon 8060S。 -->

## 其他變更

- 任何權重格式的 dense 模型，在 balance 與 fast 取得 f16 DeltaNet 與 GEMM 輸出路徑（以前只有 MXFP4 檔）：Qwen3.8-27B Q4_K_M prefill 8k 1,849 到 2,042 tok/s（+10.4%）、32k +9.9%、128k +7.0%，decode 不變。
- 溫度 > 0 時，MTP 取樣在 balance 與 fast 仍然精確（`specsample`）：同一個 seed 的文字在 MTP 開或關時都一樣。
- 外部審查的修正，輸出不變：草稿步驟丟出例外時，MTP 草稿視窗會還原；prefill 批次不會縮到小於排程器的 chunk 大小；比記錄短的快取檔會被刪除，讀取不足退回冷 prefill；還原複製失敗就視為失敗；SSD 目錄鎖失敗時關閉 SSD 層；DeltaNet 形狀是 kernel 沒實作的，載入時拒絕。
- fast 模式：重複的請求不論串流與否、冷啟動或從自己的檢查點接續，文字都一樣。

<!-- src：CHANGELOG.md Unreleased（KG-1、specsample、FIX-FS）；audit/FIX-REVIEW2.md。 -->

## 逐位元一致與測試

候選版上的 gate（R9700）：CPU 測試 2239 + 212 + 86 + 40 + 960 + 537 項，0 失敗；kernel 測試（Q4_K_M 與 MXFP4）0 失敗；`whirl selftest` 在 dense 與 MoE 模型的三種模式皆 ok；golden 輸出 18 項中 18 項與記錄的基準相同；伺服器 gate basic + mt_cache 在 fast 與 balance 皆 27/27；fast_stream dense 與 MoE 皆 31/31；正常停止後 SSD 層還原，輸出一致。8060S：kernel 測試、selftest（6 項）、golden 輸出（18 項中 18 項）、並行修正後伺服器 gate basic + mt_cache（balance、fast）27/27、fast_stream 31/31。[TBD：在發行 zip 裡的實際檔案上重跑]

<!-- src：audit/RC4-GATES.md（R9700 與 8060S 表，「8b0dab5 R9700」重驗）、audit/CONC-8S.md 與 audit/TIER-8S.md（修正後的 8060S gate）。DFlash2：precise 下輸出 hash 與 plain greedy 相同（RC4-GATES）。 -->

## 已知限制

- fast 模式不保證與單純 greedy 解碼逐位元相同，同一個請求單獨跑可能與和其他人一起跑不同。
- 接續對話時會重用上一輪的 KV 與 DeltaNet 狀態：與重新處理整段歷史在數值上等價，但不是逐位元相同（與 0.1.x 相同）。
- Dense 的 precise 在 32 GB 卡上放不下 256k context（f16 KV：KV 池 176,128 token）；用這個設定程式會拒絕啟動並說明原因。
- `--kv-ssd-mode shutdown` 下，當機或被強制結束會失去尚未寫入的 session。
- 8060S 上尚未提供圖片輸入。
- DFlash2 只在 R9700、Qwen3.8-27B Q4_K_M 上量測過。
- 執行檔沒有程式碼簽章（SmartScreen 可能會詢問；見 README）。
- 只支援 R9700 與 Radeon 8060S。RX 9070 系列（16 GB）未測試。

## 升級

- 不需要設定任何東西。`%LOCALAPPDATA%\whirl` 的 kernel 調校快取在模型、GPU 與建置相符時會沿用。
- 0.1.x 寫下的快取檔不會被還原；它們計入 64 GiB 上限並最先被刪除（也可以自己刪除 `%LOCALAPPDATA%\whirl\kvcache`）。
- Dense 模型在預設模式一律用 q8h KV（0.1.x 依放得下什麼選 f16、q8v 或 q8h）。要 f16 KV，請用 `--precise`。
- 8060S 使用者：主記憶體層現在會從作業系統拿記憶體（64 GB 的機器上 8 GiB）。用 `--kv-ram-mb 0` 可關閉。

完整變更清單：[CHANGELOG.md](https://github.com/tsaipifong/whirl-llm/blob/main/CHANGELOG.md)。
