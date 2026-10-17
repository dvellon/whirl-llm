[English](../en/vision.md) | **繁體中文**

# 圖片輸入（視覺輸入）

> **狀態。** 圖片輸入已收錄在 WHIRL 0.1.0（gfx1201）：`whirl chat --mmproj … --image …`，以及伺服器的
> `--mmproj` 搭配 OpenAI `image_url`。在我們的測試圖片上，C++ 編碼器的 embedding 與研究建置逐位元相同。
> 從 0.2.0 起 gfx1151（Radeon 8060S）也支援，見 [§7](#gfx1151)。選項見[使用參考](usage.md#serve)。

**對誰有幫助：** 任何要在記憶體受限的文字伺服器（server）上加入視覺編碼器的人 — 如何在不佔用純文字使用者 VRAM 的前提下加入圖片輸入 — 以及任何在除錯「輸出取決於伺服器在目前請求*之前*做過什麼」問題的人。

## 1. 目標與限制

預設的 27B server 在 128k + 64k 下限之上，只留給 KV 池一點點餘裕（[kv-and-caching.md](kv-and-caching.md#floor)）。常駐的視覺編碼器會讓*每一位*使用者（包括從不傳送圖片的人）的預設 KV 格式從 q8v 退到較慢的 q8h。因此需求是：**載入 mmproj 不得改變 KV 池大小，也不得改變常駐 VRAM。** 在候選建置上量測：使用 `--mmproj` 時池大小不變，載入 mmproj 時 VRAM 變化 0.0 MiB。

## 2. 設計

| 部分 | 設計 |
|---|---|
| 投影器權重 | 啟動時載入 **pinned host RAM（鎖頁主機記憶體）**；在圖片到來之前，編碼器沒有任何部分位於 VRAM |
| 編碼器生命週期 | kernel、常駐權重與一條複製串流在第一張圖片到來時**依需求**建立，並在**閒置逾時後釋放**（`--vis-idle-s`，預設 60 s）；VRAM 回到純文字時的佔用量 |
| 權重模式 | `--vis-mode auto`（預設）：若有足夠的可用 VRAM 則權重常駐，否則**每張圖片逐層串流**；`resident` / `stream` 強制指定模式。預設 27B 設定剩下的可用 VRAM 不夠，因此權重採串流：在這台 USB4 eGPU 上每張圖片約 260 ms 的 H2D（視環境而定） |
| activation（啟動值） | 借用 prefill（提示詞預填）的暫存緩衝區（不新增配置）；log 會回報一張 4096-token 圖片需要多少 |
| 前處理 | resize/normalize 與 PIL 的 bicubic 路徑逐位元相同 |
| 嵌入快取 | 投影後的嵌入依**內容雜湊**快取在主機記憶體中（LRU，`--vis-cache-mb`，預設 1024 MiB）：之後請求中的同一張圖片不會再編碼一次 |
| 提示詞整合 | 提示詞中的圖片佔位位置接收圖片嵌入列（在第一層之前複製到嵌入輸出中） |
| 位置 | **多段 RoPE（M-RoPE）：** 每一列取得一個多分量位置；旋轉維度被切成數段（依 GGUF 的 RoPE section 中繼資料），每段以一個位置分量旋轉。由專用的 `attn_prep` 變體套用；每列的位置會在 forward 時上傳。圖片之後的文字繼續使用多段位置 |
| 前綴快取 | 圖片以內容雜湊識別，因此重複出現某張圖片的對話會重用其 KV（一次測試：238 個提示詞 token 中 238 個命中快取）；在圖片結尾會儲存一個共用的 `prefix` 檢查點，讓之後的回合從其後接續 |
| API | OpenAI `image_url` 內容片段，搭配 `data:` URL；每個請求可含多張圖片 |
| 不支援的 GPU | 不含多段 RoPE kernel 的 code object 會在啟動時拒絕 `--mmproj`（`MropeUnsupported`） |

**編碼器速度。** 為編碼器寫的第二版注意力（每個 wave 32 個 query、64-key 的 softmax 區塊、延遲重新縮放、Vᵀ 放在全域記憶體、加墊的 LDS）將暖機後的 1080p 編碼從 422 ms 降到 255 ms。

## 3. 數值

| 比較 | 結果 |
|---|---|
| WHIRL 編碼器 vs f32 numpy 參考實作，4 張圖片 | 相對 L2 0.12–0.30%，最小 cosine ≥ 0.9995 |
| llama.cpp 編碼器（CPU 與 8060S GPU）vs 同一參考實作 | 相對 L2 1.6–12%，最小 cosine 0.89–0.998 |
| 帶圖片的語言模型，vs `llama-mtmd-cli` b11214，使用相同提示詞 token（Q4_K_M） | 形狀描述前 524 個字元相同（有一個字不同），程式碼轉錄兩者皆完全正確 |
| 圖片提示詞上的 MTP greedy vs 一般 greedy | 相同 |

支援 BF16 投影器（如 MoE 模型的 mmproj）；含圖片的 MoE 冒煙測試已通過。

## 4. 找到的 bug

### <a id="vis-pad"></a>4.1 未初始化的填充讓嵌入取決於伺服器歷史

**症狀。** 在 server 或 CLI 內計算的圖片嵌入與獨立編碼器工具的結果不同，而且每次執行都不同，取決於 server 先前處理過什麼。獨立工具則一直是乾淨的。

**根本原因。** 編碼器的 Q/K 準備 kernel 會把 head 維度補齊；它沒有寫入填充 lane 76–79。注意力會讀取完整的補齊寬度，因此那些位元組裡原本有什麼，就會進入內積。獨立工具配置的是全新（乾淨）的 arena；在 server 內，activation 借用的是 **prefill 暫存區**，裡面仍留著先前請求的資料。

**修正。** 在 prep kernel 中寫入填充（零）。

**現在怎麼抓。** 在任意先前流量之後，server 的嵌入必須與獨立工具的嵌入逐位元相同。通則：當緩衝區是借用或重用的，填充 lane 必須明確寫入，而且測試必須在*髒*緩衝區上執行 — 全新配置會掩蓋這類 bug。

### 4.2 圖片之後 MTP 接受率下降

在圖片提示詞之後於某個 slot 上生成的文字（從圖片的共用檢查點接續），其 MTP 接受率低於全新的 slot。修正填充之後，候選建置量測到的接受率恢復正常（47.2%，與全新 slot 相同）。TODO：確認填充 bug 是否就是全部原因。

### 4.3 不是視覺 bug：F32 alpha/beta

某個量化版本中由 F32 `ssm_alpha` / `ssm_beta` 造成的 MTP 不一致，是另一個模型檔案的問題（[quantization.md](quantization.md#f32-alpha-beta)）；它是在 MXFP4 量化工作中發現的，不是在視覺輸入工作中。

### 4.4 又是 gate 的 bug

視覺 gate 的 MXFP4 參考 server 自動選了 f16 KV（沒掛 mmproj 時可用 VRAM 較多），而受測 server 選了 q8v，所以每一項比較都「失敗」。與 system-prompt gate 的教訓相同：在比較 server 的 gate 中要固定 KV 格式。

## 5. Gate

`vis_gate`（dense 與 MXFP4）：有無 `--mmproj` 時池大小相同；載入 mmproj 時 VRAM 變化 0.0 MiB；圖片提示詞結果等於冷啟動參考；圖片之後的檢查點重用（某個案例為 2,045 個 token）；重新啟動後從 SSD 還原；閒置釋放後 VRAM 精確歸還（差值 0.0 MiB）。在候選建置上整條鏈都通過：文字提示詞的最後一個 token logits 與已安裝建置逐位元相同、兩套 server 測試套件 50/50、分段 prefill、並行、池、多回合、層、system 檢查點與還原 gate、視覺 gate dense + MXFP4。

## 6. 剩餘工作

- 已完成：編碼器、排程器整合與 gate 都已在 C++ 引擎中（`src/vision/`、`kernels/vision/`、`whirl-server-gate --suite vis`）。
- 確認 MTP 接受率下降的原因（§4.2）。
- 已完成（0.2.0）：gfx1151 多段 RoPE kernel 與編碼器 kernel（§7）。
- gfx1151：編碼器速度（§7：attention 與 GEMM 只用到 WMMA 峰值的一小部分）。

## <a id="gfx1151"></a>7. Radeon 8060S（gfx1151）

從 0.2.0 起圖片輸入也能在 8060S 上執行（`--device 8060s`，參數相同）。原本缺的是：gfx1151 kernel 組沒有多段
RoPE attention prep（`attn_prep_m*`、`set_rpos`），所以 server 拒絕 `--mmproj`；編碼器 code object 也只有
gfx1201 版。現在編碼器是同一份原始碼（`kernels/vision/vision.hip`）依架構各建一份。只有兩個 WMMA kernel 不同：
gfx11 每個 lane 持有 A / B fragment 的全部 16 個 k（gfx12 是 8 個），累加器第 v 個元素是第 `2v + lane / 16` 列
（gfx12 是 `v + 8 * (lane / 16)`）；attention 每個 block 用 8 個 wave、每個 wave 16 個 query（每 wave 32 個
query 在 gfx11 會溢出暫存器）。算術不變：f16 輸入、f32 累加，沒有 fp8。R9700 的 code object 反組譯完全相同，
embedding 與改動前逐位元相同。

| 檢查（Qwen3.8-27B mmproj F16） | 結果 |
|---|---|
| `whirl-kernel-test --only vision`（8 個編碼器 kernel 全部對 CPU 參考）、`attn_prep_m*` / `set_rpos` 檢查 | 兩張 GPU 都通過 |
| 8060S embedding 對 f32 numpy 參考，4 張圖 | 相對 L2 0.16–0.30%，最小 cosine ≥ 0.9995（R9700：0.16–0.31%） |
| 8060S 對 R9700 embedding，7 張圖（220–2,040 token） | 相對 L2 0.09–0.23%，最小 cosine ≥ 0.9995 |
| Server，`--precise`，temperature 0，8 個圖片請求（一或兩張圖、英文／中文問題、最大 1080p） | 兩張 GPU 回答文字完全相同；同一張圖再問一次兩邊都完全命中快取 |

8060S 編碼時間（`whirl-vis vis-encode`，權重常駐，第三次執行）：

| 圖片（token） | 8060S | R9700 |
|---|---|---|
| 640x352（220） | 54 ms | 17 ms |
| 512x512（256） | 59 ms | 18 ms |
| 1024x672（672） | 219 ms | 52 ms |
| 1280x736（920） | 314 ms | 78 ms |
| 1024x1024（1,024） | 332 ms | 92 ms |
| 1920x1088（2,040） | 934 ms（attention 532、GEMM 303） | 253 ms |

`whirl-server-gate --suite vis` 在 8060S 與 R9700 上都通過。VRAM 檢查依裝置類別用不同的計數器
（`include/whirl/device_mem.h`，量測見 VIS-8S-2）：統一記憶體上載入 mmproj 時 `hipMemGetInfo` 少了
1,775 MiB，是 887 MiB 鎖頁 mmproj 複本的兩倍，但這些頁面在系統 RAM（GPU「共用使用量」），不在 BIOS
劃給顯示卡的記憶體。所以 8060S 上伺服器回報自己劃分記憶體的變化（GPU「專用使用量」，即工作管理員的數字）：
1.9 MiB，是鎖頁複本的 GPU 分頁表（每 4 KiB 頁 8 位元組）；上限為 2 MiB 加上這部分。閒置釋放後與第一張圖片
之前相差 8.8 MiB 以內：HIP runtime 在第一個開著前綴快取的長圖片提示時自己長一次記憶體（8.3–8.8 MiB，
20 次載入／釋放循環持平，沒有每次循環的洩漏）。整合式 GPU 在這項允許 16 MiB（獨立顯卡 8 MiB，R9700 量到
0.0），兩種都另外檢查再做兩次載入／釋放循環不再增加（至多 2 MiB）。

8060S 上權重串流的成本差不多（1080p 949 ms）：主機記憶體與 GPU 共用同一組 DRAM。編碼器在 8060S 上約
10–17 TFLOPS，只是 WMMA 峰值的一小部分；調校尚未進行（§6）。

#### 移植時找到的 bug：跨 lane 讀到非作用中的 lane

第一版 gfx11 attention 只有 head dim 71 輸出 0。它的收尾在兩個半 wave 之間交換累加值（`permlanex16`），寫在
每個 lane 的 `d < 72` / `q < N` 判斷之前，但編譯器把交換排到判斷之後；dim 72–79 的對應 lane 已不在作用中，
而 `permlanex16` 讀非作用中 lane 會得到 0。修正：收尾改成逐元素寫出，不做跨 lane 步驟。kernel 測試抓到了它
（每個 head 恰好一個 dim 錯）。
