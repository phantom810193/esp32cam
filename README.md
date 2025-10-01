# ESP32-CAM Retail Advertising MVP

新版 MVP 聚焦在「ESP32-CAM 拍照 → 雲端 Flask 後端 → Amazon Rekognition + Gemini Text → SQLite → HDMI 看板」的最小可執行流程。攝影機即時上傳照片，後端呼叫 Amazon Rekognition 取得臉部特徵摘要並產生匿名會員 ID，透過 SQLite 撈取歷史消費資料，再用 Gemini Text 自動生成個人化廣告文案，最後輸出給電視棒（HDMI Dongle）輪播。若未啟用 Amazon Rekognition 或 Gemini，系統仍會以雜湊比對與預設模板提供完整端到端流程。

## 系統架構與技術流程

1. **ESP32-CAM 拍照上傳**：韌體每隔數秒拍照並透過 HTTP POST 將 JPEG 送至雲端 `/upload_face` API。
2. **Amazon Rekognition 雲端辨識**：Flask 後端把影像送進 Amazon Rekognition，取得臉部結構與屬性摘要並產生匿名 `MEMxxxxxxxxxx` 會員 ID。
3. **SQLite 會員 & 消費資料**：後端使用摘要向 SQLite 查詢既有會員與歷史訂單，找不到則建立新會員並寫入歡迎優惠。
4. **Vertex AI 圖文生成**：後端呼叫 Gemini (`gemini-2.5-pro`) 產生標題、副標與 CTA，再把結果交給 Imagen (`imagen-3.0-generate-001`) 生成
   1080x1080 的廣告海報。生成完立即寫入 GCS 並產製簽名網址，全流程以 **10 秒內完成** 為服務水位。
5. **廣告頁輸出**：後端將文案與歷史訂單傳入 Jinja2 模板 `/ad/<member_id>`，網頁每 5 秒自動刷新，適合放在 HDMI 電視棒或任何瀏覽器輪播。

## 專案結構

```
backend/                # Flask 後端：API、Amazon Rekognition / Gemini 介接、SQLite、廣告模板
  ai.py                 # Gemini Text 封裝與錯誤處理
  app.py                # 主要進入點（/upload_face、/ad/<id> 等）
  database.py           # SQLite 存取與 Demo 資料建立
  recognizer.py         # 使用 Amazon Rekognition 產生匿名特徵，含 hash fallback
  advertising.py        # 根據消費紀錄與 AI 文案產出頁面所需內容
  templates/            # Jinja2 模板（首頁 + 廣告頁，含自動刷新）
firmware/esp32cam_mvp/  # ESP32-CAM PlatformIO 專案
  src/main.cpp          # 拍照並 POST 至 Flask 的範例程式
  include/secrets_template.h # Wi-Fi 與伺服器設定範本
  platformio.ini        # PlatformIO 組態（Arduino framework）
```

## 後端（Flask + SQLite + Amazon Rekognition + Gemini）

1. 建議使用虛擬環境安裝依賴：

   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   cd ..  # 回到專案根目錄
   ```

   `requirements.txt` 內包含 `boto3` 與 `google-generativeai`。若無法存取網路或暫時沒有雲端憑證，系統會自動退回到純雜湊比對與固定模板。

2. 設定 Amazon Rekognition 憑證：

   ```bash
   export AWS_ACCESS_KEY_ID="<your_access_key>"
   export AWS_SECRET_ACCESS_KEY="<your_secret_key>"
   export AWS_REGION="ap-northeast-1"  # 或任何適用區域
   export AWS_REKOGNITION_COLLECTION_ID="esp32cam-members"  # 選填，預設會建立同名集合
   ```

   也可以使用 IAM Role、`~/.aws/credentials` 或容器內建的臨時憑證。若未設定，系統會自動退回純雜湊比對。後端會在啟動時自動呼叫
   [`CreateCollection`](https://docs.aws.amazon.com/rekognition/latest/dg/API_CreateCollection.html) 確保集合存在，並使用
   [`IndexFaces`](https://docs.aws.amazon.com/rekognition/latest/dg/API_IndexFaces.html) 將新會員影像訓練進集合，後續比對則透過
   `SearchFacesByImage` 直接找回先前註冊過的 `ExternalImageId`（即會員編號）。

3. 設定 Gemini API Key：

   ```bash
   export GEMINI_API_KEY="<your_api_key>"
   ```

   未設定時後端仍可運作，但不會呼叫 Gemini。部署到雲端（例如 Cloud Run）時，也可改放在環境變數或 Secret Manager。預設會使用
   `gemini-2.5-flash` 作為 Vision 與 Text 模型名稱，可視需求在程式或環境變數中覆寫。

4. 設定 Vertex AI 服務帳號與素材儲存桶：

   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
   export GCP_PROJECT_ID="<your_project_id>"
   export GCP_REGION="asia-east1"
   export ASSET_BUCKET="esp32cam-assets"
   ```

   - 服務帳號至少需要 `Vertex AI User`、`Vertex AI Service Agent`（若為新專案需啟用）與 `Storage Object Admin` 權限，
     才能呼叫 Gemini 文案模型並將 Imagen 圖片寫入指定的 GCS bucket。建議：
     1. 於 Google Cloud Console 啟用 **Vertex AI API** 與 **Cloud Storage**。
     2. 為專案建立服務帳號，例如 `esp32cam-adgen@<project-id>.iam.gserviceaccount.com`。
     3. 將上述角色授與該服務帳號，下載 JSON 金鑰並指定給 `GOOGLE_APPLICATION_CREDENTIALS`。
   - `ASSET_BUCKET` 必須事先建立（名稱可自訂），後端會將生成的海報
     以 `adgen/<SKU>/<date>/<uuid>.png` 命名存放並回傳 12 小時簽名網址。若打算跨專案存取，請記得授權服務帳號至該 bucket。
   - 若環境暫時無法連線至 Vertex AI 或 GCS，API 會退回預設模板與
     `ME0000.jpg` 圖片，仍可驗證整體流程。

   | 變數 | 說明 |
   | --- | --- |
   | `GOOGLE_APPLICATION_CREDENTIALS` | 服務帳號 JSON 金鑰路徑。 |
   | `GCP_PROJECT_ID` | Vertex AI 與 GCS 所屬專案。 |
   | `GCP_REGION` | Vertex AI 區域，預設 `asia-east1`。 |
   | `ASSET_BUCKET` | 用來儲存 Imagen 成果的 GCS bucket。 |
   | `VERTEX_TEXT_MODEL` *(選填)* | 覆寫 Gemini 文案模型，預設 `gemini-2.5-pro`。 |
   | `VERTEX_IMAGE_MODEL` *(選填)* | 覆寫 Imagen 模型，預設 `imagen-3.0-generate-001`。 |
   | `VERTEX_IMAGE_SIZE` *(選填)* | 覆寫輸出尺寸，預設 `1080x1080`。 |

5. 啟動 Flask 伺服器（於專案根目錄執行）：

   ```bash
   flask --app backend.app run --host 0.0.0.0 --port 8000 --debug
   ```

6. API 重點：

    - `POST /upload_face`：接受 `image/jpeg` 或 `multipart/form-data` 影像。回傳 JSON，內含 `member_id`、`member_code`（僅在已有商場註冊代號時帶值）、`new_member` 旗標與廣告頁 URL。
    - `GET /ad/<member_id>`：根據 SQLite + Gemini Text 的輸出生成廣告頁，內建 `<meta http-equiv="refresh" content="5">`，適合放在電視棒上自動輪播。
    - `GET /ad/latest/stream`：透過 Server-Sent Events 形式推送最新的上傳事件與廣告頁連結，前端或電視棒可直接串流更新。若僅需單次查詢，可加上 `?once=1` 取得最新事件後立即結束連線。
    - `GET /latest_upload`：顯示最新上傳影像、辨識結果、個人化廣告連結與各階段耗時分析，方便除錯與現場展示。後端會保留最新一張上傳影像，並為每位會員留存首次辨識的照片，以避免佔用過多空間又能在名單頁回溯影像。
    - `GET /members`：瀏覽預寫會員的個人資料、首次辨識影像與 2025 年消費紀錄，也可從最新上傳儀表板的「會員資料」按鈕快速進入。
    - `POST /adgen`：呼叫 Vertex AI Gemini + Imagen 生成標題、副標、CTA 與海報圖。
      需帶入 `{"sku": "SKU123", "member_profile": {...}}`，成功時會得到
      已上傳至 GCS 的 `image_url`。
    - `GET /health`：基本健康檢查。
    - `GET /healthz`：延伸健康檢查，包含廣告素材目錄的寫入權限與 GCS 連線狀態。
    - `POST /members/merge`：手動修正重複會員。請帶入 `{"source": "MEM001", "target": "MEM002"}`，系統會把來源會員的消費紀錄與 Rekognition 雲端索引合併到目標會員下。

7. 商品目錄與預測邏輯：

   - `backend/catalogue.py` 以品類為單位維護商品清單，並針對不同族群給定固定前綴的商品編號：
     - `DES###`：精緻甜點（Dessert）
     - `FIT###`：運動健身（Fitness）
     - `KID###`：幼兒教養（Kindergarten）
     - `HOM###`：家居生活（Homemaker）
     - `GEN###`：生活選品（General）
   - 每個商品皆附帶基礎查閱率（View Rate）與售價資訊，方便預測模組產生 UI 所需欄位。
   - 透過 `infer_category_from_item()` 將歷史訂單名稱對應回目錄品類，確保跨模組一致性。

8. 商品查閱率與機率計算公式：

   - `backend/prediction.py` 會取出「上一個月」的消費紀錄，若該月份沒有資料則回退至最近一個有紀錄的月份。
   - 先根據商品目錄中的 `view_rate`（0~1）搭配會員上一期的品類權重，計算查閱率：

     ```text
     adjusted_view_rate = min(1.0, base_view_rate * (0.9 + category_weight(category)))
     view_rate_percent = round(adjusted_view_rate * 100, 1)
     ```

   - 再針對所有候選商品計算分數：

     ```text
     score = 0.45 * category_weight
           + 0.25 * recency_bonus
           + 0.20 * price_similarity
           + 0.10 * novelty
     ```

     - `category_weight`：當月各品類的購買比重（頻率正規化）。
     - `recency_bonus`：依據最近 5 筆交易的品類倒序加權（越新的比重越高）。
     - `price_similarity`：商品售價與歷史平均單價的差異，越接近越高。
     - `novelty`：未購買過的新商品加權 1.0，已出現過則降至 0.3，會員身份再額外帶入 0.8 的基礎值。
   - 將上述分數送入 softmax 取得 `probability`，並四捨五入至 0.1% 輸出 `probability_percent`；查閱率亦以 0.1% 顯示在 UI 上。

9. 驗證介面：

   - 管理者頁面：`http://<server-ip>:8000/manager`，預設載入第一位有會員 ID 的顧客，可透過下拉選單切換其他人員。此頁面會呈現：
     1. ESP32-CAM 上傳的顧客影像（若未有首張影像則使用 hero fallback）。
     2. 本月潛在熱銷清單（七筆）、上一期熱銷前三名與上一個月的消費紀錄。
     3. 會員基本資料、職業 / 產業別欄位與參與活動紀錄。
   - 顧客端廣告頁：`GET /ad/<member_id>?v2=1` 可直接預覽新版樣式；若需要除錯，可加上 `&debug=1` 觀察 hero 圖來源與情境代碼。
   - 若要重跑推薦邏輯，可從 VM 目錄中挑任一張測試照片呼叫 `/upload_face`，後台會自動辨識身份、寫入上一個月資料並更新上述兩個畫面。

10. 服務啟動時會先刪除並重建 Amazon Rekognition 人臉集合，確保所有雲端特徵從零開始訓練；同時 SQLite 也會重設
   `member_profiles` 中預留的五筆示範資料，讓 `member_id` 欄位保持空白。辨識到新臉孔時，系統會先以空集合比對，
   若找不到則建立匿名 `MEMxxxxxxxxxx` 會員、寫入歡迎禮並將影像訓練進集合，然後由程式自動把最新 5 位新會員
   依序填入預留的示範資料列。示範資料涵蓋甜點收藏家、幼兒園家長、健身族、家庭採買者與健康養生客層，其中
   前 3 位已在商場完成註冊（`member_profiles` 會帶上 ME0001~ME0003），後 2 位則維持匿名狀態。每當新會員佔據
   其中一筆預留資料，系統會立即灌入對應的 2025 年消費紀錄並延續後續流程；待五筆都完成綁定後，新的臉孔就會
   依照一般規則自動生成新的系統編號與歷史資料。

11. 修正重複會員：若同一張臉誤判成不同會員，可呼叫 `POST /members/merge`，將重複的會員 ID 合併。API 會：

   1. 將來源會員 (`source`) 的所有消費紀錄轉移到目標會員 (`target`)。
   2. 從 SQLite 刪除來源會員。
   3. 透過 Amazon Rekognition 刪除來源會員的 `ExternalImageId`，避免後續再命中舊編號。
   4. 如果來源的雲端特徵較完整（或設定 `"prefer_source_encoding": true`），會自動覆蓋目標會員的特徵值，確保未來比對以正確編號返回。

12. 手動測試（使用任何 JPEG）：

   ```bash
   curl -X POST \
        -H "Content-Type: image/jpeg" \
        --data-binary @sample.jpg \
        http://localhost:8000/upload_face
   ```

   回傳範例：

   ```json
   {
     "status": "ok",
     "member_id": "MEM6A9C2A41F2",
     "member_code": "",
     "new_member": false,
     "ad_url": "http://localhost:8000/ad/MEM6A9C2A41F2"
   }
   ```

   測試 Vertex AI 文生圖流程（需先設定步驟 4 的環境變數）：

   ```bash
   curl -X POST \
        -H "Content-Type: application/json" \
        -d '{"sku": "GEN001", "member_profile": {"segment": "registered:dessert", "name": "ME0001"}}' \
        http://localhost:8000/adgen
   ```

   成功時會收到：

   ```json
   {
     "title": "GEN001 限時禮遇",
     "subline": "...",
     "cta": "立即體驗",
     "image_url": "https://storage.googleapis.com/..."
   }
   ```

   若 Vertex AI 或 GCS 暫時無法連線，API 會返回 `503` 及預設模板，
   `image_url` 會指向 `/static/images/ads/ME0000.jpg`，方便持續驗證前後台 UI。

### 風險與失敗 fallback

- **Vertex AI 模型初始化失敗**：回傳 `503` 並改用固定文案與 `ME0000.jpg`。
- **GCS 無法寫入**：記錄錯誤並顯示 fallback 圖片，後台可透過 `/healthz` 檢查 bucket 狀態。
- **服務帳號權限不足**：請確認 `GOOGLE_APPLICATION_CREDENTIALS` 指向的帳號擁有 Vertex AI 及 Storage 權限。
- **超過 10 秒 SLA**：後端會在日誌輸出警告，可藉此調整 prompt、模型或網路設定。

### Vertex AI 生成流程與效能追蹤

1. **取得推薦上下文**：`POST /adgen` 接收 `sku` 與可選 `member_profile`，由推薦模組提供七筆 SKU 與對應機率。
2. **Gemini 文案生成**：呼叫 `gemini-2.5-pro`，回傳 `title`、`subline`、`cta`。失敗時退回模板字串。
3. **Imagen 海報繪製**：以文案與客群特徵組合 prompt，使用 `imagen-3.0-generate-001` 產出 1080x1080 PNG。
4. **GCS 儲存**：檔案上傳至 `gs://<ASSET_BUCKET>/adgen/<SKU>/<date>/<uuid>.png`，並回傳 12 小時簽名網址。
5. **效能紀錄**：後端會在日誌中輸出 copy / image / upload 各階段耗時，若超過 10 秒會額外顯示警告方便排查。

上述流程遇到任一階段失敗即回傳 `503` 及 `ME0000.jpg`，確保前台仍有素材可播。

## 前端展示（電視棒 / 螢幕）

- 在電視棒或任何瀏覽器打開 `http://<server-ip>:8000/ad/<member_id>` 即可。
- 網頁每 5 秒刷新一次，Gemini Text 產出的主標、副標、促購亮點會即時更新。若 AI 功能未啟用，則會顯示預設模板。

## ESP32-CAM（PlatformIO 範例）

1. 安裝 [PlatformIO](https://platformio.org/)（VS Code 擴充或 CLI）。
2. 複製 Wi-Fi / 後端設定：

   ```bash
   cp firmware/esp32cam_mvp/include/secrets_template.h firmware/esp32cam_mvp/include/secrets.h
   ```

   編輯 `secrets.h`，填入 Wi-Fi SSID/Password 以及雲端 Flask `/upload_face` 的 HTTPS URL。

3. 使用 USB-TTL 連線並上傳：

   ```bash
   cd firmware/esp32cam_mvp
   pio run --target upload
   pio device monitor  # 查看序列埠輸出
   ```

4. 裝置會每 5 秒拍照並透過 HTTP POST 上傳。成功時序列埠會顯示伺服器回覆的 JSON 以及廣告頁連結，方便在雲端驗證。

## Demo 建議流程

1. 啟動雲端或本地的 Flask 伺服器並設定 Amazon Rekognition 憑證與 `GEMINI_API_KEY`。
2. 將 HDMI 電視棒固定在 `/ad/<member_id>` 頁面，或使用瀏覽器展示。
3. ESP32-CAM 開機 → 拍照上傳 → 後端呼叫 Amazon Rekognition → 查詢 SQLite → Gemini Text 生成廣告 → 頁面自動刷新，整體流程預期 < 10 秒。
4. 錄製 < 1 分鐘 Demo 影片展示「上傳 → 廣告更新」的完整閉環。

## 後續擴充想法（Nice-to-Have）

- 可接軌 YOLO 行為追蹤、多人排隊分析，或串接雲端資料倉儲。
- 替換 SQLite 為雲端資料庫（Cloud SQL / Firestore），並把 Flask 佈署到 Cloud Run、App Engine 或 Kubernetes。
- 擴充 Gemini 提示，讓 AI 文案依照天氣、時段或會員屬性動態調整促銷內容。

祝開發順利！
