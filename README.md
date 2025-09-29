# ESP32-CAM Retail Advertising MVP

新版 MVP 聚焦在「ESP32-CAM 拍照 → 雲端 Flask 後端 → Amazon Rekognition + Gemini Text → SQLite → HDMI 看板」的最小可執行流程。攝影機即時上傳照片，後端呼叫 Amazon Rekognition 取得臉部特徵摘要並產生匿名會員 ID，透過 SQLite 撈取歷史消費資料，再用 Gemini Text 自動生成個人化廣告文案，最後輸出給電視棒（HDMI Dongle）輪播。若未啟用 Amazon Rekognition 或 Gemini，系統仍會以雜湊比對與預設模板提供完整端到端流程。

## 系統架構與技術流程

1. **ESP32-CAM 拍照上傳**：韌體每隔數秒拍照並透過 HTTP POST 將 JPEG 送至雲端 `/upload_face` API。
2. **Amazon Rekognition 雲端辨識**：Flask 後端把影像送進 Amazon Rekognition，取得臉部結構與屬性摘要並產生匿名 `MEMxxxxxxxxxx` 會員 ID。
3. **SQLite 會員 & 消費資料**：後端使用摘要向 SQLite 查詢既有會員與歷史訂單，找不到則建立新會員並寫入歡迎優惠。
4. **Gemini Text 生成廣告**：把會員歷史紀錄整理成 JSON，交給 Gemini Text 回傳包含主標、副標、促購亮點的廣告文案。
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

4. 啟動 Flask 伺服器（於專案根目錄執行）：

   ```bash
   flask --app backend.app run --host 0.0.0.0 --port 8000 --debug
   ```

5. API 重點：

    - `POST /upload_face`：接受 `image/jpeg` 或 `multipart/form-data` 影像。回傳 JSON，內含 `member_id`、`member_code`（僅在已有商場註冊代號時帶值）、`new_member` 旗標與廣告頁 URL。
    - `GET /ad/<member_id>`：根據 SQLite + Gemini Text 的輸出生成廣告頁，適合在除錯或需要鎖定特定會員時手動檢視。
    - `GET /ad/latest`：即時廣告看板。使用 Server-Sent Events (SSE) 監聽最新辨識結果，電視棒只要固定開啟這個網址就會自動切換成剛辨識完成的會員廣告，不需重新整理頁面。
    - `GET /latest_upload`：顯示最新上傳影像、辨識結果、個人化廣告連結與各階段耗時分析，方便除錯與現場展示。後端會保留最新一張上傳影像，並為每位會員留存首次辨識的照片，以避免佔用過多空間又能在名單頁回溯影像。
    - `GET /members`：瀏覽預寫會員的個人資料、首次辨識影像與 2025 年消費紀錄，也可從最新上傳儀表板的「會員資料」按鈕快速進入。
    - `GET /health`：基本健康檢查。
    - `POST /members/merge`：手動修正重複會員。請帶入 `{"source": "MEM001", "target": "MEM002"}`，系統會把來源會員的消費紀錄與 Rekognition 雲端索引合併到目標會員下。

6. 服務啟動時會先刪除並重建 Amazon Rekognition 人臉集合，確保所有雲端特徵從零開始訓練；同時 SQLite 也會重設
   `member_profiles` 中預留的五筆示範資料，讓 `member_id` 欄位保持空白。辨識到新臉孔時，系統會先以空集合比對，
   若找不到則建立匿名 `MEMxxxxxxxxxx` 會員、寫入歡迎禮並將影像訓練進集合，然後由程式自動把最新 5 位新會員
   依序填入預留的示範資料列。示範資料涵蓋甜點收藏家、幼兒園家長、健身族、家庭採買者與健康養生客層，其中
   前 3 位已在商場完成註冊（`member_profiles` 會帶上 ME0001~ME0003），後 2 位則維持匿名狀態。每當新會員佔據
   其中一筆預留資料，系統會立即灌入對應的 2025 年消費紀錄並延續後續流程；待五筆都完成綁定後，新的臉孔就會
   依照一般規則自動生成新的系統編號與歷史資料。

7. 修正重複會員：若同一張臉誤判成不同會員，可呼叫 `POST /members/merge`，將重複的會員 ID 合併。API 會：

   1. 將來源會員 (`source`) 的所有消費紀錄轉移到目標會員 (`target`)。
   2. 從 SQLite 刪除來源會員。
   3. 透過 Amazon Rekognition 刪除來源會員的 `ExternalImageId`，避免後續再命中舊編號。
   4. 如果來源的雲端特徵較完整（或設定 `"prefer_source_encoding": true`），會自動覆蓋目標會員的特徵值，確保未來比對以正確編號返回。

8. 手動測試（使用任何 JPEG）：

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

## 前端展示（電視棒 / 螢幕）

- 建議在電視棒或任何瀏覽器打開 `http://<server-ip>:8000/ad/latest`，透過 SSE 自動切換成最新辨識出的會員廣告。
- 若需固定觀看指定會員，仍可開啟 `http://<server-ip>:8000/ad/<member_id>` 進行除錯。此頁會保留原本的輪播刷新行為。

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
2. 將 HDMI 電視棒固定在 `/ad/latest` 頁面，或使用瀏覽器展示，即可即時看到最新辨識的會員廣告。
3. ESP32-CAM 開機 → 拍照上傳 → 後端呼叫 Amazon Rekognition → 查詢 SQLite → Gemini Text 生成廣告 → SSE 推播至 `/ad/latest` 頁面，整體流程預期 < 10 秒。
4. 錄製 < 1 分鐘 Demo 影片展示「上傳 → 廣告更新」的完整閉環。

## 後續擴充想法（Nice-to-Have）

- 可接軌 YOLO 行為追蹤、多人排隊分析，或串接雲端資料倉儲。
- 替換 SQLite 為雲端資料庫（Cloud SQL / Firestore），並把 Flask 佈署到 Cloud Run、App Engine 或 Kubernetes。
- 擴充 Gemini 提示，讓 AI 文案依照天氣、時段或會員屬性動態調整促銷內容。

祝開發順利！
