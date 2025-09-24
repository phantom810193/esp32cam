# ESP32-CAM Retail Advertising MVP

新版 MVP 聚焦在「ESP32-CAM 拍照 → 雲端 Flask 後端 → Azure Face API → SQLite → HDMI 看板」的最小可執行流程。攝影機即時上傳照片，後端呼叫 Azure Face 取得臉部特徵摘要並產生匿名會員 ID，透過 SQLite 撈取歷史消費資料，再以簡易模板輸出客製化廣告文案。若未啟用 Azure Face，系統仍會以雜湊比對與預設模板提供完整端到端流程。

## 系統架構與技術流程

1. **ESP32-CAM 拍照上傳**：韌體每隔數秒拍照並透過 HTTP POST 將 JPEG 送至雲端 `/upload_face` API。
2. **Azure Face 雲端辨識**：Flask 後端把影像送進 Azure Face API，取得臉部特徵摘要後會先透過 Find Similar API 與我們維護的 Face List 做比對；若命中，就直接重用既有會員 ID。若未找到或帳戶已獲得 Person Group 權限，系統也會同步更新 Person Group，讓後續還是能沿用 Azure 的 Person ID。
3. **SQLite 會員 & 消費資料**：後端使用摘要向 SQLite 查詢既有會員與歷史訂單，找不到則建立新會員並寫入歡迎優惠。
4. **模板產生廣告**：把會員歷史紀錄整理成 JSON，套用內建模板產出主標、副標、促購亮點的廣告文案，確保即使未啟用雲端 AI 也能維持完整流程。
5. **廣告頁輸出**：後端將文案與歷史訂單傳入 Jinja2 模板 `/ad/<member_id>`，網頁每 5 秒自動刷新，適合放在 HDMI 電視棒或任何瀏覽器輪播。

## 專案結構

```
backend/                # Flask 後端：API、Azure Face 介接、SQLite、廣告模板
  ai.py                 # Azure Face 封裝與錯誤處理（含廣告模板）
  app.py                # 主要進入點（/upload_face、/ad/<id> 等）
  database.py           # SQLite 存取與 Demo 資料建立
  recognizer.py         # 使用 Azure Face 產生匿名特徵，含 hash fallback
  advertising.py        # 根據消費紀錄與 AI 文案產出頁面所需內容
  templates/            # Jinja2 模板（首頁 + 廣告頁，含自動刷新）
firmware/esp32cam_mvp/  # ESP32-CAM PlatformIO 專案
  src/main.cpp          # 拍照並 POST 至 Flask 的範例程式
  include/secrets_template.h # Wi-Fi 與伺服器設定範本
  platformio.ini        # PlatformIO 組態（Arduino framework）
```

## 後端（Flask + SQLite + Azure Face）

1. 建議使用虛擬環境安裝依賴：

   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   cd ..  # 回到專案根目錄
   ```

   `requirements.txt` 內包含最新版的 `azure-ai-vision-face` SDK。若暫時沒有金鑰，後端會自動退回到純雜湊比對與固定模板。

2. 設定 Azure Face 認證：

   ```bash
   export AZURE_FACE_ENDPOINT="https://<your-resource>.cognitiveservices.azure.com/"
   export AZURE_FACE_KEY="<your_face_api_key>"
   export AZURE_FACE_PERSON_GROUP_ID="esp32cam-mvp"    # 可自訂，會自動建立（選填）
   export AZURE_FACE_LIST_ID="esp32cam-mvp-faces"      # Find Similar 使用的 Face List（可省略使用預設值）
   ```

   未設定時後端仍可運作，但只會使用雜湊簽名與預設廣告模板。部署到雲端（例如 Cloud Run）時，也可改放在環境變數或 Secret Manager。

   > ⚠️ **Person Group 權限申請**：Azure Face 的 Person Group / Identify API 需額外核准
   > 「Identification/Verification」功能才能啟用。若帳號尚未被核准，後端會自動停用
   > 相關功能並僅保留臉部描述與 Face List 比對；頁面會提示前往
   > <https://aka.ms/facerecognition> 申請預覽權限。

3. 啟動 Flask 伺服器（於專案根目錄執行）：

   ```bash
   flask --app backend.app run --host 0.0.0.0 --port 8000 --debug
   ```

4. API 重點：

   - `POST /upload_face`：接受 `image/jpeg` 或 `multipart/form-data` 影像。回傳 JSON，內含 `member_id`、`new_member` 旗標與廣告頁 URL。
   - `GET /ad/<member_id>`：根據 SQLite + 模板化文案輸出生成廣告頁，內建 `<meta http-equiv="refresh" content="5">`，適合放在電視棒上自動輪播。
   - `GET /health`：基本健康檢查。
   - `GET /person-group`：提供瀏覽器表單，可輸入會員 ID 並上傳多張臉部照片，預先將該會員訓練至 Azure Face Person Group。
   - `GET /face-list`：若帳戶暫未獲得 Person Group 核准，可改用此頁面批次上傳多張照片並綁定同一個會員 ID，供 Face List / Find Similar 使用。

5. SQLite 會自動建立資料庫與 Demo 資料。辨識到新臉孔時，系統會以 Azure Face 摘要雜湊生成匿名 `MEMxxxxxxxxxx` 並寫入歡迎禮優惠，同時把臉部向量註冊到 Azure Face List（及可用時的 Person Group），以便下一次造訪可直接透過 Find Similar 命中既有會員。

6. 手動測試（使用任何 JPEG）：

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
     "new_member": false,
     "ad_url": "http://localhost:8000/ad/MEM6A9C2A41F2"
   }
   ```

7. 需要先為特定顧客補上多張照片時，可視 Azure 權限選擇以下工具：

   - `http://localhost:8000/person-group`：輸入會員 ID 後上傳多張影像，系統會在 Azure 中建立/更新 Person，並觸發訓練；同時會把 Face List 與（可用時）Person ID 存回 SQLite。
   - `http://localhost:8000/face-list`：若帳戶尚未取得 Person Group 核准，仍可透過此頁面把多張影像加入 Face List，確保 Find Similar 可以命中既有會員，並把 persisted face ID 註冊回資料庫。
   - 兩個頁面都會在資料庫找不到該會員時自動建立基礎資料，方便日後辨識。

## 前端展示（電視棒 / 螢幕）

- 在電視棒或任何瀏覽器打開 `http://<server-ip>:8000/ad/<member_id>` 即可。
- 網頁每 5 秒刷新一次，若啟用 Azure Face 會根據臉部描述帶入專屬文案；未啟用時則顯示預設模板。

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

1. 啟動雲端或本地的 Flask 伺服器並設定 `AZURE_FACE_ENDPOINT` / `AZURE_FACE_KEY`。
2. 將 HDMI 電視棒固定在 `/ad/<member_id>` 頁面，或使用瀏覽器展示。
3. ESP32-CAM 開機 → 拍照上傳 → 後端呼叫 Azure Face → 查詢 SQLite → 套用模板生成廣告 → 頁面自動刷新，整體流程預期 < 10 秒。
4. 錄製 < 1 分鐘 Demo 影片展示「上傳 → 廣告更新」的完整閉環。

## 後續擴充想法（Nice-to-Have）

- 可接軌 YOLO 行為追蹤、多人排隊分析，或串接雲端資料倉儲。
- 替換 SQLite 為雲端資料庫（Cloud SQL / Firestore），並把 Flask 佈署到 Cloud Run、App Engine 或 Kubernetes。
- 若需更智慧的文案，可再串接語言模型或其他雲端 AI 服務，依天氣、時段或會員屬性動態調整促銷內容。

祝開發順利！
