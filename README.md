# ESP32-CAM Retail Advertising MVP

這個專案實作了「ESP32-CAM → Flask → SQLite → 即時廣告頁面」的最小可行產品。硬體端的 ESP32-CAM 會定期拍照並透過 Wi-Fi 上傳 JPEG 影像到後端；Flask 伺服器使用 dlib/face_recognition 進行臉部特徵比對、查詢 SQLite 歷史消費資料，最後輸出可直接在電視棒（HDMI Dongle）播放的廣告頁。

## 專案結構

```
backend/                # Flask 後端：API、SQLite、廣告模板
  app.py                # 主要進入點（/upload_face、/ad/<id> 等）
  database.py           # SQLite 存取與 Demo 資料建立
  recognizer.py         # dlib/face_recognition 封裝，提供 hash fallback
  advertising.py        # 根據消費紀錄產生廣告文案
  templates/            # Jinja2 模板（首頁 + 廣告頁，含自動刷新）
firmware/esp32cam_mvp/  # ESP32-CAM PlatformIO 專案
  src/main.cpp          # 拍照並 POST 至 Flask 的範例程式
  include/secrets_template.h # Wi-Fi 與伺服器設定範本
  platformio.ini        # PlatformIO 組態（Arduino framework）
```

## 後端（Flask + SQLite + dlib）

1. 建議先建立虛擬環境並安裝依賴：

   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   cd ..  # 返回專案根目錄
   ```

   > `face_recognition` 套件需要系統層級的 dlib 依賴，若只想快速測試流程，可先略過安裝。系統會自動退回到「影像雜湊比對」模式，仍可串接完整流程。

2. 啟動 Flask 伺服器（於專案根目錄執行）：

   ```bash
   flask --app backend.app run --host 0.0.0.0 --port 8000 --debug
   ```

3. API 重點：

   - `POST /upload_face`：接受 `image/jpeg` 或 `multipart/form-data` 影像。回傳 JSON，內含 `member_id` 與對應的廣告頁 URL。
   - `GET /ad/<member_id>`：根據 SQLite 內容產生廣告頁，內建 `<meta http-equiv="refresh" content="5">`，適合放在電視棒上自動輪播。
   - `GET /health`：基本健康檢查。

4. SQLite 會自動建立資料庫與 Demo 資料。若辨識到新臉孔，系統會分配新的 `MEMxxx` ID 並寫入歡迎禮優惠。

5. 手動測試（使用任何 JPEG）：

   ```bash
   curl -X POST \
        -H "Content-Type: image/jpeg" \
        --data-binary @sample.jpg \
        http://localhost:8000/upload_face
   ```

## 前端展示（電視棒 / 螢幕）

- 在電視棒或瀏覽器開啟 `http://<server-ip>:8000/ad/MEM001` 等頁面即可。
- 網頁會每 5 秒自動刷新，當新的臉孔上傳成功後，自動切換到對應會員的廣告內容。

## ESP32-CAM（PlatformIO 範例）

1. 安裝 [PlatformIO](https://platformio.org/)（VS Code 擴充或 CLI）。
2. 複製 Wi-Fi / 後端設定：

   ```bash
   cp firmware/esp32cam_mvp/include/secrets_template.h firmware/esp32cam_mvp/include/secrets.h
   ```

   編輯 `secrets.h`，填入 Wi-Fi SSID/Password 以及 Flask 伺服器的 `/upload_face` URL。

3. 使用 USB-TTL 連線並上傳：

   ```bash
   cd firmware/esp32cam_mvp
   pio run --target upload
   pio device monitor  # 查看序列埠輸出
   ```

4. 裝置會每 5 秒拍照並透過 HTTP POST 上傳。成功時序列埠會顯示伺服器回覆的 JSON。

## Demo 建議流程

1. 啟動後端 Flask 伺服器與 SQLite。
2. 將電視棒固定在 `/ad/<member_id>` 頁面。
3. ESP32-CAM 開機 → 拍照 → 上傳 → 後端辨識 → 廣告頁面刷新，整體流程預期 < 10 秒。
4. 錄製 < 1 分鐘 Demo 影片展示完整流程即可。

## 後續擴充想法（Nice-to-Have）

- YOLO 行為追蹤、AI 自動生成廣告文案、Streamlit Dashboard 皆可以在現有結構上逐步擴充。
- 若要改為雲端部署，只需將 Flask API 與 SQLite 換成雲端服務（例如 Cloud Run + Cloud SQL）。

祝開發順利！
