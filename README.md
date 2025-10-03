# ESP32-CAM Retail Advertising MVP

新版 MVP 聚焦在「ESP32-CAM 拍照 → 雲端 Flask 後端 → Amazon Rekognition + Gemini Text → SQLite → HDMI 看板」的最小可執行流程。攝影機即時上傳照片，後端呼叫 Amazon Rekognition 取得臉部特徵摘要並產生匿名會員 ID，透過 SQLite 撈取歷史消費資料，再用 Gemini Text 自動生成個人化廣告文案，最後輸出給電視棒（HDMI Dongle）輪播。若未啟用 Amazon Rekognition 或 Gemini，系統仍會以雜湊比對與預設模板提供完整端到端流程。

## 系統架構與技術流程

1. **ESP32-CAM 拍照上傳**：韌體每隔數秒將 JPEG 透過 HTTP POST 傳至 `/upload_face`。
2. **Amazon Rekognition / FaceEncoder**：後端偵測人臉並回推匿名 `MEMxxxx`，同時累積首次來訪時間。
3. **會員狀態判定 + SQLite**：依資料庫紀錄區分「首次來訪」「尚未入會」「已是會員」，並抓取當年度的上一個月份消費紀錄。
4. **`predict_next_purchases` 商品預測**：以前一個月份的交易與目錄標籤推算「本月最可能購買但尚未購買」的商品及機率。
5. **Gemini 文案產生**：將預測結果與會員狀態送入 Google Gemini `gemini-2.5-flash-lite` 模型，產出 headline / subheading / highlight / CTA，尚未入會者強調加入誘因，會員則突出專屬折扣。
6. **模板套用（ME0000~ME0003）**：依預測品類映射到甜點、幼兒、健身模板；首次來訪固定顯示 ME0000。最終前端以背景圖 + 文字疊圖輸出，並透過 SSE 即時推播。

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
   `gemini-2.5-flash-lite` 文案模型，可視需求在程式或環境變數中覆寫。

4. 啟動 Flask 伺服器（於專案根目錄執行）：

   ```bash
   flask --app backend.app run --host 0.0.0.0 --port 8000 --debug
   ```

5. API 重點：

    - `POST /upload_face`：接受 `image/jpeg` 或 `multipart/form-data` 影像。回傳 JSON，內含 `member_id`、`member_code`（僅在已有商場註冊代號時帶值）、`new_member` 旗標與廣告頁 URL。
    - `GET /ad/<member_id>`：根據 SQLite + Gemini Text 的輸出生成廣告頁，適合在除錯或需要鎖定特定會員時手動檢視。
    - `GET /dashboard?member_id=MEMxxxx`：管理人員專用的會員儀表板。可輸入 `member_id` 查詢特定會員，也會自動回退至最新上傳事件或預設示範會員。頁面會顯示會員基本資料、聯絡資訊、點數餘額、Persona 標籤及購買歷史，並在有上傳首張照片時自動載入對應人像。
    - `GET /ad/latest`：即時廣告看板。使用 Server-Sent Events (SSE) 監聽最新辨識結果，電視棒只要固定開啟這個網址就會自動切換成剛辨識完成的會員廣告，不需重新整理頁面。
    - `GET /latest_upload`：顯示最新上傳影像、辨識結果、個人化廣告連結與各階段耗時分析，方便除錯與現場展示。後端會保留最新一張上傳影像，並為每位會員留存首次辨識的照片，以避免佔用過多空間又能在名單頁回溯影像。
    - `GET /members`：瀏覽預寫會員的個人資料、首次辨識影像與 2025 年消費紀錄，也可從最新上傳儀表板的「會員資料」按鈕快速進入。
    - `POST /adgen`：呼叫 Google Gemini 生成標題、副標、CTA 與亮點文案，
      並依據 `template_id`、`category` 或會員標籤自動指派 `ME0001~ME0003`
      背景圖檔，回傳 `image_url` 供前端直接套用靜態素材。
    - `POST /members/merge`?????????????? `/upload_face` ??????????? 0.32?0.40 ???????? API ???????????????????????? `{"source": "MEM001", "target": "MEM002"}`??????????????? Rekognition ?????????????

6. 商品目錄與預測邏輯：

   - `backend/catalogue.py` 以品類為單位維護商品清單，並針對不同族群給定固定前綴的商品編號：
     - `DES###`：精緻甜點（Dessert）
     - `FIT###`：運動健身（Fitness）
     - `KID###`：幼兒教養（Kindergarten）
     - `HOM###`：家居生活（Homemaker）
     - `GEN###`：生活選品（General）
   - 每個商品皆附帶基礎查閱率（View Rate）與售價資訊，方便預測模組產生 UI 所需欄位。
   - 透過 `infer_category_from_item()` 將歷史訂單名稱對應回目錄品類，確保跨模組一致性。

7. 機率計算公式：

   - `backend/prediction.py` 會取出「上一個月」的消費紀錄，若該月份沒有資料則回退至最近一個有紀錄的月份。
   - 針對所有候選商品計算分數：

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
   - 將上述分數送入 softmax 取得 `probability`，並四捨五入至 0.1% 輸出 `probability_percent`；查閱率亦以 0.1% 顯示。

8. 驗證介面：

   - 管理者頁面：`http://<server-ip>:8000/manager`，預設載入第一位有會員 ID 的顧客，可透過下拉選單切換其他人員。此頁面會呈現：
     1. ESP32-CAM 上傳的顧客影像（若未有首張影像則使用 hero fallback）。
     2. 本月潛在熱銷清單（七筆）、上一期熱銷前三名與上一個月的消費紀錄。
     3. 會員基本資料、職業 / 產業別欄位與參與活動紀錄。
   - 顧客端廣告頁：`GET /ad/<member_id>?v2=1` 可直接預覽新版樣式；若需要除錯，可加上 `&debug=1` 觀察 hero 圖來源與情境代碼。
   - 若要重跑推薦邏輯，可從 VM 目錄中挑任一張測試照片呼叫 `/upload_face`，後台會自動辨識身份、寫入上一個月資料並更新上述兩個畫面。

9. 服務啟動時會先刪除並重建 Amazon Rekognition 人臉集合，確保所有雲端特徵從零開始訓練；同時 SQLite 也會重設
   `member_profiles` 中預留的五筆示範資料，讓 `member_id` 欄位保持空白。辨識到新臉孔時，系統會先以空集合比對，
   若找不到則建立匿名 `MEMxxxxxxxxxx` 會員、寫入歡迎禮並將影像訓練進集合，然後由程式自動把最新 5 位新會員
   依序填入預留的示範資料列。示範資料涵蓋甜點收藏家、幼兒園家長、健身族、家庭採買者與健康養生客層，其中
   前 3 位已在商場完成註冊（`member_profiles` 會帶上 ME0001~ME0003），後 2 位則維持匿名狀態。每當新會員佔據
   其中一筆預留資料，系統會立即灌入對應的 2025 年消費紀錄並延續後續流程；待五筆都完成綁定後，新的臉孔就會
   依照一般規則自動生成新的系統編號與歷史資料。

10. ???????????????????? 0.32?0.40 ?????????????????????????? `POST /members/merge`??????? ID ???API ??

   1. 將來源會員 (`source`) 的所有消費紀錄轉移到目標會員 (`target`)。
   2. 從 SQLite 刪除來源會員。
   3. 透過 Amazon Rekognition 刪除來源會員的 `ExternalImageId`，避免後續再命中舊編號。
   4. 如果來源的雲端特徵較完整（或設定 `"prefer_source_encoding": true`），會自動覆蓋目標會員的特徵值，確保未來比對以正確編號返回。

11. 手動測試（使用任何 JPEG）：

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

   測試 Gemini 文案生成流程（需先設定 `GEMINI_API_KEY`）：

   ```bash
   curl -X POST \
        -H "Content-Type: application/json" \
        -d '{"sku": "GEN001", "member_profile": {"segment": "registered:dessert", "name": "ME0001"}, "template_id": "ME0001"}' \
        http://localhost:8000/adgen
   ```

   成功時會收到：

   ```json
   {
    "headline": "GEN001 限時禮遇",
    "subheading": "...",
    "highlight": "...",
    "cta": "立即體驗",
    "template_id": "ME0001",
    "image_url": "http://localhost:8000/static/images/ads/ME0001.jpg"
  }
  ```

   若 Gemini 服務暫時不可用，API 會返回 `503` 並以預設文案搭配靜態
   圖片回傳，`image_url` 仍會指向 `/static/images/ads/ME000x.jpg` 方便
   持續驗證前後台 UI。

## 前端展示（電視棒 / 螢幕）

- 建議在電視棒或任何瀏覽器打開 `http://<server-ip>:8000/ad/latest`，透過 SSE 自動切換成最新辨識出的會員廣告。
- 若需固定觀看指定會員，仍可開啟 `http://<server-ip>:8000/ad/<member_id>` 進行除錯。此頁會保留原本的輪播刷新行為。
- 管理人員可前往 `http://<server-ip>:8000/dashboard` 檢視會員儀表板，亦可透過查詢參數 `?member_id=MEMxxxx` 指定會員。此頁整合會員基本資料、聯絡方式、點數餘額與購買紀錄，方便現場客服或營運團隊快速掌握狀態。
- Demo 情境可直接拜訪 `http://<server-ip>:8000/demo/upload-ad`，上傳人像後立即取得會員辨識結果與同步產生的 Gemini 廣告文案，並可在頁面內預覽固定底圖 + 文案的輸出效果。

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

## 功能驗證步驟

1. **啟動後端服務**：依照前述方式啟動 Flask（需設定 AWS 與 Gemini/Vertex AI 的環境變數）。開啟 `http://<server>:8000/ad/latest` 作為看板預覽。
2. **上傳人臉影像**：使用 ESP32-CAM 或以 `curl` 模擬 `POST /upload_face`。後端會顯示 JSON，並於日誌紀錄會員身分判定與預測結果。
3. **確認會員狀態分類**：在伺服器日誌中可看到 audience `new` / `guest` / `member`。跨查 SQLite 可驗證首次到訪是否寫入歡迎禮。
4. **檢查商品預測**：`predict_next_purchases` 會輸出機率最高的商品，SSE payload 的 `predicted` 欄位與管理者儀表板會同步顯示該商品。
5. **驗證 Gemini 文案**：若環境已設定 Vertex AI，payload 的 `headline/subheading/highlight/cta_text` 會換成即時生成內容；若失敗則會落回預設模板並在日誌出現 warning。
6. **比對模板背景**：依 `template_id`（ME0000~ME0003）檢查前端是否切換對應底圖。新客應固定呈現 ME0000，其餘依預測品類映射。
7. **前端看板輸出**：`/ad/<member_id>` 與 `/ad/latest` 皆會顯示背景 + 文案，並於 SSE 更新時計入新的 hero URL 與 CTA。

驗證一次完整流程後，即可確認「偵測人臉 → 會員分類 → 商品預測 → Gemini 文案 → 模板套用 → 前端輸出」皆正常運作。
