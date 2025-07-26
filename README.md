**多特效影片自動化處理與原音合成專案 (Python + OpenCV + MoviePy)**

## 專案簡介          

本專案融合 OpenCV 與 MoviePy，實現批次自動將多種特效套用於影片，涵蓋畫面分割、鏡像、色彩映射、SIFT 特徵、臉部馬賽克等多項影像處理，並支援將處理後的特效影片與原始音訊自動合併。專案程式碼模組化、易於維護與擴展，展現數位創意與工程實踐、跨部門溝通設計之實力。

## 功能特色          

- **多種影像特效批次處理**：旋轉縮放、邊緣檢測、4分割畫面鏡像、16格 color map、DoG、Gamma 校正、SIFT 關鍵點、臉部辨識馬賽克等。
- **可擴充 Effects Timeline 機制**：每項特效均可自定指定時間區段、高度模組化設計。
- **畫中畫（PiP）、多視窗對比**：主畫面可疊加小畫面/分割效果，利於展示原始與處理效果即時比較。
- **即時進度與特效標註**：自動疊加「特效名稱」、「目前幀/總幀」、「FPS」等訊息，適合流程追蹤與自動化運維。
- **音訊自動合併**：用 MoviePy 將 OpenCV 處理後無聲影片與原始音訊流合併，生成完整成果。
- **程式結構嚴謹，易於維護、擴展與批次自動化改造**          

```markdown          
```
VIDEO-EFFECTS_OPENCV/
├── .venv/                         # Python 虛擬環境資料夾，用於管理專案依賴
├── .gitignore                     # Git 忽略文件，指定不應被版本控制追蹤的檔案和資料夾
├── effects_video_processing.py    # 處理影片特效的主要腳本
├── finalize_video_with_audio.py   # 用於將處理後的影片與音訊合併的腳本
├── gudetama_effects_output.mp4    # 經過特效處理後輸出的範例影片
├── gudetama_effects_with_audio.mp4# 結合了音訊的範例輸出影片
├── requirements.txt               # 列出專案所需的 Python 函式庫和它們的版本
└── ぐでたまテーマソングMV(English subtitled).mp4 # 原始輸入影片（範例），用於應用特效
```
```

## 使用說明

### 依賴安裝

```bash
pip install -r requirements.txt
```
（主要依賴 opencv-python、moviepy、numpy）

### 1. 執行批次特效生成

```bash
python gudetama_opencv_effects.py
```
- 自訂 input/output 路徑與特效順序，依註解調整細節。

### 2. 合併原始音訊

```bash
python finalize_video_with_audio.py
```
- 用 MoviePy 自動合併原音與特效影片，輸出新 MP4。

### 3. 檔案排除管理

- `.gitignore` 已排除所有大型原始與產出影音檔案，確保 repo 精簡。


## 專案畫面範例（可擷圖輔助）

- 特效影片進度/名稱標註
- 多畫面分割/鏡像效果
- PiP 畫中畫同步展示
- 原音合成後成果預覽

## 授權聲明

為尊重著作權，本專案請勿上傳或追蹤任何版權音樂/影片檔案，實作與測試時請使用自有或無版權素材。
