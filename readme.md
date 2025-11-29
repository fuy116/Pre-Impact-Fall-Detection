# 基於深度學習之主動式預跌倒偵測系統
>**Note**： 本 Repository 僅專注於 **AI Pipeline實作**，包含資料前處理、深度學習模型訓練以及針對邊緣裝置的量化工程。


![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red) ![ESP32](https://img.shields.io/badge/Hardware-ESP32--S3-green) 
## 專案簡介 
本專案旨在開發一套輕量化的跌倒預測系統，能夠部署於資源受限的邊緣裝置（如 ESP32-S3）。系統利用 **CNN-Dense** 深度學習模型處理 6 軸感測器數據（加速度計 + 陀螺儀），實現 **「跌倒前預判 (Pre-impact Prediction)」**，而非傳統的跌倒後偵測。

本系統平均可在跌倒發生前發出預警，能夠與防護氣囊等主動防護硬體結合，有效降低跌倒傷害。

### 核心特點
* **Dataset**：使用公開數據集 **KFall Dataset** 進行模型訓練與驗證，確保數據的多樣性與準確性。
* **演算法**：整合 Kalman Filter 訊號降噪與 RobustScaler 抗異常值處理。
* **模型量化**：支援 ESP-DL 量化流程，可輸出給 ESP32-S3 使用的 `.espdl` 模型檔。
* **安全機制**：訓練策略優先考量 **Recall (召回率)**，寧可誤報也不可漏報（False Negative is unacceptable）。

---


## 專案架構 

```
preimpact_fall_prediction/
├── src/
│   ├── preprocessing/          # 資料預處理模組
│   │   ├── 01_merge_label.py   # 合併感測器資料與標籤
│   │   ├── 02_kalman_6axis.py  # Kalman 濾波去噪（6軸）
│   │   ├── 03_features_6axis.py# 特徵計算 (SVM, Tilt) 與分割
│   │   ├── 04_window_6axis.py  # 滑動視窗 (Sliding Window) 處理
│   │   ├── 05_normalize_6axis.py # 資料正規化 (RobustScaler)
│   ├── model/                  # 模型訓練模組
│   │   └── cnn_dense.py        # CNN-Dense 模型定義與訓練
│   └── quantize/               # 模型量化
│       └── quantize.py         # 使用 esp_ppq 進行量化與效能評估
 ```   
    