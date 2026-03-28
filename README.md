# Combining EfficientNet and Vision Transformers for Video Deepfake Detection

本專案實作了一種結合 **CNN (Xception/EfficientNet)** 與 **Vision Transformers (ViT)** 的混合架構，專門用於偵測偽造影片（Deepfake）。專案支援雙分支 Cross-Attention 機制，能有效融合不同尺度的臉部特徵。

---

## 環境設定 (Environment Setup)

本專案建議在 **Windows WSL (Ubuntu 24.04)** 環境下執行，並使用專屬的虛擬環境。

### 核心套件版本：

* torch==2.2.2
* torchvision==0.17.2
* timm==0.6.5 (預訓練權重來源)
* einops==0.8.1 (Transformer 張量操作)
* albumentations==1.3.0 (影像增強)
* facenet-pytorch==2.6.0 (MTCNN 臉部偵測)

## 資料集來源 (Dataset Sources)

| 資料集名稱 | 描述 | 來源連結 |
| :--- | :--- | :--- |
| **FaceForensics++ (FF++)** | 包含 1000 份原始影片及其使用四種不同技術（Deepfakes, Face2Face, FaceSwap, NeuralTextures）生成的偽造版本。 | [GitHub Repo](https://github.com/ondyari/FaceForensics) |
| **Celeb-DF (v2)** | 包含從 YouTube 提取的真實與合成的明星臉部影片，具有較高的視覺品質。 | [Official Site](https://github.com/yuezunli/celeb-deepfakeforensics) |
| **Deepfake Detection Challenge (DFDC)** | 由 Facebook 發起的大規模資料集，包含多樣化的背景與光照條件。 | [Kaggle DFDC](https://www.kaggle.com/c/deepfake-detection-challenge) |

> **注意**：預處理腳本 `preprocessing/detect_faces.py` 中的 `--dataset` 參數預設支援 `FACEFORENSICS` 格式。

### 資料增強策略 (Data Augmentation)

為了提升模型在不同影片品質下的泛化能力，使用 `albumentations` 實作了自定義的增強流程：

![fig3_2](https://github.com/user-attachments/assets/2a302a29-7840-43ef-b2c6-324d81349f07)

| 技術名稱 | 實作功能 | 目的 |
| :--- | :--- | :--- |
| **IsotropicResize** | 等比例縮放至 299x299 | 維持臉部幾何特徵不變形 |
| **RandomBrightnessContrast** | 隨機亮度與對比度調整 | 模擬不同的攝影環境光影 |
| **ImageCompression** | 隨機 JPEG/WebP 壓縮 | 模擬社交媒體傳輸後的畫質損失 |
| **HorizontalFlip** | 水平翻轉 | 增加樣本多樣性，防止方位依賴 |
| **Normalize** | ImageNet 標準化 | 加速模型收斂 |

---
## 模型原理簡介

本專案利用 Xception 提取空間特徵，並將特徵圖切分為多個 Patch 傳入 Transformer Encoder。在 cross_xception_vit 中，我們引入了 Cross-Attention 機制：$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$用於融合不同層級的 CNN 特徵，強化模型對局部細微偽造痕跡的捕捉能力。

---
## Pipeline
### 資料預處理 (Preprocessing)
臉部偵測與裁剪：
使用 MTCNN 偵測影片中的人臉影格並儲存為圖片。

```bash
python preprocessing/detect_faces.py --data_path ./raw_videos --dataset FACEFORENSICS
python preprocessing/extract_crops.py --data_path ./raw_videos --output_path ./face_crops
```
資料集分配：
執行 organize_dataset.py 將圖片依照 splits/ 中的 JSON 標籤分配到 train/val/test 資料夾。

### 模型訓練 (Training)
本專案提供兩種模型架構，定義於各目錄的 model.py 中。

Cross-Attention 版本 :

```bash
cd cross_xception_vit
python train.py --config architecture.yaml
```
### 效能評估 (Testing)
評估模型在測試集上的 AUC 與 LogLoss。

```bash
python test.py --config architecture.yaml --dataset All
```
