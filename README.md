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

## 資料集來源與處理 (Dataset & Preprocessing)

本專案主要使用 **FaceForensics++ (FF++)** 資料集進行實驗與驗證。我們針對資料集進行了嚴謹的劃分與欠取樣（Under-sampling）策略：

*   **五種偽造技術**：包含 DeepFakes, Face2Face, FaceSwap, NeuralTextures 以及高保真度的 FaceShifter。
*   **資料集分割**：依據標準劃分，包含 720 支訓練影片、140 支驗證影片以及 140 支測試影片。
*   **真偽平衡 (1:1)**：為解決資料不平衡問題，訓練集從真實與偽造影片中均勻取樣，最終確保訓練、驗證與測試皆維持 1:1 的真偽影像比例（訓練集共 43,200 張影像）。

### 資料增強策略 (Data Augmentation)

為了提升模型在不同影片品質下的泛化能力，使用 `albumentations` 實作了自定義的增強流程：

![fig3_2](https://github.com/user-attachments/assets/2a302a29-7840-43ef-b2c6-324d81349f07)

---
## 模型原理簡介

本專案利用 Xception 提取空間特徵，並將特徵圖切分為多個 Patch 傳入 Transformer Encoder。在 cross_xception_vit 中，我們引入了 Cross-Attention 機制用於融合不同層級的 CNN 特徵，強化模型對局部細微偽造痕跡的捕捉能力。

---
## Pipeline

### 資料預處理 (Preprocessing)
*   **臉部偵測與裁剪**：
  使用 MTCNN 偵測影片中的人臉影格並儲存為圖片。
  
  ```bash
  python preprocessing/detect_faces.py --data_path ./raw_videos --dataset FACEFORENSICS
  python preprocessing/extract_crops.py --data_path ./raw_videos --output_path ./face_crops
  ```

*   **資料集分配**：
  執行 organize_dataset.py 將圖片依照 splits/ 中的 JSON 標籤分配到 train/val/test 資料夾。

### 模型訓練 (Training)

*   **Cross-Attention 版本** :

  ```bash
  cd cross_xception_vit
  python train.py --config architecture.yaml
  ```

### 效能評估 (Testing)

評估模型在測試集上的 AUC 與 LogLoss。

```bash
python test.py --config architecture.yaml --dataset All
```

---

## 參考資料

*   **資料集 (Dataset)**: [FaceForensics++: Learning to Detect Manipulated Facial Images](https://github.com/ondyari/FaceForensics)
*   **原始研究 (Base Project)**: [Combining EfficientNet and Vision Transformers for Video Deepfake Detection](https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection)
