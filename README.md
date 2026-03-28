# Combining EfficientNet and Vision Transformers for Video Deepfake Detection

本專案實作了一種結合 **CNN (Xception/EfficientNet)** 與 **Vision Transformers (ViT)** 的混合架構，專門用於偵測偽造影片（Deepfake）。專案支援雙分支 Cross-Attention 機制，能有效融合不同尺度的臉部特徵。

---

## 環境設定 (Environment Setup)

本專案建議在 **Windows WSL (Ubuntu 24.04)** 環境下執行，並使用專屬的虛擬環境。

### 核心套件版本：

* torch==2.2.2
* torchvision==0.17.2
* timm==0.6.5 
* einops==0.8.1 
* albumentations==1.3.0 
* facenet-pytorch==2.6.0 

## 資料集來源與處理 (Dataset & Preprocessing)

本專案主要使用 **FaceForensics++ (FF++)** 資料集進行實驗與驗證。我們針對資料集進行了嚴謹的劃分與欠取樣（Under-sampling）策略：

*   **五種偽造技術**：包含 DeepFakes, Face2Face, FaceSwap, NeuralTextures 以及高保真度的 FaceShifter。
*   **資料集分割**：依據標準劃分，包含 720 支訓練影片、140 支驗證影片以及 140 支測試影片。
*   **真偽平衡 (1:1)**：為解決資料不平衡問題，訓練集從真實與偽造影片中均勻取樣，最終確保訓練、驗證與測試皆維持 1:1 的真偽影像比例（訓練集共 43,200 張影像）。

### 資料增強策略 (Data Augmentation)

為了提升模型在不同影片品質下的泛化能力，使用 `albumentations` 實作了自定義的增強流程：

![fig3_2](https://github.com/user-attachments/assets/2a302a29-7840-43ef-b2c6-324d81349f07)

---
## 模型架構 (Architecture)

本專案核心採用 **Cross Xception ViT** 混合架構，旨在結合 **Xception (CNN)** 強大的局部特徵提取能力與 **Vision Transformer (ViT)** 卓越的全域關係建模能力。

### 1. 多尺度雙分支設計 (Dual-Branch Design)
為了同時捕捉不同層級的偽造痕跡，模型實作了兩條獨立的特徵處理路徑：

*   **S-Branch (Small-scale)**: 接收 **10x10** 的特徵圖。此分支專注於提取高解析度的**局部細節**，例如皮膚紋理的不自然破綻或微小的拼接痕跡。
*   **L-Branch (Large-scale)**: 接收 **38x38** 的特徵圖。由於具備更廣的感受野，此分支專注於分析**全域幾何結構**，例如臉部光影的一致性或跨區域的物理矛盾。



### 2. 跨尺度特徵融合 (Cross-Attention Fusion)
提取出的特徵圖會被切分為多個 Patch 並傳入 Transformer Encoder。本專案引入了 **Cross-Attention** 機制，允許兩分支交換彼此的尺度資訊，其核心注意力運算如下：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

透過此機制，模型能強化對隱蔽偽造痕跡的捕捉能力，確保在面對高擬真的 Deepfake 影像時，依然能識別出細微的局部破綻與宏觀的邏輯錯誤。

---
## 模型原理簡介

本專案利用 Xception 提取空間特徵，並將特徵圖切分為多個 Patch 傳入 Transformer Encoder。在 Cross Xception ViT 中，我們引入了 Cross-Attention 機制用於融合不同層級的 CNN 特徵，強化模型對局部細微偽造痕跡的捕捉能力。

## 模型架構細節 (Architecture Details)

本專案實作的 `CrossXceptionViT` 採用雙分支設計，分別處理不同尺度的特徵流：

*   **S-Branch (Small-scale)**: 處理 10x10 的特徵圖，專注於提取高解析度的局部細節（如皮膚紋理破綻）。
*   **L-Branch (Large-scale)**: 處理 38x38 的特徵圖，擁有更廣的感受野，專注於分析全域幾何結構的不一致性。
*   **Feature Fusion**: 透過 Cross-Attention 機制交換兩分支的資訊，強化模型對隱蔽偽造痕跡的捕捉。

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
