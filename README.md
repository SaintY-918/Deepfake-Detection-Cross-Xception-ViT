# Combining EfficientNet and Vision Transformers for Video Deepfake Detection

本專案實作了一種結合 **CNN (Xception/EfficientNet)** 與 **Vision Transformers (ViT)** 的混合架構，專門用於偵測偽造影片（Deepfake）。專案支援雙分支 Cross-Attention 機制，能有效融合不同尺度的臉部特徵。

---

## 1. 環境設定 (Environment Setup)

本專案建議在 **Windows WSL (Ubuntu 24.04)** 環境下執行，並使用專屬的虛擬環境。

### 核心套件版本：

* torch==2.2.2
* torchvision==0.17.2
* timm==0.6.5 (預訓練權重來源)
* einops==0.8.1 (Transformer 張量操作)
* albumentations==1.3.0 (影像增強)
* facenet-pytorch==2.6.0 (MTCNN 臉部偵測)

## 2. 資料集來源 (Dataset Sources)

| 資料集名稱 | 描述 | 來源連結 |
| :--- | :--- | :--- |
| **FaceForensics++ (FF++)** | 包含 1000 份原始影片及其使用四種不同技術（Deepfakes, Face2Face, FaceSwap, NeuralTextures）生成的偽造版本。 | [GitHub Repo](https://github.com/ondyari/FaceForensics) |
| **Celeb-DF (v2)** | 包含從 YouTube 提取的真實與合成的明星臉部影片，具有較高的視覺品質。 | [Official Site](https://github.com/yuezunli/celeb-deepfakeforensics) |
| **Deepfake Detection Challenge (DFDC)** | 由 Facebook 發起的大規模資料集，包含多樣化的背景與光照條件。 | [Kaggle DFDC](https://www.kaggle.com/c/deepfake-detection-challenge) |

> **注意**：預處理腳本 `preprocessing/detect_faces.py` 中的 `--dataset` 參數預設支援 `FACEFORENSICS` 格式。

---

