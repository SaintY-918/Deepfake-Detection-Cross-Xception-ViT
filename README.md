# Combining EfficientNet and Vision Transformers for Video Deepfake Detection

本專案實作了一種結合 **CNN (Xception/EfficientNet)** 與 **Vision Transformers (ViT)** 的混合架構，專門用於偵測偽造影片（Deepfake）。專案支援雙分支 Cross-Attention 機制，能有效融合不同尺度的臉部特徵。

---

## 1. 環境設定 (Environment Setup)

本專案建議在 **Windows WSL (Ubuntu 24.04)** 環境下執行，並使用專屬的虛擬環境。

### 虛擬環境路徑
- **Environment**: `/home/saint/torch_env`
- **Python**: 3.12+
