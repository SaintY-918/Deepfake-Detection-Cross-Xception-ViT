

import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 從 transforms.albu 導入 IsotropicResize
from transforms.albu import IsotropicResize

def get_train_transforms(image_size):
    """
    定義訓練時使用的資料增強流程。
    """
    return A.Compose([
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
        A.GaussNoise(p=0.3),
        A.HorizontalFlip(p=0.5),

        A.OneOf([
            IsotropicResize(target_size=image_size, interpolation=cv2.INTER_AREA),
            IsotropicResize(target_size=image_size, interpolation=cv2.INTER_LINEAR),
        ], p=1),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),

        A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(), A.HueSaturationValue()], p=0.4),
        A.ToGray(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),

        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms(image_size):
    """
    定義驗證時使用的轉換流程。
    """
    return A.Compose([
        
        IsotropicResize(target_size=image_size, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

class DeepFakesDataset(Dataset):
    def __init__(self, data_paths_with_labels, image_size, mode='train'):
        self.data = data_paths_with_labels
        self.mode = mode
        if self.mode == 'train':
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)
        # print(f"--- Dataset in '{self.mode}' mode initialized with {len(self.data)} samples. ---")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：無法讀取影像 {img_path}，將跳過此樣本。")
            return None, None
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        transformed = self.transform(image=img)
        img_tensor = transformed['image']
        
        return img_tensor, float(label)

def collate_fn_filter_none(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)