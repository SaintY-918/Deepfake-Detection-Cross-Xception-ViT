# transforms/albu.py (拼字修正版)

import numpy as np
import cv2

from albumentations.core.transforms_interface import DualTransform
# 移除未使用的 import
# from albumentations.augmentations.functional import crop 
from albumentations.augmentations import functional as F


class IsotropicResize(DualTransform):
    
    # === 核心修正：將 cv2.INTER_ĀREA 改為 cv2.INTER_AREA ===
    def __init__(self, target_size, interpolation=cv2.INTER_AREA, always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.target_size = target_size
        self.interpolation = interpolation

    def apply(self, img, interpolation=cv2.INTER_AREA, **params):
        h, w, c = img.shape
        
        if w > h:
            scale = self.target_size / w
            new_w = self.target_size
            new_h = int(h * scale)
            
            img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

            pad_h = self.target_size - new_h
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            
            img = cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, 0)

        else:
            scale = self.target_size / h
            new_h = self.target_size
            new_w = int(w * scale)
            
            img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
            
            pad_w = self.target_size - new_w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            img = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, 0)
        
        return img

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def get_transform_init_args_names(self):
        return ("target_size", "interpolation")