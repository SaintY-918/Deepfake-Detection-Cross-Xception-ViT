# cross_xception_vit_dual.py (Dropout 修正版)

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import VisionTransformer
import timm

class CrossXceptionViTDual(nn.Module):
    def __init__(self, config, pretrained=True):
        super().__init__()
        vit_conf = config['model']['vit']

        self.cnn = timm.create_model('xception', pretrained=pretrained, features_only=True, out_indices=(0, 4))
        
        mid_channels = 64
        deep_channels = 2048
        
        self.pool = nn.AdaptiveAvgPool2d((10, 10))

        self.projection = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(mid_channels + deep_channels, vit_conf['dim']), 
            nn.LayerNorm(vit_conf['dim'])
        )

        self.transformer = VisionTransformer(
            embed_dim=vit_conf['dim'],
            depth=vit_conf['depth'],
            num_heads=vit_conf['heads'],
            mlp_ratio=vit_conf['mlp-dim'] / vit_conf['dim'],
            qkv_bias=True
        )
        self.transformer.head = nn.Identity()

        # === 核心修正：將 Dropout 層加回到分類頭中 ===
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(vit_conf['dim']),
            nn.Dropout(p=0.3), # p=0.3  Dropout 比例
            nn.Linear(vit_conf['dim'], config['model']['num-classes'])
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, vit_conf['dim']))

    def forward(self, img):
        cnn_features = self.cnn(img)
        features_mid = cnn_features[0]
        features_deep = cnn_features[1]

        pooled_mid = self.pool(features_mid)
        fused_features = torch.cat((pooled_mid, features_deep), dim=1)
        tokens = self.projection(fused_features)

        b, n, _ = tokens.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, tokens), dim=1)
        
        x = self.transformer.blocks(x)
        
        x = x[:, 0]
        return self.mlp_head(x)