import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import timm

# --- 輔助類別 (從原檔案保留) ---

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        q_input = x
        context = x if context is None else context
        
        q = self.to_q(q_input)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# --- 主模型架構 (重寫為單分支) ---

class XceptionViT(nn.Module):
    def __init__(self, config, pretrained=True):
        super().__init__()
        model_conf = config['model']
        vit_conf = model_conf['transformer']
        dropout = model_conf.get('dropout', 0.)
        
        # 1. CNN 主幹網路 (使用 Xception，只取最後一層特徵)
        # out_indices=(4,) 對應 2048 個 channels
        self.cnn = timm.create_model('xception', pretrained=pretrained, features_only=True, out_indices=(4,))

        # 2. 自適應池化層 (將 CNN 特徵圖縮放到固定尺寸，例如 10x10)
        pool_size = vit_conf['pool_size']
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        # 3. 分塊 + 投影層 (Patch Embedding)
        patch_size = vit_conf['patch_size']
        patch_dim = vit_conf['cnn_out_channels'] * (patch_size ** 2)
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, vit_conf['dim']),
            nn.LayerNorm(vit_conf['dim'])
        )

        # 4. CLS Token 和位置編碼
        self.cls_token = nn.Parameter(torch.randn(1, 1, vit_conf['dim']))
        
        num_patches = (pool_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, vit_conf['dim']))
        
        # 5. Transformer 編碼器
        self.transformer = Transformer(
            dim=vit_conf['dim'],
            depth=vit_conf['depth'],
            heads=vit_conf['heads'],
            dim_head=vit_conf['dim_head'],
            mlp_dim=vit_conf['mlp_dim'],
            dropout=dropout
        )

        # 6. MLP Head 分類頭
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(vit_conf['dim']),
            nn.Linear(vit_conf['dim'], model_conf['num-classes'])
        )

    def forward(self, img):
        # 1. 提取 CNN 特徵
        # [0] 是因為 features_only=True 且 out_indices=(4,)
        feat_raw = self.cnn(img)[0] 
        
        # 2. 池化
        feat = self.pool(feat_raw) # Shape: (b, 2048, 10, 10)

        # 3. 代幣化 (Patch Embedding)
        tokens = self.to_patch_embedding(feat) # Shape: (b, 25, dim) (因為 10x10, patch=2 -> (10/2)*(10/2)=25)
        b = tokens.shape[0]

        # 4. 附加 CLS Token
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        tokens = torch.cat((cls_tokens, tokens), dim=1) # Shape: (b, 26, dim)

        # 5. 附加位置編碼
        tokens += self.pos_embedding

        # 6. 進入 Transformer
        x = self.transformer(tokens)

        # 7. 提取 CLS Token 進行分類
        cls_output = x[:, 0]
        logits = self.mlp_head(cls_output)
        
        return logits