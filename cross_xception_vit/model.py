
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import timm

# --- Helper Classes ---

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

class CrossTransformer(nn.Module):
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.Linear(lg_dim, sm_dim) if lg_dim != sm_dim else nn.Identity(),
                nn.Linear(sm_dim, lg_dim) if sm_dim != lg_dim else nn.Identity(),
                PreNorm(sm_dim, Attention(sm_dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(lg_dim, Attention(lg_dim, heads=heads, dim_head=dim_head, dropout=dropout)),
            ]))

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        for proj_lg_to_sm, proj_sm_to_lg, sm_attend_lg, lg_attend_sm in self.layers:
            lg_context = proj_lg_to_sm(lg_patch_tokens)
            sm_cls = sm_attend_lg(sm_cls, context=lg_context) + sm_cls
            sm_context = proj_sm_to_lg(sm_patch_tokens)
            lg_cls = lg_attend_sm(lg_cls, context=sm_context) + lg_cls
        
        return torch.cat((sm_cls, sm_patch_tokens), dim=1), torch.cat((lg_cls, lg_patch_tokens), dim=1)

class MultiScaleEncoder(nn.Module):
    def __init__(self, sm_dim, lg_dim, depth, sm_enc_params, lg_enc_params, cross_attn_params, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim=sm_dim, dropout=dropout, **sm_enc_params),
                Transformer(dim=lg_dim, dropout=dropout, **lg_enc_params),
                CrossTransformer(sm_dim=sm_dim, lg_dim=lg_dim, dropout=dropout, **cross_attn_params)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens = sm_enc(sm_tokens)
            lg_tokens = lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)
        return sm_tokens, lg_tokens

# 主模型架構 
class CrossXceptionViTCrossAttn(nn.Module):
    def __init__(self, config, pretrained=True):
        super().__init__()
        model_conf = config['model']
        sm_conf = model_conf['s-branch']
        lg_conf = model_conf['l-branch']
        cross_attn_conf = model_conf['cross-attention']
        dropout = model_conf.get('dropout', 0.)
        
        # 1. CNN 主幹網路
        self.cnn_s = timm.create_model('xception', pretrained=pretrained, features_only=True, out_indices=(4,))
        self.cnn_l = timm.create_model('xception', pretrained=pretrained, features_only=True, out_indices=(2,))

        # 2. 自適應池化層 (確保尺寸可以被整除)
        s_patch_size = sm_conf['patch-size']
        l_patch_size = lg_conf['patch-size']
        
        # S-Branch 目標尺寸設為 10x10，因為 10 可以被 patch_size=2 整除
        self.pool_s = nn.AdaptiveAvgPool2d((10, 10))
        # L-Branch 目標尺寸設為 36x36，因為 36 可以被 patch_size=4 整除
        self.pool_l = nn.AdaptiveAvgPool2d((36, 36))

        # 3. 二次分塊 + 投影層
        s_patch_dim = sm_conf['cnn_channels'] * (s_patch_size ** 2)
        self.to_patch_embedding_s = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=s_patch_size, p2=s_patch_size),
            nn.Linear(s_patch_dim, sm_conf['dim']),
            nn.LayerNorm(sm_conf['dim'])
        )

        l_patch_dim = lg_conf['cnn_channels'] * (l_patch_size ** 2)
        self.to_patch_embedding_l = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=l_patch_size, p2=l_patch_size),
            nn.Linear(l_patch_dim, lg_conf['dim']),
            nn.LayerNorm(lg_conf['dim'])
        )
        
        # 4. CLS Token 和位置編碼
        self.cls_token_s = nn.Parameter(torch.randn(1, 1, sm_conf['dim']))
        self.cls_token_l = nn.Parameter(torch.randn(1, 1, lg_conf['dim']))

        s_num_patches = (10 // s_patch_size) ** 2
        l_num_patches = (36 // l_patch_size) ** 2
        self.pos_embedding_s = nn.Parameter(torch.randn(1, s_num_patches + 1, sm_conf['dim']))
        self.pos_embedding_l = nn.Parameter(torch.randn(1, l_num_patches + 1, lg_conf['dim']))
        
        # 5. Multi-Scale Encoder
        self.multi_scale_encoder = MultiScaleEncoder(
            depth=model_conf['depth'],
            sm_dim=sm_conf['dim'],
            lg_dim=lg_conf['dim'],
            sm_enc_params=dict(depth=sm_conf['depth'], heads=sm_conf['heads'], dim_head=sm_conf['dim_head'], mlp_dim=sm_conf['mlp_dim']),
            lg_enc_params=dict(depth=lg_conf['depth'], heads=lg_conf['heads'], dim_head=lg_conf['dim_head'], mlp_dim=lg_conf['mlp_dim']),
            cross_attn_params=dict(depth=cross_attn_conf['depth'], heads=cross_attn_conf['heads'], dim_head=cross_attn_conf['dim_head']),
            dropout=dropout
        )

        # 6. MLP Heads 分類頭
        self.mlp_head_s = nn.Sequential(nn.LayerNorm(sm_conf['dim']), nn.Linear(sm_conf['dim'], model_conf['num-classes']))
        self.mlp_head_l = nn.Sequential(nn.LayerNorm(lg_conf['dim']), nn.Linear(lg_conf['dim'], model_conf['num-classes']))

    def forward(self, img):
        # 提取 CNN 特徵
        feat_s_raw = self.cnn_s(img)[0]
        feat_l_raw = self.cnn_l(img)[0]

        # 使用池化層確保尺寸
        feat_s = self.pool_s(feat_s_raw)
        feat_l = self.pool_l(feat_l_raw)

        # 代幣化
        tokens_s = self.to_patch_embedding_s(feat_s)
        tokens_l = self.to_patch_embedding_l(feat_l)
        b = tokens_s.shape[0]

        # 附加 CLS Token
        cls_s = repeat(self.cls_token_s, '() n d -> b n d', b=b)
        tokens_s = torch.cat((cls_s, tokens_s), dim=1)
        
        cls_l = repeat(self.cls_token_l, '() n d -> b n d', b=b)
        tokens_l = torch.cat((cls_l, tokens_l), dim=1)

        # 附加位置編碼
        tokens_s += self.pos_embedding_s
        tokens_l += self.pos_embedding_l

        # 進入 Multi-Scale Encoder
        sm_tokens, lg_tokens = self.multi_scale_encoder(tokens_s, tokens_l)

        # 提取融合後的 CLS Token
        sm_cls, lg_cls = sm_tokens[:, 0], lg_tokens[:, 0]
        sm_logits = self.mlp_head_s(sm_cls)
        lg_logits = self.mlp_head_l(lg_cls)
        
        # 將兩個分支的結果相加
        return sm_logits + lg_logits