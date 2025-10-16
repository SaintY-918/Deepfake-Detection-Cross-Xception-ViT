
import torch
from torch import nn
from einops import rearrange, repeat
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

# --- MultiScaleEncoder (從原作者模型中引入) ---
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

# --- 主模型架構 ---
class CrossXceptionViTCrossAttn(nn.Module):
    def __init__(self, config, pretrained=True):
        super().__init__()
        model_conf = config['model']
        sm_conf = model_conf['s-branch']
        lg_conf = model_conf['l-branch']
        cross_attn_conf = model_conf['cross-attention']
        dropout = model_conf.get('dropout', 0.)
        
        self.cnn_s = timm.create_model('xception', pretrained=pretrained, features_only=True, out_indices=(4,))
        self.cnn_l = timm.create_model('xception', pretrained=pretrained, features_only=True, out_indices=(2,))

        self.projection_s = nn.Linear(sm_conf['cnn_channels'], sm_conf['dim'])
        self.projection_l = nn.Linear(lg_conf['cnn_channels'], lg_conf['dim'])

        self.cls_token_s = nn.Parameter(torch.randn(1, 1, sm_conf['dim']))
        self.cls_token_l = nn.Parameter(torch.randn(1, 1, lg_conf['dim']))
        
        self.multi_scale_encoder = MultiScaleEncoder(
            depth=model_conf['depth'],
            sm_dim=sm_conf['dim'],
            lg_dim=lg_conf['dim'],
            sm_enc_params=dict(depth=sm_conf['depth'], heads=sm_conf['heads'], dim_head=sm_conf['dim_head'], mlp_dim=sm_conf['mlp_dim']),
            lg_enc_params=dict(depth=lg_conf['depth'], heads=lg_conf['heads'], dim_head=lg_conf['dim_head'], mlp_dim=lg_conf['mlp_dim']),
            cross_attn_params=dict(depth=cross_attn_conf['depth'], heads=cross_attn_conf['heads'], dim_head=cross_attn_conf['dim_head']),
            dropout=dropout
        )

        self.mlp_head_s = nn.Sequential(nn.LayerNorm(sm_conf['dim']), nn.Linear(sm_conf['dim'], model_conf['num-classes']))
        self.mlp_head_l = nn.Sequential(nn.LayerNorm(lg_conf['dim']), nn.Linear(lg_conf['dim'], model_conf['num-classes']))

    def forward(self, img):
        # 初始 Tokenization
        feat_s = self.cnn_s(img)[0]
        tokens_s = rearrange(feat_s, 'b c h w -> b (h w) c')
        tokens_s = self.projection_s(tokens_s)
        b = tokens_s.shape[0]
        cls_s = repeat(self.cls_token_s, '() n d -> b n d', b=b)
        tokens_s = torch.cat((cls_s, tokens_s), dim=1)

        feat_l = self.cnn_l(img)[0]
        tokens_l = rearrange(feat_l, 'b c h w -> b (h w) c')
        tokens_l = self.projection_l(tokens_l)
        b = tokens_l.shape[0]
        cls_l = repeat(self.cls_token_l, '() n d -> b n d', b=b)
        tokens_l = torch.cat((cls_l, tokens_l), dim=1)

        # 進入 Multi-Scale Encoder 進行多輪融合
        sm_tokens, lg_tokens = self.multi_scale_encoder(tokens_s, tokens_l)

        # MLP Heads
        sm_cls, lg_cls = sm_tokens[:, 0], lg_tokens[:, 0]
        sm_logits = self.mlp_head_s(sm_cls)
        lg_logits = self.mlp_head_l(lg_cls)
        
        return sm_logits + lg_logits