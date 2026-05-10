import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=4, channels=[32, 64, 64],
                 kernels=[8, 4, 3], strides=[4, 2, 1]):
        super().__init__()
        layers = []
        ch_in = in_channels
        for ch_out, k, s in zip(channels, kernels, strides):
            layers.append(nn.Conv2d(ch_in, ch_out, k, stride=s))
            layers.append(nn.ReLU())
            ch_in = ch_out
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            out = self.conv(dummy)
            self.output_size = out.view(1, -1).size(1)
            self.spatial_shape = out.shape[2:]
            self.num_channels = out.shape[1]

    def forward(self, x):
        return self.conv(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.gate = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        x = x + self.gate * self.attn(self.norm1(x))
        x = x + self.gate * self.mlp(self.norm2(x))
        return x


class HybridCNNTransformerEncoder(nn.Module):
    def __init__(self, in_channels=4, channels=[32, 64, 64],
                 kernels=[8, 4, 3], strides=[4, 2, 1],
                 embed_dim=64, num_heads=4, num_layers=2,
                 mlp_ratio=2.0, dropout=0.0, use_cls_token=True):
        super().__init__()
        self.cnn = CNNEncoder(in_channels, channels, kernels, strides)
        cnn_channels = self.cnn.num_channels
        spatial_h, spatial_w = self.cnn.spatial_shape
        self.num_patches = spatial_h * spatial_w
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        self.patch_proj = nn.Linear(cnn_channels, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + (1 if use_cls_token else 0), embed_dim) * 0.02
        )
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
              for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        if use_cls_token:
            self.output_size = embed_dim
        else:
            self.output_size = embed_dim

    def forward(self, x):
        feat = self.cnn(x)
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.patch_proj(tokens)
        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        if self.use_cls_token:
            out = tokens[:, 0]
        else:
            out = tokens.mean(dim=1)
        return out


class AttentionAugmentedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 num_heads=4, dk=None, dv=None):
        super().__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dk = dk or out_channels // 2
        self.dv = dv or out_channels // 2
        self.conv_out = out_channels - self.dv
        assert self.conv_out > 0
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, self.conv_out, kernel_size,
                              stride=stride, padding=padding)
        self.qkv_conv = nn.Conv2d(in_channels, 2 * self.dk + self.dv, 1)
        self.attn_out = nn.Conv2d(self.dv, self.dv, 1)
        self.stride = stride

    def forward(self, x):
        conv_out = self.conv(x)
        B, _, H_in, W_in = x.shape
        qkv = self.qkv_conv(x)
        if self.stride > 1:
            qkv = F.avg_pool2d(qkv, self.stride)
        B, _, H, W = qkv.shape
        q, k, v = torch.split(qkv, [self.dk, self.dk, self.dv], dim=1)
        q = q.reshape(B, self.num_heads, self.dk // self.num_heads, H * W)
        k = k.reshape(B, self.num_heads, self.dk // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, self.dv // self.num_heads, H * W)
        scale = (self.dk // self.num_heads) ** -0.5
        attn = torch.matmul(q.transpose(-2, -1), k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(v, attn)
        out = out.reshape(B, self.dv, H, W)
        out = self.attn_out(out)
        return torch.cat([conv_out, out], dim=1)


class AAConvEncoder(nn.Module):
    def __init__(self, in_channels=4, channels=[32, 64, 64],
                 kernels=[8, 4, 3], strides=[4, 2, 1], num_heads=4):
        super().__init__()
        layers = []
        ch_in = in_channels
        for i, (ch_out, k, s) in enumerate(zip(channels, kernels, strides)):
            if i == len(channels) - 1:
                layers.append(AttentionAugmentedConv(ch_in, ch_out, k, s, num_heads))
            else:
                layers.append(nn.Conv2d(ch_in, ch_out, k, stride=s))
            layers.append(nn.ReLU())
            ch_in = ch_out
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            out = self.conv(dummy)
            self.output_size = out.view(1, -1).size(1)

    def forward(self, x):
        out = self.conv(x)
        return out.view(out.size(0), -1)


def build_encoder(config):
    enc_cfg = config.get("encoder", {})
    trans_cfg = config.get("transformer", {})
    enc_type = enc_cfg.get("type", "cnn")
    channels = enc_cfg.get("cnn_channels", [32, 64, 64])
    kernels = enc_cfg.get("cnn_kernels", [8, 4, 3])
    strides = enc_cfg.get("cnn_strides", [4, 2, 1])
    frame_stack = config.get("env", {}).get("frame_stack", 4)

    if enc_type == "cnn":
        enc = CNNEncoder(frame_stack, channels, kernels, strides)
        return enc, enc.output_size
    elif enc_type == "hybrid":
        enc = HybridCNNTransformerEncoder(
            in_channels=frame_stack,
            channels=channels, kernels=kernels, strides=strides,
            embed_dim=trans_cfg.get("embed_dim", 64),
            num_heads=trans_cfg.get("num_heads", 4),
            num_layers=trans_cfg.get("num_layers", 2),
            mlp_ratio=trans_cfg.get("mlp_ratio", 2.0),
            dropout=trans_cfg.get("dropout", 0.0),
            use_cls_token=trans_cfg.get("use_cls_token", True),
        )
        return enc, enc.output_size
    elif enc_type == "aaconv":
        enc = AAConvEncoder(
            in_channels=frame_stack,
            channels=channels, kernels=kernels, strides=strides,
            num_heads=trans_cfg.get("num_heads", 4),
        )
        return enc, enc.output_size
    else:
        raise ValueError(f"Unknown encoder type: {enc_type}")
