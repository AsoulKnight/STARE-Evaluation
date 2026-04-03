import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Basic blocks
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# -------------------------
# Patch embedding
# -------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, in_ch=3, embed_dim=256, patch_size=8):
        super().__init__()
        self.patch = nn.Conv2d(
            in_ch,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.patch(x)                  # [B, C, H/8, W/8]
        x = x.flatten(2).transpose(1, 2)   # [B, N, C]
        return x


# -------------------------
# Transformer (lighter)
# -------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        return self.encoder(x)


# -------------------------
# UNETR 2D (FIXED)
# -------------------------
class UNETR2D(nn.Module):
    def __init__(self, img_size=512, in_ch=3, out_ch=1):
        super().__init__()

        self.patch_size = 8
        self.embed_dim = 256

        # -------------------------
        # CNN Encoder (for skip connections)
        # -------------------------
        self.enc1 = ConvBlock(in_ch, 64)       # 512
        self.enc2 = ConvBlock(64, 128)         # 256
        self.enc3 = ConvBlock(128, 256)        # 128

        self.pool = nn.MaxPool2d(2)

        # -------------------------
        # Transformer
        # -------------------------
        self.patch_embed = PatchEmbedding(in_ch, self.embed_dim, self.patch_size)
        self.transformer = SimpleTransformer(self.embed_dim)

        # -------------------------
        # Decoder
        # -------------------------
        self.up1 = DeconvBlock(self.embed_dim, 256)   # 64x64 → 128x128
        self.up2 = DeconvBlock(256 + 256, 128)        # skip enc3
        self.up3 = DeconvBlock(128 + 128, 64)         # skip enc2
        self.up4 = DeconvBlock(64 + 64, 32)           # skip enc1

        self.final = nn.Conv2d(32, out_ch, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]

        # -------------------------
        # CNN encoder (skip features)
        # -------------------------
        e1 = self.enc1(x)              # [B,64,512,512]
        e2 = self.enc2(self.pool(e1))  # [B,128,256,256]
        e3 = self.enc3(self.pool(e2))  # [B,256,128,128]

        # -------------------------
        # Transformer
        # -------------------------
        t = self.patch_embed(x)        # [B, N, C]
        t = self.transformer(t)

        h = w = int((t.shape[1]) ** 0.5)
        t = t.transpose(1, 2).view(B, self.embed_dim, h, w)  # [B,256,64,64]

        # -------------------------
        # Decoder + skip connections
        # -------------------------
        x = self.up1(t)                         # → 128x128
        x = torch.cat([x, e3], dim=1)

        x = self.up2(x)                         # → 256x256
        x = torch.cat([x, e2], dim=1)

        x = self.up3(x)                         # → 512x512
        x = torch.cat([x, e1], dim=1)

        x = self.up4(x)                         # → 1024 (fix below)

        ###### Resize back to input size#######
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)

        return self.final(x)