"""CINet v2: Compact Spatio-Temporal CNN with Local Extremum Attention (LEA)
and Sub-window Convective Cell Center Heatmap Head.

Key meteorological innovations:
1. Local Extremum Attention (LEA):
   Specifically addresses the 0.14% area small convective initiation (CI) cores
   by combining channel extremum pooling (GAP + GMP) and multi-scale spatial
   dilated extremum attention between Block 3 and GeM pooling.
2. Sub-window Keypoint Heatmap Branch:
   A lateral feature-fusion decoder branch predicting continuous Gaussian heatmaps
   (sigma=2 px) of CI cell centroids (28x28 or 56x56 grid).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(2, 3)).pow(1.0 / self.p)


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = nn.GroupNorm(min(8, cout), cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = nn.GroupNorm(min(8, cout), cout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False),
                nn.GroupNorm(min(8, cout), cout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        s = x if self.skip is None else self.skip(x)
        return F.gelu(h + s)


class LocalExtremumAttention(nn.Module):
    """Multi-Scale Spatial-Channel Local Extremum Attention (LEA).

    Captures localized extreme brightness temperature drops and water vapor
    gradients in convective clouds:
    1. Channel Extremum: Captures peak cooling and water vapor differences
       via both Global Average Pooling (GAP) and Global Max Pooling (GMP).
    2. Spatial Extremum: Combines channel-wise max/mean statistics with
       multi-scale receptive fields (3x3 dilation=1, dilation=2) to isolate
       0.14% area point-like convective cores from expansive anvil clouds.
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid_channels = max(16, in_channels // reduction)

        # Channel Extremum Attention
        self.fc1 = nn.Linear(in_channels, mid_channels, bias=False)
        self.fc2 = nn.Linear(mid_channels, in_channels, bias=False)

        # Multi-scale Spatial Extremum Attention
        self.sp_conv1 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(2, 8),
            nn.GELU(),
        )
        self.sp_conv2 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(2, 8),
            nn.GELU(),
        )
        self.sp_out = nn.Conv2d(16, 1, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # 1. Channel Extremum Modulation
        avg_pool = x.mean(dim=(2, 3))
        max_pool = x.amax(dim=(2, 3))
        ch_attn = torch.sigmoid(
            self.fc2(F.gelu(self.fc1(avg_pool)))
            + self.fc2(F.gelu(self.fc1(max_pool)))
        ).view(b, c, 1, 1)

        x_ch = x * ch_attn

        # 2. Multi-scale Spatial Extremum Modulation
        sp_max = x_ch.amax(dim=1, keepdim=True)
        sp_avg = x_ch.mean(dim=1, keepdim=True)
        sp_feat = torch.cat([sp_max, sp_avg], dim=1)

        s1 = self.sp_conv1(sp_feat)
        s2 = self.sp_conv2(sp_feat)
        sp_attn = torch.sigmoid(self.sp_out(torch.cat([s1, s2], dim=1)))

        x_mod = x_ch * sp_attn
        return x + self.gamma * x_mod


class HeatmapHead(nn.Module):
    """Sub-window convective cell center heatmap regression head.

    Fuses deep features from Block 3 / LEA with high-resolution skip features
    from Block 2 to predict continuous 2D Gaussian heatmaps (sigma=2 px) of
    convective cell centroids.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int = 1,
        heatmap_size: int = 28,
    ):
        super().__init__()
        self.heatmap_size = heatmap_size
        mid_channels = skip_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(min(8, mid_channels), mid_channels),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels // 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(min(8, mid_channels // 2), mid_channels // 2),
            nn.GELU(),
            nn.Conv2d(mid_channels // 2, out_channels, kernel_size=1, bias=True),
        )

    def forward(
        self, feat_deep: torch.Tensor, feat_skip: torch.Tensor
    ) -> torch.Tensor:
        up = F.interpolate(
            feat_deep,
            size=feat_skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = self.fuse(torch.cat([up, feat_skip], dim=1))
        out = self.head(fused)
        if out.shape[-2:] != (self.heatmap_size, self.heatmap_size):
            out = F.interpolate(
                out,
                size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear",
                align_corners=False,
            )
        return out


class CINet_v2(nn.Module):
    """CINet v2: Multi-task CNN with Local Extremum Attention (LEA)

    and auxiliary Sub-window Cell Center Heatmap Head.
    """

    def __init__(
        self,
        in_ch: int = 41,
        width: int = 32,
        emb_dim: int = 0,
        heatmap_size: int = 28,
    ):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, 2, 1, bias=False),
            nn.GroupNorm(min(8, width), width),
            nn.GELU(),
        )
        self.b1 = Block(width, width * 2, 2)
        self.b2 = Block(width * 2, width * 4, 2)
        self.b3 = Block(width * 4, width * 8, 2)

        # Local Extremum Attention (LEA) between b3 and GeM pooling
        self.lea = LocalExtremumAttention(width * 8)
        self.pool = GeM()
        self.feature_dim = width * 8
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, 1),
        )

        # Sub-window convective cell center heatmap head
        self.heatmap_head = HeatmapHead(
            in_channels=width * 8,
            skip_channels=width * 4,
            out_channels=1,
            heatmap_size=heatmap_size,
        )

        self.projector = None
        if emb_dim:
            self.projector = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.GELU(),
                nn.Linear(self.feature_dim, emb_dim),
            )

    def extract_features(self, x: torch.Tensor):
        x = self.stem(x)
        x1 = self.b1(x)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        x3_lea = self.lea(x3)
        return x2, x3_lea

    def features(self, x: torch.Tensor) -> torch.Tensor:
        _, x3_lea = self.extract_features(x)
        return self.pool(x3_lea)

    def forward(self, x: torch.Tensor, return_heatmap: bool = False):
        x2, x3_lea = self.extract_features(x)
        pooled = self.pool(x3_lea)
        cls_logits = self.head(pooled).squeeze(-1)

        if return_heatmap:
            hm_logits = self.heatmap_head(x3_lea, x2)
            return cls_logits, hm_logits
        return cls_logits

    def predict_heatmap(self, x: torch.Tensor) -> torch.Tensor:
        x2, x3_lea = self.extract_features(x)
        hm_logits = self.heatmap_head(x3_lea, x2)
        return torch.sigmoid(hm_logits)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.projector is None:
            raise ValueError("Model was not initialized with emb_dim > 0")
        return self.projector(self.features(x))
