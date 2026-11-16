"""Shared CNN definitions (used by supervised, SSL and inference scripts)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p=3.0):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))

    def forward(self, x):
        return x.clamp(min=1e-6).pow(self.p).mean(dim=(2, 3)).pow(1.0 / self.p)


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = nn.GroupNorm(min(8, cout), cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = nn.GroupNorm(min(8, cout), cout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                      nn.GroupNorm(min(8, cout), cout))

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        s = x if self.skip is None else self.skip(x)
        return F.gelu(h + s)


class CINet(nn.Module):
    def __init__(self, in_ch=41, width=32, emb_dim=0):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, width, 3, 2, 1, bias=False),
                                  nn.GroupNorm(8, width), nn.GELU())
        self.b1 = Block(width, width * 2, 2)
        self.b2 = Block(width * 2, width * 4, 2)
        self.b3 = Block(width * 4, width * 8, 2)
        self.pool = GeM()
        self.feature_dim = width * 8
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.feature_dim, 1))
        self.projector = None
        if emb_dim:
            self.projector = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim),
                                           nn.GELU(), nn.Linear(self.feature_dim, emb_dim))

    def features(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return self.pool(x)

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)

    def embed(self, x):
        return self.projector(self.features(x))
