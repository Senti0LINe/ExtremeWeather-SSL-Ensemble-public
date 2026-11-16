"""Self-supervised helpers: meteorological augmentations and NT-Xent."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def augment_view(x: torch.Tensor) -> torch.Tensor:
    """Two views of the same window: flips, brightness scaling, cutout."""
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3])
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[2])
    x = x * (0.95 + 0.10 * torch.rand(x.shape[0], 1, 1, 1, device=x.device))
    h, w = x.shape[-2:]
    bh, bw = h // 5, w // 5
    for b in range(x.shape[0]):
        if torch.rand(1).item() < 0.5:
            top = torch.randint(0, h - bh, (1,)).item()
            left = torch.randint(0, w - bw, (1,)).item()
            x[b, :, top : top + bh, left : left + bw] = 0.0
    return x


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    b = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(-1e4)
    targets = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(sim, targets)
