# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Matthew Priester
"""Small linear-algebra helpers: the randomized Walsh-Hadamard input rotation and the Cholesky factor used by the error
feedback of the trellis search."""
import math
import torch


def pow2_block(d, cap=None):
    """Largest power-of-two block dividing width d (at most cap)."""
    b = d & -d
    return min(b, cap) if cap else b


def had_signs(seed, d):
    """The +-1 sign vector of a rotation: torch.Generator().manual_seed(seed), randint(0, 2, (d,)) * 2 - 1 (float32)."""
    g = torch.Generator().manual_seed(int(seed))
    return (torch.randint(0, 2, (int(d),), generator=g) * 2 - 1).float()


def rand_hadamard(d, block=None, seed=0, dev="cuda"):
    """Orthogonal randomized Walsh-Hadamard rotation R = diag(s) * blockdiag(H_b) / sqrt(b) (dense, float32).
    H_b is the Sylvester construction; s = had_signs(seed, d). A rotated weight is W @ R; the engine computes
    W x = (W R)(R^T x) by rotating the activation."""
    b = block or d
    assert d % b == 0 and (b & (b - 1)) == 0
    Hb = torch.ones(1, 1)
    while Hb.shape[0] < b:
        Hb = torch.cat([torch.cat([Hb, Hb], 1), torch.cat([Hb, -Hb], 1)], 0)
    Hb = Hb.to(dev) / math.sqrt(b)
    R = torch.block_diag(*([Hb] * (d // b)))
    g = torch.Generator().manual_seed(seed)
    s = (torch.randint(0, 2, (d,), generator=g) * 2 - 1).float().to(dev)
    return s[:, None] * R


def chol_inv_upper(H, damp=0.01):
    """Upper Cholesky factor of (H + damp * mean(diag H) * I)^-1 (the GPTQ convention)."""
    C = H.shape[-1]
    d = torch.diagonal(H, dim1=-2, dim2=-1).mean(-1)
    H = H + damp * d[..., None, None] * torch.eye(C, device=H.device, dtype=H.dtype)
    L = torch.linalg.cholesky(H)
    Hi = torch.cholesky_inverse(L)
    return torch.linalg.cholesky(Hi, upper=True)
