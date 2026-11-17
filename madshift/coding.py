# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Matthew Priester
"""Coding of one matrix with the trellis code, and of one SwiGLU FFN (gate, up, down) with its refit.

code_matrix: rotate W (W @ R), take the rotated input second moment R^T H R, and run the trellis search with error feedback.
ffn_code: the FFN's gate / up on the FFN-input moment, then the down projection refit to its target on the coded gate / up's
hidden activations (ridge least squares), rotated on the hidden width and coded on the hidden second moment."""
import os, time
import torch, torch.nn.functional as F
from madshift import trellis
from madshift.linalg import chol_inv_upper, rand_hadamard as _rand_hadamard, pow2_block

DEV = "cuda"
L_STATE = 14                                                           # trellis state bits
BDEC, WWIN = 8, 32                                                     # committed columns per window, window length
GAIN = {1: 1.0, 2: 1.15, 3: 1.2, 4: 1.2, 5: 1.2, 6: 1.2}               # row scale = GAIN[k] * std(row) / 147.8
CODE_STD = 147.8                                                       # standard deviation of the 1MAD values
USE_KERNELS = os.environ.get("MADSHIFT_REFERENCE_SEARCH", "0") != "1"
_ROT, _CB = {}, {}
STATS = {"search_s": 0.0, "search_w": 0, "chol_s": 0.0, "damp_fallbacks": 0}
ROT_CACHE = 12                                                         # rotation matrices kept (least recently used dropped)


def tf32(on):
    torch.backends.cuda.matmul.allow_tf32 = bool(on)


def sync():
    torch.cuda.synchronize()


def rand_hadamard(d, block, seed, dev=DEV):
    """linalg.rand_hadamard, cached (a pure function of its arguments)."""
    key = (d, block, seed, str(dev))
    if key in _ROT:
        _ROT[key] = _ROT.pop(key)
        return _ROT[key]
    _ROT[key] = _rand_hadamard(d, block=block, seed=seed, dev=dev)
    while len(_ROT) > ROT_CACHE:
        del _ROT[next(iter(_ROT))]
    return _ROT[key]


def codebook():
    if "1mad" not in _CB:
        _CB["1mad"] = trellis.code_1mad(L_STATE, DEV)
    return _CB["1mad"]


def tcq(W, Hg, k, damp=0.01):
    """Trellis-code W (R, C) fp32 (already rotated) against its input second moment Hg (C, C).
    Returns (Q fp32 dequantized, entry {k, L, states, scale, code} with CPU tensors, coded bits)."""
    sdev = W.std(1)
    t0 = time.time()
    try:
        Hu = chol_inv_upper(Hg, damp)
    except torch.linalg.LinAlgError:                 # H not positive definite at this damping: a larger one
        for d2 in (0.05, 0.2):
            try:
                Hu = chol_inv_upper(Hg, d2)
                print(f"  WARNING: Cholesky failed at damp {damp}, used {d2} ({tuple(W.shape)}, k {k})", flush=True)
                STATS["damp_fallbacks"] += 1
                break
            except torch.linalg.LinAlgError:
                continue
        else:
            raise
    sig = (GAIN[k] * sdev / CODE_STD).half().float()                   # fp16 row scale (stored in every 512-weight block)
    cb = codebook()
    sync(); t1 = time.time(); STATS["chol_s"] += t1 - t0
    if USE_KERNELS:
        from madshift import kernels
        Q, st = kernels.search(W, Hu, sig, L=L_STATE, k=k, Bdec=BDEC, Wwin=WWIN)
    else:
        Q, st = trellis.search(W, Hu, sig, L=L_STATE, k=k, Bdec=BDEC, Wwin=WWIN, cb=cb)
    sync(); STATS["search_s"] += time.time() - t1; STATS["search_w"] += W.numel()
    e = dict(k=k, L=L_STATE, states=st.cpu(), scale=sig.half().cpu(), code="1mad")
    return Q, e, W.numel() * k + 16 * W.shape[0]


def code_matrix(W, H, k, seed, rot, name, block_cap=4096, damp=0.01):
    """W (out, in) fp32, H (in, in) its input second moment (TF32 must be off). Returns (dequantized W fp32 in the original
    basis, entry with 'rot', bits per weight)."""
    blk = pow2_block(W.shape[1], block_cap)
    R = rand_hadamard(W.shape[1], blk, seed) if rot else None
    Wr, Hr = (W @ R, R.T @ H @ R) if R is not None else (W, H)
    try:
        Qr, e, bits = tcq(Wr.contiguous(), Hr, k, damp=damp)
    except torch.linalg.LinAlgError:
        print(f"  WARNING {name}: Cholesky failed at damp 0.01, retrying at 0.05", flush=True)
        Qr, e, bits = tcq(Wr.contiguous(), Hr, k, damp=0.05)
    e["rot"] = dict(seed=seed, block=blk) if R is not None else None
    return (Qr @ R.T if R is not None else Qr), e, bits / W.numel()


def dequant_entry(e, dev=DEV):
    """Entry -> the dequantized matrix in the rotated basis: code[states] * scale (bit-identical to the coder's Q)."""
    cb = codebook()
    st = e["states"].to(dev).long()
    sig = e["scale"].to(dev).float()
    return cb[st] * sig[:, None]


def second_moment_rows(get_rows, N, d, chunk=32768):
    """H = sum over row chunks of x^T x (fp32, first chunk assigned, then H = H + x^T x), divided by N."""
    H = None
    for i in range(0, N, chunk):
        x = get_rows(i, min(N, i + chunk)).float()
        H = x.T @ x if H is None else H + x.T @ x
    return H / N


def mm32(a, b):
    """a @ b with fp32 output; bf16 inputs use the tensor cores with fp32 accumulation."""
    if a.dtype == torch.bfloat16:
        return torch.mm(a, b, out_dtype=torch.float32)
    return a @ b


def ffn_code(G, U, D, n_rows, y_rows, N, kgud, hblock=512, lam=1e-4, chunk=32768, damp=0.01, prec="tf32", refit_mu=0.0):
    """Code one SwiGLU FFN. G, U (F, d), D (d, F) fp32; n_rows(i0, i1) -> the FFN inputs of calibration rows [i0, i1) (bf16);
    y_rows(i0, i1) -> the down projection's target (bf16); kgud = [k_gate, k_up, k_down].
      1. Hn = Rin^T (sum x^T x / N) Rin; gate and up coded on Hn (input rotation seed 1, block = largest power of two of d).
      2. hq = silu(x Gq^T) * (x Uq^T) with the coded gate / up; Hh = hq^T hq / N, rhs = hq^T y / N.
      3. Dt = (Hh + (lam + refit_mu) mean(diag Hh) I)^-1 (rhs + refit_mu mean(diag Hh) D^T): the down projection refit to the
         target, with a ridge toward the original D (refit_mu).
      4. Dt rotated on the hidden width (seed 2, block hblock) and coded on Rh^T Hh Rh.
    prec: the token-loop GEMMs in TF32 ("tf32") or with bf16 inputs and fp32 accumulation ("bf16").
    Returns (Gq, Uq, Dq fp32 in the original basis, entries {G, U, D, hadamard}, bits per weight)."""
    dev = G.device
    Rin = rand_hadamard(G.shape[1], pow2_block(G.shape[1]), 1, dev)
    Rh = rand_hadamard(G.shape[0], min(hblock, G.shape[0]), 2, dev)
    lo = prec == "bf16"
    cast = (lambda t: t.to(torch.bfloat16)) if lo else (lambda t: t.float())
    S = torch.zeros(G.shape[1], G.shape[1], device=dev)
    for i in range(0, N, chunk):
        x = cast(n_rows(i, min(N, i + chunk)))
        S += mm32(x.T, x)
    Hn = Rin.T @ (S / N) @ Rin
    Gq, eG, bg = tcq(G @ Rin, Hn, kgud[0], damp=damp)
    Uq, eU, bu = tcq(U @ Rin, Hn, kgud[1], damp=damp)
    Gq = Gq @ Rin.T; Uq = Uq @ Rin.T
    Gc, Uc = cast(Gq).T.contiguous(), cast(Uq).T.contiguous()
    Hh = torch.zeros(G.shape[0], G.shape[0], device=dev)
    rhs = torch.zeros(G.shape[0], D.shape[0], device=dev)
    for i in range(0, N, chunk):
        x = cast(n_rows(i, min(N, i + chunk)))
        hq = F.silu(mm32(x, Gc)) * mm32(x, Uc)
        y = y_rows(i, min(N, i + chunk))
        hqc = cast(hq)
        Hh += mm32(hqc.T, hqc); rhs += mm32(hqc.T, cast(y))
        del hq, hqc, y
    Hh /= N; rhs /= N
    mu = refit_mu * Hh.diagonal().mean()
    Dt = torch.linalg.solve(Hh + (lam * Hh.diagonal().mean() + mu) * torch.eye(Hh.shape[0], device=dev), rhs + mu * D.T).T
    Dq, eD, bd = tcq(Dt @ Rh, Rh.T @ Hh @ Rh, kgud[2], damp=damp)
    Dq = Dq @ Rh.T
    pk = dict(G=eG, U=eU, D=eD)
    pk["hadamard"] = dict(in_seed=1, in_block=pow2_block(G.shape[1]), h_seed=2, h_block=min(hblock, G.shape[0]))
    bits = (bg + bu + bd) / (G.numel() + U.numel() + D.numel())
    return Gq, Uq, Dq, pk, bits
