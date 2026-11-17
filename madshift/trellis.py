# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Matthew Priester
"""The bitshift trellis code and the reference trellis search (pure PyTorch).

State s has L = 14 bits. Each weight shifts in k new bits: s' = ((s << k) | b) mod 2^L. The weight's value is
code(s') * d with the row scale d. code() is the computed "1MAD" function of QTIP (Tseng et al., NeurIPS 2024):
x = 34038481 * s + 76625530 mod 2^32, value = (sum of the four bytes of x) - 510, an integer in [-510, 510] with
standard deviation about 147.8.

search(): sliding-window Viterbi with error feedback. A window looks Wwin columns ahead and commits the first Bdec of them;
each committed column's error is fed forward to the later columns through the upper Cholesky factor Hu of H^-1, as in
GPTQ / LDLQ, before the next window is searched. The path is continuous along the whole row (the committed state is the
start of the next window). The branch metric is piv_c * (v^2 - 2 x v) with piv_c = 1 / Hu[c, c]^2 (the row constant x^2
cannot change a decision). madshift/kernels.cu implements the same search with the same floating-point operations in the
same order; this file is the readable reference and the fallback when the kernels are not built."""
import torch

L_STATE = 14


def code_1mad(L=L_STATE, dev="cuda"):
    """Values of the 1MAD code for every L-bit state (float32)."""
    s = torch.arange(2 ** L, dtype=torch.int64)
    x = (34038481 * s + 76625530) & 0xFFFFFFFF
    v = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255) - 510
    return v.float().to(dev)


@torch.no_grad()
def search(W, Hu, sig, L=L_STATE, k=2, Bdec=8, Wwin=32, cb=None, rchunk=1024):
    """W (R, C) fp32 targets, Hu (C, C) upper Cholesky factor of H^-1, sig (R,) row scales, cb the code values.
    Returns (Q (R, C) dequantized fp32, states (R, C) int16)."""
    R, C = W.shape
    dev = W.device
    cb = code_1mad(L, dev) if cb is None else cb
    S = 2 ** L; A = 2 ** k; J = S // A
    piv = (1.0 / torch.diagonal(Hu) ** 2).tolist()
    Wc = W.clone()
    Q = torch.zeros_like(W)
    states_all = torch.empty(R, C, dtype=torch.int16, device=dev)
    state = None
    bsel = torch.arange(A, device=dev)
    chunks = [(r0, min(R, r0 + rchunk)) for r0 in range(0, R, rchunk)]
    vcache = {}
    for c0 in range(0, C, Bdec):
        c1 = min(C, c0 + Wwin)
        T = c1 - c0
        nd = min(Bdec, C - c0)
        dec_states = torch.empty(R, nd, dtype=torch.long, device=dev)
        for (r0, r1) in chunks:
            Rr = r1 - r0
            if len(chunks) == 1 and vcache:
                vals, vsq = vcache[0]
            else:
                vals = cb[None, :] * sig[r0:r1, None]; vsq = vals * vals
                if len(chunks) == 1:
                    vcache[0] = (vals, vsq)
            X = Wc[r0:r1, c0:c1]
            cost = torch.addcmul(vsq, vals, X[:, :1], value=-2.0).mul_(piv[c0])
            if state is not None:                                       # only successors of the committed state
                succ = ((state[r0:r1, None] << k) | bsel[None, :]) & (S - 1)
                c_s = cost.gather(1, succ)
                cost.fill_(float("inf")).scatter_(1, succ, c_s)
            bp = torch.empty(T, Rr, J, dtype=torch.uint8, device=dev)
            for t in range(1, T):
                # predecessors of s' = (j << k) | b are (a << (L - k)) | j: the best one depends only on j
                m, a = cost.view(Rr, A, J).min(1)
                inc = torch.addcmul(vsq, vals, X[:, t:t + 1], value=-2.0)
                cost = torch.add(m[:, :, None], inc.view(Rr, J, A), alpha=piv[c0 + t]).view(Rr, S)
                bp[t] = a
            s = cost.argmin(-1)
            ar = torch.arange(Rr, device=dev)
            path = torch.empty(Rr, T, dtype=torch.long, device=dev)
            path[:, T - 1] = s
            for t in range(T - 1, 0, -1):
                j = s >> k
                s = (bp[t][ar, j].long() << (L - k)) | j
                path[:, t - 1] = s
            dec_states[r0:r1] = path[:, :nd]
        state = dec_states[:, -1]
        states_all[:, c0:c0 + nd] = dec_states.to(torch.int16)
        q_all = cb[dec_states] * sig[:, None]
        for j in range(nd):                                             # commit: error feedback, column by column
            c = c0 + j
            q = q_all[:, j]
            Q[:, c] = q
            e = (Wc[:, c] - q) / Hu[c, c]
            if c + 1 < C:
                Wc[:, c + 1:] -= e[:, None] * Hu[c, c + 1:][None, :]
    return Q, states_all
