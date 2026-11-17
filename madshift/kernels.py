# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Matthew Priester
"""ctypes wrapper of kernels.cu: the trellis search of trellis.search() on fused CUDA kernels, same decisions and values bit
for bit, with the error feedback applied lazily (O(R C) memory traffic per window instead of O(R C^2)).

search(W, Hu, sig, L, k, Bdec, Wwin) -> (Q fp32, states int16), the arguments and results of trellis.search() for L = 14 and
the 1MAD code. The library is madshift/madshift_kernels.dll (Windows) or madshift/libmadshift_kernels.so (Linux), built by
build_kernels.bat / build_kernels.sh; the environment variable MADSHIFT_KERNELS may name another path."""
import ctypes, os, sys
import torch

_LIB = []
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "madshift_kernels.dll" if sys.platform == "win32" else "libmadshift_kernels.so")
DLL = os.environ.get("MADSHIFT_KERNELS") or DEFAULT
MODE = 2                       # rounding mode of the reference (torch.addcmul / torch.add on CUDA: two roundings, fused update)


def available():
    return os.path.exists(DLL)


def lib():
    if not _LIB:
        if not os.path.exists(DLL):
            raise FileNotFoundError(f"{DLL} missing: build it with madshift/build_kernels.bat (Windows) or build_kernels.sh (Linux)")
        L = ctypes.CDLL(DLL)
        P, I = ctypes.c_void_p, ctypes.c_int
        for f in (L.msc_vit2, L.msc_vit3, L.msc_vit4):
            f.argtypes = [P, I, P, P, P, P, I, I, I, I, I, P, P, I, I, P]
            f.restype = I
        L.msc_commit.argtypes = [P, I, P, I, P, I, P, I, I, P, P, P, I, P, I, P, P]
        L.msc_apply.argtypes = [P, I, I, I, I, P, P, I, I, I, P]
        L.msc_commit.restype = I
        L.msc_apply.restype = I
        _LIB.append(L)
    return _LIB[0]


def _chk(rc, what):
    if rc != 0:
        raise RuntimeError(f"madshift kernels: {what}: CUDA error {rc}")


@torch.no_grad()
def search(W, Hu, sig, L=14, k=2, Bdec=8, Wwin=32, pb=256, bp_budget=256 << 20):
    """W (R, C) fp32 targets, Hu (C, C) upper Cholesky factor of H^-1, sig (R,) row scales. Returns (Q fp32, states int16)."""
    assert L == 14 and W.is_cuda and Wwin <= 64 and Bdec <= Wwin
    f = lib()
    vit = f.msc_vit4 if 2 <= k <= 6 else f.msc_vit3          # vit4: register-resident path costs; vit3 for k = 1; same results
    R, C = W.shape
    dev = W.device
    J = (1 << L) >> k
    Wc = W.float().contiguous().clone()
    Huc = Hu.float().contiguous()
    piv = (1.0 / torch.diagonal(Hu) ** 2).float().contiguous()
    sigc = sig.float().contiguous()
    Q = torch.empty(R, C, dtype=torch.float32, device=dev)
    ST = torch.empty(R, C, dtype=torch.int16, device=dev)
    E = torch.empty(C, R, dtype=torch.float32, device=dev)             # committed errors, column-major
    rows_per = max(1, min(R, bp_budget // (min(Wwin, C) * J)))
    bp = torch.empty(rows_per * min(Wwin, C) * J, dtype=torch.uint8, device=dev)
    dec = torch.empty(R * Bdec, dtype=torch.int64, device=dev)
    st = torch.zeros(R, dtype=torch.int64, device=dev)
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    pW, pE, pH, pQ, pS = Wc.data_ptr(), E.data_ptr(), Huc.data_ptr(), Q.data_ptr(), ST.data_ptr()

    def apply(j0, j1, ca, cb_):
        if j1 > j0 and cb_ > ca:
            _chk(f.msc_apply(pW, C, R, j0, j1, pE, pH, C, ca, cb_, stream), "apply")

    a_far, F, prev = 0, 0, None
    for c0 in range(0, C, Bdec):
        c1 = min(C, c0 + Wwin); T = c1 - c0; nd = min(Bdec, C - c0)
        if c0 - a_far >= pb and F < C:                                  # far columns: all errors since the last bulk, in order
            apply(F, C, a_far, c0); a_far = c0
        if prev is not None:                                            # columns of the last window: its committed errors
            apply(c0, min(F, c1), prev, c0)
        apply(max(F, c0), c1, a_far, c0)                                # columns entering the window
        F = max(F, c1)
        for r0 in range(0, R, rows_per):
            n = min(rows_per, R - r0)
            _chk(vit(pW + 4 * (r0 * C + c0), C, sigc.data_ptr() + 4 * r0, None, piv.data_ptr() + 4 * c0,
                     st.data_ptr() + 8 * r0, int(prev is not None), n, k, T, nd, bp.data_ptr(),
                     dec.data_ptr() + 8 * r0 * nd, nd, MODE, stream), "viterbi")
        _chk(f.msc_commit(pW, C, pE, R, pH, C, dec.data_ptr(), nd, c0, sigc.data_ptr(), None, pQ, C, pS, C,
                          st.data_ptr(), stream), "commit")
        prev = c0
    return Q, ST
