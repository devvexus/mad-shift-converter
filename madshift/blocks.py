# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Matthew Priester
"""The MS block encoding of a coded matrix (the GGUF tensor data of the MS_C1 .. MS_C6 types).

Per 512-weight block of a row: fp16 scale d, uint16 start state s0 (the 14-bit state of the block's first weight), then the
511 k-bit symbols of the other weights, most significant bit first, plus k zero pad bits: 4 + 64 k bytes. The states form a
bitshift trellis along the row (state_i = ((state_{i-1} << k) | symbol_i) & (2^14 - 1)), so the block holds exactly the
information of the per-weight states, and storing s0 lets every block be decoded on its own.

compact_entry() encodes an entry's states (int16 (R, C)) into its block bytes and decodes them back to check, bit for bit,
before anything is saved; expand_entry() does the reverse."""
import torch

QK = 512
DEV = "cuda"


def _scale_bytes(e, R, nb):
    """(R, nb, 2) uint8: the fp16 scale of every block, little-endian as in the GGUF."""
    sc = e["scale"].to(torch.float16).contiguous()
    return sc.view(torch.uint8).view(R, 1, 2).expand(R, nb, 2)


def _encode_rows(S, scb, k, L):
    """S: (R, nb, 512) int32 states (GPU); scb: (R, nb, 2) uint8 (GPU) -> (R, nb, 4 + 64k) uint8 blocks."""
    R, nb, _ = S.shape
    MASK = (1 << L) - 1
    if int(S.min()) < 0 or int(S.max()) > MASK:
        raise ValueError(f"states outside [0, {MASK}]")
    mask = (1 << k) - 1
    prev, cur = S[:, :, :-1], S[:, :, 1:]
    if bool(((((prev << k) | (cur & mask)) & MASK) != cur).any()):
        raise ValueError("trellis rule violated inside a block")
    sym = (cur & mask).to(torch.uint8)
    sh = torch.arange(k - 1, -1, -1, device=S.device, dtype=torch.uint8)
    bits = ((sym[..., None] >> sh) & 1).reshape(R, nb, (QK - 1) * k)
    bits = torch.cat([bits, torch.zeros(R, nb, k, dtype=torch.uint8, device=S.device)], -1).view(R, nb, 64 * k, 8)
    qs = bits[..., 0] << 7
    for i in range(1, 8):
        qs |= bits[..., i] << (7 - i)
    out = torch.empty(R, nb, 4 + 64 * k, dtype=torch.uint8, device=S.device)
    out[:, :, 0:2] = scb
    s0 = S[:, :, 0]
    out[:, :, 2] = (s0 & 255).to(torch.uint8); out[:, :, 3] = (s0 >> 8).to(torch.uint8)
    out[:, :, 4:] = qs
    return out


def _decode_rows(blk, k, L):
    """(R, nb, 4 + 64k) uint8 (GPU) -> states (R, nb, 512) int32."""
    R, nb, _ = blk.shape
    MASK = (1 << L) - 1
    i32 = dict(device=blk.device, dtype=torch.int32)
    s0 = (blk[:, :, 2].to(torch.int32) | (blk[:, :, 3].to(torch.int32) << 8)) & MASK
    by = blk[:, :, 4:].to(torch.int32)
    bits = ((by[..., None] >> torch.arange(7, -1, -1, **i32)) & 1).reshape(R, nb, 512 * k)
    s0b = (s0[..., None] >> torch.arange(L - 1, -1, -1, **i32)) & 1
    allb = torch.cat([s0b, bits], -1)
    st = torch.zeros(R, nb, QK, **i32)
    base = k * torch.arange(QK, device=blk.device)
    for i in range(L):                       # the state of weight i is the L-bit window starting at bit k * i
        st |= allb[:, :, base + i] << (L - 1 - i)
    return st


def is_entry(e):
    return isinstance(e, dict) and "k" in e and ("states" in e or "blocks" in e)


def compact_entry(e, chunk_rows=2048):
    st = e["states"]
    R, C = st.shape
    if C % QK:
        raise ValueError(f"cols {C} not a multiple of {QK}")
    nb, k, L = C // QK, int(e["k"]), int(e.get("L", 14))
    scb_all = _scale_bytes({"scale": e["scale"].to(DEV)}, R, nb)
    out = torch.empty(R, nb * (4 + 64 * k), dtype=torch.uint8)
    for r0 in range(0, R, chunk_rows):
        r1 = min(R, r0 + chunk_rows)
        S = st[r0:r1].to(DEV).to(torch.int32).view(r1 - r0, nb, QK)
        b = _encode_rows(S, scb_all[r0:r1], k, L)
        if not torch.equal(_decode_rows(b, k, L), S) or not torch.equal(b[:, :, 0:2], scb_all[r0:r1]):
            raise RuntimeError("block encoding: decode(encode(states)) != states")
        out[r0:r1] = b.view(r1 - r0, -1).cpu()
    c = {key: v for key, v in e.items() if key != "states"}
    c["blocks"] = out
    c["shape"] = [int(R), int(C)]
    return c


def expand_entry(e, chunk_rows=2048):
    if "blocks" not in e:
        return e
    R, C = e["shape"]
    nb, k, L = C // QK, int(e["k"]), int(e.get("L", 14))
    st = torch.empty(R, C, dtype=torch.int16)
    blk = e["blocks"]
    for r0 in range(0, R, chunk_rows):
        r1 = min(R, r0 + chunk_rows)
        b = blk[r0:r1].to(DEV).view(r1 - r0, nb, 4 + 64 * k)
        st[r0:r1] = _decode_rows(b, k, L).view(r1 - r0, C).to(torch.int16).cpu()
    x = {key: v for key, v in e.items() if key not in ("blocks", "shape")}
    x["states"] = st
    return x


def compact_pack(pk):
    return {key: (compact_entry(v) if is_entry(v) and "states" in v else v) for key, v in pk.items()}


def expand_pack(pk):
    return {key: (expand_entry(v) if is_entry(v) else v) for key, v in pk.items()}


def entry_shape(e):
    return tuple(e["shape"]) if "blocks" in e else tuple(e["states"].shape)
