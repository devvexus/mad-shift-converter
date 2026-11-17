# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Matthew Priester
"""File size of an MS tier from its config, before converting, and the byte budget left for fill.py.

For a tier whose widths follow an unsloth file (unsloth_widths.py), the MS file is the unsloth file ("copy_from")
  - minus its MTP block (tensors blk.N.* with N >= the trunk layer count; MS files do not contain it),
  - minus unsloth's bytes of every tensor MS codes (attention projections, FFN matrices, the head),
  - plus those tensors as MS_Ck (rows x cols / 512 blocks of 64 k + 4 bytes; a split head adds a 4-byte row order),
  - plus the difference of the input table when it is copied from another file ("table_from"),
  - plus HEADER for the MS metadata.
The size target is the unsloth file without its MTP block; the fill budget is target - estimate - MARGIN.

usage: python size.py <config.json> [<config.json> ...] --ref-dir <directory holding the unsloth files>
"""
import argparse
import json
import os
import re

import gguf

ATTN = {"qkv": "attn_qkv", "z": "attn_gate", "sout": "ssm_out", "q": "attn_q", "k": "attn_k", "v": "attn_v",
        "o": "attn_output"}
FFN = ["ffn_gate", "ffn_up", "ffn_down"]
HEADER = 3_100_000      # MS header and metadata beyond the unsloth file's (measured on Qwen3.8-27B: 3,046,336 bytes)
MARGIN = 8_000_000      # bytes kept free below the target


def ms_bytes(n, k):
    return n // 512 * (64 * k + 4)


def estimate(cfg, ref_dir):
    """(estimated MS file size, size target) in bytes."""
    ref = os.path.join(ref_dir, cfg["copy_from"])
    T = {t.name: t for t in gguf.GGUFReader(ref).tensors}
    size = os.path.getsize(ref)
    n_layers = 1 + max(int(l) for l in cfg["ffn"]["k_layer"])
    mtp = 0
    for name, t in T.items():
        m = re.match(r"blk\.(\d+)\.", name)
        if m and int(m.group(1)) >= n_layers:
            size -= int(t.n_bytes)
            mtp += int(t.n_bytes)
    target = os.path.getsize(ref) - mtp
    size += HEADER
    for c, g in ATTN.items():
        for l, s in cfg.get(f"{c}_layer", {}).items():
            t = T[f"blk.{l}.{g}.weight"]
            size += ms_bytes(int(t.n_elements), s["ms"]) - int(t.n_bytes)
    for l, ks in cfg["ffn"]["k_layer"].items():
        for g, k in zip(FFN, ks):
            t = T[f"blk.{l}.{g}.weight"]
            size += ms_bytes(int(t.n_elements), k) - int(t.n_bytes)
    out = cfg.get("out")
    if out and "output.weight" in T:
        t = T["output.weight"]
        V, d = int(t.shape[1]), int(t.shape[0])
        if out.get("lo"):                                     # split head: n_hi rows at ms, the rest at lo
            nh = int(out["n_hi"])
            size += 4 * V + 4096                              # the int32 row order
            size += ms_bytes(nh * d, out["ms"]) + ms_bytes((V - nh) * d, out["lo"]) - int(t.n_bytes)
        else:
            size += ms_bytes(int(t.n_elements), out["ms"]) - int(t.n_bytes)
    table = cfg.get("table_from")
    if table and table != cfg["copy_from"]:
        te = next(t for t in gguf.GGUFReader(os.path.join(ref_dir, table)).tensors if t.name == "token_embd.weight")
        size += int(te.n_bytes) - int(T["token_embd.weight"].n_bytes)
    return size, target


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("configs", nargs="+", help="tier configs (JSON)")
    ap.add_argument("--ref-dir", required=True, help="directory holding the files named in copy_from / table_from")
    a = ap.parse_args()
    for path in a.configs:
        est, target = estimate(json.load(open(path)), a.ref_dir)
        print(f"{path}: estimated {est:,} B, target {target:,} B, slack {(target - est) / 1e6:.2f} MB, "
              f"fill budget {target - est - MARGIN:,} B")


if __name__ == "__main__":
    main()
