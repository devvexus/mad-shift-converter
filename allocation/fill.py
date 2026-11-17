# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Matthew Priester
"""Spend the bytes left below a tier's size target as +1 MS bit on the tensors with the most KL per byte.

Signal: a leave-one-out table measured on the whole MS model. For each part p (one layer's attention block or one layer's
FFN), S_p is the share of the whole model's KL that disappears when only that part is restored to the original weights.
Parts with S_p <= 0 (restoring them alone makes the model worse) are left alone. Within a part, tensor t's share is modelled
as n_t 4^-k_t / sum_j n_j 4^-k_j (squared error ~ 4^-k per weight); +1 bit removes 3/4 of it at a cost of n_t / 8 bytes.

  1. Measured phase: greedy by modelled KL removed per byte; after each bump the part's remaining S_p drops by the gain;
     widths stop at 6.
  2. Fill phase: bytes the measured phase cannot spend go, one bit at a time, to the remaining tensors: lowest k first;
     within a k, the tensors unsloth gave the most bits relative to k first (unsloth bits per weight - k), then kind order
     ffn_down, ffn_up, ffn_gate, attn_v, attn_output, ssm_out, attn_qkv, attn_gate, attn_q, attn_k, then layer.
Both phases stop when no tensor fits in what is left of the budget.

usage: python fill.py <config.json> <out.json> --ref-dir <dir> --loo data/qwen3.8-27b_loo.json [--budget BYTES]
  The budget defaults to size target - estimate - 8 MB (size.py). Tensor sizes and unsloth's bits per weight come from the
  config's "copy_from" file in --ref-dir.
"""
import argparse
import json
import os

import gguf

import size

ATTN = {"qkv": "attn_qkv", "z": "attn_gate", "sout": "ssm_out", "q": "attn_q", "k": "attn_k", "v": "attn_v",
        "o": "attn_output"}
FFN = ["ffn_gate", "ffn_up", "ffn_down"]
KIND_ORDER = {g: i for i, g in enumerate(["ffn_down", "ffn_up", "ffn_gate", "attn_v", "attn_output", "ssm_out", "attn_qkv",
                                          "attn_gate", "attn_q", "attn_k"])}
KMAX = 6


def part_shares(path):
    """Leave-one-out table -> {part: S_p} for the parts with S_p > 0, in table order."""
    d = json.load(open(path))
    full = sum(d["full"].values()) / len(d["full"])
    S = {}
    for part, v in d["parts"].items():
        s = (full - sum(v.values()) / len(v)) / full
        if s > 0:
            S[part] = s                                        # e.g. "L33_ffn": 0.025
    return S


def get_k(cfg, where):
    c, l, i = where
    return cfg["ffn"]["k_layer"][str(l)][i] if c == "ffn" else cfg[f"{c}_layer"][str(l)]["ms"]


def set_k(cfg, where, k):
    c, l, i = where
    if c == "ffn":
        cfg["ffn"]["k_layer"][str(l)][i] = k
    else:
        cfg[f"{c}_layer"][str(l)]["ms"] = k


def part_tensors(cfg, numel, part):
    """[(kind, k, weights, where)] of a part such as "L33_ffn" or "L33_attn"."""
    l, kind = part[1:].split("_")
    l = int(l)
    out = []
    if kind == "ffn":
        ks = cfg["ffn"]["k_layer"][str(l)]
        for i, g in enumerate(FFN):
            out.append((g, ks[i], numel[f"blk.{l}.{g}"], ("ffn", l, i)))
    else:
        for c, g in ATTN.items():
            lay = cfg.get(f"{c}_layer", {})
            if str(l) in lay and f"blk.{l}.{g}" in numel:
                out.append((g, lay[str(l)]["ms"], numel[f"blk.{l}.{g}"], (c, l, None)))
    return out


def measured_phase(cfg, numel, S, budget):
    """Greedy +1 bit by modelled KL removed per byte. Returns (bytes spent, bumps, modelled share of KL removed)."""
    S = dict(S)
    spent, bumps, removed = 0, [], 0.0
    while True:
        best = None
        for part, s in S.items():
            ts = part_tensors(cfg, numel, part)
            tot = sum(n * 4.0 ** -k for _, k, n, _ in ts)
            for g, k, n, where in ts:
                if k >= KMAX or spent + n // 8 > budget:
                    continue
                gain = s * (n * 4.0 ** -k / tot) * 0.75
                score = gain / (n / 8)
                if best is None or score > best[0]:
                    best = (score, gain, part, g, k, n, where)
        if best is None:
            break
        _, gain, part, g, k, n, where = best
        set_k(cfg, where, k + 1)
        S[part] -= gain
        spent += n // 8
        removed += gain
        bumps.append(f"{part.split('_')[0]}.{g} {k}->{k + 1}")
    return spent, bumps, removed


def fill_phase(cfg, numel, bpw, budget, spent):
    """+1 bit on the remaining tensors, lowest k first (see the module docstring). Returns (bytes spent, bumps)."""
    tensors = []
    for l in cfg["ffn"]["k_layer"]:
        for i, g in enumerate(FFN):
            tensors.append((g, int(l), ("ffn", int(l), i)))
    for c, g in ATTN.items():
        for l in cfg.get(f"{c}_layer", {}):
            if f"blk.{l}.{g}" in numel:
                tensors.append((g, int(l), (c, int(l), None)))
    bumps = []
    while True:
        cand = [(get_k(cfg, w), -(bpw[f"blk.{l}.{g}"] - get_k(cfg, w)), KIND_ORDER[g], l, g, w) for g, l, w in tensors
                if get_k(cfg, w) < KMAX and spent + numel[f"blk.{l}.{g}"] // 8 <= budget]
        if not cand:
            break
        k, _, _, l, g, w = min(cand)
        set_k(cfg, w, k + 1)
        spent += numel[f"blk.{l}.{g}"] // 8
        bumps.append(f"L{l}.{g} {k}->{k + 1} (fill)")
    return spent, bumps


def reference_tables(path):
    """Weights per tensor and unsloth's bits per weight, keyed by tensor name without ".weight"."""
    ts = [t for t in gguf.GGUFReader(path).tensors if t.name.endswith(".weight")]
    numel = {t.name[:-len(".weight")]: int(t.n_elements) for t in ts}
    bpw = {t.name[:-len(".weight")]: int(t.n_bytes) * 8 / int(t.n_elements) for t in ts}
    return numel, bpw


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="tier config (JSON), e.g. from unsloth_widths.py")
    ap.add_argument("out", help="tier config to write (JSON)")
    ap.add_argument("--ref-dir", required=True, help="directory holding the files named in copy_from / table_from")
    ap.add_argument("--loo", required=True, help="leave-one-out table (JSON), e.g. data/qwen3.8-27b_loo.json")
    ap.add_argument("--budget", type=int, help="bytes to spend (default: size target - estimate - 8 MB)")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    budget = a.budget
    if budget is None:
        est, target = size.estimate(cfg, a.ref_dir)
        budget = target - est - size.MARGIN
        print(f"{cfg['name']}: estimate {est:,} B, target {target:,} B -> budget {budget:,} B")
    numel, bpw = reference_tables(os.path.join(a.ref_dir, cfg["copy_from"]))
    spent, bumps, removed = measured_phase(cfg, numel, part_shares(a.loo), budget)
    spent, fills = fill_phase(cfg, numel, bpw, budget, spent)
    with open(a.out, "w") as f:
        json.dump(cfg, f, indent=1)
    print(f"+1 bit on {len(bumps) + len(fills)} tensors ({len(fills)} in the fill phase), {spent:,} of {budget:,} bytes; "
          f"modelled KL removed {removed * 100:.1f}% (measured phase)")
    if bumps or fills:
        print("  " + ", ".join(bumps + fills))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
