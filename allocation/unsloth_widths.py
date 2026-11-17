# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Matthew Priester
"""MS tier config that follows the per-tensor bit allocation of an Unsloth Dynamic GGUF (the method used for Qwen3.8-27B).

The starting point is unsloth's per-tensor quantization types: this tool reads them from an Unsloth Dynamic GGUF of the
same model (only the header is read) and gives every tensor MS codes the MS width closest to unsloth's choice, without
using more bytes. Credit for the per-tensor allocation it starts from goes to Unsloth (Unsloth Dynamic GGUFs).

Unsloth's types have fractional sizes (IQ3_XXS 3.0625 bits per weight, Q3_K 3.4375, Q4_K 4.5, Q6_K 6.5625, ...); an MS_Ck
tensor costs k + 0.0625 bits per weight, k = 1..6. So:
  1. per tensor: floor = the widest k with k + 0.0625 <= unsloth's bits per weight (clamped to KMIN..6); the choices are
     floor, floor + 1 (if floor < 6 and floor + 0.0625 <= unsloth's bits per weight) and floor - 1 (if >= KMIN);
  2. per layer, over all its MS-coded tensors (attention projections and FFN), every combination of choices is scored and
     the best one kept (see solve_layer); the running MS byte total over the layers never exceeds unsloth's;
  3. output head: floor k of unsloth's head type, then split into the 65,536 most frequent rows at k and the rest at
     k - 1 (k - 2 when k = 6).
Tensors MS does not code (norms, conv, ssm parameters) are copied from the unsloth file ("copy_from"); the input table is
copied from "table_from" (by default the same file). The spare bytes up to the size target are spent afterwards by fill.py.

usage: python unsloth_widths.py <unsloth.gguf> <out.json> --name MS4 --model Qwen/Qwen3.8-27B
          [--table-from <gguf>] [--head-split auto|none|HI,LO] [--raise-to K]
"""
import argparse
import collections
import itertools
import json
import os

import gguf

import recipe

QK_MS = 512                 # weights per MS block
KMIN, KMAX = 1, 6           # MS widths
HEAD_ROWS_HI = 65536        # rows of the split head kept at the higher width
KIND_CLASS = {"attn_qkv": "qkv", "attn_gate": "z", "ssm_out": "sout", "attn_q": "q", "attn_k": "k", "attn_v": "v",
              "attn_output": "o", "ffn_gate": "ffn0", "ffn_up": "ffn1", "ffn_down": "ffn2"}
KIND_ORDER = ["ffn_down", "ffn_up", "ffn_gate", "attn_v", "attn_output", "ssm_out", "attn_qkv", "attn_gate", "attn_q",
              "attn_k"]
COMPONENT = {"attn_qkv": "linear attention", "attn_gate": "linear attention", "ssm_out": "linear attention",
             "attn_q": "full attention", "attn_k": "full attention", "attn_v": "full attention",
             "attn_output": "full attention", "ffn_gate": "FFN", "ffn_up": "FFN", "ffn_down": "FFN"}


def ms_bytes(rows, cols, k):
    assert cols % QK_MS == 0, (rows, cols)
    return rows * (cols // QK_MS) * (4 + 64 * k)


def floor_k(b):
    return max(KMIN, min(KMAX, int((b - 0.0625) + 1e-9)))


def read_reference(path):
    """Tensor table of a GGUF: name -> dict(type, n, nbytes, rows, cols); and the trunk layer count (MTP block excluded)."""
    r = gguf.GGUFReader(path, "r")
    arch = r.get_field(gguf.Keys.General.ARCHITECTURE).contents()

    def field(key):
        return r.get_field(f"{arch}.{key}")

    nextn = field("nextn_predict_layers")
    n_layers = int(field("block_count").contents()) - (int(nextn.contents()) if nextn is not None else 0)
    T = {}
    for t in r.tensors:
        T[t.name] = dict(type=t.tensor_type.name, n=int(t.n_elements), nbytes=int(t.n_bytes),
                         cols=int(t.shape[0]), rows=int(t.shape[1]) if len(t.shape) > 1 else 1)
    return T, n_layers


def solve_layer(items, comp_t, kind_t, budget):
    """Choose k for each MS-coded tensor of one layer.

    items: the layer's tensors (dict rows, cols, b, n, kind, comp, floor, rank). Every combination of floor / floor + 1 /
    floor - 1 is scored; hard limit: MS bytes <= budget (unsloth's bytes so far minus MS's bytes so far, plus this layer's).
    Score, smallest first:
      1. fewest below-floor steps;
      2. 2 x sum over components (FFN, linear attention, full attention) of |MS bytes - component target|
         + sum over tensor kinds of |MS bytes - kind target|
         (targets: unsloth's bytes in this layer plus each component's / kind's deficit carried from earlier layers; a
         kind target holds only what MS can store, so the excess of a tensor above MS_C6 (Q6_K, Q8_0) goes to the
         component, and the factor 2 lets a neighbour of the same component take it);
      3. per-tensor faithfulness: sum of n |MS bits - unsloth bits| rounded to 3 decimals (the largest rounding
         remainder rounds up first);
      4. favour (unsloth's bits relative to its kind's mean; the tensor with the lowest rank weighs most), then the
         first combination in enumeration order.
    Returns (k per item, MS bytes)."""
    opts = []
    for it in items:
        f = it["floor"]
        o = [(f, 0)]
        if f < KMAX and it["b"] >= f + 0.0625 - 1e-9:   # also for an exact match, so slack can move to this tensor
            o.append((f + 1, 1))
        if f - 1 >= KMIN:
            o.append((f - 1, -1))
        opts.append(o)
    best = None
    n = len(items)
    for combo in itertools.product(*opts):
        mb = [ms_bytes(it["rows"], it["cols"], k) for it, (k, _) in zip(items, combo)]
        tot = sum(mb)
        if tot > budget:
            continue
        downs = sum(-s for _, s in combo if s < 0)
        cd = dict((c, -v) for c, v in comp_t.items())
        kd = dict((c, -v) for c, v in kind_t.items())
        for it, m in zip(items, mb):
            cd[it["comp"]] += m
            kd[it["kind"]] += m
        score = 2 * sum(abs(v) for v in cd.values()) + sum(abs(v) for v in kd.values())
        tdev = sum(it["n"] * abs(k + 0.0625 - it["b"]) for it, (k, _) in zip(items, combo))
        fav = sum(2.0 ** (n - it["rank"]) * s for it, (_, s) in zip(items, combo))
        key = (downs, score, round(tdev, 3), -fav)
        if best is None or key < best[0]:
            best = (key, [k for k, _ in combo], tot)
    if best is None:
        raise SystemExit(f"no allocation fits the layer budget {budget}")
    return best[1], best[2]


def match_widths(T, n_layers):
    """Per-tensor MS widths matched to the reference tensors: list of row dicts (layer, kind, k, ...), and the head k
    (None for a model with a tied head)."""
    rows = []
    for l in range(n_layers):
        for kind in KIND_CLASS:
            name = f"blk.{l}.{kind}.weight"
            if name in T:
                t = T[name]
                rows.append(dict(layer=l, kind=kind, n=t["n"], rows=t["rows"], cols=t["cols"], ub=t["nbytes"],
                                 b=t["nbytes"] * 8 / t["n"], comp=COMPONENT[kind]))
    kind_mean = {}
    for kind in KIND_CLASS:
        v = [r for r in rows if r["kind"] == kind]
        if v:
            kind_mean[kind] = sum(r["b"] * r["n"] for r in v) / sum(r["n"] for r in v)
    for r in rows:
        r["favour"] = r["b"] - kind_mean[r["kind"]]
        r["floor"] = floor_k(r["b"])
        r["rem"] = r["b"] - (r["floor"] + 0.0625)
    carry = collections.defaultdict(int)       # per component: unsloth bytes - MS bytes so far
    kcarry = collections.defaultdict(int)      # per tensor kind: the same
    run = 0                                    # over all layers so far; never negative
    for l in range(n_layers):
        items = [r for r in rows if r["layer"] == l]
        order = sorted(items, key=lambda r: (-round(r["favour"], 6), -round(r["rem"], 6), KIND_ORDER.index(r["kind"])))
        for i, r in enumerate(order):
            r["rank"] = i
        cu, ku = collections.defaultdict(int), collections.defaultdict(int)
        for r in items:
            cu[r["comp"]] += r["ub"]
            ku[r["kind"]] += min(max(r["ub"], ms_bytes(r["rows"], r["cols"], KMIN)), ms_bytes(r["rows"], r["cols"], KMAX))
        comp_t = {c: v + carry[c] for c, v in cu.items()}
        kind_t = {c: v + kcarry[c] for c, v in ku.items()}
        lb = sum(cu.values())
        ks, tot = solve_layer(items, comp_t, kind_t, lb + run)
        ms_c, ms_k = collections.defaultdict(int), collections.defaultdict(int)
        for r, k in zip(items, ks):
            r["k"] = k
            r["mb"] = ms_bytes(r["rows"], r["cols"], k)
            ms_c[r["comp"]] += r["mb"]
            ms_k[r["kind"]] += r["mb"]
        for c in cu:
            carry[c] = comp_t[c] - ms_c[c]
        for c in ku:
            kcarry[c] = kind_t[c] - ms_k[c]
        run += lb - tot
    head_k = None
    if "output.weight" in T:
        h = T["output.weight"]
        head_k = floor_k(h["nbytes"] * 8 / h["n"])
    return rows, head_k


def head_split(spec, k):
    if spec == "none":
        return {"ms": k}
    if spec == "auto":
        hi, lo = k, (k - 2 if k >= 6 else k - 1)
    else:
        hi, lo = (int(x) for x in spec.split(","))
    return {"ms": hi, "lo": lo, "n_hi": HEAD_ROWS_HI}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reference", help="Unsloth Dynamic GGUF of the model whose per-tensor types are followed")
    ap.add_argument("out", help="tier config to write (JSON)")
    ap.add_argument("--name", required=True, help="tier name, e.g. MS4")
    ap.add_argument("--model", required=True, help="model id written to the config, e.g. Qwen/Qwen3.8-27B")
    ap.add_argument("--table-from", help="GGUF whose token_embd.weight is copied (default: the reference file)")
    ap.add_argument("--head-split", default="auto",
                    help="auto (default): matched head width k for the 65,536 most frequent rows, k - 1 for the rest "
                         "(k - 2 when k = 6); none: one width; or HI,LO")
    ap.add_argument("--raise-to", type=int, default=0,
                    help="after matching, raise every MS tensor below this width to it; nothing else moves")
    ap.add_argument("--stats-precision", default="tf32", choices=["tf32", "bf16"],
                    help="GEMM precision of the FFN statistics (the released MS3.5 and MS5 used bf16)")
    a = ap.parse_args()
    T, n_layers = read_reference(a.reference)
    rows, head_k = match_widths(T, n_layers)

    layer_ks, default_ks = {}, {}
    for c in recipe.ATTN_CLASSES:
        lay = {r["layer"]: r["k"] for r in rows if KIND_CLASS[r["kind"]] == c}
        if lay:
            layer_ks[c] = lay
            default_ks[c] = collections.Counter(lay.values()).most_common(1)[0][0]   # most common width
    ffn_ks = {}
    for l in range(n_layers):
        g = {r["kind"]: r["k"] for r in rows if r["layer"] == l and r["kind"].startswith("ffn_")}
        if g:
            ffn_ks[l] = [g["ffn_gate"], g["ffn_up"], g["ffn_down"]]
    raised = []
    if a.raise_to:
        for l, ks in ffn_ks.items():
            for i, k in enumerate(ks):
                if k < a.raise_to:
                    raised.append(f"L{l}.{('ffn_gate', 'ffn_up', 'ffn_down')[i]} {k}->{a.raise_to}")
                    ks[i] = a.raise_to
        for c, lay in layer_ks.items():
            for l, k in lay.items():
                if k < a.raise_to:
                    raised.append(f"L{l}.{c} {k}->{a.raise_to}")
                    lay[l] = a.raise_to

    cfg = recipe.tier_config(a.name, a.model, 1 + max(ffn_ks), layer_ks, ffn_ks, default_ks, stats_chunk=8192,
                             stats_precision=a.stats_precision)
    if head_k is not None:
        cfg["out"] = head_split(a.head_split, head_k)
    cfg["copy_from"] = os.path.basename(a.reference)
    cfg["table_from"] = os.path.basename(a.table_from or a.reference)
    with open(a.out, "w") as f:
        json.dump(cfg, f, indent=1)

    comp = collections.OrderedDict()
    for r in rows:
        c = comp.setdefault(r["comp"], [0, 0, 0])
        c[0] += r["n"]
        c[1] += r["ub"]
        c[2] += r["mb"]
    print(f"{a.name}: {n_layers} layers matched to {os.path.basename(a.reference)}")
    for c, (n, ub, mb) in comp.items():
        print(f"  {c:16s} unsloth {ub * 8 / n:.4f} bpw  MS {mb * 8 / n:.4f} bpw  ({mb - ub:+,} bytes)")
    if head_k is not None:
        print(f"  head: matched k{head_k} -> {cfg['out']}")
    if a.raise_to:
        print(f"  raised to k{a.raise_to} ({len(raised)}): " + ", ".join(raised))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
