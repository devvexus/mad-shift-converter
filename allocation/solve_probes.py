# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Matthew Priester
"""Bit allocation for an MS tier at a fixed file size, from probe measurements (the method used for Qwen3.5-2B).

A probe is one conversion in which a single tensor class (for example the linear-attention qkv projection of every layer) is
MS-coded at one width k while the rest of the model stays BF16; its result is the KL divergence of the converted model's
next-token distribution from the original's. The probe file holds these measurements. From them the solver builds one curve
per class (log2 KL piecewise linear in bits per weight) and hands out bits greedily: every unit (one layer's tensor of a class,
one FFN matrix of one layer, the embedding table or one of its two row groups) starts at --kmin, and each further bit goes to
the unit with the largest predicted KL drop per byte until the file size is reached. The model's shapes, vocabulary, layer
types and the bytes outside the coded tensors come from its BF16 GGUF.

usage: python solve_probes.py <bf16.gguf> <probes.json> <out.json> --size-mb 1263 --name MS4 [--emb split64kmix] [--kmin 2]
"""
import argparse
import json
import math
import re

import gguf

import recipe

SCALE_BITS = 0.0625     # an MS_Ck block stores 512 weights in 64 k + 4 bytes: k + 0.0625 bits per weight
MS_METADATA = 16_480    # MS header data other than the rotation signs, in bytes
CODED = re.compile(r"(token_embd|output|blk\.\d+\.(attn_qkv|attn_gate|ssm_out|attn_q|attn_k|attn_v|attn_output|"
                   r"ffn_gate|ffn_up|ffn_down))\.weight$")


def model_info(path):
    """Constants of the model in a BF16 GGUF: trunk layer count, hidden size, vocabulary, (out, in) of the first tensor of
    each kind, linear- and full-attention layers, and the bytes of the finished MS file outside its MS-coded tensors.

    Those bytes are the GGUF header, every tensor that stays unquantized (norms, conv, ssm parameters; the MTP block is
    dropped), 4 bytes of rotation sign per input feature of each rotated tensor family (embedding, each layer's attention
    read and output projections, FFN input and down input) and MS_METADATA."""
    r = gguf.GGUFReader(path, "r")
    arch = r.get_field("general.architecture").contents()
    nextn = r.get_field(f"{arch}.nextn_predict_layers")
    trunk = int(r.get_field(f"{arch}.block_count").contents()) - (int(nextn.contents()) if nextn is not None else 0)
    hidden = int(r.get_field(f"{arch}.embedding_length").contents())
    inter = int(r.get_field(f"{arch}.feed_forward_length").contents())
    T = {t.name: t for t in r.tensors}

    def is_mtp(name):
        return name.startswith("blk.") and int(name.split(".")[1]) >= trunk

    small = sum(int(t.n_bytes) for t in r.tensors if not CODED.match(t.name) and not is_mtp(t.name))
    out_in = 0
    for l in range(trunk):
        t = T.get(f"blk.{l}.ssm_out.weight")
        if t is None:
            t = T[f"blk.{l}.attn_output.weight"]
        out_in += int(t.shape[0])
    signs = hidden + trunk * hidden + out_in + hidden + inter
    shapes = {}
    for l in range(trunk):
        for s in ("attn_qkv", "attn_gate", "ssm_out", "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate"):
            t = T.get(f"blk.{l}.{s}.weight")
            if t is not None and s not in shapes:
                shapes[s] = (int(t.shape[1]), int(t.shape[0]))          # GGUF shape = (in, out)
    lin = [l for l in range(trunk) if f"blk.{l}.ssm_out.weight" in T]
    return dict(arch=arch, trunk=trunk, hidden=hidden, V=int(T["token_embd.weight"].shape[1]),
                overhead=r.data_offset + small + 4 * signs + MS_METADATA, shapes=shapes, lin=lin,
                full=[l for l in range(trunk) if l not in lin])


def model_units(m):
    """Weights per unit (N) and the units of each class, outer layers first (last, first, second to last, second, ...)."""
    nl, s, lin, full = m["trunk"], m["shapes"], m["lin"], m["full"]
    edge = sorted(range(nl), key=lambda l: min(l, nl - 1 - l) - (0.5 if l > (nl - 1) // 2 else 0))

    def cnt(t):
        return s[t][0] * s[t][1]

    N, units = dict(emb=m["V"] * m["hidden"]), dict(emb=[0])
    if lin:
        N.update(qkv=cnt("attn_qkv"), z=cnt("attn_gate"), sout=cnt("ssm_out"))
        units.update({c: [l for l in edge if l in lin] for c in ("qkv", "z", "sout")})
    if full:
        N.update(qk=cnt("attn_q") + cnt("attn_k"), v=cnt("attn_v"), o=cnt("attn_output"))
        units.update({c: [l for l in edge if l in full] for c in ("qk", "v", "o")})
    N["ffn"] = cnt("ffn_gate")
    units["ffn"] = [(l, mat) for mat in (2, 1, 0) for l in edge]          # down, up, gate
    return N, units


def load_probes(path):
    """Probe file -> (model id, points). points[class] = [(bits per weight, KL), ...]; a split-embedding probe is stored
    under its name's split key (e.g. "split64kmix") as dict(rows, hi, lo, kl).

    Probe names: "probe <class> k<k>" or "probe emb split<N>k[mix] <hi>/<lo>" (N x 1024 frequent rows at k = hi, the rest
    at lo). A probe's k is its measured bits rounded; its KL is the mean over its evaluation sets. A probe listed twice
    counts once (the last entry)."""
    d = json.load(open(path))
    last = {}
    for p in d["probes"]:
        last[p["name"]] = p
    pts = {}
    for p in last.values():
        kl = sum(p["kl"].values()) / len(p["kl"])
        w = p["name"].split()
        if "split" in p["name"]:
            hi, lo = map(int, w[3].split("/"))
            rows = int(re.search(r"split(\d+)k", w[2]).group(1)) * 1024
            pts[w[2]] = dict(rows=rows, hi=hi, lo=lo, kl=kl)
        else:
            k = round(p["bits_by_class"]["q" if w[1] == "qk" else w[1]])
            pts.setdefault(w[1], []).append((k + SCALE_BITS, kl))
    return d["model"], pts


def curve_fn(points, slope):
    """KL as a function of bits per weight: log2 KL piecewise linear between the points, extrapolated with the end
    segments; a class with one point falls by half every `slope` bits."""
    p = sorted(points)

    def f(b):
        if len(p) == 1:
            return p[0][1] * 2 ** (-(b - p[0][0]) / slope)
        i = 0 if b < p[0][0] else (len(p) - 2 if b >= p[-1][0] else max(j for j in range(len(p) - 1) if p[j][0] <= b))
        (b0, k0), (b1, k1) = p[i], p[i + 1]
        return 2 ** (math.log2(k0) + (b - b0) / (b1 - b0) * (math.log2(k1) - math.log2(k0)))
    return f


def solve(m, pts, size_mb, emb="plain", kmin=2, kmax=6):
    """Greedy allocation. Returns ({class: {unit: k}}, report lines)."""
    N, units = model_units(m)
    overhead = m["overhead"]
    two = [c for c in pts if isinstance(pts[c], list) and len(pts[c]) >= 2 and c != "ffn"]
    if not two:
        raise SystemExit("need two probes of at least one class other than the FFN (for the mean slope)")
    slopes = [(max(pts[c])[0] - min(pts[c])[0]) / math.log2(min(pts[c])[1] / max(pts[c])[1]) for c in two]
    s = sum(slopes) / len(slopes)                                        # mean bits per halving of KL
    curves = {c: curve_fn(pts[c], s) for c in N if isinstance(pts.get(c), list)}
    n, units = dict(N), dict(units)
    lines = []
    if emb != "plain":
        # Split embedding: the frequent rows follow the plain-table curve; the rare rows add the measured penalty
        # KL(split) - KL(plain at the same k_hi), which halves every s bits.
        sp = pts[emb]
        plain = curves.pop("emb")
        units.pop("emb")
        n.pop("emb")
        pen = sp["kl"] - plain(sp["hi"] + SCALE_BITS)
        lines.append(f"split {emb}: KL {sp['kl']:.5f} vs plain k{sp['hi']} {plain(sp['hi'] + SCALE_BITS):.5f} "
                     f"-> rare-row penalty {pen:.5f} at k{sp['lo']}")
        curves["emb_hi"] = plain
        curves["emb_lo"] = lambda b, p0=max(pen, 1e-6), b0=sp["lo"] + SCALE_BITS: p0 * 2 ** (-(b - b0) / s)
        n["emb_hi"], n["emb_lo"] = sp["rows"] * m["hidden"], (m["V"] - sp["rows"]) * m["hidden"]
        units["emb_hi"], units["emb_lo"] = [0], [0]
        overhead += m["V"] * 4                                           # row order of the split table
    missing = [c for c in N if c not in curves and not (c == "emb" and emb != "plain")]
    if missing:
        raise SystemExit(f"missing probes: {missing}")
    k = {(c, u): kmin for c in curves for u in units[c]}
    budget = (size_mb * 1e6 - overhead) * 8                              # bits for the MS-coded tensors
    used = sum(n[c] * (kk + SCALE_BITS) for (c, u), kk in k.items())

    def share(c, kk):
        return curves[c](kk + SCALE_BITS) / len(units[c])

    while True:
        best = None
        for (c, u), kk in k.items():
            if kk >= kmax or used + n[c] > budget:
                continue
            g = (share(c, kk) - share(c, kk + 1)) / n[c]                 # predicted KL drop per bit of file
            if best is None or g > best[0]:
                best = (g, c, u)
        if best is None:
            break
        k[(best[1], best[2])] += 1
        used += n[best[1]]
    lines.append(f"file {size_mb:.1f} MB: coded-tensor budget {budget / 8e6:.1f} MB, used {used / 8e6:.1f} MB; "
                 f"mean slope {s:.3f} bits per KL halving")
    out, tot = {}, 0.0
    for c in curves:
        ks = [k[(c, u)] for u in units[c]]
        kl = sum(share(c, kk) for kk in ks)
        tot += kl
        avg = sum(ks) / len(ks) + SCALE_BITS
        out[c] = dict(zip(units[c], ks))
        lines.append(f"  {c:6s} {avg:5.2f} bpw  k {sorted(set(ks))}  {n[c] * len(ks) * avg / 8e6:7.1f} MB  "
                     f"predicted KL {kl:.5f}")
    lines.append(f"predicted KL (sum of the classes measured one at a time): {tot:.5f}")
    return out, lines


def to_config(name, model, m, out, pts, emb):
    layer_ks, default_ks = {}, {}
    for c, names in [("qkv", ["qkv"]), ("z", ["z"]), ("sout", ["sout"]), ("qk", ["q", "k"]), ("v", ["v"]), ("o", ["o"])]:
        if c in out:
            for cc in names:
                layer_ks[cc] = dict(out[c])
                default_ks[cc] = max(out[c].values())
    ffn_ks = {l: [0, 0, 0] for l in range(m["trunk"])}
    for (l, mat), kk in out["ffn"].items():
        ffn_ks[l][mat] = kk
    cfg = recipe.tier_config(name, model, m["trunk"], layer_ks, ffn_ks, default_ks)
    if "emb" in out:
        cfg["emb"] = {"ms": out["emb"][0]}
    else:
        cfg["emb"] = {"ms": out["emb_hi"][0], "lo": out["emb_lo"][0], "split": pts[emb]["rows"],
                      **({"rank": "mix"} if "mix" in emb else {})}
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gguf", help="BF16 GGUF of the model")
    ap.add_argument("probes", help="probe measurements (JSON)")
    ap.add_argument("out", help="tier config to write (JSON)")
    ap.add_argument("--size-mb", type=float, required=True, help="target file size in decimal MB (MTP block not included)")
    ap.add_argument("--name", required=True, help="tier name, e.g. MS4")
    ap.add_argument("--emb", default="plain",
                    help="plain, or the split-embedding probe to use (e.g. split64kmix: frequent and rare rows get "
                         "separate widths)")
    ap.add_argument("--kmin", type=int, default=2, help="lowest width (default 2)")
    ap.add_argument("--kmax", type=int, default=6, help="highest width (default 6, the widest MS type)")
    a = ap.parse_args()
    m = model_info(a.gguf)
    print(f"{m['arch']}: {m['trunk']} layers, hidden {m['hidden']}, vocabulary {m['V']}, "
          f"bytes outside the coded tensors {m['overhead']:,}")
    model, pts = load_probes(a.probes)
    out, lines = solve(m, pts, a.size_mb, a.emb, a.kmin, a.kmax)
    print("\n".join(lines))
    cfg = to_config(a.name, model, m, out, pts, a.emb)
    with open(a.out, "w") as f:
        json.dump(cfg, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
