# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Matthew Priester
"""Write an MS GGUF from a converted tier (convert.py's output) and the model's BF16 GGUF.

usage:
  python write_gguf.py configs/qwen3.5-2b/MS4.json Qwen3.5-2B-BF16.gguf work/qwen3.5-2b out/Qwen3.5-2B-MS4.gguf
  python write_gguf.py configs/qwen3.8-27b/MS4.json Qwen3.8-27B-BF16.gguf work/qwen3.8-27b out/Qwen3.8-27B-MS4.gguf --gguf-dir unsloth/

The BF16 GGUF is the one llama.cpp's convert_hf_to_gguf.py makes from the same checkpoint. The file gets:
  - the BF16 file's metadata, with the MTP (next-token prediction) block dropped: block_count = the trunk layers,
    nextn_predict_layers removed, recurrent_layers trimmed; general.name = <model>-<tier>;
  - every MS-coded tensor as its MS_C<k> blocks (ggml types MS_C1..MS_C6 = 43..48 of the MS llama.cpp fork), in the BF16
    file's tensor order; a split table / head part without a BF16 counterpart (token_embd_lo, output_lo) after them;
  - every other tensor copied unchanged: from the BF16 file, or (configs with copy_from / table_from) from those GGUFs;
  - the rotation metadata (prism.hadamard.*: block sizes, sign vectors, which tensors), the split order
    (msffn.emb_split_order) and the MS keys (msffn.*). These key names are the format's metadata keys; the engine reads them.
  - Qwen3.5 models with more linear-attention value heads than key heads (the 27B): llama.cpp's converter stores the value heads
    tiled, so the value rows of attn_qkv and all rows of attn_gate are permuted the same way here (block rows move with their
    row); ssm_out stays in grouped order (it is rotated, its columns cannot be permuted) and prism.hadamard.gdn_v_grouped = true
    tells the engine to permute the activation instead.
The written file is read back and every tensor is compared with its source, byte for byte."""
import os, sys, re, glob, time, json, argparse
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from madshift import blocks
from madshift.linalg import had_signs

try:
    import gguf
except ImportError:
    raise SystemExit("gguf-py of the MS llama.cpp fork is needed: pip install <ms-llama.cpp checkout>/gguf-py")

QK_MS = 512
L_STATE = 14
PACK_TO_GGUF = {"G": "ffn_gate", "U": "ffn_up", "D": "ffn_down"}
EXTRA_PACK_KEYS = ["attn_qkv", "attn_gate", "ssm_out", "attn_q", "attn_k", "attn_v", "attn_output"]
AUTHOR = {
    "general.author": "Matthew Priester",
    "general.url": "https://github.com/devvexus/ms-llama.cpp",
    "general.description": "MS (Mad Shift) quantization by Matthew Priester (ORCID 0009-0000-0032-5292). "
                           "Runs on the MS llama.cpp fork: https://github.com/devvexus/ms-llama.cpp",
}


def ms_types():
    T = {k: getattr(gguf.GGMLQuantizationType, f"MS_C{k}", None) for k in range(1, 7)}
    if any(v is None or int(v) != 42 + k for k, v in T.items()):
        raise SystemExit("this gguf-py has no MS_C1..MS_C6 = 43..48: install the gguf-py of the MS llama.cpp fork")
    return T


def vhead_perm(n_k, n_v, d):
    """Row index: tiled row j comes from grouped row idx[j] (the value-head order of llama.cpp's converter)."""
    return torch.arange(n_v * d).reshape(n_k, n_v // n_k, d).permute(1, 0, 2).reshape(-1)


class MSInfo:
    """An MS tensor of the tier: where its entry is, its k, its block shape."""
    def __init__(self, tn, path, key, e, label, reorder=None):
        self.tn, self.path, self.key, self.label, self.reorder = tn, path, key, label, reorder
        if "blocks" not in e:
            raise ValueError(f"{label}: entry without block bytes")
        self.k = int(e["k"])
        if int(e.get("L", L_STATE)) != L_STATE or e.get("code") != "1mad" or self.k not in range(1, 7):
            raise ValueError(f"{label}: unsupported entry (k {self.k}, L {e.get('L')}, code {e.get('code')})")
        R_, C_ = (int(v) for v in e["shape"])
        sc = e["scale"]
        if C_ % QK_MS or sc.dtype != torch.float16 or tuple(sc.shape) != (R_,) or not torch.isfinite(sc.float()).all():
            raise ValueError(f"{label}: bad scale / shape")
        if tuple(e["blocks"].shape) != (R_, C_ // QK_MS * (4 + 64 * self.k)) or e["blocks"].dtype != torch.uint8:
            raise ValueError(f"{label}: blocks shape {tuple(e['blocks'].shape)}")
        self.rows, self.cols = R_, C_
        self.shape = (self.rows, self.cols // QK_MS * (4 + 64 * self.k))
        self.nbytes = self.shape[0] * self.shape[1]


def check_rot(rot, cols, name):
    if rot is None:
        return None
    seed, blk = int(rot["seed"]), int(rot["block"])
    if blk <= 0 or (blk & (blk - 1)) or cols % blk:
        raise ValueError(f"{name}: rotation block {blk} must be a power of two dividing the input width {cols}")
    return seed, blk


def load_pack(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config"); ap.add_argument("bf16_gguf"); ap.add_argument("work"); ap.add_argument("out_gguf")
    ap.add_argument("--gguf-dir", default=None, help="directory with the reference GGUF files named by copy_from / table_from")
    ap.add_argument("--general-name", default=None, help="general.name (default: <BF16 file's basename without -BF16>-<tier>)")
    ap.add_argument("--no-author", action="store_true", help="leave out general.author / general.url / general.description")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    name = cfg["name"]
    tier_dir = os.path.join(a.work, name)
    QT = ms_types()
    res = lambda f: (os.path.join(a.gguf_dir, f) if a.gguf_dir else f) if f else None
    embd_from, copy_from = res(cfg.get("table_from")), res(cfg.get("copy_from"))
    if os.path.lexists(a.out_gguf):
        sys.exit(f"{a.out_gguf} exists (never overwritten)")
    t0 = time.time()
    reader = gguf.GGUFReader(a.bf16_gguf, "r")
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    names = {t.name: t for t in reader.tensors}
    general_name = a.general_name
    if general_name is None:
        base = os.path.basename(a.bf16_gguf)
        general_name = re.sub(r"(-BF16)?\.gguf$", "", base, flags=re.I) + "-" + name
    donor_t = None
    if embd_from:
        donor = gguf.GGUFReader(embd_from, "r")
        donor_t = {t.name: t for t in donor.tensors}["token_embd.weight"]
        if tuple(donor_t.shape) != tuple(names["token_embd.weight"].shape):
            sys.exit(f"table_from: token_embd shape {tuple(donor_t.shape)} != BF16 {tuple(names['token_embd.weight'].shape)}")
    cdonor = gguf.GGUFReader(copy_from, "r") if copy_from else None
    ctens = {t.name: t for t in cdonor.tensors} if cdonor is not None else {}
    _f = lambda k_: reader.get_field(f"{arch}.{k_}")
    n_k = int(_f("ssm.group_count").contents()) if _f("ssm.group_count") is not None else 0
    n_v = int(_f("ssm.time_step_rank").contents()) if _f("ssm.time_step_rank") is not None else 0
    vr = bool(n_k and n_v != n_k)
    if vr:
        inner, dk = int(_f("ssm.inner_size").contents()), int(_f("ssm.state_size").contents())
        dv = inner // n_v
        vidx, qk_rows = vhead_perm(n_k, n_v, dv), 2 * n_k * dk

    # ---- the tier's entries --------------------------------------------------------------------------------------------
    pack_files = {int(re.fullmatch(r"L(\d+)\.pt", os.path.basename(p)).group(1)): p
                  for p in glob.glob(os.path.join(tier_dir, "L*.pt")) if re.fullmatch(r"L\d+\.pt", os.path.basename(p))}
    emb_file = os.path.join(tier_dir, "emb.pt")
    emb_file = emb_file if os.path.isfile(emb_file) else None
    if not pack_files:
        sys.exit(f"no L*.pt in {tier_dir}")
    mats, stat, kinfo, had_cfgs, dims, rots = {}, {"ffn": [0, 0, 0], "all": [0, 0, 0]}, {}, {}, {}, {}
    emb_split = None

    def add_entry(tn, path, key, e, label, cls, hadamard_cfg=None, src_name=None, n_rows=None, reorder=None):
        mi = MSInfo(tn, path, key, e, label, reorder)
        src_name = src_name or tn
        if src_name not in names:
            sys.exit(f"{label}: the GGUF has no tensor {src_name}")
        t = names[src_name]
        cols, rows = int(t.shape[0]), int(t.shape[1])
        if n_rows is not None:
            rows = n_rows
        if len(t.shape) != 2 or (mi.rows, mi.cols) != (rows, cols):
            sys.exit(f"{tn}: GGUF shape (ne) {tuple(t.shape)} does not match the entry {(mi.rows, mi.cols)}")
        if tn in mats:
            sys.exit(f"{tn}: given twice")
        mats[tn] = mi
        for g in (("ffn", "all") if cls == "ffn" else ("all",)):
            stat[g][0] += rows * cols
            stat[g][1] += rows * cols * mi.k
            stat[g][2] += mi.nbytes
        kinfo[tn] = mi.k
        r = check_rot(e.get("rot"), cols, label)
        if r is not None:
            if hadamard_cfg is not None:
                sys.exit(f"{label}: an FFN entry cannot carry 'rot'")
            rots[tn] = (r[0], r[1], cols)
        return cols

    for n in sorted(pack_files):
        pk = load_pack(pack_files[n])
        had_cfgs[n] = pk.get("hadamard")
        for key, base in PACK_TO_GGUF.items():
            if key in pk:
                dims[key] = add_entry(f"blk.{n}.{base}.weight", pack_files[n], key, pk[key], f"L{n}.{key}", "ffn", had_cfgs[n])
        for key in EXTRA_PACK_KEYS:
            if key in pk:
                ro = None
                if vr and key == "attn_qkv":
                    if blocks.entry_shape(pk[key])[0] != qk_rows + n_v * dv:
                        sys.exit(f"L{n} attn_qkv rows {blocks.entry_shape(pk[key])}")
                    ro = qk_rows
                if vr and key == "attn_gate":
                    if blocks.entry_shape(pk[key])[0] != n_v * dv:
                        sys.exit(f"L{n} attn_gate rows")
                    ro = 0
                add_entry(f"blk.{n}.{key}.weight", pack_files[n], key, pk[key], f"L{n}.{key}", "extra", reorder=ro)
        del pk
    tied = "output.weight" not in names
    head_split = None
    emb_order = None
    if emb_file:
        pk = load_pack(emb_file)
        if "token_embd" in pk and donor_t is not None:
            sys.exit("table_from with an MS-coded token_embd")
        V = int(names["token_embd.weight"].shape[1])
        if "emb_order" in pk:
            emb_order = pk["emb_order"].to(torch.int64).reshape(-1).numpy()
            if emb_order.shape != (V,) or not (np.sort(emb_order) == np.arange(V)).all():
                sys.exit("split: 'emb_order' is not a permutation of the vocabulary")

        def split_rows(key, label):
            R_ = int(blocks.entry_shape(pk[key])[0])
            if R_ <= 0 or R_ >= V or int(blocks.entry_shape(pk[key + "_lo"])[0]) != V - R_:
                sys.exit(f"split {label}: bad row counts")
            return R_

        if "output" in pk:                      # untied head: plain, or split (output + output_lo in the stored order)
            if tied:
                sys.exit("emb.pt has 'output' but the BF16 file has no output.weight (tied model)")
            if "output_lo" not in pk:
                add_entry("output.weight", emb_file, "output", pk["output"], "emb.output", "extra")
            else:
                R_h = split_rows("output", "head")
                if (pk["output"].get("rot") is None) != (pk["output_lo"].get("rot") is None):
                    sys.exit("split head: both halves rotated or both unrotated")
                add_entry("output.weight", emb_file, "output", pk["output"], "emb.output", "extra", n_rows=R_h)
                add_entry("output_lo.weight", emb_file, "output_lo", pk["output_lo"], "emb.output_lo", "extra",
                          src_name="output.weight", n_rows=V - R_h)
                head_split = R_h
        if "token_embd" in pk:                  # MS-coded table: plain, or split (one shared rotation)
            if "token_embd_lo" not in pk:
                add_entry("token_embd.weight", emb_file, "token_embd", pk["token_embd"], "emb.token_embd", "extra")
            else:
                hi, lo = pk["token_embd"], pk["token_embd_lo"]
                R = split_rows("token_embd", "table")
                rh, rl = hi.get("rot"), lo.get("rot")
                if (rh is None) != (rl is None) or (rh is not None and (int(rh["seed"]), int(rh["block"])) != (int(rl["seed"]), int(rl["block"]))):
                    sys.exit("split table: hi and lo must carry the same rotation")
                add_entry("token_embd.weight", emb_file, "token_embd", hi, "emb.token_embd", "extra", n_rows=R)
                lo2 = dict(lo); lo2["rot"] = None
                add_entry("token_embd_lo.weight", emb_file, "token_embd_lo", lo2, "emb.token_embd_lo", "extra",
                          src_name="token_embd.weight", n_rows=V - R)
                emb_split = R
        del pk
    n_weights, code_bits, stored_bytes = stat["ffn"]
    bpw = stored_bytes * 8 / n_weights if n_weights else 0.0
    code_bpw = code_bits / n_weights if n_weights else 0.0
    all_w, all_cb, all_by = stat["all"]

    # ---- metadata: the BF16 file's fields (MTP block dropped), in the BF16 file's order ------------------------------------
    w = gguf.GGUFWriter(a.out_gguf, arch=arch)
    _bc = reader.get_field(f"{arch}.block_count"); _nx = reader.get_field(f"{arch}.nextn_predict_layers")
    N_TRUNK = int(_bc.contents()) - (int(_nx.contents()) if _nx is not None else 0)
    skip_keys = {gguf.Keys.General.ARCHITECTURE, gguf.Keys.General.FILE_TYPE}
    for field in reader.fields.values():
        n = field.name
        if n in skip_keys or n.startswith("GGUF.") or n.endswith(".nextn_predict_layers"):
            continue
        if n == gguf.Keys.General.ALIGNMENT:
            w.add_custom_alignment(int(field.contents()))
            continue
        v = field.contents()
        if n.endswith(".block_count"):
            v = N_TRUNK
        if n.endswith(".attention.recurrent_layers"):
            v = list(v)[:N_TRUNK]
        if n == "general.name":
            v = general_name
        vt = field.types[0]
        w.add_key_value(n, v, vt, sub_type=field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)

    # ---- rotation metadata ------------------------------------------------------------------------------------------------
    ffn_cfgs = [c for n, c in had_cfgs.items() if any(f"blk.{n}.{b}.weight" in mats for b in PACK_TO_GGUF.values())]
    cfg0 = ffn_cfgs[0] if ffn_cfgs else None
    if any(c != cfg0 for c in ffn_cfgs):
        sys.exit("'hadamard' differs between the layer files")
    if cfg0 is not None:
        in_seed, in_block, h_seed, h_block = int(cfg0["in_seed"]), int(cfg0["in_block"]), int(cfg0["h_seed"]), int(cfg0["h_block"])
        for tn in mats:
            mt = re.fullmatch(r"blk\.\d+\.ffn_(gate|up|down)\.weight", tn)
            if mt:
                rots[tn] = ((in_seed, in_block, dims["G"]) if mt.group(1) != "down" else (h_seed, h_block, dims["D"]))
    rot_order = [t.name for t in reader.tensors if t.name in rots] + [tn for tn in mats if tn not in names and tn in rots]
    if rots:
        I8, I32, U32 = gguf.GGUFValueType.INT8, gguf.GGUFValueType.INT32, gguf.GGUFValueType.UINT32
        A = gguf.GGUFValueType.ARRAY
        emb_rot = rots.get("token_embd.weight")
        tied_rot = emb_rot is not None and tied        # tied: the rotated table is also the head (version 2 + tied_output)
        fwd_names = [tn for tn in rot_order if tn != "token_embd.weight"]
        sets, set_id = [], {}
        for tn in fwd_names + (["token_embd.weight"] if emb_rot else []):
            key = (rots[tn][0], rots[tn][2])
            if key not in set_id:
                set_id[key] = len(sets)
                sets.append(key)
        sign_vals = [int(v) for key in sets for v in had_signs(key[0], key[1]).to(torch.int8).tolist()]
        if cfg0 is not None:
            w.add_bool("msffn.hadamard", True)
            for pre, seed_, blk_, d_ in (("in", in_seed, in_block, dims["G"]), ("h", h_seed, h_block, dims["D"])):
                w.add_uint32(f"msffn.had_{pre}_block", blk_)
                w.add_uint32(f"msffn.had_{pre}_seed", seed_)
                w.add_key_value(f"msffn.had_{pre}_signs", had_signs(seed_, d_).to(torch.int8).tolist(), A, sub_type=I8)
        w.add_key_value("msffn.rot_weight_names", rot_order, A, sub_type=gguf.GGUFValueType.STRING)
        w.add_key_value("msffn.rot_seeds", [int(rots[tn][0]) for tn in rot_order], A, sub_type=I32)
        w.add_key_value("msffn.rot_blocks", [int(rots[tn][1]) for tn in rot_order], A, sub_type=U32)
        blocks_all = [rots[tn][1] for tn in rot_order]
        w.add_uint32("prism.hadamard.version", 2 if tied_rot else 1)
        if tied_rot:
            w.add_bool("prism.hadamard.tied_output", True)
        w.add_uint32("prism.hadamard.block_size", min(blocks_all))
        w.add_string("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard")
        w.add_string("prism.hadamard.axis", "input-last-dimension")
        w.add_string("prism.hadamard.sign_mode", "explicit")
        w.add_key_value("prism.hadamard.sign_widths", [k_[1] for k_ in sets], A, sub_type=I32)
        w.add_key_value("prism.hadamard.sign_values", sign_vals, A, sub_type=I32)
        if fwd_names:
            w.add_key_value("prism.hadamard.weight_names", fwd_names, A, sub_type=gguf.GGUFValueType.STRING)
            w.add_key_value("prism.hadamard.weight_block_sizes", [rots[tn][1] for tn in fwd_names], A, sub_type=U32)
            w.add_key_value("prism.hadamard.weight_sign_ids", [set_id[(rots[tn][0], rots[tn][2])] for tn in fwd_names], A, sub_type=U32)
        if emb_rot:
            w.add_key_value("prism.hadamard.inverse_weight_names", ["token_embd.weight"], A, sub_type=gguf.GGUFValueType.STRING)
            w.add_key_value("prism.hadamard.inverse_block_sizes", [emb_rot[1]], A, sub_type=U32)
            w.add_key_value("prism.hadamard.inverse_sign_ids", [set_id[(emb_rot[0], emb_rot[2])]], A, sub_type=U32)
    if vr:
        w.add_bool("prism.hadamard.gdn_v_grouped", True)
    if emb_split is not None:
        w.add_uint32("msffn.emb_split_rows", emb_split)
    if head_split is not None:
        w.add_uint32("msffn.head_split_rows", head_split)
    if emb_split is not None or head_split is not None:
        w.add_key_value("msffn.emb_split_order", [int(v) for v in emb_order], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.INT32)
    w.add_file_type(42)                                              # mixed MS types
    w.add_string("msffn.quant_name", name)
    w.add_string("msffn.code", "1mad")
    w.add_float32("msffn.bits_per_ffn_weight", float(bpw))
    w.add_float32("msffn.code_bits_per_ffn_weight", float(code_bpw))
    if all_w > n_weights:
        w.add_float32("msffn.bits_per_ms_weight", float(all_by * 8 / all_w))
        w.add_float32("msffn.code_bits_per_ms_weight", float(all_cb / all_w))
        w.add_uint32("msffn.ms_tensor_count", len(mats))
    w.add_string("msffn.nonffn_quant", "whole-MS")
    if donor_t is not None:
        w.add_string("msffn.token_embd_source", f"{os.path.basename(embd_from)} (copied unchanged, {donor_t.tensor_type.name})")
    if cdonor is not None:
        w.add_string("msffn.copy_source", f"{os.path.basename(copy_from)} (every non-MS tensor copied unchanged)")
    if not a.no_author:
        for k_, v_ in AUTHOR.items():
            w.add_string(k_, v_)

    # ---- tensor infos: the BF16 file's order, MTP blocks dropped, then MS tensors without a BF16 counterpart --------------
    plan = []
    for t in reader.tensors:
        if t.name.startswith("blk.") and int(t.name.split(".")[1]) >= N_TRUNK:
            continue
        if t.name in mats:
            plan.append(("ms", t.name, mats[t.name]))
        elif t.name == "token_embd.weight" and donor_t is not None:
            plan.append(("donor", t.name, donor_t))
        elif cdonor is not None:
            if t.name not in ctens:
                sys.exit(f"copy_from: {t.name} not in {copy_from}")
            ct = ctens[t.name]
            if tuple(int(x) for x in ct.shape) != tuple(int(x) for x in t.shape):
                sys.exit(f"copy_from: {t.name} shape {tuple(ct.shape)} != BF16 {tuple(t.shape)}")
            plan.append(("donor", t.name, ct))
        else:
            plan.append(("copy", t.name, t))
    for tn, mi in mats.items():
        if tn not in names:
            plan.append(("ms", tn, mi))
    for kind, tn, o in plan:
        if kind == "ms":
            w.add_tensor_info(tn, o.shape, np.dtype(np.uint8), o.nbytes, QT[o.k])
        else:
            w.add_tensor_info(tn, o.data.shape, o.data.dtype, o.data.nbytes, o.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()

    # ---- tensor data: one layer file open at a time --------------------------------------------------------------------
    cache = {"path": None, "pk": None}

    def entry_of(mi):
        if cache["path"] != mi.path or cache["pk"] is None:
            cache["pk"] = None
            cache["pk"], cache["path"] = load_pack(mi.path), mi.path
        e = cache["pk"][mi.key]
        if mi.reorder is not None:                     # value-head reorder: a row's blocks depend only on that row
            out = dict(e)
            for key in ("blocks", "scale"):
                t_ = e[key].clone(); r0 = mi.reorder
                t_[r0:r0 + len(vidx)] = e[key][r0:r0 + len(vidx)][vidx]
                out[key] = t_
            e = out
        return e

    for kind, tn, o in plan:
        if kind == "ms":
            e = entry_of(o)
            b = e["blocks"].numpy()
            if b.shape != o.shape:
                sys.exit(f"{tn}: block shape {b.shape} != planned {o.shape}")
            w.write_tensor_data(b)
            del b, e
        else:
            w.write_tensor_data(o.data)
    w.close()
    cache["pk"] = None
    size = os.path.getsize(a.out_gguf)
    print(f"wrote {a.out_gguf}: {size} bytes ({size / 1e9:.3f} GB) in {time.time() - t0:.0f}s; {len(mats)} MS tensors, "
          f"{sum(1 for p in plan if p[0] != 'ms')} copied; MTP blocks >= {N_TRUNK} dropped; value-head reorder {'yes' if vr else 'no'}")
    print(f"  FFN bits/weight (stored) {bpw:.6f}; all MS tensors {all_by * 8 / max(all_w, 1):.6f} bits/weight")

    # ---- verify -----------------------------------------------------------------------------------------------------
    if not a.no_verify:
        t1 = time.time()
        r2 = gguf.GGUFReader(a.out_gguf, "r")
        t2 = {t.name: t for t in r2.tensors}
        bad = 0
        if [t.name for t in r2.tensors] != [p[1] for p in plan]:
            print("  VERIFY FAIL tensor order"); bad += 1
        for kind, tn, o in plan:
            got = t2[tn]
            if kind == "ms":
                if got.tensor_type != QT[o.k]:
                    print(f"  VERIFY FAIL {tn} type"); bad += 1; continue
                e = entry_of(o)                          # (the blocks were decode-checked against the states when saved)
                ok = np.array_equal(np.asarray(got.data), e["blocks"].numpy())
                if not ok:
                    print(f"  VERIFY FAIL {tn}"); bad += 1
                del e
            else:
                g = np.asarray(got.data).reshape(-1).view(np.uint8)
                s = np.asarray(o.data).reshape(-1).view(np.uint8)
                step = 1 << 28
                same = g.shape == s.shape and all(np.array_equal(g[j:j + step], s[j:j + step]) for j in range(0, g.shape[0], step))
                if not same or got.tensor_type != o.tensor_type:
                    print(f"  VERIFY FAIL {tn} ({kind})"); bad += 1
        cache["pk"] = None
        del r2, t2
        print(f"  verify ({len(plan)} tensors re-read and compared): {'OK' if bad == 0 else f'{bad} FAILED'} ({time.time() - t1:.0f}s)")
        if bad:
            sys.exit(1)


if __name__ == "__main__":
    main()
