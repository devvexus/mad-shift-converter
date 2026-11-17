# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Matthew Priester
"""Tensors an MS file copies unchanged from a reference GGUF (the Qwen3.8-27B tiers: unsloth's GGUF of the same size class).

Every tensor MS does not code (norms, conv, ssm_a / dt, ssm_alpha / ssm_beta, and the input token table) is copied into the MS
file unchanged. The converter uses the same values while it converts: after a layer's original weights are loaded into the
converted model's layer module, every reference tensor of that layer that
  - maps to a module parameter (GGUF2HF below),
  - is not MS-coded by this tier in this layer, and
  - is stored in a lossy type (anything but F32 / BF16)
is dequantized (gguf-py) and written into the module, so the later layers' conversion absorbs its error. Rows are matched to
the checkpoint's rows: identity if the dequantized tensor is close to the checkpoint's; for small matrices (<= 1024 rows, e.g.
ssm_alpha / ssm_beta, whose value heads llama.cpp reorders) a row permutation by nearest cosine is used if it is a bijection.
The input table works the same way (input_table()): the converted model's streams start from the dequantized reference table."""
import numpy as np
import torch

GGUF2HF = {
    "attn_qkv": "linear_attn.in_proj_qkv.weight", "attn_gate": "linear_attn.in_proj_z.weight",
    "ssm_out": "linear_attn.out_proj.weight", "ssm_alpha": "linear_attn.in_proj_a.weight", "ssm_beta": "linear_attn.in_proj_b.weight",
    "attn_q": "self_attn.q_proj.weight", "attn_k": "self_attn.k_proj.weight", "attn_v": "self_attn.v_proj.weight",
    "attn_output": "self_attn.o_proj.weight",
    "ffn_gate": "mlp.gate_proj.weight", "ffn_up": "mlp.up_proj.weight", "ffn_down": "mlp.down_proj.weight",
}
MS_SPEC = {"attn_qkv": "qkv", "attn_gate": "z", "ssm_out": "sout", "attn_q": "q", "attn_k": "k", "attn_v": "v", "attn_output": "o"}
LOSSLESS = ("F32", "BF16")


def _gguf():
    import gguf
    return gguf


class Reference:
    _readers = {}

    def __init__(self, path):
        gguf = _gguf()
        if path not in Reference._readers:
            Reference._readers[path] = gguf.GGUFReader(path, "r")
        self.r = Reference._readers[path]
        self.path = path
        self.T = {t.name: t for t in self.r.tensors}
        self.cache = {}

    def dequant(self, name):
        gguf = _gguf()
        t = self.T[name]
        x = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def coded_kinds(cfg, l, spec_of):
    """GGUF tensor kinds this tier MS-codes in layer l."""
    out = set()
    for kind, c in MS_SPEC.items():
        if spec_of(cfg, c, l) is not None:
            out.add(kind)
    if cfg.get("ffn") is not None and str(l) in cfg["ffn"].get("k_layer", {}):
        out |= {"ffn_gate", "ffn_up", "ffn_down"}
    return out


def match_rows(Q, W):
    """Q (reference, dequantized) vs W (checkpoint), same shape: identity if close, else a row permutation (small matrices)."""
    rel = float((Q - W).norm() / W.norm())
    if rel < 0.25:
        return None, rel
    if Q.shape[0] > 1024:
        raise SystemExit(f"reference tensor differs from the checkpoint's (rel {rel:.3f}) and is too large for a row search")
    qn = Q / Q.norm(dim=1, keepdim=True).clamp_min(1e-12)
    wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
    perm = (qn @ wn.T).argmax(1)                         # reference row i = checkpoint row perm[i]
    if len(set(perm.tolist())) != Q.shape[0]:
        raise SystemExit(f"reference tensor: no row bijection to the checkpoint (rel {rel:.3f})")
    inv = torch.empty_like(perm); inv[perm] = torch.arange(len(perm), device=perm.device)
    rel2 = float((Q[inv] - W).norm() / W.norm())
    if rel2 > 0.25:
        raise SystemExit(f"reference tensor: row-matched rel {rel2:.3f} still large")
    return inv, rel2


@torch.no_grad()
def apply(t, l, Lt, spec_of, log, logged):
    """Write the reference GGUF's dequantized copies of layer l's non-MS tensors into the tier's layer module Lt."""
    src = t.copy_from
    if not src:
        return []
    D = getattr(t, "_ref", None)
    if D is None:
        D = t._ref = Reference(src)
    coded = coded_kinds(t.cfg, l, spec_of)
    params = dict(Lt.named_parameters())
    done = []
    pre = f"blk.{l}."
    for name, tt in D.T.items():
        if not name.startswith(pre):
            continue
        kind = name[len(pre):].rsplit(".", 1)[0] if name.endswith(".weight") or name.endswith(".bias") else name[len(pre):]
        ty = tt.tensor_type.name
        if kind in coded or ty in LOSSLESS:
            continue
        if kind not in GGUF2HF:
            raise SystemExit(f"reference GGUF {src}: {name} is {ty} (lossy) and has no module mapping")
        p = params[GGUF2HF[kind]]
        key = (src, name)
        if key not in D.cache:
            Q = D.dequant(name)
            if tuple(Q.shape) != tuple(p.shape):
                raise SystemExit(f"{name}: reference shape {tuple(Q.shape)} != module {tuple(p.shape)}")
            inv, rel = match_rows(Q.to(p.device), p.float())
            if inv is not None:
                Q = Q[inv.cpu()]
            D.cache = {k: v for k, v in D.cache.items() if k[1].startswith(pre)}      # keep this layer only
            D.cache[key] = (Q, rel, inv is not None)
        Q, rel, permuted = D.cache[key]
        p.copy_(Q.to(device=p.device, dtype=p.dtype))
        done.append((name, ty, rel, permuted))
    if done and (l, t.name) not in logged:
        logged.add((l, t.name))
        log(f"  {t.name} L{l}: reference tensors used in the stream: " +
            ", ".join(f"{n.split('.', 2)[2]} {ty} rel {rel:.2e}{' (rows permuted)' if pm else ''}" for n, ty, rel, pm in done))
    return done


def input_table(path, W, dev, log):
    """The reference file's token_embd.weight dequantized (V, d) fp32 on dev, checked against the checkpoint's table W."""
    gguf = _gguf()
    r = gguf.GGUFReader(path, "r")
    tt = next(x for x in r.tensors if x.name == "token_embd.weight")
    V, d = int(tt.shape[1]), int(tt.shape[0])
    if (V, d) != tuple(W.shape):
        raise SystemExit(f"table GGUF {path}: shape {(V, d)} != the checkpoint's table {tuple(W.shape)}")
    Q = torch.empty(V, d, dtype=torch.float32, device=dev)
    data = tt.data
    for r0 in range(0, V, 8192):
        x = gguf.quants.dequantize(np.asarray(data[r0:r0 + 8192]), tt.tensor_type)
        Q[r0:r0 + x.shape[0]] = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev)
    rel = float((Q - W).norm() / W.norm())
    log(f"  input table: token_embd.weight of {path} ({tt.tensor_type.name}), dequantized; relative difference to the "
        f"checkpoint's table {rel:.4f}")
    return Q
