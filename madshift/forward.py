# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Matthew Priester
"""Decoder-layer forwards over a residual stream held on the GPU.

A stream is (N, d) bf16 for N = n_seq * T tokens. Layers run in batches of `batch` sequences of T tokens with the Hugging Face
Qwen3.5 modules (masks, rotary embeddings and keyword arguments as the model's own forward), so every value is what a forward
of the model gives. Pieces:
  core   input of the attention output projection (out_proj / o_proj): the attention module with its output projection swapped
         for the identity
  mid = x + out_proj(core),  n = post_attention_layernorm(mid),  out = mid + mlp(n)   (the ops of Qwen3_5DecoderLayer.forward)
Optionally each piece is replayed from a CUDA graph (same kernels)."""
import sys
import torch

GRAPHS = False                                        # replay the per-batch layer pieces as CUDA graphs
FAST_GDN = True                                       # linear-attention core with a batched triangular solve (madshift.gdn)


class Geometry:
    """Masks and rotary embeddings for batches of `nb` sequences of length T (computed once per shape)."""

    def __init__(self, model, rotary, T, dev="cuda"):
        self.m, self.rot, self.T, self.dev = model, rotary, T, dev
        self.cache = {}

    def get(self, x):
        key = (x.shape[0], x.dtype)
        if key not in self.cache:
            mm = sys.modules[self.m.mm().__name__]
            T = self.T
            pos = torch.arange(T, device=x.device).view(1, 1, -1).expand(4, x.shape[0], -1)
            tpos, pos = pos[0], pos[1:]
            mk = dict(config=self.m.tc, inputs_embeds=x, attention_mask=None, past_key_values=None, position_ids=tpos)
            masks = {"full_attention": mm.create_causal_mask(**mk), "linear_attention": mm.create_recurrent_attention_mask(**mk)}
            pe = self.rot(x, pos)
            self.cache[key] = (tpos, masks, pe)
        return self.cache[key]


def proj_of(L):
    """The attention output projection module of a decoder layer."""
    return L.linear_attn.out_proj if hasattr(L, "linear_attn") else L.self_attn.o_proj


class _Id(torch.nn.Module):
    def forward(self, x):
        return x


_ID = _Id()


def core_direct(L, x, geo, ltype):
    """Input of the attention output projection for the batch x (nb, T, d), bf16."""
    tpos, masks, pe = geo.get(x)
    h = L.input_layernorm(x)
    if ltype == "linear_attention" and FAST_GDN:
        from madshift import gdn
        return gdn.gdn_core(L.linear_attn, h)
    if ltype == "linear_attention":
        A = L.linear_attn; op = A.out_proj; A.out_proj = _ID
        try:
            return A(hidden_states=h, cache_params=None, attention_mask=masks[ltype], use_cache=False)
        finally:
            A.out_proj = op
    A = L.self_attn; op = A.o_proj; A.o_proj = _ID
    try:
        return A(hidden_states=h, attention_mask=masks[ltype], position_ids=tpos, past_key_values=None,
                 position_embeddings=pe, use_cache=False)[0]
    finally:
        A.o_proj = op


def _k_core(L, geo, ty):
    return lambda x: (core_direct(L, x, geo, ty),)


def _k_full(L, geo, ty):
    def f(x):
        core = core_direct(L, x, geo, ty)
        mid = x + proj_of(L)(core)
        return mid + L.mlp(L.post_attention_layernorm(mid)), mid, core
    return f


def _k_mid1(L, geo, ty):
    return lambda x: (x + proj_of(L)(core_direct(L, x, geo, ty)),)


def _k_mid1_core(L, geo, ty):
    return lambda x, core: (x + proj_of(L)(core),)


def _k_adv_mid1(L, geo, ty):
    return lambda mid: (mid + L.mlp(L.post_attention_layernorm(mid)),)


def _k_adv_core(L, geo, ty):
    def f(x, core):
        mid = x + proj_of(L)(core)
        return (mid + L.mlp(L.post_attention_layernorm(mid)),)
    return f


def _k_norm2(L, geo, ty):
    return lambda mid: (L.post_attention_layernorm(mid),)


KINDS = dict(core=_k_core, full=_k_full, mid1=_k_mid1, mid1_core=_k_mid1_core, adv_mid1=_k_adv_mid1, adv_core=_k_adv_core,
             norm2=_k_norm2)
_GRAPHS = {}


class _Graph:
    def __init__(self, fn, inputs):
        self.inp = [x.clone() for x in inputs]
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fn(*self.inp)
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self.out = fn(*self.inp)

    def __call__(self, *inputs):
        for a, x in zip(self.inp, inputs):
            a.copy_(x)
        self.g.replay()
        return self.out


def piece(kind, L, geo, ty, *inputs):
    """One layer piece on one batch: eager, or replayed from a CUDA graph captured for (module, kind, TF32 flag, shapes)."""
    if not GRAPHS:
        return KINDS[kind](L, geo, ty)(*inputs)
    key = (id(L), kind, ty, bool(torch.backends.cuda.matmul.allow_tf32), FAST_GDN, tuple(tuple(x.shape) for x in inputs))
    g = _GRAPHS.get(key)
    if g is None:
        g = _GRAPHS[key] = _Graph(KINDS[kind](L, geo, ty), inputs)
    return g(*inputs)


def batches(N, T, batch):
    rows = batch * T
    for r0 in range(0, N, rows):
        yield r0, min(N, r0 + rows)


def view(h, r0, r1, T):
    return h[r0:r1].view(-1, T, h.shape[-1])


@torch.no_grad()
def layer_full(L, h, geo, ltype, T, batch, mid_out=None):
    """h <- layer(h) in place. Optionally stores the post-attention residual (mid_out, (N, d)) of every token."""
    for r0, r1 in batches(h.shape[0], T, batch):
        out, mid, core = piece("full", L, geo, ltype, view(h, r0, r1, T))
        if mid_out is not None:
            mid_out[r0:r1] = mid.reshape(r1 - r0, -1)
        h[r0:r1] = out.reshape(r1 - r0, -1)


@torch.no_grad()
def cores(L, h, geo, ltype, T, batch, out):
    """out[rows] = projection input for every token of stream h (no write to h)."""
    for r0, r1 in batches(h.shape[0], T, batch):
        out[r0:r1] = piece("core", L, geo, ltype, view(h, r0, r1, T))[0].reshape(r1 - r0, -1)


@torch.no_grad()
def advance_from_core(L, h, core, T, batch):
    """h <- (h + out_proj(core)) + mlp(post_norm(.)) in place, with the stored projection input `core`."""
    for r0, r1 in batches(h.shape[0], T, batch):
        x = view(h, r0, r1, T)
        out = piece("adv_core", L, None, None, x, core[r0:r1].view(x.shape[0], T, -1))[0]
        h[r0:r1] = out.reshape(r1 - r0, -1)


@torch.no_grad()
def embed_rows(table_bf16, ids, batch=64):
    """Embedding lookup of ids (n_seq, T) -> (N, d) bf16 (batches of 64 sequences; a gather is exact)."""
    d = table_bf16.shape[1]
    return torch.cat([torch.nn.functional.embedding(ids[b:b + batch].to(table_bf16.device), table_bf16).reshape(-1, d)
                      for b in range(0, ids.shape[0], batch)])


@torch.no_grad()
def mids_inplace(L, h, geo, ltype, T, batch, core=None):
    """h <- post-attention residual of layer L (from the stored projection input `core` when given), in place."""
    for r0, r1 in batches(h.shape[0], T, batch):
        x = view(h, r0, r1, T)
        if core is None:
            mid = piece("mid1", L, geo, ltype, x)[0]
        else:
            mid = piece("mid1_core", L, None, None, x, core[r0:r1].view(x.shape[0], T, -1))[0]
        h[r0:r1] = mid.reshape(r1 - r0, -1)


@torch.no_grad()
def norm_rows(L, mid, i0, i1, T, batch):
    """post_attention_layernorm of residual rows [i0, i1), computed per batch of `batch` sequences."""
    out = torch.empty(i1 - i0, mid.shape[1], dtype=mid.dtype, device=mid.device)
    for r0, r1 in batches(i1 - i0, T, batch):
        out[r0:r1] = piece("norm2", L, None, None, view(mid, i0 + r0, i0 + r1, T))[0].reshape(r1 - r0, -1)
    return out


@torch.no_grad()
def advance_from_mid1(L, h, T, batch):
    """h holds the post-attention residual; h <- mid + mlp(post_attention_layernorm(mid)) in place."""
    for r0, r1 in batches(h.shape[0], T, batch):
        out = piece("adv_mid1", L, None, None, view(h, r0, r1, T))[0]
        h[r0:r1] = out.reshape(r1 - r0, -1)
