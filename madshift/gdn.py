# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Matthew Priester
#
# Derived from torch_chunk_gated_delta_rule and Qwen3_5GatedDeltaNet.forward in Hugging Face transformers
# (src/transformers/models/qwen3_5/modeling_qwen3_5.py), Copyright 2025 The Qwen Team and The HuggingFace Inc. team,
# licensed under the Apache License, Version 2.0. Changes: the intra-chunk forward substitution is one batched
# unit-lower-triangular solve; no cache, no padding mask; the function returns the input of the output projection.
"""Gated DeltaNet core of the Qwen3.5 linear-attention layers (the forward up to the output projection), the math of
transformers' torch fallback with one change: the intra-chunk forward substitution (63 sequential row updates) is one batched
unit-lower-triangular solve, attn = (I - A)^-1, which is what the loop computes. Rounding-level differences only."""
import torch
import torch.nn.functional as F


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def chunk_gated_delta_rule_fast(query, key, value, g, beta, chunk_size=64):
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size)); key = F.pad(key, (0, 0, 0, pad_size)); value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size)); g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1] ** 0.5))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
                                         for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    eye = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    attn = torch.linalg.solve_triangular(eye - attn, eye.expand_as(attn), upper=False, unitriangular=True)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    S = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
    out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        a_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_new = v_i - k_cumdecay[:, :, i] @ S
        out[:, :, i] = (q_i * g[:, :, i, :, None].exp()) @ S + a_i @ v_new
        S = S * g[:, :, i, -1, None, None].exp() + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
    out = out.reshape(out.shape[0], out.shape[1], -1, out.shape[-1])[:, :, :sequence_length]
    return out.transpose(1, 2).contiguous().to(initial_dtype)


def gdn_core(A, hidden_states):
    """Qwen3_5GatedDeltaNet.forward (no cache, no padding mask) up to the input of out_proj."""
    batch_size, seq_len, _ = hidden_states.shape
    mixed_qkv = A.in_proj_qkv(hidden_states).transpose(1, 2)
    z = A.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, A.head_v_dim)
    b = A.in_proj_b(hidden_states)
    a = A.in_proj_a(hidden_states)
    w = A.conv1d.weight.squeeze(1)
    mixed_qkv = F.conv1d(mixed_qkv.to(w.dtype), weight=w.unsqueeze(1), bias=A.conv1d.bias, padding=w.shape[-1] - 1,
                         groups=mixed_qkv.shape[1])[:, :, :seq_len]
    mixed_qkv = F.silu(mixed_qkv).transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [A.key_dim, A.key_dim, A.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, A.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, A.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, A.head_v_dim)
    beta = b.sigmoid()
    g = -A.A_log.float().exp() * F.softplus(a.float() + A.dt_bias)
    if A.num_v_heads // A.num_k_heads > 1:
        query = query.repeat_interleave(A.num_v_heads // A.num_k_heads, dim=2)
        key = key.repeat_interleave(A.num_v_heads // A.num_k_heads, dim=2)
    core = chunk_gated_delta_rule_fast(query, key, value, g, beta)
    core = A.norm(core.reshape(-1, A.head_v_dim), z.reshape(-1, A.head_v_dim))
    return core.reshape(batch_size, seq_len, -1)
