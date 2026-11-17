# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Matthew Priester
"""Calibration and conversion settings shared by every MS tier config, and the tier config layout.

The allocation tools decide only the bit widths. Everything else a tier config carries (calibration pools, rotation, refit
settings, FFN coder settings) is the same for every tier and comes from here.
"""

# Calibration pools (token files of 512 sequences x 512 tokens) and the number of extra sequences drawn from them.
POOLS = [
    "real_web_512x512.pt",
    "real_wiki_512x512.pt",
    "real_code_512x512.pt",
    "real_math_512x512.pt",
    "real_web2_512x512.pt",
    "real_wiki2_512x512.pt",
    "selfgen_div_512x512_s5_t1p1.pt",
    "selfgen_div_512x512_s6_t1p1.pt",
]
EXTRA_SEQ = 3904

# Model-wide conversion settings.
SETTINGS = {"rot": 1, "refit_out": 1, "refit_mu_out": 0.05, "asym_read": 1, "asym_ffn": 1, "asym_mu": 0.01}

# FFN settings; "plain_from" (first layer with the plain FFN target) is half the trunk layer count.
FFN_SETTINGS = {"absorb": 1, "hblock": 512, "refit_mu": 0.05}

# Per-layer attention classes of a tier config, in config order.
ATTN_CLASSES = ["qkv", "z", "sout", "q", "k", "v", "o"]


def tier_config(name, model, n_layers, layer_ks, ffn_ks, default_ks, stats_chunk=32768, stats_precision="tf32"):
    """Assemble a tier config.

    layer_ks:   {class: {layer: k}} for the attention classes the model has (see ATTN_CLASSES)
    ffn_ks:     {layer: [k_gate, k_up, k_down]}
    default_ks: {class: k}, the class-wide width written next to each per-layer table
    stats_chunk / stats_precision: rows per chunk and GEMM precision of the FFN statistics (released tiers: 32768 for
    Qwen3.5-2B, 8192 for Qwen3.8-27B; "tf32", except "bf16" for the 27B MS3.5 and MS5).
    Table / head / reference keys are added by the caller.
    """
    cfg = {"name": name, "model": model, "pools": list(POOLS), "extra_seq": EXTRA_SEQ,
           "stats_precision": stats_precision, "stats_chunk": stats_chunk, **SETTINGS}
    for c in ATTN_CLASSES:
        if c in layer_ks:
            cfg[c] = {"ms": default_ks[c]}
            cfg[c + "_layer"] = {str(l): {"ms": k} for l, k in sorted(layer_ks[c].items())}
    cfg["ffn"] = dict(FFN_SETTINGS, plain_from=n_layers // 2,
                      k_layer={str(l): list(ks) for l, ks in sorted(ffn_ks.items())})
    return cfg
