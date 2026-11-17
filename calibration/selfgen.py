# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Matthew Priester
"""Model-written calibration pools: the model writes its own text from content-free openers (no corpus, no prompts).

  s0  selfgen_256x512_t0.9_s0.pt          256 sequences of 512 tokens (the converter uses the first 192). Openers: in blocks of
                                          32 sequences, two of every four blocks start a chat turn ("<|im_start|>user\\n", so the
                                          model invents the question and the answer), one starts a bare "<|im_start|>", one starts
                                          from one random token id below 150,000. Sampling: temperature 0.9, top-p 0.95, top-k 50,
                                          torch.manual_seed(0), numpy seed 0.
  d5  selfgen_div_512x512_s5_t1p1.pt      512 sequences: 64 per format opener, sampled at temperature 1, top-p 1, top-k 50 (the
  d6  selfgen_div_512x512_s6_t1p1.pt      model's full distribution), torch.manual_seed(100 + set); rows shuffled with
                                          torch.randperm(512, generator=manual_seed(set)). Openers below (STYLES).

usage: python calibration/selfgen.py Qwen/Qwen3.5-2B --out calib/qwen3.5-2b [--pools s0,d5,d6]

Sampling on another GPU or another PyTorch / transformers version gives different tokens; calibration/MANIFEST.json lists the
SHA-256 of the pools the released files were converted with (calibration/manifest.py checks a directory against it)."""
import os, sys, time, glob, json, argparse
import numpy as np
import torch

STYLES = {
    5: ["```python\n", "```\n", "#include <", "{\n  \"", "# ", "\\begin{", "1. ", "From Wikipedia, the free encyclopedia\n\n"],
    6: ["def ", "import ", "class ", "diff --git a/", "#!/bin/bash\n", "SELECT ", "Abstract\n", "== "],
}
DEV = "cuda"


def snapshot(model_id):
    if os.path.isdir(model_id):
        return model_id
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
    snaps = sorted(glob.glob(os.path.join(hub, "models--" + model_id.replace("/", "--"), "snapshots", "*")))
    if not snaps:
        raise SystemExit(f"no local snapshot of {model_id} under {hub}")
    return snaps[0]


def load_on_gpu(model_id):
    """AutoModelForCausalLM in bf16, built on the GPU and filled shard by shard from the safetensors files (the 27B does not
    fit in host memory). Checkpoint keys model.language_model.* -> model.*, lm_head kept; the vision tower and the MTP block
    are not part of the causal LM. The snapshot's generation_config.json is attached, as from_pretrained does."""
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
    snap = snapshot(model_id)
    cfg = AutoConfig.from_pretrained(snap)
    with torch.device(DEV):
        m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    want = {n for n, _ in m.named_parameters()}
    got = set()
    idx = os.path.join(snap, "model.safetensors.index.json")
    shards = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else \
        [os.path.basename(p) for p in glob.glob(os.path.join(snap, "*.safetensors"))]
    for shard in shards:
        sd = load_file(os.path.join(snap, shard), device=DEV)
        mapped = {}
        for k, v in sd.items():
            if k.startswith("model.language_model."):
                mapped["model." + k[len("model.language_model."):]] = v
            elif k == "lm_head.weight":
                mapped[k] = v
        m.load_state_dict(mapped, strict=False)
        got |= set(mapped) & want
        del sd, mapped
    missing = want - got
    if missing and missing != {"lm_head.weight"}:
        raise SystemExit(f"parameters not in the checkpoint: {sorted(missing)[:10]}")
    if missing:                                          # tied table: the head is the embedding
        m.tie_weights()
    m.generation_config = GenerationConfig.from_pretrained(snap)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def self_generate(model, tok, path, n_seq=256, seq_len=512, batch=32, temp=0.9, top_p=0.95, seed=0):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    chat = tok("<|im_start|>user" + chr(10))["input_ids"]
    bare = tok("<|im_start|>")["input_ids"]
    out = []
    for b in range(0, n_seq, batch):
        nb = min(batch, n_seq - b)
        kind = (b // batch) % 4
        if kind in (0, 1):
            pre = torch.tensor(chat).repeat(nb, 1)
        elif kind == 2:
            pre = torch.tensor(bare).repeat(nb, 1)
        else:
            pre = torch.tensor(rng.randint(0, min(150000, tok.vocab_size), size=(nb, 1)))
        pre = pre.to(DEV)
        need = seq_len - pre.shape[1]
        res = model.generate(pre, attention_mask=torch.ones_like(pre), max_new_tokens=need, min_new_tokens=need,
                             do_sample=True, temperature=temp, top_p=top_p, top_k=50, pad_token_id=0)
        out.append(res[:, :seq_len].cpu())
        print(f"  s0 {b + nb}/{n_seq}", flush=True)
    x = torch.cat(out)
    torch.save(x, path)
    return x


@torch.no_grad()
def diverse(model, tok, sset, path, temp=1.0, topp=1.0):
    out = []
    torch.manual_seed(100 + sset)
    t = time.time()
    for i, s in enumerate(STYLES[sset]):
        pre = torch.tensor(tok(s)["input_ids"]).repeat(64, 1).to(DEV)
        need = 512 - pre.shape[1]
        r = model.generate(pre, attention_mask=torch.ones_like(pre), max_new_tokens=need, min_new_tokens=need,
                           do_sample=True, temperature=temp, top_p=topp, top_k=50, pad_token_id=0)
        out.append(r[:, :512].cpu())
        print(f"  d{sset} opener {i} {s!r} ({time.time() - t:.0f}s)", flush=True)
    x = torch.cat(out)
    x = x[torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(sset))]
    torch.save(x, path)
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="Hugging Face model id (local cache) or checkpoint directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pools", default="s0,d5,d6")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(snapshot(a.model))
    m = load_on_gpu(a.model)
    for w in a.pools.split(","):
        t = time.time()
        if w == "s0":
            p = os.path.join(a.out, "selfgen_256x512_t0.9_s0.pt")
            if not os.path.exists(p):
                self_generate(m, tok, p)
        elif w in ("d5", "d6"):
            p = os.path.join(a.out, f"selfgen_div_512x512_s{w[1]}_t1p1.pt")
            if not os.path.exists(p):
                diverse(m, tok, int(w[1]), p)
        else:
            raise SystemExit(f"unknown pool {w}")
        print(f"{w}: {p} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
