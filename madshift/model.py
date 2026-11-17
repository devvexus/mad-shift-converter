# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Matthew Priester
"""The original model, one layer at a time: the Hugging Face snapshot, its config, per-layer weights read from the
safetensors files (with a prefetch thread), and the Qwen3.5 modules (decoder layer, rotary embedding, final norm) built empty
and filled per layer. No full model is ever held in memory."""
import os, glob, json, threading, queue, time
import torch
from safetensors import safe_open

TEXT_PREFIXES = ("model.language_model.", "model.")      # multimodal Qwen3.5 checkpoints, then text-only ones


def snapshot(model_id):
    """Local snapshot directory of a Hugging Face model id (or the path itself if it is a directory)."""
    if os.path.isdir(model_id):
        return model_id
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
    snaps = sorted(glob.glob(os.path.join(hub, "models--" + model_id.replace("/", "--"), "snapshots", "*")))
    if not snaps:
        raise FileNotFoundError(f"no local snapshot of {model_id} under {hub}: download it first (huggingface-cli download {model_id})")
    return snaps[0]


def check_complete(snap):
    """Every shard named by the index exists and holds all the bytes its header promises. Returns the {tensor: file} map."""
    idx = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
    else:
        fs = glob.glob(os.path.join(snap, "*.safetensors"))
        if len(fs) != 1:
            raise FileNotFoundError(f"{snap}: no index and {len(fs)} safetensors files")
        with safe_open(fs[0], "pt") as f:
            wm = {k: os.path.basename(fs[0]) for k in f.keys()}
    for fn in sorted(set(wm.values())):
        p = os.path.join(snap, fn)
        if not os.path.exists(p):
            raise FileNotFoundError(f"incomplete snapshot: {p} missing")
        with open(p, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little"); hdr = json.loads(fh.read(n))
        end = max(v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__")
        if os.path.getsize(p) < 8 + n + end:
            raise FileNotFoundError(f"incomplete snapshot: {p} has {os.path.getsize(p)} bytes, header needs {8 + n + end}")
    return wm


class Model:
    """Config and weight index of one Hugging Face Qwen3.5 checkpoint."""

    def __init__(self, model_id, dev="cuda"):
        from transformers import AutoConfig
        self.id = model_id
        self.snap = snapshot(model_id)
        self.wmap = check_complete(self.snap)
        cfg = AutoConfig.from_pretrained(self.snap)
        tc = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        if getattr(tc, "_attn_implementation", None) in (None, "eager") and getattr(tc, "_attn_implementation_internal", None) is None:
            tc._attn_implementation = "sdpa"                     # what from_pretrained picks
        self.tc = tc
        self.dev = dev
        self.prefix = next(p for p in TEXT_PREFIXES if any(k.startswith(p + "layers.") for k in self.wmap))
        self.NL = tc.num_hidden_layers
        self.d = tc.hidden_size
        self.tied = bool(getattr(cfg, "tie_word_embeddings", getattr(tc, "tie_word_embeddings", False)))
        self.head_name = None if self.tied else next(k for k in ("lm_head.weight", "model.lm_head.weight") if k in self.wmap)
        self.layer_types = list(tc.layer_types)
        self._files = {}

    def _f(self, fn):
        """safe_open handle of a shard; at most two kept open (each open handle maps its whole shard)."""
        if fn in self._files:
            self._files[fn] = self._files.pop(fn)
            return self._files[fn]
        while len(self._files) >= int(os.environ.get("MADSHIFT_SHARDS_OPEN", "2")):
            del self._files[next(iter(self._files))]
        self._files[fn] = safe_open(os.path.join(self.snap, fn), "pt")
        return self._files[fn]

    def get(self, name):
        """One tensor (CPU, checkpoint dtype)."""
        return self._f(self.wmap[name]).get_tensor(name)

    def layer_names(self, l):
        p = f"{self.prefix}layers.{l}."
        return sorted(k for k in self.wmap if k.startswith(p))

    def layer_state(self, l):
        """State dict of decoder layer l with module-relative names (CPU, checkpoint dtype)."""
        p = f"{self.prefix}layers.{l}."
        return {k[len(p):]: self.get(k) for k in self.layer_names(l)}

    def embed(self):
        return self.get(self.prefix + "embed_tokens.weight")

    def head(self):
        return self.embed() if self.tied else self.get(self.head_name)

    def final_norm(self):
        return self.get(self.prefix + "norm.weight")

    def mm(self):
        import transformers.models.qwen3_5.modeling_qwen3_5 as MM
        return MM

    def new_layer(self, l):
        """An empty decoder layer module of layer l's type on the GPU (bf16), to be filled with fill_layer."""
        MM = self.mm()
        with torch.device("meta"):
            m = MM.Qwen3_5DecoderLayer(self.tc, l)
        m = m.to_empty(device=self.dev).to(torch.bfloat16).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    def rotary(self):
        MM = self.mm()
        return MM.Qwen3_5TextRotaryEmbedding(config=self.tc).to(self.dev)

    def norm_module(self):
        MM = self.mm()
        n = MM.Qwen3_5RMSNorm(self.d, eps=self.tc.rms_norm_eps)
        n.weight.data = self.final_norm().to(torch.bfloat16)
        return n.to(self.dev).eval()


def fill_layer(m, sd):
    """Copy a layer state dict (any dtype, CPU or GPU) into module m (bf16 parameters, as from_pretrained(dtype=bf16) loads)."""
    with torch.no_grad():
        params = dict(m.named_parameters())
        missing = set(params) - set(sd)
        if missing:
            raise KeyError(f"layer weights missing: {sorted(missing)}")
        for n, p in params.items():
            p.copy_(sd[n].to(device=p.device, dtype=p.dtype))


class Prefetch:
    """Reads layer weights ahead on a thread (disk -> CPU), so disk time overlaps GPU work. get(l) -> GPU state dict."""

    def __init__(self, model, layers, depth=1, pin=True):
        self.m, self.q, self.layers, self.pin = model, queue.Queue(maxsize=depth), list(layers), pin
        self.read_s = 0.0
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start()

    def _run(self):
        for l in self.layers:
            t = time.time()
            sd = self.m.layer_state(l)
            if self.pin:
                sd = {k: v.pin_memory() for k, v in sd.items()}
            self.read_s += time.time() - t
            self.q.put((l, sd))

    def get(self, l):
        l2, sd = self.q.get()
        assert l2 == l, (l2, l)
        return {k: v.to(self.m.dev, non_blocking=True) for k, v in sd.items()}
