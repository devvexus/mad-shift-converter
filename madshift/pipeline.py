# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Matthew Priester
"""One streamed pass over the layers of a model for one MS tier.

Two residual streams of the calibration tokens live on the GPU: the ORIGINAL model's and the CONVERTED model's (the layers
converted so far). Per layer l:
  1. load     the original weights of layer l from the safetensors files (prefetched on a thread)
  2. maps     (asym_read) the cross second moment of the original and converted streams' normed layer inputs
  3. advance  the original stream advances one layer; kept: its post-attention residual (the attention-output refit target)
              and its output (the FFN target)
  4. tier     on the converted stream: the attention read projections are coded on the second moment of their converted inputs
              (optionally mapped toward the original inputs first), the attention output projection is refit to the original
              residual and coded, the FFN is coded (gate / up on their converted inputs, down refit to its target), then the
              converted stream advances through the converted layer
  5. write    L<l>.pt (the layer's coded entries as MS blocks) and a checkpoint
Before the layers, the token table / output head is coded against the original model's final normed hidden state (a pre-pass
of the original model over the calibration), and the converted stream starts from the converted model's own input table.

Resume: rerun the same command; finished layers are replayed from their files (the streams are rebuilt exactly)."""
import os, json, time
import torch
import torch.nn.functional as F
from madshift import model as hf, forward as fwd, coding as K, blocks
from madshift import reference_gguf as REF
from madshift.coding import tf32, sync

READ_KEYS = {"qkv": "attn_qkv", "z": "attn_gate", "q": "attn_q", "k": "attn_k", "v": "attn_v"}
FIRST_POOL = ("selfgen_256x512_t0.9_s0.pt", 192)        # the first 192 calibration sequences come from this pool
SEQ = 512                                               # calibration sequence length


def cross_rows(fa, fb, N, d, chunk=32768):
    """sum over rows [0, N) of a^T b / N (fp32; TF32 off at the call sites)"""
    C = torch.zeros(d, d, device=K.DEV)
    for i in range(0, N, chunk):
        i1 = min(N, i + chunk)
        C += fa(i, i1).float().T @ fb(i, i1).float()
    return C / N


def asym_map(Hqq, Crq, amu):
    """Input map M such that W M = argmin over W* of sum ||W x_ref - W* x_q||^2 + r ||W* - W||^2 for every W reading these
    inputs (x_ref = the original model's inputs, x_q = the converted model's): M = (C_rq + r I)(H_qq + r I)^-1 with
    C_rq = sum x_ref x_q^T / N, H_qq = sum x_q x_q^T / N, r = amu * mean diag H_qq. W M is then coded on H_qq."""
    r = amu * Hqq.diagonal().mean()
    I = torch.eye(Hqq.shape[0], device=Hqq.device)
    return torch.linalg.solve(Hqq + r * I, Crq.T + r * I).T


def spec_of(cfg, c, l):
    return cfg.get(c + "_layer", {}).get(str(l), cfg.get(c))


def ms_k(spec, what):
    if spec is None:
        return None
    if set(spec) != {"ms"}:
        raise SystemExit(f"{what}: only {{'ms': k}} specs are supported (got {spec})")
    return int(spec["ms"])


def load_pack(path):
    """A layer file with its block-encoded entries decoded back to states."""
    return blocks.expand_pack(torch.load(path))


def save_atomic(obj, path):
    obj = blocks.compact_pack(obj)
    d = os.path.join(os.path.dirname(path), ".tmp"); os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, os.path.basename(path))
    torch.save(obj, tmp)
    os.replace(tmp, path)


class Tier:
    def __init__(self, cfg, pack_dir, gguf_dir=None):
        self.name, self.cfg, self.pack_dir = cfg["name"], cfg, pack_dir
        self.h = {}                     # TF32 flag -> stream (N, d) bf16
        self.bits = {}
        res = lambda f: (os.path.join(gguf_dir, f) if gguf_dir else f) if f else None
        self.copy_from = res(cfg.get("copy_from"))
        self.table_from = res(cfg.get("table_from"))
        for p in (self.copy_from, self.table_from):
            if p and not os.path.exists(p):
                raise SystemExit(f"{self.name}: reference GGUF {p} not found (--gguf-dir)")
        os.makedirs(pack_dir, exist_ok=True)

    def add_bits(self, cls, b, n):
        a = self.bits.setdefault(cls, [0.0, 0]); a[0] += b * n; a[1] += n


class Pass:
    def __init__(self, model_id, calib_dir, tier, cache_dir, batch=8, T=SEQ, core_policy="gpu", log=print):
        self.m = hf.Model(model_id)
        self.batch, self.T, self.tiers, self.log = batch, T, [tier], log
        c0 = tier.cfg
        self.prec = c0.get("stats_precision", "tf32")           # FFN statistics GEMMs: "tf32" or "bf16"
        self.chunk = int(c0.get("stats_chunk", 32768))          # FFN statistics rows per chunk
        self.core_policy = core_policy                          # "gpu": keep the attention core of all tokens; "recompute"
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.state_path = os.path.join(tier.pack_dir, "state.json")
        self.flags = (False,)
        fwd.FAST_GDN = True
        need = [FIRST_POOL[0]] + list(c0["pools"])
        missing = [f for f in need if not os.path.exists(os.path.join(calib_dir, f))]
        if missing:
            raise SystemExit(f"calibration pools missing in {calib_dir}: {missing} (calibration/README in this repository)")
        cal = torch.load(os.path.join(calib_dir, FIRST_POOL[0]))[:FIRST_POOL[1]]
        pools = [cal] + [torch.load(os.path.join(calib_dir, f)) for f in c0["pools"]]
        self.ids = torch.cat(pools)[:cal.shape[0] + c0["extra_seq"]]
        self.N = self.ids.numel()
        self.geo = fwd.Geometry(self.m, self.m.rotary(), T)
        self.rot = bool(c0.get("rot", 1))
        if self.core_policy == "auto":            # the attention cores of all tokens cost N x d_in x 2 bytes (same values either way)
            tc = self.m.tc
            din = max(getattr(tc, "linear_num_value_heads", 0) * getattr(tc, "linear_value_head_dim", 0),
                      tc.num_attention_heads * getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads), tc.hidden_size)
            self.core_policy = "gpu" if self.N * din * 2 < 0.25 * torch.cuda.mem_get_info()[0] else "recompute"
        self._ref_logged = set()
        self.log(f"model {model_id} ({self.m.NL} layers, d {self.m.d}, tied {self.m.tied}) | tier {tier.name} | {self.N} calibration "
                 f"tokens | FFN statistics {self.prec}, chunk {self.chunk} | trellis search "
                 f"{'CUDA kernels' if K.USE_KERNELS else 'PyTorch reference'} | out {tier.pack_dir}")

    # ------------------------------------------------------------------ state
    def load_state(self):
        if os.path.exists(self.state_path):
            return json.load(open(self.state_path))
        return dict(layers_done=0, emb_done=False)

    def save_state(self, st):
        tmp = self.state_path + ".tmp"
        json.dump(st, open(tmp, "w")); os.replace(tmp, self.state_path)

    # ------------------------------------------------------------------ pre-pass
    @torch.no_grad()
    def prepass(self):
        """The original model's final normed hidden state over the calibration: its second moment H (TF32 off, chunks of 32768
        rows) and, for a split tied table, the token importance (mean softmax probability of the original model over the
        calibration + the token's calibration frequency). One stream of the original model through every layer."""
        fH = os.path.join(self.cache_dir, f"final_hidden_moment_{self.N}.pt")
        fI = os.path.join(self.cache_dir, f"token_importance_{self.ids.shape[0]}.pt")
        need_imp = any("split" in (t.cfg.get("emb") or {}) for t in self.tiers) and not os.path.exists(fI)
        if os.path.exists(fH) and not need_imp:
            return torch.load(fH).to(K.DEV)
        m, T, B = self.m, self.T, self.batch
        tf32(False)
        t0 = time.time()
        E0 = m.embed().to(K.DEV, torch.bfloat16)
        h = fwd.embed_rows(E0, self.ids)
        pf = hf.Prefetch(m, range(m.NL))
        self.Lo, self.Lt = {}, {}
        for l in range(m.NL):
            sd = pf.get(l)
            Lo, _, ty = self.modules(l)
            hf.fill_layer(Lo, sd); del sd
            fwd.layer_full(Lo, h, self.geo, ty, T, B)
        norm = m.norm_module()
        H = None
        Wh = m.head().to(K.DEV, torch.bfloat16) if need_imp else None
        V = Wh.shape[0] if need_imp else 0
        p = torch.zeros(V, device=K.DEV) if need_imp else None
        chunk = 32768
        for i in range(0, self.N, chunk):
            rows = []
            for r0, r1 in fwd.batches(min(self.N, i + chunk) - i, T, B):
                rows.append(norm(fwd.view(h, i + r0, i + r1, T)).reshape(r1 - r0, -1))
            X = torch.cat(rows)
            x = X.float()
            H = x.T @ x if H is None else H + x.T @ x
            if need_imp:
                for j in range(0, X.shape[0], 1024):
                    p += torch.softmax((X[j:j + 1024] @ Wh.T).float(), -1).sum(0)
            del x, X
        H = H / self.N
        if need_imp:
            p /= self.N
            fr = torch.bincount(self.ids.reshape(-1).to(K.DEV), minlength=V).float(); fr /= fr.sum()
            torch.save((p + fr).cpu(), fI)
        torch.save(H.cpu(), fH)
        del h, E0
        self.log(f"  pre-pass: final-hidden second moment over {m.NL} layers ({time.time() - t0:.0f}s)")
        return H

    # ------------------------------------------------------------------ table and head
    @torch.no_grad()
    def embedding(self, H, replay=False):
        """Codes the tier's token table (tied models: also the output head) or its output head (untied models), and starts the
        tier's stream from the converted model's input table. replay: rebuild the table from emb.pt instead."""
        m = self.m
        tf32(False)
        W = m.embed().to(K.DEV).float()
        V = W.shape[0]
        for t in self.tiers:
            es = t.cfg.get("emb")
            t0 = time.time()
            if not m.tied and t.cfg.get("out") and not replay:
                # Untied model: the output head is coded on the final-hidden second moment H (rotation seed 12). With a split
                # ('lo'), rows are ranked by token frequency in the calibration (stable sort); the n_hi most frequent rows
                # are coded at 'ms' bits, the rest at 'lo' bits; one stored order serves both halves.
                ko = ms_k({"ms": t.cfg["out"]["ms"]}, "out")
                sh = bool(t.cfg["out"].get("lo"))
                pk_u = {}
                Wh = m.head().to(K.DEV).float()
                if sh:
                    cnt = torch.bincount(self.ids.reshape(-1).long(), minlength=W.shape[0])[:W.shape[0]]
                    order = torch.argsort(cnt.cpu(), descending=True, stable=True)
                    nh = int(t.cfg["out"]["n_hi"]); hi, lo = order[:nh], order[nh:]
                    self.log(f"  split head: {nh} rows cover {float(cnt[hi].sum() / cnt.sum()) * 100:.2f}% of calibration tokens")
                    pk_u["emb_order"] = order.to(torch.int32)
                    for part, rows, k_ in (("output", hi, ko), ("output_lo", lo, int(t.cfg["out"]["lo"]))):
                        Qh, eh, bh = K.code_matrix(Wh[rows.to(K.DEV)], H, k_, 12, True, part)
                        pk_u[part] = eh; t.add_bits("out", bh, Qh.numel()); del Qh
                else:
                    Qh, eh, bh = K.code_matrix(Wh, H, ko, 12, self.rot, "output")
                    t.add_bits("out", bh, Wh.numel()); del Qh
                    pk_u["output"] = eh
                del Wh
                if es is not None:
                    raise SystemExit("untied models: the input table is copied from 'table_from', not MS-coded")
                save_atomic(pk_u, os.path.join(t.pack_dir, "emb.pt"))
            if es is None:
                Q = self.input_table(t, W)
            elif replay:
                pk = load_pack(os.path.join(t.pack_dir, "emb.pt"))
                Q = torch.empty_like(W)
                if "token_embd_lo" in pk:
                    order = pk["emb_order"].to(K.DEV); R_ = pk["token_embd"]["states"].shape[0]
                    for part, rows in (("token_embd", order[:R_]), ("token_embd_lo", order[R_:])):
                        Q[rows] = self.undo_rot(K.dequant_entry(pk[part]), pk["token_embd"]["rot"])
                else:
                    Q = self.undo_rot(K.dequant_entry(pk["token_embd"]), pk["token_embd"]["rot"])
            elif not m.tied:
                raise SystemExit("untied models: the input table is copied from 'table_from', not MS-coded")
            else:
                # Tied model: one table is the input lookup and the output head, coded on the final-hidden second moment H
                # (rotation seed 11). Split ('split' rows at 'ms' bits, the rest at 'lo' bits): see token_order().
                pk = {}
                if "split" in es:
                    order = self.token_order(es, V)
                    hi, lo = order[:es["split"]], order[es["split"]:]
                    Q = torch.empty_like(W)
                    for part, rows, k in (("token_embd", hi, es["ms"]), ("token_embd_lo", lo, es["lo"])):
                        Qp, e, b = K.code_matrix(W[rows], H, k, 11, self.rot, part)
                        Q[rows] = Qp; pk[part] = e; t.add_bits("emb", b, Qp.numel())
                    pk["emb_order"] = order.cpu()
                else:
                    Q, e, b = K.code_matrix(W, H, es["ms"], 11, self.rot, "token_embd")
                    pk["token_embd"] = e; t.add_bits("emb", b, Q.numel())
                save_atomic(pk, os.path.join(t.pack_dir, "emb.pt"))
            tab = Q.to(torch.bfloat16)
            t.h = {False: fwd.embed_rows(tab, self.ids)}
            del tab, Q
            self.log(f"  table/head {t.name}: {'replayed' if replay else 'coded'} ({time.time() - t0:.0f}s)")
        del W
        torch.cuda.empty_cache()

    def token_order(self, es, V):
        """Row order of a split tied table: with rank 'mix', the split/2 most important tokens (token_importance) plus the
        lowest token ids not among them fill the hi part; both parts are stored in token-id order."""
        imp = torch.load(os.path.join(self.cache_dir, f"token_importance_{self.ids.shape[0]}.pt")).to(K.DEV)
        order = torch.argsort(imp, descending=True)
        if es.get("rank") == "mix":
            R = es["split"]; take = torch.zeros(V, dtype=torch.bool, device=order.device)
            take[order[:R // 2]] = True
            free = (~take).nonzero().squeeze(1)[:R - R // 2]; take[free] = True
            order = torch.cat([take.nonzero().squeeze(1), (~take).nonzero().squeeze(1)])
        return order

    def input_table(self, t, W):
        """Untied models: the input table the converted model uses: 'table_from' GGUF's token_embd.weight (copied unchanged
        into the MS file), dequantized, so the conversion calibrates on the real inputs. Otherwise the checkpoint's own table."""
        if not t.table_from:
            return W
        return REF.input_table(t.table_from, W, K.DEV, self.log)

    def undo_rot(self, Qr, rot):
        if rot is None:
            return Qr
        return Qr @ K.rand_hadamard(Qr.shape[1], rot["block"], rot["seed"]).T

    # ------------------------------------------------------------------ layers
    def run(self):
        st = self.load_state()
        m = self.m
        H = self.prepass() if any(t.cfg.get("emb") or t.cfg.get("out") for t in self.tiers) else None
        self.ref_h = None
        self.embedding(H, replay=st["emb_done"])
        if not st["emb_done"]:
            st["emb_done"] = True; self.save_state(st)
        del H
        tf32(False)
        h0 = fwd.embed_rows(m.embed().to(K.DEV, torch.bfloat16), self.ids)
        self.ref_h = {False: h0}
        del h0
        self.ref_mid = torch.empty(self.N, m.d, dtype=torch.bfloat16, device=K.DEV)
        start = st["layers_done"]
        pf = hf.Prefetch(m, range(m.NL))
        self.Lo, self.Lt = {}, {}
        for l in range(m.NL):
            sd = pf.get(l)
            if l < start:
                self.replay_layer(l, sd)
            else:
                self.layer(l, sd)
                st["layers_done"] = l + 1; self.save_state(st)
            del sd
        return st

    def modules(self, l):
        ty = self.m.layer_types[l]
        for D in (self.Lo, self.Lt):
            if ty not in D:
                D[ty] = self.m.new_layer(l)
            if ty == "linear_attention":
                D[ty].linear_attn.layer_idx = l
            else:
                D[ty].self_attn.layer_idx = l
        return self.Lo[ty], self.Lt[ty], ty

    @torch.no_grad()
    def ref_advance(self, Lo, ty):
        tf32(False)
        fwd.layer_full(Lo, self.ref_h[False], self.geo, ty, self.T, self.batch, mid_out=self.ref_mid)
        tf32(False)

    @torch.no_grad()
    def layer(self, l, sd):
        m, T, B, N = self.m, self.T, self.batch, self.N
        t0 = time.time()
        Lo, Lt, ty = self.modules(l)
        lin = ty == "linear_attention"
        hf.fill_layer(Lo, sd)
        cr = {}                                    # asym_read: sum ln(x_ref) ln(x_q)^T before the original stream advances
        for t in self.tiers:
            if t.cfg.get("asym_read"):
                tf32(False)
                ln = lambda h: (lambda i0, i1: torch.cat([Lo.input_layernorm(fwd.view(h, i0 + r0, i0 + r1, T)).reshape(r1 - r0, -1)
                                                          for r0, r1 in fwd.batches(i1 - i0, T, B)]))
                cr[t.name] = cross_rows(ln(self.ref_h[False]), ln(t.h[False]), N, m.d)
        self.ref_advance(Lo, ty)
        last = l + 1 >= m.NL
        for t in self.tiers:
            self.tier_layer(t, l, sd, Lt, ty, lin, last, cr.pop(t.name, None))
        if torch.cuda.memory_reserved() - torch.cuda.memory_allocated() > 2 * 2**30:
            torch.cuda.empty_cache()
        self.log(f"  L{l} ({'lin' if lin else 'full'}) {time.time() - t0:.1f}s")

    @torch.no_grad()
    def tier_layer(self, t, l, sd, Lt, ty, lin, last, c_read=None):
        """The conversion of layer l for one tier on its own stream."""
        T, B, N, cfg = self.T, self.batch, self.N, t.cfg
        hF = t.h[False]
        damp = 0.01
        hf.fill_layer(Lt, sd)
        REF.apply(t, l, Lt, spec_of, self.log, self._ref_logged)          # tensors copied from the reference GGUF
        pk = {}
        mats = ([("qkv", Lt.linear_attn.in_proj_qkv), ("z", Lt.linear_attn.in_proj_z)] if lin else
                [("q", Lt.self_attn.q_proj), ("k", Lt.self_attn.k_proj), ("v", Lt.self_attn.v_proj)])
        tf32(False)
        specs = {c: ms_k(spec_of(cfg, c, l), c) for c, _ in mats}
        # ---- attention read projections: coded on the second moment of their converted inputs (seed 100 + l)
        if any(v is not None for v in specs.values()):
            def ln_rows(i0, i1):
                return torch.cat([Lt.input_layernorm(fwd.view(hF, i0 + r0, i0 + r1, T)).reshape(r1 - r0, -1)
                                  for r0, r1 in fwd.batches(i1 - i0, T, B)])
            H = K.second_moment_rows(ln_rows, N, self.m.d)
            Mr = asym_map(H, c_read, cfg.get("asym_mu", 0.01)) if c_read is not None else None
            for c, mod in mats:
                k = specs[c]
                if k is None:
                    continue
                Q, e, b = K.code_matrix(mod.weight.float() if Mr is None else mod.weight.float() @ Mr, H, k, 100 + l, self.rot,
                                        READ_KEYS[c], damp=damp)
                mod.weight.copy_(Q.to(mod.weight.dtype)); pk[READ_KEYS[c]] = e; t.add_bits(c, b, mod.weight.numel())
            del H, Mr
        # ---- attention output projection: refit to the original residual, coded on its input moment (seed 200 + l)
        om = fwd.proj_of(Lt); oc = "sout" if lin else "o"
        ko = ms_k(spec_of(cfg, oc, l), oc)
        core = None
        if ko is not None:
            din = om.weight.shape[1]
            keep = self.core_policy == "gpu"
            if keep:
                core = torch.empty(N, din, dtype=torch.bfloat16, device=K.DEV)
            else:
                cbuf = torch.empty(32768, din, dtype=torch.bfloat16, device=K.DEV)
            HH = torch.zeros(din, din, device=K.DEV)
            refit = cfg.get("refit_out", 1)
            C = torch.zeros(din, om.weight.shape[0], device=K.DEV) if refit else None
            for i in range(0, N, 32768):
                i1 = min(N, i + 32768)
                cb = core[i:i1] if keep else cbuf[:i1 - i]
                fwd.cores(Lt, hF[i:i1], self.geo, ty, T, B, cb)
                h = cb.float()
                HH += h.T @ h
                if refit:
                    y = self.ref_mid[i:i1].float() - hF[i:i1].float()
                    C += h.T @ y
                del h
            if refit:
                lam = 1e-4 * HH.diagonal().mean()
                mu_o = cfg.get("refit_mu_out", 0.0) * HH.diagonal().mean()      # ridge toward the original weight
                Wt = torch.linalg.solve(HH + (lam + mu_o) * torch.eye(HH.shape[0], device=K.DEV),
                                        C + mu_o * om.weight.float().T).T if mu_o else torch.linalg.solve(HH + lam * torch.eye(HH.shape[0], device=K.DEV), C).T
                del C
            else:
                Wt = om.weight.float()
            if not keep:
                del cbuf
            Q, e, b = K.code_matrix(Wt, HH / N, ko, 200 + l, self.rot, "ssm_out" if lin else "attn_output", damp=damp)
            om.weight.copy_(Q.to(om.weight.dtype)); pk["ssm_out" if lin else "attn_output"] = e
            t.add_bits(oc, b, om.weight.numel())
            del HH, Wt, Q
        # ---- FFN
        fc = cfg.get("ffn")
        midT = None
        fpk = {}
        if fc is not None:
            kgud = fc["k_layer"][str(l)]
            tf32(False)
            fwd.mids_inplace(Lt, hF, self.geo, ty, T, B, core=core)    # the converted stream now holds its post-attention residual
            midT = hF
            n_rows = lambda i0, i1: fwd.norm_rows(Lt, midT, i0, i1, T, B)
            tf32(True)
            G = sd["mlp.gate_proj.weight"].float(); U = sd["mlp.up_proj.weight"].float(); D = sd["mlp.down_proj.weight"].float()
            G0, U0 = G, U
            if cfg.get("asym_ffn"):           # gate / up mapped toward the original model's FFN inputs (asym_map)
                tf32(False)
                Hq = K.second_moment_rows(n_rows, N, self.m.d)
                Mf = asym_map(Hq, cross_rows(lambda i0, i1: fwd.norm_rows(Lt, self.ref_mid, i0, i1, T, B), n_rows, N, self.m.d),
                              cfg.get("asym_mu", 0.01))
                G, U = G @ Mf, U @ Mf
                del Hq, Mf
                tf32(True)
            rout = self.ref_h[False]
            if fc.get("absorb") and l < fc.get("plain_from", 10 ** 9):
                # target: the original model's output minus the converted residual (the FFN also absorbs upstream error)
                y_rows = lambda i0, i1: (rout[i0:i1].float() - midT[i0:i1].float()).to(torch.bfloat16)
            elif fc.get("absorb"):
                # from layer plain_from on: the ORIGINAL FFN applied to the converted model's own FFN inputs
                def y_rows(i0, i1):
                    out = torch.empty(i1 - i0, D.shape[0], dtype=torch.bfloat16, device=K.DEV)
                    for j in range(i0, i1, 4096):
                        x = n_rows(j, min(i1, j + 4096)).float()
                        out[j - i0:j - i0 + x.shape[0]] = ((F.silu(x @ G0.T) * (x @ U0.T)) @ D.T).to(torch.bfloat16)
                    return out
            else:
                raise SystemExit("ffn.absorb must be 1")
            Gq, Uq, Dq, fpk, bits = K.ffn_code(G, U, D, n_rows, y_rows, N, kgud, hblock=fc.get("hblock", 512), lam=1e-4,
                                               prec=self.prec, chunk=self.chunk, refit_mu=fc.get("refit_mu", 0.0), damp=damp)
            for mod, Wq in ((Lt.mlp.gate_proj, Gq), (Lt.mlp.up_proj, Uq), (Lt.mlp.down_proj, Dq)):
                mod.weight.copy_(Wq.to(mod.weight.dtype))
            t.add_bits("ffn", bits, 3 * G.numel())
            del G, U, D, Gq, Uq, Dq
            tf32(False)
        pk = dict(fpk, **pk)
        if pk:
            save_atomic(pk, os.path.join(t.pack_dir, f"L{l}.pt"))
        if not last:                          # the converted stream advances through the converted layer
            tf32(False)
            if midT is not None:
                fwd.advance_from_mid1(Lt, hF, T, B)
            elif core is not None:
                fwd.advance_from_core(Lt, hF, core, T, B)
            else:
                fwd.layer_full(Lt, hF, self.geo, ty, T, B)
        del core

    # ------------------------------------------------------------------ replay (resume)
    @torch.no_grad()
    def set_from_pack(self, Lt, l, pk, lin):
        """Converted weights of layer l from its file (exactly the values the conversion set: dequantized, un-rotated)."""
        tf32(False)
        names = {"attn_qkv": "linear_attn.in_proj_qkv", "attn_gate": "linear_attn.in_proj_z", "ssm_out": "linear_attn.out_proj",
                 "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
                 "attn_output": "self_attn.o_proj"}
        mods = dict(Lt.named_modules())
        for key, mn in names.items():
            if key in pk:
                mods[mn].weight.copy_(self.undo_rot(K.dequant_entry(pk[key]), pk[key]["rot"]).to(torch.bfloat16))
        if "G" in pk:
            tf32(True)
            hd = pk["hadamard"]
            Gq, Uq, Dq = (K.dequant_entry(pk[x]) for x in "GUD")
            Rin = K.rand_hadamard(Gq.shape[1], hd["in_block"], hd["in_seed"])
            Rh = K.rand_hadamard(Dq.shape[1], hd["h_block"], hd["h_seed"])
            Dq = Dq @ Rh.T; Gq = Gq @ Rin.T; Uq = Uq @ Rin.T
            for mod, Wq in ((Lt.mlp.gate_proj, Gq), (Lt.mlp.up_proj, Uq), (Lt.mlp.down_proj, Dq)):
                mod.weight.copy_(Wq.to(torch.bfloat16))
            tf32(False)

    @torch.no_grad()
    def replay_layer(self, l, sd):
        T, B = self.T, self.batch
        t0 = time.time()
        Lo, Lt, ty = self.modules(l)
        hf.fill_layer(Lo, sd)
        self.ref_advance(Lo, ty)
        for t in self.tiers:
            hf.fill_layer(Lt, sd)
            REF.apply(t, l, Lt, spec_of, self.log, self._ref_logged)
            f = os.path.join(t.pack_dir, f"L{l}.pt")
            if os.path.exists(f):
                self.set_from_pack(Lt, l, load_pack(f), ty == "linear_attention")
            for fl, h in t.h.items():
                tf32(fl); fwd.layer_full(Lt, h, self.geo, ty, T, B)
            tf32(False)
        self.log(f"  L{l} replayed from its file ({time.time() - t0:.1f}s)")
