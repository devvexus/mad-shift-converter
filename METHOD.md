# MS (Mad Shift): the conversion method

MS (Mad Shift) is a trellis-coded GGUF quantization format and the conversion method that produces it, by Matthew Priester
(https://github.com/devvexus, ORCID 0009-0000-0032-5292). This document describes, completely and as precisely as the code in
this repository, how the released files were made:

- Qwen3.5-2B: tiers MS2.7, MS3, MS3.5, MS3.7, MS4, MS4.3, MS5 (`devvexus/Qwen3.5-2B-MS`);
- Qwen3.8-27B: tiers MS2.7, MS3, MS3.5, MS3.7, MS4, MS4.3, MS5 (`devvexus/Qwen3.8-27B-MS`).

The 27B repository's separate MTP file is made by a different script that is not part of this repository (section 11).

Every statement below is about the code in this repository and the released files. Where the method builds on earlier work, the
credit is given where it applies (section 13 lists it in one place).

---

## 1. Overview

A released MS file is a GGUF file in which every large matrix of the model (the attention and linear-attention projections, the
FFN matrices, and the token table or output head) is stored as a bitshift trellis code with k bits per weight, k = 1..6 chosen
per tensor, after a randomized Hadamard rotation of its input dimension. The engine ([ms-llama.cpp](https://github.com/devvexus/ms-llama.cpp),
MIT) decodes these tensors on the fly.

The conversion is layer by layer over a fixed calibration set of 2,097,152 tokens. Two residual streams of the calibration
tokens are kept on the GPU: the original model's and the converted model's (the layers converted so far, computed with their
decoded weights). In each layer, every matrix is coded against the second moment of its inputs in the converted model; the
attention output projection and the FFN down projection are first refit by ridge least squares so that the converted model
reproduces the original model's residual stream (absorbing the error of everything converted before); the read projections and
the FFN gate/up are first mapped toward the original model's inputs. The search that turns a matrix into trellis states is a
sliding-window Viterbi search with error feedback through the inverse second moment.

Inputs: the Hugging Face BF16 checkpoint, the calibration token pools (section 5), a tier config (per-tensor widths, section 9),
for the 27B the unsloth GGUF files whose small tensors and input table are copied (section 8), and the model's BF16 GGUF made by
llama.cpp's `convert_hf_to_gguf.py` (for the metadata and the tensors that stay unquantized).

---

## 2. Storage format

### 2.1 Blocks

An MS tensor of type `MS_Ck` (ggml type ids 43..48 for k = 1..6 in the MS fork) stores each row in blocks of 512 weights along
the input dimension. A block is, packed and little-endian:

```
fp16   d          the row scale (the same value in every block of a row)
uint16 s0         the 14-bit trellis state of the block's first weight (top 2 bits zero)
uint8  qs[64 k]   the k-bit symbols b_1 .. b_511 of the other 511 weights, most significant bit first, then k zero bits
```

so a block is 4 + 64k bytes and a tensor costs exactly k + 0.0625 bits per weight (MS_C1 68 B, MS_C2 132 B, MS_C3 196 B,
MS_C4 260 B, MS_C5 324 B, MS_C6 388 B per 512 weights). Every input width of a coded matrix must be a multiple of 512.

### 2.2 Decoding: the bitshift trellis and the 1MAD code

The state of weight i of a block is

    s_0 = s0,   s_i = ((s_{i-1} << k) | b_i) & (2^14 - 1)

(equivalently: the 14-bit window starting at bit k*i of the bit string [s0 as 14 bits][qs]). The weight's value is

    x = (34038481 * s + 76625530) mod 2^32
    v = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24) - 510        (an integer in [-510, 510])
    w = float32(v) * float32(d)

The bitshift trellis and this computed code ("1MAD") are QTIP's (A. Tseng, Q. Sun, D. Hou, C. De Sa, "QTIP: Quantization with
Trellises and Incoherence Processing", NeurIPS 2024, arXiv:2406.11235): MS uses the bitshift-trellis construction and the 1MAD
decoding function described there, with its published constants. MS's choices in the format are: a 14-bit state (L = 14);
row-aligned blocks of 512 weights, each storing the state of its first weight, so that every block decodes on its own and the
trellis needs no tail-biting; an fp16 scale stored per block; and one width k per tensor, from 1 to 6 bits, mixed freely within a
model. The values of v have standard deviation of about 147.8 over uniformly distributed states; the scale carries the units.

### 2.3 Encoding the states into blocks

The encoder (section 4) produces one continuous trellis path per row: a state for every weight of the row, with
s_i = ((s_{i-1} << k) | b_i) & (2^14 - 1) holding along the whole row. Writing the blocks is then lossless: block j's s0 is the
path's state at weight 512 j and its symbols are the low k bits of the following 511 states. The converter checks every
encoded tensor by decoding the blocks back to states and comparing them with the search's states bit for bit
(`madshift/blocks.py`); the GGUF writer copies these bytes into the file and compares the written file with them.

---

## 3. Rotation

Every coded matrix W (out x in) is stored as the codes of W R, where

    R = diag(s) * blockdiag(H_b, ..., H_b) / sqrt(b)

with H_b the b x b Sylvester-Hadamard matrix and s a +-1 vector of length `in`:
`s = torch.randint(0, 2, (in,), generator=torch.Generator().manual_seed(seed)) * 2 - 1`. The engine computes W x = (W R)(R^T x):
it multiplies the input by s and applies a block fast Walsh-Hadamard transform (`prism.hadamard.*` metadata: block sizes, the
sign vectors, which tensors; this is llama.cpp's input-rotation mechanism from its PR #29600, which MS extends with per-weight
block sizes and sign vectors). Tensors that read the same input with the same rotation share one rotated activation.
Random-sign Hadamard rotations ("incoherence processing") are the technique of QuIP (Chee, Cai, Kuleshov, De Sa, NeurIPS 2023)
and QuIP# (Tseng, Chee, Sun, Kuleshov, De Sa, ICML 2024); MS applies them on the input side of each matrix only.

Seeds and blocks (block = the largest power of two dividing the input width, at most 4096; for the FFN hidden width 512):

| tensor | seed | 2B block | 27B block |
|---|---|---|---|
| attention / linear-attention read projections of layer l (attn_qkv, attn_gate; attn_q, attn_k, attn_v) | 100 + l | 2048 | 1024 |
| attention output projections of layer l (ssm_out, attn_output) | 200 + l | 2048 | 2048 (input width 6144) |
| FFN gate and up (input = hidden size) | 1 | 2048 | 1024 |
| FFN down (input = FFN hidden size) | 2 | 512 | 512 |
| tied token table / head (2B) | 11 | 2048 | - |
| output head, both split parts (27B) | 12 | - | 1024 |

For the tied 2B table the stored rows are rows of E R; the engine applies R after the lookup and R^T to the final hidden state
before the head (`prism.hadamard.version` 2, `tied_output`). For the 27B the input table is not MS-coded (section 8).

---

## 4. Coding one matrix

### 4.1 Objective

A matrix W with input second moment H = E[x x^T] (estimated over the calibration tokens, with x the matrix's input in the
converted model) is coded to minimize the proxy tr((W - Q) H (W - Q)^T), the objective of GPTQ / LDLQ, over the values Q the
trellis code can represent with the row's scale. All of this happens in the rotated basis: W R and R^T H R.

### 4.2 Scale

For each row r of the rotated target, d_r = gain_k * std(row r) / 147.8, rounded to fp16 (this fp16 value is what the blocks
store), with gain_1 = 1.0, gain_2 = 1.15 and gain_k = 1.2 for k >= 3. The scale is fixed before the search.

### 4.3 Error feedback

H is damped and inverted: Hu = the upper Cholesky factor of (H + 0.01 * mean(diag H) * I)^-1 (if the Cholesky factorization
fails, damping 0.05, then 0.2). Columns are processed left to right; when column c is committed with values q, its error
e = (w_c - q) / Hu[c, c] is fed forward to every later column: W[:, c+1:] -= e * Hu[c, c+1:]. This is the error feedback of
GPTQ (Frantar et al., ICLR 2023) / LDLQ (Chee et al., NeurIPS 2023), descended from Optimal Brain Quantization (Frantar,
Singh, Alistarh, NeurIPS 2022).

### 4.4 Trellis search

A Viterbi search (dynamic programming over the 2^14 trellis states) chooses the states, in sliding windows:

- the window covers the next 32 columns of the error-fed matrix (fewer at the end of the row);
- the branch cost of state s at column c is piv_c * (v(s)^2 d^2 - 2 x_c v(s) d), with piv_c = 1 / Hu[c, c]^2 (the squared
  error weighted by the feedback's pivot; the per-row constant x_c^2 is dropped because it cannot change a decision);
- the first window of a row starts from any state; every later window must start from a successor of the last committed state,
  so the path is continuous along the whole row;
- after the search over the window, the first 8 columns of the best path are committed (their states, values and errors), the
  errors are fed forward (4.3), and the next window starts 8 columns further on.

In a bitshift trellis the predecessors of state s' = (j << k) | b are the 2^k states (a << (14 - k)) | j, so the best predecessor
depends only on j = s' >> k: each step is one minimum over a (2^k x 2^(14-k)) view of the costs, O(2^14) work per weight. Ties
choose the lowest index. `madshift/trellis.py` is the readable PyTorch implementation; `madshift/kernels.cu` implements the same
search on CUDA (one thread block per row, path costs in registers, lazy blocked application of the error feedback) with the same
floating-point operations in the same order, so the states are identical (checked on random matrices for k = 1..6 against the
PyTorch search). The stored start state of each 512-weight block (section 2.3) is just the path's state at that position: the
search does not restart at block boundaries.

The QTIP paper also pairs trellis coding with LDLQ-style feedback (its BlockLDLQ, with tail-biting trellises over 16 x 16
tiles); MS's search, window, block layout and stored start states are as described here.

---

## 5. Calibration data

Every tier of both models uses the same composition: 4,096 sequences of 512 tokens (2,097,152 tokens), the first 192
sequences of the model-written pool s0 followed by the other pools in this order, cut at 4,096 sequences:

| pool | sequences used | content |
|---|---|---|
| `selfgen_256x512_t0.9_s0.pt` | 192 | written by the model being converted, from content-free openers (below) |
| `real_web_512x512.pt` | 512 | FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, sample-10BT), first rows |
| `real_wiki_512x512.pt` | 512 | English Wikipedia (`wikimedia/wikipedia`, 20231101.en), first rows |
| `real_code_512x512.pt` | 512 | `codeparrot/github-code-clean`, Python 30%, JavaScript 10%, TypeScript 8%, Java 10%, C++ 10%, C 8%, Go 7%, Rust 7%, Shell 5%, SQL 5% of the tokens (configs the rows API did not serve were skipped and the first working source topped the pool up) |
| `real_math_512x512.pt` | 512 | `open-web-math/open-web-math`, first rows |
| `real_web2_512x512.pt` | 512 | FineWeb-Edu from row 30,000 |
| `real_wiki2_512x512.pt` | 512 | Wikipedia from row 30,000 |
| `selfgen_div_512x512_s5_t1p1.pt` | 512 | written by the model from format openers, set 5 |
| `selfgen_div_512x512_s6_t1p1.pt` | 320 | written by the model from format openers, set 6 |

Real text (`calibration/realtext.py`): documents read from the Hugging Face datasets-server rows API, tokenized with the model's
tokenizer, each followed by "\n\n" and cut at 1,536 tokens, joined and cut into 512-token rows. The two models share the same
tokenizer, so the real-text pools are identical for both.

Model-written text (`calibration/selfgen.py`), generated by the model that is converted (each model writes its own pools):

- s0: 256 sequences in blocks of 32; blocks 0 and 1 of every 4 open a chat turn (`<|im_start|>user\n`, so the model invents the
  question and the answer), block 2 opens a bare `<|im_start|>`, block 3 starts from one random token id below 150,000.
  Sampling: temperature 0.9, top-p 0.95, top-k 50, `torch.manual_seed(0)`, numpy seed 0.
- set 5 openers: "```python\n", "```\n", "#include <", "{\n  \"", "# ", "\\begin{", "1. ", "From Wikipedia, the free encyclopedia\n\n";
  set 6 openers: "def ", "import ", "class ", "diff --git a/", "#!/bin/bash\n", "SELECT ", "Abstract\n", "== ".
  64 sequences per opener, temperature 1, top-p 1, top-k 50 (the model's full distribution), `torch.manual_seed(100 + set)`;
  the 512 rows are then shuffled with `torch.randperm(512, generator=torch.Generator().manual_seed(set))`.

Sampled text depends on the GPU and the software versions, so `calibration/MANIFEST.json` records the SHA-256 of the token ids
of every pool the released files were converted with.

---

## 6. The conversion, layer by layer

### 6.1 Pre-pass

Before the layers, the original model runs once over the calibration and the second moment of its final normed hidden state is
taken: H_out = sum x x^T / N (fp32, rows in chunks of 32,768). For a split tied table (2B) the same pass also records each
token's importance: the original model's next-token probability averaged over all calibration positions, plus the token's
frequency in the calibration ids.

### 6.2 Table and head

- **Qwen3.5-2B (tied: one table is the input lookup and the output head).** The table is coded on H_out (rotation seed 11).
  Split tiers code two row groups at two widths: 65,536 rows at k_hi and the other 182,784 rows at k_lo. The 65,536 rows are the
  32,768 most important tokens (6.1) plus the 32,768 lowest token ids not already among them; both groups are stored in
  token-id order, and `msffn.emb_split_order` maps stored rows back to token ids. Widths (k_hi / k_lo): MS5 6 (not split), MS4.3
  6/3, MS4 6/3, MS3.7 6/3, MS3.5 6/2, MS3 5/3, MS2.7 4/3. The converted stream starts from the decoded table, so the later layers
  absorb its error.
- **Qwen3.8-27B (untied).** The output head is coded on H_out (rotation seed 12) and split: the rows are ranked by the token's
  frequency in the calibration ids (stable sort, most frequent first); the first 65,536 are coded at k_hi (`output.weight`) and
  the other 182,784 at k_lo (`output_lo.weight`), in that stored order (`msffn.emb_split_order`, `msffn.head_split_rows`).
  Widths: MS2.7 4/3, MS3 5/4, MS3.5 5/4, MS3.7 6/4, MS4 6/4, MS4.3 6/4, MS5 6/4. The input table is not MS-coded: it is copied
  unchanged from an unsloth GGUF (section 8), and the converted stream starts from that table's dequantized values, so the
  conversion calibrates on the inputs the file will have.

### 6.3 Per layer

The original stream advances through the original layer and keeps, for every calibration token, the post-attention residual
(mid_ref) and the layer output (out_ref). On the converted stream (h, the converted model's residual entering layer l), with the
layer's original weights loaded (and, for the 27B, the copied small tensors substituted, section 8):

1. **Read projections** (linear attention: attn_qkv, attn_gate; full attention: attn_q, attn_k, attn_v). Inputs x_q =
   input_layernorm(h) and x_ref = the same norm of the original stream's residual. With H_qq = sum x_q x_q^T / N,
   C_rq = sum x_ref x_q^T / N and r = 0.01 * mean(diag H_qq), the input map M = (C_rq + r I)(H_qq + r I)^-1 is applied,
   W' = W M (W' minimizes sum ||W x_ref - W' x_q||^2 + r ||W' - W||^2), and W' is coded on H_qq (section 4). The coded weights
   replace the originals in the converted layer.
2. **Attention output projection** (ssm_out or attn_output). Its inputs z (the attention core of the converted layer, computed
   with the coded read projections) give HH = sum z z^T and C = sum z (mid_ref - h)^T. The projection is refit:
   W_t^T = (HH + (lam + mu) I)^-1 (C + mu W^T), lam = 1e-4 * mean(diag HH), mu = 0.05 * mean(diag HH): the projection now
   reproduces the original post-attention residual from the converted residual, absorbing the error accumulated so far, with
   a ridge toward the original weight. W_t is coded on HH / N.
3. **FFN.** The converted stream becomes its post-attention residual mid = h + o_proj(z) (coded weights); FFN inputs
   n = post_attention_layernorm(mid).
   - Gate and up: input map as in step 1 between n and the original FFN inputs post_attention_layernorm(mid_ref)
     (r = 0.01 * mean diag); G' = G M, U' = U M; both coded on Hn = R_in^T (sum n n^T / N) R_in (seed 1).
   - Hidden activations of the coded gate / up: a = silu(n G_q^T) * (n U_q^T); Hh = sum a a^T / N.
   - Target y: in the first half of the layers (l < NL / 2: 2B layers 0-11, 27B layers 0-31), y = out_ref - mid (the FFN
     reproduces the original layer output and absorbs the residual error); from NL / 2 on, y = the ORIGINAL FFN applied to the
     converted model's own inputs, silu(n G^T) * (n U^T) D^T.
   - Down refit: D_t = (rhs + mu D^T)^T (Hh + (lam + mu) I)^-1 with rhs = sum a y^T / N, lam = 1e-4 * mean(diag Hh),
     mu = 0.05 * mean(diag Hh); rotated on the hidden width (seed 2, block 512) and coded on R_h^T Hh R_h.
4. The converted stream advances through the converted layer (the decoded weights, in bf16), and the layer's coded entries are
   saved.

Precision: the attention steps (statistics, input maps, refit, coding) run in fp32 with TF32 off, as do the gate / up input
map statistics. The rest of the FFN step runs with TF32 enabled for its matrix products: the token loops (Hn, the hidden
activations, Hh, rhs) as TF32 GEMMs in chunks of 32,768 rows (2B) or 8,192 rows (27B), except that the 27B MS3.5 and MS5 tiers
ran these loops with bf16 inputs and fp32 accumulation (config key `stats_precision`); the trellis search itself is fp32. Forwards use the Hugging Face Qwen3.5 modules in
bf16, batches of 8 sequences of 512 tokens; the linear-attention core uses a batched triangular solve in place of the
reference implementation's row-by-row forward substitution (`madshift/gdn.py`, derived from Hugging Face transformers).

---

## 7. What every tier codes

Every tier codes every large matrix: in every layer the read projections, the output projection and the three FFN matrices,
plus the 2B's tied table or the 27B's output head. Not coded: norms, the linear-attention convolution, ssm_a, ssm_dt,
ssm_alpha, ssm_beta, and (27B) the input token table. Bits per MS-coded weight in the released files
(`msffn.bits_per_ms_weight`, including the 0.0625 of scales and start states):

| tier | Qwen3.5-2B | Qwen3.8-27B |
|---|---|---|
| MS5 | 6.06 | 5.83 |
| MS4.3 | 5.47 | 5.17 |
| MS4 | 5.30 | 4.79 |
| MS3.7 | 4.88 | 4.45 |
| MS3.5 | 4.56 | 4.10 |
| MS3 | 3.58 | 3.76 |
| MS2.7 | 3.22 | 3.12 |

The number in a tier's name is a size class (the unsloth file it is sized against), not its bits per weight.

---

## 8. Tensors copied from unsloth's GGUFs (Qwen3.8-27B)

Every tensor of a 27B MS file that is not MS-coded is copied unchanged (type and bytes) from unsloth's GGUFs of Qwen3.8-27B
([unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), Apache License 2.0): the input token table, the
norms and the small SSM and convolution tensors (the 96 `ssm_alpha` / `ssm_beta` matrices are Q8_0 in those files, the rest F32).

| tier | input token table from | every other non-MS tensor from | revision |
|---|---|---|---|
| MS5 | UD-Q4_K_M (Q4_K) | UD-Q5_K_M | `4ca72078` |
| MS4.3 | UD-Q4_K_M (Q4_K) | UD-Q4_K_XL | `4ca72078` (table), `ce331efc` (UD-Q4_K_XL) |
| MS4 | UD-Q4_K_M (Q4_K) | UD-Q4_K_M | `4ca72078` |
| MS3.7 | UD-Q4_K_M (Q4_K) | UD-Q4_K_S | `4ca72078` |
| MS3.5 | UD-Q4_K_M (Q4_K) | UD-IQ4_XS | `4ca72078` |
| MS3 | UD-Q4_K_M (Q4_K) | UD-Q3_K_XL | `4ca72078` |
| MS2.7 | UD-Q3_K_XL (Q3_K) | UD-IQ3_XXS | `4ca72078` |

Each file names its sources in `msffn.token_embd_source` and `msffn.copy_source`. During the conversion the converted stream
uses the copied tensors' dequantized values: every lossy (not F32 / BF16) copied tensor of a layer replaces the checkpoint's
tensor in the converted layer before that layer is converted (rows matched to the checkpoint's; ssm_alpha / ssm_beta, whose
value heads llama.cpp reorders, by nearest cosine), so the conversion absorbs their error too. The 2B files copy nothing from
unsloth: their non-MS tensors come from the BF16 GGUF.

---

## 9. Per-tensor bit widths

### 9.1 Qwen3.5-2B: probes and a greedy solver (`allocation/solve_probes.py`)

1. Probes: 12 conversions of the 2B in which one tensor class is MS-coded in every layer at one width and everything else stays
   BF16, each scored by the KL divergence of the converted model's next-token distribution from the original's, averaged over
   five held-out token sets (model-written text, FineWeb-Edu, Wikipedia, MATH, SWE-Gym patches; 32 x 512 tokens each). Classes
   and points: table k4, k6 and split 6/3; linear-attention qkv k3, k5; z k4; ssm_out k5; full-attention q+k k4; v k5; o k4;
   FFN k4, k5 (`allocation/data/qwen3.5-2b_probes.json`). The probes used an earlier, model-written-only calibration set; the
   released tiers kept the resulting allocations and were converted with the calibration of section 5.
2. Curves: for a class with two or more points, log2 KL is piecewise linear in bits per weight (extrapolated beyond the end
   points); a one-point class halves its KL every s bits, s = the mean over the two-point non-FFN classes of
   (b_max - b_min) / log2(KL(b_min) / KL(b_max)) (0.747 bits on the 2B). Split table: the hi part follows the plain table's curve;
   the lo part's KL is P * 2^(-(b - (lo + 0.0625)) / s) with P = KL(split probe) - KL_plain(6.0625).
3. Units: one per layer and class (q and k of a full-attention layer share one width; each FFN matrix is its own unit), plus the
   table (or its two row groups). A unit's predicted KL is its class curve at k + 0.0625 divided by the number of units of the class.
4. Budget: the target file size minus everything that is not an MS-coded tensor (the BF16 file's header and unquantized tensors,
   MTP block dropped, rotation sign vectors, 16,480 bytes of other MS metadata, and 4 V bytes of split order).
5. Greedy: every unit starts at kmin (2; 3 for MS3 and MS2.7); repeatedly the unit with the largest predicted KL drop per byte for
   +1 bit (k < 6, fitting the budget) gets it. Ties go to the first unit in a fixed order (classes qkv, z, sout, qk, v, o, ffn,
   table; layers outermost first; FFN down before up before gate).

Targets (decimal MB): MS5 1460 (plain table), MS4.3 1330, MS4 1263, MS3.7 1166, MS3.5 1088, MS3 858, MS2.7 775 (split table).
Regenerating the seven tiers with this tool from the probe data gives the released widths exactly.

### 9.2 Qwen3.8-27B: unsloth's per-tensor types as the starting point (`allocation/unsloth_widths.py`, `fill.py`, `size.py`)

The 27B allocation starts from unsloth's per-tensor quantization types: each tier follows the per-tensor allocation of the
Unsloth Dynamic GGUF of its size class (the "every other non-MS tensor" file of section 8: MS2.7 UD-IQ3_XXS, MS3 UD-Q3_K_XL,
MS3.5 UD-IQ4_XS, MS3.7 UD-Q4_K_S, MS4 UD-Q4_K_M, MS4.3 UD-Q4_K_XL, MS5 UD-Q5_K_M). Credit for that per-tensor allocation goes to
Unsloth.

1. Matching. For each MS-coded tensor, b = unsloth's bits per weight (8 * bytes / weights) and floor = the largest k (1..6) with
   k + 0.0625 <= b. Candidates per tensor: floor, floor + 1 (if floor < 6 and floor + 0.0625 <= b) and floor - 1. Per layer,
   every combination of candidates is scored and the best one taken, subject to the running MS byte total never exceeding
   unsloth's; the score, in order: fewest steps below floor; closest match of the bytes per component (FFN, linear attention,
   full attention; weight 2) and per tensor kind (weight 1), each with its deficit carried to later layers; smallest
   sum of n |k + 0.0625 - b|; then preference for the tensors unsloth gave more bits than its kind's mean. The head gets the floor
   width of unsloth's head type and is split: k_hi = k, k_lo = k - 1 (k - 2 when k = 6), 65,536 rows at k_hi.
2. MS2.7 only: every width below 2 raised to 2 after matching (nothing else moves).
3. Size target: the unsloth file's size without its MTP block (its `blk.64.*` tensors), minus 8,000,000 bytes. The MS4.3 target
   is the UD-Q4_K_XL of revision `ce331efc`; MS4.3 is 0.09 GB larger than the UD-Q4_K_XL of revision `4ca72078`.
4. Fill. The bytes left below the target are spent as +1 bit on single tensors, by modelled KL removed per byte. The model: a
   leave-one-out table measured once on an earlier 27B MS4 conversion (for each part p, one layer's attention block or FFN: the
   share S_p of the whole model's KL removed by restoring only that part to the original weights, in PyTorch, on four development
   texts of 16 x 512 tokens: Wikipedia, FineWeb-Edu, code, math); within a part, tensor t's share S_p * n_t 4^-k_t / sum_j n_j 4^-k_j; +1 bit removes 3/4 of it at n_t / 8 bytes. Greedy
   by gain per byte (parts with S_p <= 0 untouched, widths capped at 6, S_p reduced by each gain); bytes left after that go to the
   remaining tensors lowest k first (then unsloth's bits minus k, kind order, layer).
   Bumps: MS3 2, MS3.5 2, MS3.7 17, MS4 41, MS4.3 45, MS5 54 tensors; MS2.7 none (its raise used up the budget).

Regenerating six tiers (all but MS4.3) from the unsloth files of revision `4ca72078` with these tools gives the released widths
exactly; MS4.3 needs the UD-Q4_K_XL of revision `ce331efc`, which was not on disk to re-check (its fill step was re-checked).

### 9.3 Released file sizes

| tier | Qwen3.5-2B bytes | Qwen3.8-27B bytes |
|---|---|---|
| MS5 | 1,440,732,096 | 19,411,597,664 |
| MS4.3 | 1,301,347,392 | 17,300,027,744 |
| MS4 | 1,262,025,792 | 16,105,306,464 |
| MS3.7 | 1,164,508,224 | 14,999,058,784 |
| MS3.5 | 1,087,568,960 | 13,893,466,464 |
| MS3 | 857,799,744 | 12,787,218,784 |
| MS2.7 | 774,700,096 | 10,582,485,344 |

---

## 10. The GGUF file (`write_gguf.py`)

- Metadata: the BF16 GGUF's keys in its order, with the MTP block dropped (`block_count` = trunk layers, `nextn_predict_layers`
  removed, `attention.recurrent_layers` trimmed); `general.name` = `<model>-<tier>`; `general.file_type` 42 (mixed MS types).
  Then the format's metadata keys: `msffn.hadamard`, `msffn.had_in_*` / `msffn.had_h_*` (FFN rotations), `msffn.rot_weight_names`,
  `msffn.rot_seeds`, `msffn.rot_blocks`, `prism.hadamard.*` (version, block sizes, sign vectors per distinct (seed, width),
  `weight_names`, `weight_block_sizes`, `weight_sign_ids`, and for the tied table `inverse_*` and `tied_output`),
  `prism.hadamard.gdn_v_grouped` (27B), `msffn.emb_split_rows` / `msffn.head_split_rows`, `msffn.emb_split_order`,
  `msffn.quant_name`, `msffn.code` = "1mad", `msffn.bits_per_ffn_weight`, `msffn.code_bits_per_ffn_weight`,
  `msffn.bits_per_ms_weight`, `msffn.code_bits_per_ms_weight`, `msffn.ms_tensor_count`, `msffn.nonffn_quant`,
  `msffn.token_embd_source`, `msffn.copy_source`; last `general.author`, `general.url`, `general.description`. These key names
  are the format's; the engine reads them.
- Tensors: the BF16 file's order with the MTP block dropped; MS tensors as their blocks; `token_embd_lo` / `output_lo` after the
  others; every other tensor copied byte for byte (from the BF16 GGUF, or for the 27B from the unsloth files of section 8).
- Value heads (27B: 16 key heads, 48 value heads per linear-attention layer): llama.cpp's converter stores the value heads in
  tiled order, while the codes are in the checkpoint's grouped order. The value rows of attn_qkv and all rows of attn_gate are
  permuted to the tiled order (a row's blocks move with the row); ssm_out, being rotated on its input, stays in grouped order and
  `prism.hadamard.gdn_v_grouped = true` tells the engine to permute its activation instead.
- The file is read back and every tensor compared with its source byte for byte.

The released 27B MS2.7 file carries a development name in `msffn.quant_name`; a file written by this repository from the same
config says "MS2.7". That string is the only header difference expected for that file; the tensors are not affected.

---

## 11. The MTP file (Qwen3.8-27B)

`mtp-Qwen3.8-27B-MS.gguf` (302,452,608 bytes) holds only the MTP block (`blk.64`, 15 tensors) and borrows the loaded tier's token
table and output head at run time (an engine feature). Its attention and FFN matrices are MS-coded at 5 bits with rotation
against the ORIGINAL model (its hidden states and its exact table; one block for every tier), `eh_proj` is Q8_0, the norm vectors
F32. It was made by a separate script with its own coding settings (a 64-column search window, a staged fit of the
block's matrices); that script is not part of this repository, and this repository cannot produce the MTP file.

---

## 12. Compute

On the RTX Spark laptop GPU (NVIDIA GB10, 128 GB unified memory; Windows on ARM) with the CUDA kernels:

- Qwen3.5-2B: about 32 minutes per tier (MS4: 1,894 s); peak GPU memory about 37 GB (both residual streams of 2.1M tokens,
  the attention cores and the statistics).
- Qwen3.8-27B: about 5 to 5.5 hours per tier (MS3.5, bf16 statistics: 17,451 s; tiers with TF32 statistics about 19,750 s);
  peak GPU memory about 80 GB (79.5 GB allocated), with the attention cores recomputed instead of stored and the weights
  read one layer at a time.

Most of the time goes to the FFN statistics GEMMs (about 46% of a TF32 27B tier) and the forwards of the two streams; the
trellis search took 11% (2B MS4: 209 s) to 14% (27B MS3.5: 2,412 s for 25.6 billion weights).

---

## 13. Prior work and credit

- **QTIP** (A. Tseng, Q. Sun, D. Hou, C. De Sa, NeurIPS 2024, arXiv:2406.11235): the bitshift-trellis construction and the 1MAD
  decoding function (section 2.2). QTIP also pairs its trellis quantizer with LDLQ-style feedback (BlockLDLQ) and tail-biting
  trellises. No QTIP code is used.
- **QuIP / QuIP#** (Chee, Cai, Kuleshov, De Sa, NeurIPS 2023; Tseng, Chee, Sun, Kuleshov, De Sa, ICML 2024): incoherence
  processing with random-sign Hadamard transforms (section 3). No QuIP# code is used.
- **GPTQ / LDLQ / OBQ** (Frantar, Ashkboos, Hoefler, Alistarh, ICLR 2023; Chee et al., NeurIPS 2023; Frantar, Singh,
  Alistarh, NeurIPS 2022): the Hessian-based error feedback and its upper Cholesky form (section 4.3).
- **Viterbi** (A. J. Viterbi, 1967): the dynamic-programming search over the trellis (section 4.4); trellis-coded quantization
  (Marcellin and Fischer, 1990).
- **GPTAQ** (Y. Li, R. Yin, D. Lee, S. Xiao, P. Panda, "GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric
  Calibration", ICML 2025, arXiv:2504.02692): the asymmetric calibration objective of matching the original model's outputs on the original
  inputs while the quantized layer sees the converted model's inputs; MS uses this objective in the closed-form input maps of
  section 6.3 (steps 1 and 3).
- **Unsloth** (Unsloth Dynamic GGUFs, unsloth/Qwen3.8-27B-GGUF, Apache 2.0): the per-tensor allocation the 27B tiers start from
  (section 9.2) and the tensors copied into the 27B files (section 8).
- **Hugging Face transformers** (Apache 2.0): the Qwen3.5 modules used for the forwards; `madshift/gdn.py` is derived from its
  Gated DeltaNet implementation. **llama.cpp** (MIT): the GGUF format, `gguf-py`, the BF16 GGUF conversion, and the
  `prism.hadamard` input-rotation mechanism (PR #29600) that MS's rotation metadata extends (section 3).
- **Adaptive softmax** (Grave, Joulin, Cissé, Grangier, Jégou, ICML 2017) and **adaptive input representations** (Baevski and
  Auli, ICLR 2019) split a vocabulary by token frequency; MS's split tables and heads (section 6.2) store two row groups at two
  code widths.
- Related work not used: **YAQA** (adaptive rounding against Kronecker-factored approximations of the full model's KL Hessian,
  Tseng et al., 2025): MS does not use it and is not equivalent to it.

The rest is MS's own, by Matthew Priester: the format of section 2 beyond the bitshift trellis and the 1MAD function (14-bit
state, 512-weight row blocks with stored start states and fp16 scales, k = 1..6 per tensor), the search of section 4.4 as
specified there, the calibration recipe (section 5), the conversion of section 6 as specified there (its closed-form input
maps, the ridge refits of the output and down projections toward the original residual stream, and the FFN target that
changes at half depth), the split tables and heads and their row orders (section 6.2), the use of the
copied tensors' values during conversion (section 8), the probe solver and the fill step (section 9), and the file layout
(section 10).

---

## 14. Reproducing a released file

`convert.py` + `write_gguf.py` with a config from `configs/`, the calibration pools of `calibration/MANIFEST.json`, the same
checkpoint revision (Qwen3.5-2B `15852e8c`, Qwen3.8-27B `1d4bf0f2`, as in each file's `general.finetune`), for the 27B the
unsloth files of section 8, and the same GPU and software versions (the released files: PyTorch 2.14 with CUDA 13.4,
transformers 5.16, an NVIDIA GB10) reproduce the coded tensors. Floating-point GEMMs differ across GPUs and library versions;
on other hardware the result is an equivalent conversion, not the same bytes. Verify with llama-perplexity's KL divergence
against the BF16 model (README.md).

Check made with this repository: Qwen3.5-2B MS4, converted with `configs/qwen3.5-2b/MS4.json` and the released calibration pools
on the machine and software that made the release (1,789 s), was byte for byte identical to the released file
(SHA-256 `ad20ec95f126bfdada3ab49db3ede64a8dee3b0dd991ad4d5bc61e9f971d5dfd`), header and tensors. The Qwen3.8-27B path
(the same code plus the steps of sections 6.2 and 8) was not re-run end to end with this repository (about 5 hours per tier);
its allocation tools were (section 9.2).
