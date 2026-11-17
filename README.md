# MS (Mad Shift) converter

The converter that made the released MS (Mad Shift) files of
[Qwen3.5-2B](https://huggingface.co/devvexus/Qwen3.5-2B-MS) and [Qwen3.8-27B](https://huggingface.co/devvexus/Qwen3.8-27B-MS):
seven tiers each (MS2.7, MS3, MS3.5, MS3.7, MS4, MS4.3, MS5). MS is a trellis-coded GGUF quantization format by Matthew Priester.
Every large matrix is stored as a bitshift trellis code with 1 to 6 bits per weight after a random Hadamard rotation, and is
converted layer by layer against calibration statistics of the partly converted model. The files run on the MS llama.cpp fork,
[ms-llama.cpp](https://github.com/devvexus/ms-llama.cpp); stock llama.cpp cannot load them.

**[METHOD.md](METHOD.md)** describes the whole method precisely. The bitshift-trellis construction and the 1MAD decoding
function are from QTIP (Tseng, Sun, Hou, De Sa, NeurIPS 2024); the 27B tiers start from unsloth's per-tensor allocation and
copy unsloth's small tensors and input table (METHOD.md, sections 8, 9 and 13).

## Requirements

- An NVIDIA GPU with CUDA and enough memory for two residual streams of 2.1 million calibration tokens:
  about 40 GB for Qwen3.5-2B and about 80 GB for Qwen3.8-27B (the released files were made on an RTX Spark laptop GPU,
  NVIDIA GB10, 128 GB unified memory).
- Time on that machine: about 30 minutes per Qwen3.5-2B tier, about 5 hours per Qwen3.8-27B tier.
- Disk: the Hugging Face checkpoint (2B: 4.6 GB, 27B: 56 GB), the model's BF16 GGUF (2B: 3.9 GB, 27B: 55 GB), the converted
  tier (about its file size) and the GGUF; for the 27B also unsloth's GGUF files of the tiers you make.
- Python 3.10 or newer with a CUDA build of PyTorch, `transformers` with Qwen3.5 support, `safetensors`, `numpy`
  (`requirements.txt`), and the `gguf-py` of [ms-llama.cpp](https://github.com/devvexus/ms-llama.cpp) (it knows the MS tensor
  types). The released files were made with Python 3.13, PyTorch 2.14 (CUDA 13.4) and transformers 5.16.
- The CUDA Toolkit and a C++ compiler for the trellis-search kernels (Visual Studio on Windows, gcc on Linux).
- llama.cpp's `convert_hf_to_gguf.py` (in ms-llama.cpp) to make the BF16 GGUF, and the ms-llama.cpp binaries to check the result.

## Install

```sh
pip install -r requirements.txt
git clone https://github.com/devvexus/ms-llama.cpp
pip install ./ms-llama.cpp/gguf-py
# trellis-search kernels (CUDA_ARCH defaults to this machine's GPU):
madshift\build_kernels.bat        # Windows
sh madshift/build_kernels.sh      # Linux
```

Without the kernels, `convert.py --reference-search` runs the same search in PyTorch (same results, much slower).

## Convert

1. **Model.** Download the checkpoint into the Hugging Face cache (`hf download Qwen/Qwen3.5-2B`), or pass
   `--model-dir`. Make its BF16 GGUF with ms-llama.cpp's `convert_hf_to_gguf.py --outtype bf16`.
2. **Calibration pools** (section 5 of METHOD.md), written into one directory per model:
   ```sh
   python calibration/selfgen.py Qwen/Qwen3.5-2B --out calib/qwen3.5-2b
   python calibration/realtext.py Qwen/Qwen3.5-2B --out calib/qwen3.5-2b
   python calibration/realtext.py Qwen/Qwen3.5-2B --out calib/qwen3.5-2b --offset 30000 --tag 2 --only web,wiki
   python calibration/manifest.py calib/qwen3.5-2b --model qwen3.5-2b
   ```
   `manifest.py` says whether your pools are the ones the released files were made with. Model-written text depends on the GPU
   and the library versions, and the rows API serves the datasets' current contents, so regenerated pools can differ; the
   conversion is then equivalent, not identical.
3. **Qwen3.8-27B only:** unsloth's GGUFs named in the tier configs (`copy_from`, `table_from`), from
   [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) at revision `4ca72078` (MS4.3's UD-Q4_K_XL: revision
   `ce331efc`), in one directory (`--gguf-dir`).
4. **Convert and write one tier:**
   ```sh
   python convert.py configs/qwen3.5-2b/MS4.json --calib-dir calib/qwen3.5-2b --work work/qwen3.5-2b
   python write_gguf.py configs/qwen3.5-2b/MS4.json Qwen3.5-2B-BF16.gguf work/qwen3.5-2b out/Qwen3.5-2B-MS4.gguf

   python convert.py configs/qwen3.8-27b/MS4.json --calib-dir calib/qwen3.8-27b --work work/qwen3.8-27b --gguf-dir unsloth
   python write_gguf.py configs/qwen3.8-27b/MS4.json Qwen3.8-27B-BF16.gguf work/qwen3.8-27b out/Qwen3.8-27B-MS4.gguf --gguf-dir unsloth
   ```
   Any other tier: replace `MS4` with `MS2.7`, `MS3`, `MS3.5`, `MS3.7`, `MS4.3` or `MS5`. An interrupted `convert.py` resumes
   when the same command runs again. The calibration statistics of a model are cached in `<work>/cache` and shared by its tiers.
   `--cuda-graphs` replays the per-batch forwards as CUDA graphs (faster on small models; the released 2B tiers were converted
   with it, the 27B tiers without).

The configs in `configs/` are the released tiers: per-tensor bit widths, table / head split, and the conversion settings.
`allocation/` has the tools that made them: `solve_probes.py` (Qwen3.5-2B: probe measurements and a greedy KL-per-byte
solver), `unsloth_widths.py`, `fill.py` and `size.py` (Qwen3.8-27B: unsloth's per-tensor types matched in MS widths, then the spare bytes
spent by KL per byte), with the measurements they use in `allocation/data/`.

## What is in this repository

| path | what |
|---|---|
| `convert.py` | converts one tier: the layer-by-layer conversion, output in `<work>/<tier>/` |
| `write_gguf.py` | writes the MS GGUF from a converted tier and the model's BF16 GGUF |
| `madshift/` | the converter: `pipeline.py` (the pass over the layers), `coding.py` (coding one matrix, one FFN), `trellis.py` (the trellis code and the reference search), `kernels.cu` / `kernels.py` (the same search in CUDA), `blocks.py` (the MS block encoding), `model.py`, `forward.py`, `gdn.py` (the checkpoint and its layers), `reference_gguf.py` (tensors taken from unsloth's GGUFs), `linalg.py` |
| `configs/` | the released tiers of Qwen3.5-2B and Qwen3.8-27B |
| `calibration/` | the scripts that make the calibration pools, and `MANIFEST.json` with the released pools' fingerprints |
| `allocation/` | the tools that chose the per-tensor bit widths of the configs, and their measurement data |
| `METHOD.md` | the method |

## Check a result

Measure the KL divergence from the BF16 model with ms-llama.cpp's `llama-perplexity`, as the model cards do (four English texts,
[devvexus/ms-eval-texts](https://huggingface.co/datasets/devvexus/ms-eval-texts); 512-token context, 128 chunks):

```sh
llama-perplexity -m Qwen3.5-2B-BF16.gguf -f eval_large_wiki.txt -c 512 -b 512 -ub 512 --chunks 128 -ngl 99 --kl-divergence-base wiki.kld
GGML_CUDA_MMMS_ACC=f32 llama-perplexity -m out/Qwen3.5-2B-MS4.gguf -f eval_large_wiki.txt -c 512 -b 512 -ub 512 --chunks 128 -ngl 99 \
    --kl-divergence-base wiki.kld --kl-divergence
```

The "Mean KLD" line should match the model card's number for that tier and text within run-to-run noise (Qwen3.5-2B MS4, wiki:
0.00323). Check made for this repository: Qwen3.5-2B MS4 converted with the released calibration pools on the machine and
software that made the release came out byte for byte identical to the released file (same SHA-256).

## Credit and citation

MS (Mad Shift) and this converter: Matthew Priester (https://github.com/devvexus, ORCID 0009-0000-0032-5292). Please cite
`CITATION.cff`, and QTIP for the bitshift trellis and the 1MAD code:
A. Tseng, Q. Sun, D. Hou, C. De Sa, "QTIP: Quantization with Trellises and Incoherence Processing", NeurIPS 2024, arXiv:2406.11235.

Technical report: Priester, M. (2026). MS (Mad Shift): Trellis-Coded GGUF Quantization for llama.cpp (Version 1.0). Zenodo. https://doi.org/10.5281/zenodo.24145285

## License

Apache License 2.0 (`LICENSE`). Keep the `NOTICE` file when you redistribute this software or work derived from it.
