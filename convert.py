# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Matthew Priester
"""Convert a Hugging Face Qwen3.5-family checkpoint to one MS tier (the coded tensors; write_gguf.py then writes the GGUF).

usage:
  python convert.py configs/qwen3.5-2b/MS4.json --calib-dir calib/qwen3.5-2b --work work/qwen3.5-2b
  python convert.py configs/qwen3.8-27b/MS4.json --calib-dir calib/qwen3.8-27b --work work/qwen3.8-27b --gguf-dir unsloth/

The model is read from the local Hugging Face cache (HF_HOME / HF_HUB_CACHE) or from --model-dir. Output: <work>/<tier>/
(emb.pt and L<l>.pt: the coded entries as MS blocks, state.json); <work>/cache/ holds the pre-pass statistics shared by all
tiers of a model. Interrupted runs resume when the same command is run again."""
import os, sys, json, time, argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="tier config (configs/<model>/<tier>.json)")
    ap.add_argument("--calib-dir", required=True, help="directory with the calibration token pools (calibration/)")
    ap.add_argument("--work", required=True, help="output directory for this model's tiers")
    ap.add_argument("--model-dir", default=None, help="local checkpoint directory (default: the config's model in the HF cache)")
    ap.add_argument("--gguf-dir", default=None, help="directory with the reference GGUF files named by copy_from / table_from")
    ap.add_argument("--core", default="auto", choices=["auto", "gpu", "recompute"],
                    help="keep the attention core of all calibration tokens on the GPU, or recompute it (same values)")
    ap.add_argument("--cuda-graphs", action="store_true", help="replay the per-batch layer pieces as CUDA graphs (same kernels)")
    ap.add_argument("--reference-search", action="store_true",
                    help="use the PyTorch trellis search instead of the CUDA kernels (same results, much slower)")
    a = ap.parse_args()
    if a.reference_search:
        os.environ["MADSHIFT_REFERENCE_SEARCH"] = "1"
    from madshift import pipeline as PL, coding as K, forward as FW
    cfg = json.load(open(a.config))
    if not K.USE_KERNELS:
        print("trellis search: PyTorch reference", flush=True)
    else:
        from madshift import kernels
        if not kernels.available():
            raise SystemExit(f"{kernels.DLL} not found: build it (madshift/build_kernels.bat or build_kernels.sh) or pass --reference-search")
    FW.GRAPHS = a.cuda_graphs
    os.makedirs(a.work, exist_ok=True)
    logf = open(os.path.join(a.work, f"{cfg['name']}.log"), "a")

    def log(s):
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    t0 = time.time()
    tier = PL.Tier(cfg, os.path.join(a.work, cfg["name"]), a.gguf_dir)
    p = PL.Pass(a.model_dir or cfg["model"], a.calib_dir, tier, os.path.join(a.work, "cache"), core_policy=a.core, log=log)
    p.run()
    b = " ".join(f"{c} {v[0] / v[1]:.2f}" for c, v in tier.bits.items())
    log(f"DONE {tier.name} in {time.time() - t0:.0f}s | bits per weight by class: {b} | trellis search {K.STATS['search_s']:.0f}s "
        f"for {K.STATS['search_w'] / 1e6:.0f}M weights | output {tier.pack_dir}")


if __name__ == "__main__":
    main()
