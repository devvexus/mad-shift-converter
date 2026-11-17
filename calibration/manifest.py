# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Matthew Priester
"""Check a calibration directory against MANIFEST.json (the pools the released files were converted with), or print the
fingerprints of a directory. A pool's fingerprint is the SHA-256 of its token ids as little-endian int64, row-major.

usage: python calibration/manifest.py calib/qwen3.5-2b --model qwen3.5-2b
       python calibration/manifest.py calib/qwen3.5-2b --print"""
import os, sys, json, hashlib, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def fingerprint(path):
    x = torch.load(path)
    a = np.ascontiguousarray(x.numpy().astype("<i8"))
    return dict(shape=list(a.shape), sha256=hashlib.sha256(a.tobytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir"); ap.add_argument("--model", default=None); ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    if a.print:
        out = {f: fingerprint(os.path.join(a.dir, f)) for f in sorted(os.listdir(a.dir)) if f.endswith(".pt")}
        print(json.dumps(out, indent=1)); return
    man = json.load(open(os.path.join(HERE, "MANIFEST.json")))["models"][a.model]["pools"]
    bad = 0
    for f, want in man.items():
        p = os.path.join(a.dir, f)
        if not os.path.exists(p):
            print(f"MISSING  {f}"); bad += 1; continue
        got = fingerprint(p)
        ok = got == {k: want[k] for k in ("shape", "sha256")}
        print(f"{'OK      ' if ok else 'DIFFERS '} {f}")
        bad += not ok
    print("all pools identical to the released conversions" if not bad else f"{bad} pools missing or different")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
