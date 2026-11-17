# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Matthew Priester
"""Real-text calibration pools, read from the Hugging Face datasets-server rows API (the first rows of each source's train
split; no whole-file downloads). Documents are tokenized with the model's tokenizer (each followed by "\\n\\n", at most 1,536
tokens per document), joined, and cut into 512 rows of 512 tokens -> <out>/real_<name><tag>_512x512.pt (int64 [512, 512]).

  web   HuggingFaceFW/fineweb-edu, config sample-10BT, column text
  wiki  wikimedia/wikipedia, config 20231101.en, column text
  code  codeparrot/github-code-clean, column code: Python 30%, JavaScript 10%, TypeScript 8%, Java 10%, C++ 10%, C 8%, Go 7%,
        Rust 7%, Shell 5%, SQL 5% of the tokens (configs the rows API does not serve are skipped; the first source that worked
        tops the pool up)
  math  open-web-math/open-web-math, column text

The released files used: real_{web,wiki,code,math}_512x512.pt (rows from 0) and real_{web,wiki}2_512x512.pt (--offset 30000
--tag 2). --exclude: text files; a document whose first 200 characters occur in one of them is skipped.

usage: python calibration/realtext.py Qwen/Qwen3.5-2B --out calib/qwen3.5-2b
       python calibration/realtext.py Qwen/Qwen3.5-2B --out calib/qwen3.5-2b --offset 30000 --tag 2 --only web,wiki
The rows API serves the datasets' current contents; calibration/MANIFEST.json has the SHA-256 of the released pools."""
import os, sys, time, json, argparse

SRC = {"web": [("HuggingFaceFW/fineweb-edu", "sample-10BT", "text", 1.0)],
       "wiki": [("wikimedia/wikipedia", "20231101.en", "text", 1.0)],
       "code": [("codeparrot/github-code-clean", c, "code", q) for c, q in
                (("Python-all", .3), ("JavaScript-all", .1), ("TypeScript-all", .08), ("Java-all", .1), ("C++-all", .1),
                 ("C-all", .08), ("GO-all", .07), ("Rust-all", .07), ("Shell-all", .05), ("SQL-all", .05))],
       "math": [("open-web-math/open-web-math", "default", "text", 1.0)]}


def rows(dataset, config, offset, length=100):
    import urllib.request, urllib.parse
    q = urllib.parse.urlencode(dict(dataset=dataset, config=config, split="train", offset=offset, length=length))
    req = urllib.request.Request("https://datasets-server.huggingface.co/rows?" + q, headers={"User-Agent": "madshift-calibration/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return [x["row"] for x in json.load(r)["rows"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="Hugging Face model id or directory (for its tokenizer)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--offset", type=int, default=0, help="first source row")
    ap.add_argument("--tag", default="", help="pool name suffix: real_<name><tag>_512x512.pt")
    ap.add_argument("--only", default="", help="comma list of web,wiki,code,math")
    ap.add_argument("--exclude", nargs="*", default=[], help="text files whose documents must not be used")
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    os.makedirs(a.out, exist_ok=True)
    only = a.only.split(",") if a.only else None
    excl = "".join(open(p, encoding="utf-8").read() for p in a.exclude)
    seen = lambda txt: bool(excl) and len(txt.strip()) >= 80 and txt.strip()[:200] in excl
    skipped = [0]
    for name, parts in SRC.items():
        if only and name not in only:
            continue
        out = os.path.join(a.out, f"real_{name}{a.tag}_512x512.pt")
        if os.path.exists(out):
            print("exists", out); continue
        t0 = time.time()
        need = a.rows * 512
        ids, ends = [], {}
        for dataset, config, col, q in parts:
            part, off, quota = [], a.offset, int(need * q) + 1
            try:
                while len(part) < quota and off < a.offset + 5000:
                    for r in rows(dataset, config, off):
                        if r.get(col) and seen(r[col]):
                            skipped[0] += 1
                        elif r.get(col):
                            part += tok(r[col] + "\n\n", add_special_tokens=False)["input_ids"][:1536]
                    off += 100
            except Exception as e:                       # e.g. configs the rows API does not serve: skipped
                print(f"{name}: {dataset} {config}: skipped ({str(e)[:60]})", flush=True)
                continue
            ends[(dataset, config, col)] = off
            print(f"{name}: {dataset} {config}: rows {a.offset}..{off - 1}, {len(part)} tokens ({time.time() - t0:.0f}s)", flush=True)
            ids += part[:quota]
        if len(ids) < need and ends:                     # top up from the first source that worked, continuing after its rows
            (dataset, config, col), off = next(iter(ends.items()))
            while len(ids) < need and off < a.offset + 20000:
                for r in rows(dataset, config, off):
                    if r.get(col) and seen(r[col]):
                        skipped[0] += 1
                    elif r.get(col):
                        ids += tok(r[col] + "\n\n", add_special_tokens=False)["input_ids"][:1536]
                off += 100
            print(f"{name}: topped up from {config} to row {off - 1}", flush=True)
        if len(ids) < need:
            raise SystemExit(f"{name}: only {len(ids)} tokens")
        x = torch.tensor(ids[:need], dtype=torch.int64).reshape(a.rows, 512)
        torch.save(x, out)
        print(f"saved {out} {tuple(x.shape)}; {skipped[0]} excluded documents skipped so far", flush=True)


if __name__ == "__main__":
    main()
