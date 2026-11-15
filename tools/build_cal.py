#!/usr/bin/env python3
"""Build the GPTQ calibration set (cal.jsonl), deterministically.

WikiText-2 TRAIN articles (the KLD evaluation set uses TEST) + Python source files from an
exllamav3 checkout, round-robin interleaved so early batches mix prose and code. Asserts
zero shared texts with the KLD evaluation set. Inputs are read-only; nothing is
redistributed by this script.
"""
import argparse
import hashlib
import itertools
import json
import os

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser(description="build the GPTQ calibration set (cal.jsonl)")
    ap.add_argument("--wt2-parquet", required=True,
                    help="parquet with a 'text' column holding raw WikiText-2 train lines")
    ap.add_argument("--exllamav3-src", required=True,
                    help="path to an exllamav3 source checkout (its .py files become calibration code)")
    ap.add_argument("--kld-eval", required=True,
                    help="kld_eval.jsonl built by build_kld_eval.py; overlap with it must be zero")
    ap.add_argument("--out", default="cal.jsonl")
    args = ap.parse_args()

    lines = pq.read_table(args.wt2_parquet).column("text").to_pylist()
    wt, cur = [], []
    for ln in lines:
        if ln.startswith(" = ") and not ln.startswith(" = = ") and cur:
            wt.append(("wikitext2-train", "".join(cur)))
            cur = []
        cur.append(ln)
    if cur:
        wt.append(("wikitext2-train", "".join(cur)))
    code = []
    for dp, _, fs in os.walk(args.exllamav3_src):
        for f in sorted(fs):
            if f.endswith(".py"):
                p = os.path.join(dp, f)
                t = open(p, encoding="utf-8", errors="replace").read()
                if len(t) > 2000:
                    code.append(("exllamav3:" + os.path.relpath(p, args.exllamav3_src), t))
    code.sort()
    docs = [d for pair in itertools.zip_longest(wt, code) for d in pair if d is not None]
    ev = {hashlib.sha256(json.loads(l)["text"].encode()).hexdigest() for l in open(args.kld_eval)}
    dropped = [(s, t[:60]) for s, t in docs if hashlib.sha256(t.encode()).hexdigest() in ev]
    docs = [(s, t) for s, t in docs if hashlib.sha256(t.encode()).hexdigest() not in ev]  # drop exact duplicates of eval texts
    assert not [1 for s, t in docs if hashlib.sha256(t.encode()).hexdigest() in ev]
    with open(args.out, "w") as f:
        for s, t in docs:
            f.write(json.dumps({"source": s, "text": t}) + "\n")
    print(json.dumps({"file": args.out, "docs": len(docs), "prose": len(wt), "code": len(code),
                      "overlap_with_eval": 0, "dropped_dupes": dropped,
                      "sha256": hashlib.sha256(open(args.out, "rb").read()).hexdigest()}))


if __name__ == "__main__":
    main()
