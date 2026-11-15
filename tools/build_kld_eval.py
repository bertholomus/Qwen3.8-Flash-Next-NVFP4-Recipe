#!/usr/bin/env python3
"""Build the KLD evaluation set (kld_eval.jsonl), deterministically.

Mix, in this fixed order: WikiText-2 raw test articles (prose, the common published-KLD
baseline), then TensorFold v1.2.6 source + docs files over 2000 chars (code + technical
prose, sorted by path), round-robin interleaved so the first --nseq x --seq-len tokens are
a prose/code mix, not prose only. Needs pyarrow.
"""
import argparse
import hashlib
import itertools
import json
import os

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser(description="build the KLD evaluation set (kld_eval.jsonl)")
    ap.add_argument("--wt2-parquet", required=True,
                    help="parquet with a 'text' column holding raw WikiText-2 test lines")
    ap.add_argument("--tensorfold-src", required=True,
                    help="path to a TensorFold v1.2.6 source tree (zig/src and docs are used)")
    ap.add_argument("--out", default="kld_eval.jsonl")
    args = ap.parse_args()

    docs = []
    # WikiText-2 test: join lines into articles at top-level " = Title = " headers
    lines = pq.read_table(args.wt2_parquet).column("text").to_pylist()
    cur = []
    for ln in lines:
        if ln.startswith(" = ") and not ln.startswith(" = = ") and cur:
            docs.append(("wikitext2", "".join(cur)))
            cur = []
        cur.append(ln)
    if cur:
        docs.append(("wikitext2", "".join(cur)))
    # TensorFold v1.2.6 code + docs
    root = args.tensorfold_src
    paths = []
    for d in ("zig/src", "docs"):
        for dp, _, fs in os.walk(os.path.join(root, d)):
            for f in fs:
                if f.endswith((".zig", ".md")):
                    paths.append(os.path.join(dp, f))
    for p in sorted(paths):
        t = open(p, encoding="utf-8", errors="replace").read()
        if len(t) > 2000:
            docs.append(("tensorfold-1.2.6:" + os.path.relpath(p, root), t))
    wt = [d for d in docs if d[0] == "wikitext2"]
    tf = [d for d in docs if d[0] != "wikitext2"]
    docs = [d for pair in itertools.zip_longest(wt, tf) for d in pair if d is not None]
    with open(args.out, "w") as f:
        for src, t in docs:
            f.write(json.dumps({"source": src, "text": t}) + "\n")
    h = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(json.dumps({"file": args.out, "docs": len(docs),
                      "wikitext2": sum(s == "wikitext2" for s, _ in docs),
                      "chars": sum(len(t) for _, t in docs), "sha256": h}))


if __name__ == "__main__":
    main()
