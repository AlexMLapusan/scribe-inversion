#!/usr/bin/env python3
"""
Build a PubMed evaluation CSV comparable to Zero2Text's PubMed setting.

Zero2Text (arXiv 2602.01757) cites PubMed = Cohan et al. 2018 (the PMC
articles+abstracts summarization dataset), evaluates on 200 sampled texts, and
says sampling "follows ALGEN". It does NOT release the subset, the random seed,
the source split, the source-truncation rule (their T=32 is the decoder's max
generation length, not a stated source cut), or any preprocessing, and ships no
code/data. This rebuild is therefore COMPARABLE, not identical; every knob is
fixed by a disclosed convention:

  - Source : ccdv/pubmed-summarization (Cohan 2018), 'abstract' field, test split
             (falls back to armanc/scientific_papers config 'pubmed').
  - Filter : keep abstracts with >= --min-tokens tokenizer tokens so the
             truncation is meaningful (drops stubs).
  - Truncate: first --max-tokens tokens under --tokenizer (default t5-base, i.e.
             the GTR-Base tokenizer), so each source is exactly the length the
             GTR victim encoder sees. Matches Zero2Text's T=32 decoder cap and
             this repo's msmarco/enron '32tok' construction.
  - Sample : uniform random --n rows with fixed --seed (default 42), no other
             filtering.

Output: dataset_csvs/pubmed_<n>_<max-tokens>tok.csv with a single 'text' column
(consumed by load_texts_from_csv in invert_gpt2.py / zsinvert.py; --dataset
pubmed resolves to pubmed_500_32tok.csv).

Usage:
    python build_pubmed_dataset.py               # 500 rows, 32 t5 tokens, seed 42
    python build_pubmed_dataset.py --n 200       # match Zero2Text's N exactly
"""

import argparse
import csv
import random
import re
import sys
from pathlib import Path


def clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def load_pubmed():
    from datasets import load_dataset
    attempts = [
        ("ccdv/pubmed-summarization", dict(split="test")),
        ("ccdv/pubmed-summarization", dict(split="test", trust_remote_code=True)),
        ("armanc/scientific_papers", dict(name="pubmed", split="test", trust_remote_code=True)),
        ("scientific_papers", dict(name="pubmed", split="test", trust_remote_code=True)),
    ]
    last = None
    for repo, kwargs in attempts:
        try:
            print(f"[load] {repo} {kwargs}", flush=True)
            return load_dataset(repo, **kwargs)
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:150]}"
            print(f"  failed: {last}", flush=True)
    sys.exit(f"Could not load a PubMed source ({last}). Try `pip install -U datasets` and check HF network access.")


def main():
    ap = argparse.ArgumentParser(
        description="Build a PubMed eval CSV comparable to (not identical to) Zero2Text's PubMed setting.")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--min-tokens", type=int, default=40)
    ap.add_argument("--tokenizer", type=str, default="t5-base")
    ap.add_argument("--field", type=str, default="abstract")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    ds = load_pubmed()
    field = args.field if args.field in ds.column_names else (
        "abstract" if "abstract" in ds.column_names else ds.column_names[0])
    print(f"[field] using '{field}'  (available: {ds.column_names})", flush=True)

    kept = []
    for row in ds:
        t = clean(row[field])
        if not t:
            continue
        ids = tok.encode(t, add_special_tokens=False)
        if len(ids) >= args.min_tokens:
            snippet = tok.decode(ids[:args.max_tokens], skip_special_tokens=True).strip()
            if snippet:
                kept.append(snippet)
    print(f"[filter] {len(kept)} abstracts with >= {args.min_tokens} {args.tokenizer} tokens", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    picked = kept[:args.n]
    if len(picked) < args.n:
        print(f"[warn] only {len(picked)} rows available (< requested {args.n})", flush=True)

    out = Path(args.output or f"dataset_csvs/pubmed_{args.n}_{args.max_tokens}tok.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["text"])
        for t in picked:
            w.writerow([t])

    tl = sorted(len(tok.encode(t, add_special_tokens=False)) for t in picked)
    wl = sorted(len(t.split()) for t in picked)
    print(f"[write] {out}  ({len(picked)} rows, seed={args.seed})", flush=True)
    print(f"[check] {args.tokenizer} tokens: min={tl[0]} median={tl[len(tl)//2]} max={tl[-1]}  |  "
          f"words median={wl[len(wl)//2]}", flush=True)
    print(f"[sample] {picked[0][:120]}", flush=True)


if __name__ == "__main__":
    main()
