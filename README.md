# SCRIBE

Reference implementation of the three-phase, query-only embedding inversion
attack described in the paper. The attacker holds a stored embedding `e*` and
black-box query access to the embedding model that produced it, and
reconstructs the source text.

## Pipeline

| stage | script | produces |
|---|---|---|
| Phase 1 | `invert_gpt2.py` | KV-cache seed optimisation, column `c1` |
| Phase 2 | `hill_climb.py` | token substitution search, column `c1_hc` |
| Phase 3 | `refine_seeds.py` | LLM beam refinement, column `refined_text` |

Each stage writes a CSV that the next stage reads.

```bash
# Phase 1
python invert_gpt2.py --encoder-model facebook/contriever --dataset mimic \
    --sentence-iterations 35 --num-boost-tokens 100 --output out/phase1.csv

# Phase 2
python hill_climb.py --input out/phase1.csv --output out/phase2.csv \
    --column c1 --encoder contriever --top-k 150 --max-passes 7 \
    --min-pass-gain 0.001 --llm-pool-size 200 --saliency --min-token-length 4

# Phase 3
python refine_seeds.py --input out/phase2.csv --output out/phase3.csv \
    --column c1_hc --target-column source_text --encoder contriever \
    --llm Qwen/Qwen2.5-0.5B-Instruct
```

The flags above are the values used for the reported results.
`SentenceTransformerWrapper.py` wraps the target model and counts every token
submitted to it, which is the cost metric in the paper. `run_metadata.py`
records the arguments of each run.

## Data

The clinical corpora derive from MIMIC-III and MIMIC-IV and are **not**
included here. They are available from PhysioNet under its credentialed data
use agreement and may not be redistributed. Enron, MS MARCO and PubMed are
public. Every patient name used in our experiments is a synthetic surrogate;
no real identifier appears in this repository.

## Requirements

```bash
pip install -r requirements.txt
```

## Use of AI assistants

Coding assistants based on large language models were used during development
for boilerplate and templating work: argument parsing, CSV and checkpoint
plumbing, docstrings, and repetitive refactoring. The attack design, the
experimental protocol, and the analysis are the authors' own, and the authors
are responsible for the correctness of all code and results reported in the
paper.
