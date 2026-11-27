# SCRIBE

Code for the paper "Compliant but Not Anonymous: Zero-Shot Inversion of
Clinical Embeddings with SCRIBE" (EACL 2027).

SCRIBE is a three-phase, query-only embedding inversion attack. The attacker
holds a stored embedding `e*` and black-box query access to the embedding
model that produced it, and reconstructs the source text. No paired data and
no training are needed.

## Repository layout

All scripts sit at the top level and are run from there.

| purpose | files |
|---|---|
| Attack | `invert_gpt2.py` (Phase 1), `hill_climb.py` (Phase 2), `refine_seeds.py` (Phase 3) |
| Shared by the attack | `SentenceTransformerWrapper.py` wraps the target model and counts every token submitted to it, which is the cost metric in the paper. `run_metadata.py` records the arguments of each run. |
| Data preparation | `build_pubmed_dataset.py` |

## Setup

```bash
pip install -r requirements.txt
```

The reported runs used torch 2.5, transformers 4.46 and sentence-transformers
3.1.1 on GPUs with 32 GB of memory. At the settings below a clinical note
takes about 6 minutes on one GPU against the encoder-based models and about
24 minutes against GTE-Qwen2.

## Data

Datasets are CSV files with a single `text` column, placed in `dataset_csvs/`.
`--dataset` takes one of the short names below or a path to any such CSV.

| short name | file | source |
|---|---|---|
| `mimic` | `MIMIC_III_420_32tok.csv` | MIMIC-III notes condensed to summaries of about 32 words |
| `mimic-iv` | `mimic_IV_256tok.csv` | 256-token MIMIC-IV excerpts |
| `enron` | `enron_aeslc_500_32tok.csv` | Enron e-mails |
| `msmarco` | `msmarco_500_32tok.csv` | MS MARCO passages |
| `pubmed` | `pubmed_500_32tok.csv` | PubMed abstracts, built with `build_pubmed_dataset.py` |

The clinical corpora derive from MIMIC-III and MIMIC-IV and are **not**
included here. They are available from PhysioNet under its credentialed data
use agreement and may not be redistributed. Enron, MS MARCO and PubMed are
public. Every patient name used in our experiments is a synthetic surrogate;
no real identifier appears in this repository.

The scripts that prepare the two clinical corpora are not part of this
repository. For these, or for any additional information, please contact the
authors.

## Running the attack

Each phase writes a CSV that the next phase reads.

| stage | script | produces |
|---|---|---|
| Phase 1 | `invert_gpt2.py` | KV-cache seed optimisation, column `c1` |
| Phase 2 | `hill_climb.py` | token substitution search, column `c1_hc` |
| Phase 3 | `refine_seeds.py` | LLM beam refinement, column `refined_text` |

```bash
# Phase 1
python invert_gpt2.py --encoder-model facebook/contriever --dataset mimic \
    --sentence-iterations 35 --sampling-top-k 100 --num-loss-computations 3 \
    --entity-penalty 5 --num-boost-tokens 100 --boost-strength 5 \
    --capitalize-boost 1.2 --max-length-mode auto --max-length-cap 96 \
    --seed 42 --output out/p1.csv

# Phase 2
python hill_climb.py --input out/p1.csv --output out/p2.csv \
    --column c1 --target-column source_text --encoder contriever \
    --top-k 150 --max-passes 7 --min-pass-gain 0.001 --saliency \
    --llm Qwen/Qwen2.5-0.5B-Instruct --llm-pool-size 200

# Phase 3
python refine_seeds.py --input out/p2.csv --output out/p3.csv \
    --column c1_hc --target-column source_text --encoder contriever \
    --llm Qwen/Qwen2.5-0.5B-Instruct --max-length-mode auto --max-length-cap 96
```

The flags above are the values used for the reported results. The defaults of
`hill_climb.py` differ from them, so pass the Phase 2 flags explicitly.

Variations used in the paper:

- **Generation budget.** `--max-length-mode auto --max-length-cap 96` sets the
  per-instance budget and is used on MIMIC-III and PubMed. Leave both flags out
  of Phases 1 and 3 for the fixed budget of 32 steps, which is what Enron and
  MS MARCO use.
- **Other embedding models.** Phase 1 takes the full model name
  (`sentence-transformers/gtr-t5-base`, `thenlper/gte-base`,
  `Alibaba-NLP/gte-Qwen2-1.5B-instruct`). Phases 2 and 3 take `gtr`, `gte`, or
  the full name for GTE-Qwen2. GTE-Qwen2 also needs
  `--use-query-prompt --trust-remote-code` in Phase 1 and `--trust-remote-code`
  in Phase 2.
- **Seeds.** Only Phase 1 samples, so `--seed` goes there.
- **Without Phase 2 (SCRIBE-Lite).** Run Phase 3 directly on the Phase 1 output
  with `--column c1`.
- **Long notes.** On `mimic-iv`, Phase 1 uses `--sentence-iterations 70
  --num-boost-tokens 500 --max-sentence-length 256` and Phase 3 uses
  `--max-steps 256`. The budget sweep sets both length flags to 16, 32, 64
  or 128.

## Outputs

Each phase stores, for every instance, its reconstruction, the cosine
similarity to the target embedding and the number of tokens it sent to the
target model:

| stage | text | cosine similarity | query tokens |
|---|---|---|---|
| Phase 1 | `c1` | `c1_sim` | `kv_tokens` |
| Phase 2 | `c1_hc` | `c1_hc_sim` | `c1_hc_tokens` |
| Phase 3 | `refined_text` | `refined_cos_sim` | `refined_tokens` |

The cost reported in the paper is the sum of the three token columns.

## Credits

The Phase 1 generator is adapted from the implementation of "Zero-Shot Video
Captioning by Evolving Pseudo-tokens" (Tewel et al., BMVC 2023),
https://github.com/YoadTew/zero-shot-video-to-text. Phase 3 follows the
LLM-guided beam search of ZSInvert (Zhang et al., 2025). The ZSInvert baseline
itself is not part of this repository.

## Citation

```bibtex
@inproceedings{lapusan2027scribe,
  title     = {Compliant but Not Anonymous: Zero-Shot Inversion of Clinical Embeddings with SCRIBE},
  author    = {Lapusan, Alex-Mihai and Lemnaru, Camelia},
  booktitle = {Proceedings of EACL 2027},
  year      = {2027}
}
```

## Use of AI assistants

Coding assistants based on large language models were used during development
for boilerplate and templating work: argument parsing, CSV and checkpoint
plumbing, docstrings, and repetitive refactoring. The attack design, the
experimental protocol, and the analysis are the authors' own, and the authors
are responsible for the correctness of all code and results reported in the
paper.
